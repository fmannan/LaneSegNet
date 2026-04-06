"""
Group OpenLane-V2 segments by curvature and topology, then optionally emit
stratified train/val/test splits.

This script is designed for the raw OpenLane-V2 directory layout:

  {data_root}/{split}/{segment_id}/info/{timestamp}.json

It reads the per-frame lane centerlines and, when available, the official
lane-lane topology adjacency matrix:

  annotation.topology_lclc

Why segment-level grouping?
- Adjacent frames within the same segment are highly correlated.
- Splitting by frame leaks nearly identical geometry into train/val/test.
- Grouping and splitting by segment_id is the safer default.

Outputs
-------
The output directory receives:
- frame_summary.jsonl
  Fields include: segment_id, timestamp, relative_path, n_lanes,
  length_mean/min/max/std_m, straightness_mean/min/max, min_radius_m,
  min_radius_mean_m, x_min/max_m, y_min/max_m (spatial range), topology_*, etc.
- segment_summary.jsonl
  Fields include: segment_id, n_frames, n_lanes_mean, length_mean_m,
  straightness_mean, min_radius_m_p10, curvature_bucket, topology_bucket,
  combined_bucket, frame_paths, and many more percentile statistics.
- segment_groups.json
- dataset_stats.json
  Full distribution statistics (mean, std, p5/p25/p50/p75/p95/p99, min, max)
  for every lane-level and frame-level metric across the entire scanned split.
- stats_summary.txt
  Human-readable table of all dataset statistics, spatial range, and
  per-lane distributions — similar in spirit to debug_queries.py summary.txt.
- split_segments_{train,val,test}.json       (when --make_splits)
- split_frames_{train,val,test}.json         (when --make_splits)
- split_manifest.json                        (when --make_splits)

Examples
--------
python tools/group_openlane_by_geometry.py \
    --data_root ./data/OpenLane-V2 \
    --split train \
    --output_dir ./openlane_groups_train

python tools/group_openlane_by_geometry.py \
    --data_root ./data/OpenLane-V2 \
    --split train \
    --output_dir ./openlane_groups_train \
    --make_splits \
    --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Group OpenLane-V2 data by curvature and topology.")
    p.add_argument("--data_root", required=True, help="OpenLane-V2 root, e.g. ./data/OpenLane-V2")
    p.add_argument("--split", default="train", help="Dataset split directory to scan")
    p.add_argument("--output_dir", required=True, help="Directory for summaries and split JSON files")
    p.add_argument("--max_segments", type=int, default=None, help="Optional cap on number of segments to scan")
    p.add_argument("--max_frames", type=int, default=None, help="Optional cap on number of frames to scan total")
    p.add_argument("--seed", type=int, default=42, help="Random seed for split generation")
    p.add_argument("--make_splits", action="store_true", help="Emit stratified train/val/test segment splits")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--radius_straight_m", type=float, default=120.0)
    p.add_argument("--radius_gentle_m", type=float, default=50.0)
    p.add_argument("--radius_moderate_m", type=float, default=15.0)
    p.add_argument("--turn_straight_deg", type=float, default=15.0)
    p.add_argument("--turn_gentle_deg", type=float, default=45.0)
    p.add_argument("--turn_moderate_deg", type=float, default=90.0)
    return p.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")


def safe_mean(values: Sequence[float], default: float = float("nan")) -> float:
    vals = [float(v) for v in values if not math.isnan(float(v)) and not math.isinf(float(v))]
    return float(np.mean(vals)) if vals else default


def safe_max(values: Sequence[float], default: float = float("nan")) -> float:
    vals = [float(v) for v in values if not math.isnan(float(v)) and not math.isinf(float(v))]
    return float(np.max(vals)) if vals else default


def safe_min(values: Sequence[float], default: float = float("nan")) -> float:
    vals = [float(v) for v in values if not math.isnan(float(v)) and not math.isinf(float(v))]
    return float(np.min(vals)) if vals else default


def safe_percentile(values: Sequence[float], q: float, default: float = float("nan")) -> float:
    vals = [float(v) for v in values if not math.isnan(float(v)) and not math.isinf(float(v))]
    return float(np.percentile(vals, q)) if vals else default


def iter_annotation_files(data_root: Path, split: str) -> Iterable[Path]:
    split_dir = data_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    for segment_dir in sorted(split_dir.iterdir()):
        info_dir = segment_dir / "info"
        if not info_dir.exists():
            continue
        for json_path in sorted(info_dir.glob("*.json")):
            if json_path.stem.endswith("-ls"):
                continue
            yield json_path


def compute_curvature(pts_xy: np.ndarray) -> np.ndarray:
    if len(pts_xy) < 3:
        return np.zeros((0,), dtype=np.float32)
    xy = pts_xy[:, :2].astype(np.float64)
    a = xy[:-2]
    b = xy[1:-1]
    c = xy[2:]
    ab = b - a
    bc = c - b
    ac = c - a
    cross = np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0])
    len_ab = np.linalg.norm(ab, axis=1)
    len_bc = np.linalg.norm(bc, axis=1)
    len_ac = np.linalg.norm(ac, axis=1)
    kappa = 2.0 * cross / (len_ab * len_bc * len_ac + 1e-9)
    return kappa.astype(np.float32)


def polyline_length_m(pts_xy: np.ndarray) -> float:
    if len(pts_xy) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts_xy[:, :2], axis=0), axis=1).sum())


def polyline_chord_length_m(pts_xy: np.ndarray) -> float:
    if len(pts_xy) < 2:
        return 0.0
    return float(np.linalg.norm(pts_xy[-1, :2] - pts_xy[0, :2]))


def straightness_ratio(pts_xy: np.ndarray) -> float:
    arc = polyline_length_m(pts_xy)
    if arc < 1e-9:
        return 0.0
    chord = polyline_chord_length_m(pts_xy)
    return float(min(chord / arc, 1.0))


def total_turning_deg(pts_xy: np.ndarray) -> float:
    if len(pts_xy) < 3:
        return 0.0
    segs = pts_xy[1:, :2] - pts_xy[:-1, :2]
    norms = np.linalg.norm(segs, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-9
    if valid.sum() < 2:
        return 0.0
    unit = np.where(norms > 1e-9, segs / (norms + 1e-12), 0.0)
    dots = np.sum(unit[:-1] * unit[1:], axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.degrees(np.arccos(dots))
    return float(np.sum(np.abs(angles)))


def min_radius_m(pts_xy: np.ndarray, cap_m: float = 1e6) -> float:
    kappa = compute_curvature(pts_xy)
    if len(kappa) == 0:
        return cap_m
    kappa = np.clip(np.abs(kappa), 1.0 / cap_m, np.inf)
    return float(np.min(1.0 / kappa))


def lane_spatial_extent(lanes: List[dict]) -> dict:
    """
    Compute the spatial bounding box of all lane annotation points in a frame.
    Returns ego-frame x (forward) and y (lateral) ranges in metres.
    Used to determine actual annotation coverage vs the configured BEV window.
    """
    xs, ys = [], []
    for lane in lanes:
        pts = np.asarray(lane.get("points", []), dtype=np.float32)
        if pts.ndim == 2 and pts.shape[0] > 0 and pts.shape[1] >= 2:
            xs.append(pts[:, 0])
            ys.append(pts[:, 1])
    if not xs:
        nan = float("nan")
        return {"x_min_m": nan, "x_max_m": nan, "y_min_m": nan, "y_max_m": nan}
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    return {
        "x_min_m": float(x.min()),
        "x_max_m": float(x.max()),
        "y_min_m": float(y.min()),
        "y_max_m": float(y.max()),
    }


def lane_stats(lane: dict) -> dict:
    pts = np.asarray(lane.get("points", []), dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return {
            "length_m": 0.0,
            "chord_m": 0.0,
            "straightness": 0.0,
            "total_turning_deg": 0.0,
            "max_abs_curvature": 0.0,
            "mean_abs_curvature": 0.0,
            "min_radius_m": 1e6,
            "n_pts": 0,
            "x_start_m": float("nan"),
            "x_end_m": float("nan"),
            "y_start_m": float("nan"),
            "y_end_m": float("nan"),
            "is_connector": bool(lane.get("is_intersection_or_connector", False)),
        }
    kappa = np.abs(compute_curvature(pts[:, :2]))
    return {
        "length_m":          polyline_length_m(pts),
        "chord_m":           polyline_chord_length_m(pts),
        "straightness":      straightness_ratio(pts),
        "total_turning_deg": total_turning_deg(pts),
        "max_abs_curvature": float(np.max(kappa)) if len(kappa) else 0.0,
        "mean_abs_curvature":float(np.mean(kappa)) if len(kappa) else 0.0,
        "min_radius_m":      min_radius_m(pts),
        "n_pts":             int(pts.shape[0]),
        # First and last annotation point in ego-frame x (forward) and y (lateral)
        "x_start_m":         float(pts[0, 0]),
        "x_end_m":           float(pts[-1, 0]),
        "y_start_m":         float(pts[0, 1]),
        "y_end_m":           float(pts[-1, 1]),
        "is_connector":      bool(lane.get("is_intersection_or_connector", False)),
    }


def parse_topology_lclc(annotation: dict, n_lanes: int) -> Tuple[Optional[np.ndarray], str]:
    raw = annotation.get("topology_lclc")
    if raw is None:
        return None, "missing"
    arr = np.asarray(raw, dtype=np.int32)
    if arr.ndim != 2:
        return None, "invalid_shape"
    if arr.shape[0] != n_lanes or arr.shape[1] != n_lanes:
        return None, "size_mismatch"
    return arr, "ok"


def frame_topology_stats(annotation: dict, lanes: List[dict]) -> dict:
    n_lanes = len(lanes)
    topo, topo_status = parse_topology_lclc(annotation, n_lanes)
    connector_count = int(sum(bool(l.get("is_intersection_or_connector", False)) for l in lanes))
    if topo is None:
        return {
            "topology_status": topo_status,
            "topology_available": False,
            "has_merge": False,
            "has_split": False,
            "n_merge_nodes": 0,
            "n_split_nodes": 0,
            "n_branch_nodes": 0,
            "max_in_degree": 0,
            "max_out_degree": 0,
            "connector_count": connector_count,
        }
    out_degree = topo.sum(axis=1)
    in_degree = topo.sum(axis=0)
    n_merge_nodes = int(np.sum(in_degree > 1))
    n_split_nodes = int(np.sum(out_degree > 1))
    n_branch_nodes = int(np.sum((in_degree > 1) | (out_degree > 1)))
    return {
        "topology_status": topo_status,
        "topology_available": True,
        "has_merge": bool(np.any(in_degree > 1)),
        "has_split": bool(np.any(out_degree > 1)),
        "n_merge_nodes": n_merge_nodes,
        "n_split_nodes": n_split_nodes,
        "n_branch_nodes": n_branch_nodes,
        "max_in_degree": int(in_degree.max()) if len(in_degree) else 0,
        "max_out_degree": int(out_degree.max()) if len(out_degree) else 0,
        "connector_count": connector_count,
    }


def classify_curvature_bucket(
    min_radius_value: float,
    max_turn_value: float,
    args: argparse.Namespace,
) -> str:
    if min_radius_value >= args.radius_straight_m and max_turn_value <= args.turn_straight_deg:
        return "straight"
    if min_radius_value >= args.radius_gentle_m and max_turn_value <= args.turn_gentle_deg:
        return "gentle"
    if min_radius_value >= args.radius_moderate_m and max_turn_value <= args.turn_moderate_deg:
        return "moderate"
    return "sharp"


def classify_topology_bucket(
    has_merge: bool,
    has_split: bool,
    connector_fraction: float,
    topology_available_fraction: float,
) -> str:
    if has_merge and has_split:
        return "merge_and_split"
    if has_merge:
        return "merge"
    if has_split:
        return "split"
    if connector_fraction > 0.2:
        return "connector_heavy"
    if topology_available_fraction == 0.0:
        return "topology_missing"
    return "simple"


def build_frame_summary(json_path: Path) -> Tuple[dict, List[dict]]:
    """
    Returns (frame_row, per_lane_stats_list).
    per_lane_stats_list is used by the caller to accumulate dataset-wide
    distributions without bloating the JSONL output.
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    annotation = data.get("annotation", {})
    lanes = annotation.get("lane_centerline", [])
    lane_metrics = [lane_stats(lane) for lane in lanes]
    topo = frame_topology_stats(annotation, lanes)
    extent = lane_spatial_extent(lanes)

    lengths = [m["length_m"] for m in lane_metrics]
    straightnesses = [m["straightness"] for m in lane_metrics]

    frame_row = {
        "segment_id":          data.get("segment_id", json_path.parent.parent.name),
        "timestamp":           str(data.get("timestamp")),
        "relative_path":       json_path.as_posix(),
        "n_lanes":             len(lanes),
        "connector_count":     topo["connector_count"],
        # Spatial range of all annotation points in this frame
        "x_min_m":             extent["x_min_m"],
        "x_max_m":             extent["x_max_m"],
        "y_min_m":             extent["y_min_m"],
        "y_max_m":             extent["y_max_m"],
        # Lane length distribution within frame
        "length_mean_m":       safe_mean(lengths, default=0.0),
        "length_min_m":        safe_min(lengths, default=0.0),
        "length_max_m":        safe_max(lengths, default=0.0),
        "length_std_m":        float(np.std(lengths)) if lengths else 0.0,
        # Straightness distribution within frame
        "straightness_mean":   safe_mean(straightnesses, default=0.0),
        "straightness_min":    safe_min(straightnesses, default=0.0),
        "straightness_max":    safe_max(straightnesses, default=0.0),
        # Curvature / turning
        "max_total_turning_deg":  safe_max([m["total_turning_deg"] for m in lane_metrics], default=0.0),
        "mean_abs_curvature":     safe_mean([m["mean_abs_curvature"] for m in lane_metrics], default=0.0),
        "max_abs_curvature":      safe_max([m["max_abs_curvature"] for m in lane_metrics], default=0.0),
        "min_radius_m":           safe_min([m["min_radius_m"] for m in lane_metrics], default=1e6),
        "min_radius_mean_m":      safe_mean([m["min_radius_m"] for m in lane_metrics], default=1e6),
        **topo,
    }
    return frame_row, lane_metrics


def aggregate_segment(segment_id: str, frames: List[dict], args: argparse.Namespace) -> dict:
    connector_fraction = sum(f["connector_count"] for f in frames) / max(sum(f["n_lanes"] for f in frames), 1)
    topology_available_fraction = safe_mean([1.0 if f["topology_available"] else 0.0 for f in frames], default=0.0)
    curvature_bucket = classify_curvature_bucket(
        min_radius_value=safe_percentile([f["min_radius_m"] for f in frames], 10, default=1e6),
        max_turn_value=safe_max([f["max_total_turning_deg"] for f in frames], default=0.0),
        args=args,
    )
    topology_bucket = classify_topology_bucket(
        has_merge=any(f["has_merge"] for f in frames),
        has_split=any(f["has_split"] for f in frames),
        connector_fraction=connector_fraction,
        topology_available_fraction=topology_available_fraction,
    )
    combined_bucket = f"{curvature_bucket}__{topology_bucket}"
    return {
        "segment_id": segment_id,
        "n_frames": len(frames),
        "n_lanes_mean": safe_mean([f["n_lanes"] for f in frames], default=0.0),
        "connector_fraction": connector_fraction,
        "topology_available_fraction": topology_available_fraction,
        "has_merge": any(f["has_merge"] for f in frames),
        "has_split": any(f["has_split"] for f in frames),
        "max_in_degree": int(safe_max([f["max_in_degree"] for f in frames], default=0.0)),
        "max_out_degree": int(safe_max([f["max_out_degree"] for f in frames], default=0.0)),
        "max_branch_nodes": int(safe_max([f["n_branch_nodes"] for f in frames], default=0.0)),
        "length_mean_m": safe_mean([f["length_mean_m"] for f in frames], default=0.0),
        "straightness_mean": safe_mean([f["straightness_mean"] for f in frames], default=0.0),
        "mean_abs_curvature_p90": safe_percentile([f["mean_abs_curvature"] for f in frames], 90, default=0.0),
        "max_abs_curvature_p90": safe_percentile([f["max_abs_curvature"] for f in frames], 90, default=0.0),
        "min_radius_m_p10": safe_percentile([f["min_radius_m"] for f in frames], 10, default=1e6),
        "max_total_turning_deg": safe_max([f["max_total_turning_deg"] for f in frames], default=0.0),
        "curvature_bucket": curvature_bucket,
        "topology_bucket": topology_bucket,
        "combined_bucket": combined_bucket,
        "frame_paths": [f["relative_path"] for f in frames],
        "straightness_mean_p10": safe_percentile([f["straightness_mean"] for f in frames], 10, default=0.0),
        "straightness_mean_p90": safe_percentile([f["straightness_mean"] for f in frames], 90, default=0.0),
        "n_lanes_p10":           safe_percentile([f["n_lanes"] for f in frames], 10, default=0.0),
        "n_lanes_p90":           safe_percentile([f["n_lanes"] for f in frames], 90, default=0.0),
        "length_p10_m":          safe_percentile([f["length_mean_m"] for f in frames], 10, default=0.0),
        "length_p90_m":          safe_percentile([f["length_mean_m"] for f in frames], 90, default=0.0),
    }


def stratified_segment_split(
    segments: List[dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[dict]]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")
    rng = random.Random(seed)
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for seg in segments:
        buckets[seg["combined_bucket"]].append(seg)

    splits = {"train": [], "val": [], "test": []}
    for bucket_name, bucket_segments in sorted(buckets.items()):
        bucket_copy = list(bucket_segments)
        rng.shuffle(bucket_copy)
        n = len(bucket_copy)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        n_test = n - n_train - n_val
        if n > 0 and n_train == 0 and train_ratio > 0:
            n_train = 1
        if n_train + n_val + n_test > n:
            overflow = n_train + n_val + n_test - n
            n_test = max(0, n_test - overflow)
        while n_train + n_val + n_test < n:
            if train_ratio >= max(val_ratio, test_ratio):
                n_train += 1
            elif val_ratio >= test_ratio:
                n_val += 1
            else:
                n_test += 1
        splits["train"].extend(bucket_copy[:n_train])
        splits["val"].extend(bucket_copy[n_train:n_train + n_val])
        splits["test"].extend(bucket_copy[n_train + n_val:n_train + n_val + n_test])
        print(
            f"[split] bucket={bucket_name:<24} n={n:4d} -> "
            f"train={n_train:4d} val={n_val:4d} test={n_test:4d}"
        )
        if n > 0 and (n_val == 0 or n_test == 0) and (val_ratio > 0 or test_ratio > 0):
            print(f"[split] WARNING: bucket={bucket_name!r} has only {n} segment(s) — val={n_val} test={n_test} (too small for full stratification)")
    return splits


def distribution_stats(values: Sequence[float]) -> dict:
    """Return a comprehensive stats dict for a flat list of numeric values."""
    vals = [float(v) for v in values if not math.isnan(float(v)) and not math.isinf(float(v))]
    if not vals:
        nan = float("nan")
        return {k: nan for k in ("mean", "std", "min", "p5", "p25", "p50", "p75", "p95", "p99", "max", "count")}
    a = np.array(vals)
    return {
        "count": int(len(a)),
        "mean":  float(np.mean(a)),
        "std":   float(np.std(a)),
        "min":   float(np.min(a)),
        "p5":    float(np.percentile(a,  5)),
        "p25":   float(np.percentile(a, 25)),
        "p50":   float(np.percentile(a, 50)),
        "p75":   float(np.percentile(a, 75)),
        "p95":   float(np.percentile(a, 95)),
        "p99":   float(np.percentile(a, 99)),
        "max":   float(np.max(a)),
    }


def compute_dataset_stats(frame_rows: List[dict], all_lane_metrics: List[dict]) -> dict:
    """
    Compute dataset-wide distribution statistics from accumulated frame and
    per-lane data.  Returns a dict written to dataset_stats.json.
    """
    def frame_vals(key):
        return [f[key] for f in frame_rows if not math.isnan(float(f.get(key, float("nan"))))]

    def lane_vals(key):
        return [m[key] for m in all_lane_metrics if not math.isnan(float(m.get(key, float("nan"))))]

    return {
        # ── Spatial range ──────────────────────────────────────────────────
        # Distribution of per-frame min/max coordinates tells us how far
        # annotations actually extend relative to the ego vehicle.
        "spatial_range": {
            "x_min_m":  distribution_stats(frame_vals("x_min_m")),   # how far behind
            "x_max_m":  distribution_stats(frame_vals("x_max_m")),   # how far ahead
            "y_min_m":  distribution_stats(frame_vals("y_min_m")),   # how far right
            "y_max_m":  distribution_stats(frame_vals("y_max_m")),   # how far left
        },
        # ── Per-frame ──────────────────────────────────────────────────────
        "frame": {
            "n_lanes":              distribution_stats(frame_vals("n_lanes")),
            "length_mean_m":        distribution_stats(frame_vals("length_mean_m")),
            "straightness_mean":    distribution_stats(frame_vals("straightness_mean")),
            "min_radius_m":         distribution_stats(frame_vals("min_radius_m")),
            "min_radius_mean_m":    distribution_stats(frame_vals("min_radius_mean_m")),
            "max_total_turning_deg":distribution_stats(frame_vals("max_total_turning_deg")),
        },
        # ── Per-lane (individual lanes, not per-frame means) ───────────────
        "lane": {
            "length_m":           distribution_stats(lane_vals("length_m")),
            "chord_m":            distribution_stats(lane_vals("chord_m")),
            "straightness":       distribution_stats(lane_vals("straightness")),
            "total_turning_deg":  distribution_stats(lane_vals("total_turning_deg")),
            "mean_abs_curvature": distribution_stats(lane_vals("mean_abs_curvature")),
            "max_abs_curvature":  distribution_stats(lane_vals("max_abs_curvature")),
            "min_radius_m":       distribution_stats(lane_vals("min_radius_m")),
            "n_pts":              distribution_stats(lane_vals("n_pts")),
            # Endpoint X positions reveal where lanes start/end relative to ego
            "x_start_m":         distribution_stats(lane_vals("x_start_m")),
            "x_end_m":           distribution_stats(lane_vals("x_end_m")),
            "y_start_m":         distribution_stats(lane_vals("y_start_m")),
            "y_end_m":           distribution_stats(lane_vals("y_end_m")),
        },
    }


def write_stats_summary(
    stats: dict,
    frame_rows: List[dict],
    segment_rows: List[dict],
    out_dir: Path,
    split: str,
) -> None:
    """
    Write stats_summary.txt — a human-readable overview of all dataset
    statistics, analogous to debug_queries.py summary.txt.
    """
    lines: List[str] = []
    W = 100

    def section(title: str) -> None:
        lines.append("=" * W)
        lines.append(f"  {title}")
        lines.append("=" * W)

    def sub(title: str) -> None:
        lines.append(f"\n  ── {title} ──")

    n_frames   = len(frame_rows)
    n_segments = len(segment_rows)
    n_lanes    = int(stats["lane"]["length_m"]["count"])

    section(f"OpenLane-V2 Dataset Statistics  |  split={split}  |  {n_frames:,} frames  {n_segments:,} segments  {n_lanes:,} lanes")

    # ── Spatial range ─────────────────────────────────────────────────────
    section("SPATIAL RANGE  (annotation bounding box per frame, ego frame metres)")
    sr = stats["spatial_range"]
    lines.append(f"\n  Axis     Direction    p5      p25     p50     p75     p95     p99     min     max")
    lines.append(f"  {'─'*88}")

    def range_row(d):
        return (
            f"{d['p5']:7.1f}  {d['p25']:7.1f}  {d['p50']:7.1f}  "
            f"{d['p75']:7.1f}  {d['p95']:7.1f}  {d['p99']:7.1f}  "
            f"{d['min']:7.1f}  {d['max']:7.1f}"
        )

    lines.append(f"\n  Per-frame extremes (worst/best case in each frame):")
    lines.append(f"  x_min (rear)   {range_row(sr['x_min_m'])}")
    lines.append(f"  x_max (front)  {range_row(sr['x_max_m'])}")
    lines.append(f"  y_min (right)  {range_row(sr['y_min_m'])}")
    lines.append(f"  y_max (left)   {range_row(sr['y_max_m'])}")

    x_min_p5  = sr["x_min_m"]["p5"]
    x_max_p95 = sr["x_max_m"]["p95"]
    y_min_p5  = sr["y_min_m"]["p5"]
    y_max_p95 = sr["y_max_m"]["p95"]
    lines.append(f"\n  Recommended BEV window (covers 90 % of annotation points):")
    lines.append(f"    x_range: [{x_min_p5:.0f}, {x_max_p95:.0f}] m   (p5 rear → p95 front)")
    lines.append(f"    y_range: [{y_min_p5:.0f}, {y_max_p95:.0f}] m   (p5 right → p95 left)")
    lines.append(f"\n  Absolute extremes across all frames:")
    lines.append(f"    x: [{sr['x_min_m']['min']:.1f}, {sr['x_max_m']['max']:.1f}] m")
    lines.append(f"    y: [{sr['y_min_m']['min']:.1f}, {sr['y_max_m']['max']:.1f}] m")

    # ── Lane endpoint distribution ─────────────────────────────────────────
    section("LANE ENDPOINT POSITIONS  (where lanes start/end in ego frame)")
    le = stats["lane"]
    lines.append(f"\n  {'Metric':<22}  {'mean':>7}  {'std':>7}  {'p5':>7}  {'p25':>7}  {'p50':>7}  {'p75':>7}  {'p95':>7}  {'p99':>7}")
    lines.append(f"  {'─'*90}")

    def stat_row(label, d, fmt=".1f"):
        f = fmt
        return (
            f"  {label:<22}  {d['mean']:7{f}}  {d['std']:7{f}}  {d['p5']:7{f}}  "
            f"{d['p25']:7{f}}  {d['p50']:7{f}}  {d['p75']:7{f}}  {d['p95']:7{f}}  {d['p99']:7{f}}"
        )

    lines.append(stat_row("x_start_m (fwd)",  le["x_start_m"]))
    lines.append(stat_row("x_end_m (fwd)",    le["x_end_m"]))
    lines.append(stat_row("y_start_m (lat)",  le["y_start_m"]))
    lines.append(stat_row("y_end_m (lat)",    le["y_end_m"]))

    # ── Per-frame distribution ─────────────────────────────────────────────
    section("PER-FRAME STATISTICS")
    fr = stats["frame"]
    lines.append(f"\n  {'Metric':<26}  {'mean':>7}  {'std':>7}  {'p5':>7}  {'p25':>7}  {'p50':>7}  {'p75':>7}  {'p95':>7}  {'p99':>7}")
    lines.append(f"  {'─'*94}")
    lines.append(stat_row("n_lanes",                fr["n_lanes"],              fmt=".1f"))
    lines.append(stat_row("length_mean_m",          fr["length_mean_m"],        fmt=".1f"))
    lines.append(stat_row("straightness_mean",      fr["straightness_mean"],    fmt=".3f"))
    lines.append(stat_row("min_radius_m (frame)",   fr["min_radius_m"],         fmt=".1f"))
    lines.append(stat_row("min_radius_mean_m",      fr["min_radius_mean_m"],    fmt=".1f"))
    lines.append(stat_row("max_turning_deg",        fr["max_total_turning_deg"],fmt=".1f"))

    # ── Per-lane distribution ──────────────────────────────────────────────
    section(f"PER-LANE STATISTICS  ({n_lanes:,} individual lanes)")
    lines.append(f"\n  {'Metric':<24}  {'mean':>8}  {'std':>8}  {'p5':>8}  {'p25':>8}  {'p50':>8}  {'p75':>8}  {'p95':>8}  {'p99':>8}")
    lines.append(f"  {'─'*98}")

    def lane_row(label, key, fmt=".2f"):
        d = le[key]
        f = fmt
        return (
            f"  {label:<24}  {d['mean']:8{f}}  {d['std']:8{f}}  {d['p5']:8{f}}  "
            f"{d['p25']:8{f}}  {d['p50']:8{f}}  {d['p75']:8{f}}  {d['p95']:8{f}}  {d['p99']:8{f}}"
        )

    lines.append(lane_row("arc length (m)",      "length_m",           fmt=".1f"))
    lines.append(lane_row("chord length (m)",    "chord_m",            fmt=".1f"))
    lines.append(lane_row("straightness",        "straightness",       fmt=".3f"))
    lines.append(lane_row("total turning (deg)", "total_turning_deg",  fmt=".1f"))
    lines.append(lane_row("mean |κ| (1/m)",      "mean_abs_curvature", fmt=".4f"))
    lines.append(lane_row("max  |κ| (1/m)",      "max_abs_curvature",  fmt=".4f"))
    lines.append(lane_row("min radius (m)",      "min_radius_m",       fmt=".1f"))
    lines.append(lane_row("n annotation pts",    "n_pts",              fmt=".0f"))

    # ── Bucket summary ─────────────────────────────────────────────────────
    section("BUCKET DISTRIBUTION")
    from collections import Counter
    curv_counts = Counter(f.get("curvature_bucket", "?") for f in segment_rows)
    topo_counts = Counter(f.get("topology_bucket", "?")  for f in segment_rows)
    comb_counts = Counter(f.get("combined_bucket", "?")  for f in segment_rows)

    sub("Curvature buckets (segments)")
    for k, v in sorted(curv_counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * v / max(n_segments, 1)
        lines.append(f"    {k:<30}  {v:5d}  ({pct:5.1f} %)")

    sub("Topology buckets (segments)")
    for k, v in sorted(topo_counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * v / max(n_segments, 1)
        lines.append(f"    {k:<30}  {v:5d}  ({pct:5.1f} %)")

    sub("Combined buckets (top 15 by count)")
    for k, v in sorted(comb_counts.items(), key=lambda x: -x[1])[:15]:
        pct = 100.0 * v / max(n_segments, 1)
        lines.append(f"    {k:<40}  {v:5d}  ({pct:5.1f} %)")

    lines.append("\n" + "=" * W)
    txt = "\n".join(lines)
    (out_dir / "stats_summary.txt").write_text(txt, encoding="utf-8")
    print(txt)


def build_manifest(splits: Dict[str, List[dict]]) -> dict:
    manifest = {}
    for split_name, segments in splits.items():
        manifest[split_name] = {
            "n_segments": len(segments),
            "n_frames": int(sum(seg["n_frames"] for seg in segments)),
            "bucket_counts": dict(Counter(seg["combined_bucket"] for seg in segments)),
            "segment_ids": [seg["segment_id"] for seg in segments],
        }
    return manifest


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_rows: List[dict] = []
    all_lane_metrics: List[dict] = []
    frames_by_segment: Dict[str, List[dict]] = defaultdict(list)

    total_frames = 0
    scanned_segments = set()
    for json_path in iter_annotation_files(data_root, args.split):
        segment_id = json_path.parent.parent.name
        if args.max_segments is not None and segment_id not in scanned_segments and len(scanned_segments) >= args.max_segments:
            break
        scanned_segments.add(segment_id)
        frame, lane_metrics = build_frame_summary(json_path)
        try:
            frame["relative_path"] = json_path.relative_to(data_root).as_posix()
        except ValueError:
            pass  # keep absolute path as fallback
        frame_rows.append(frame)
        all_lane_metrics.extend(lane_metrics)
        frames_by_segment[frame["segment_id"]].append(frame)
        total_frames += 1
        if args.max_frames is not None and total_frames >= args.max_frames:
            break
        if total_frames % 500 == 0:
            print(f"[scan] processed {total_frames} frames across {len(frames_by_segment)} segments", flush=True)

    segment_rows = [
        aggregate_segment(segment_id, frames, args)
        for segment_id, frames in sorted(frames_by_segment.items())
    ]
    segment_rows.sort(key=lambda row: row["segment_id"])

    segment_groups: Dict[str, List[str]] = defaultdict(list)
    for seg in segment_rows:
        segment_groups[seg["combined_bucket"]].append(seg["segment_id"])

    write_jsonl(out_dir / "frame_summary.jsonl", frame_rows)
    write_jsonl(out_dir / "segment_summary.jsonl", segment_rows)
    write_json(out_dir / "segment_groups.json", dict(sorted(segment_groups.items())))

    # Dataset-wide distribution statistics
    stats = compute_dataset_stats(frame_rows, all_lane_metrics)
    write_json(out_dir / "dataset_stats.json", stats)
    write_stats_summary(stats, frame_rows, segment_rows, out_dir, args.split)

    print(f"[done] wrote {len(frame_rows)} frame rows, {len(segment_rows)} segment rows, "
          f"{len(all_lane_metrics)} lane records to {out_dir}")

    if args.make_splits:
        splits = stratified_segment_split(
            segment_rows,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        manifest = build_manifest(splits)
        for split_name, segs in splits.items():
            write_json(out_dir / f"split_segments_{split_name}.json", [seg["segment_id"] for seg in segs])
            write_json(
                out_dir / f"split_frames_{split_name}.json",
                [path for seg in segs for path in seg["frame_paths"]],
            )
        write_json(out_dir / "split_manifest.json", manifest)
        print(f"[done] wrote split manifest to {out_dir / 'split_manifest.json'}")


if __name__ == "__main__":
    main()
