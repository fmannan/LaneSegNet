"""
Visualize the outputs of group_openlane_by_geometry.py.

Reads:
- segment_summary.jsonl
- optional split_manifest.json

Writes PNG charts and a short text summary into the requested output dir:
- curvature_buckets.png          segment count per curvature bucket
- topology_buckets.png           segment count per topology bucket
- combined_buckets.png           segment count per combined bucket (sorted by count)
- curvature_buckets_fw.png       curvature bucket counts weighted by frame count
- topology_buckets_fw.png        topology bucket counts weighted by frame count
- min_radius_hist.png            distribution of min_radius_m_p10 (clipped at 500 m)
- max_turn_hist.png              distribution of max_total_turning_deg
- n_lanes_hist.png               distribution of mean lane count per frame
- length_hist.png                distribution of mean lane length (clipped at 80 m)
- straightness_hist.png          distribution of mean straightness ratio
- topology_coverage.png          distribution of topology_available_fraction
- branch_degree_hist.png         overlaid max in/out degree histograms
- curvature_topology_scatter.png min_radius vs max_turn coloured by topology bucket
- split_bucket_heatmap.png       per-split bucket counts (when split_manifest.json present)
- summary.txt                    text statistics

Example
-------
python tools/visualize_openlane_groups.py \
    --group_dir ./openlane_groups_train \
    --output_dir ./openlane_groups_train/viz
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize OpenLane grouping outputs.")
    p.add_argument("--group_dir", required=True, help="Directory containing segment_summary.jsonl")
    p.add_argument("--output_dir", required=True, help="Directory for plots")
    return p.parse_args()


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_bar(counter: Counter, title: str, xlabel: str, out_path: Path, rotate: bool = False) -> None:
    labels = list(counter.keys())
    values = [counter[k] for k in labels]
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(labels)), 5))
    ax.bar(range(len(labels)), values, color="#3a78b8")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45 if rotate else 0, ha="right" if rotate else "center")
    ax.set_ylabel("Segments")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_hist(values: np.ndarray, bins: int, title: str, xlabel: str, out_path: Path, xlim=None, vlines=None) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(values, bins=bins, color="#d4832f", edgecolor="black", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Segments")
    ax.grid(axis="y", alpha=0.25)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if vlines:
        for vx in vlines:
            ax.axvline(vx, color="red", linestyle="--", linewidth=0.9, alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_scatter(x: np.ndarray, y: np.ndarray, labels: List[str], out_path: Path) -> None:
    unique = list(dict.fromkeys(labels))
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(unique), 1)))
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique)}

    x = np.clip(x, 0, 500)
    fig, ax = plt.subplots(figsize=(7, 6))
    for label in unique:
        mask = np.array([l == label for l in labels], dtype=bool)
        ax.scatter(
            x[mask], y[mask],
            s=32,
            alpha=0.85,
            color=color_map[label],
            label=label,
        )
    ax.set_xlim(0, 500)
    ax.set_xlabel("min_radius_m_p10  (clipped at 500 m)")
    ax.set_ylabel("max_total_turning_deg")
    ax.set_title("Curvature Space By Topology Bucket")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_split_heatmap(split_manifest: dict, out_path: Path) -> None:
    split_names = [name for name in ("train", "val", "test") if name in split_manifest]
    buckets = sorted({bucket for split in split_manifest.values() for bucket in split.get("bucket_counts", {})})
    if not split_names or not buckets:
        return
    matrix = np.zeros((len(split_names), len(buckets)), dtype=np.float32)
    for i, split_name in enumerate(split_names):
        counts = split_manifest[split_name].get("bucket_counts", {})
        for j, bucket in enumerate(buckets):
            matrix[i, j] = counts.get(bucket, 0)

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(buckets)), 3 + 0.4 * len(split_names)))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(buckets)))
    ax.set_xticklabels(buckets, rotation=45, ha="right")
    ax.set_yticks(range(len(split_names)))
    ax.set_yticklabels(split_names)
    ax.set_title("Bucket Counts Per Split")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black", fontsize=8)
    plt.colorbar(im, ax=ax, label="Segments")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def write_summary(rows: List[dict], split_manifest: dict, out_path: Path) -> None:
    curvature_counts = Counter(row["curvature_bucket"] for row in rows)
    topology_counts = Counter(row["topology_bucket"] for row in rows)
    combined_counts = Counter(row["combined_bucket"] for row in rows)

    min_radius = np.array([row["min_radius_m_p10"] for row in rows], dtype=np.float32)
    max_turn = np.array([row["max_total_turning_deg"] for row in rows], dtype=np.float32)

    lines = []
    lines.append("OpenLane Grouping Summary")
    lines.append("")
    lines.append(f"Segments: {len(rows)}")
    lines.append(f"Curvature buckets: {dict(curvature_counts)}")
    lines.append(f"Topology buckets: {dict(topology_counts)}")
    lines.append(f"Combined buckets: {dict(combined_counts)}")
    lines.append("")
    if len(rows):
        lines.append(
            "min_radius_m_p10: "
            f"mean={float(np.mean(min_radius)):.3f} "
            f"median={float(np.median(min_radius)):.3f} "
            f"p90={float(np.percentile(min_radius, 90)):.3f}"
        )
        lines.append(
            "max_total_turning_deg: "
            f"mean={float(np.mean(max_turn)):.3f} "
            f"median={float(np.median(max_turn)):.3f} "
            f"p90={float(np.percentile(max_turn, 90)):.3f}"
        )
        straight_arr2 = np.array([row.get("straightness_mean", 0) for row in rows], dtype=np.float32)
        n_lanes_arr2  = np.array([row.get("n_lanes_mean", 0) for row in rows], dtype=np.float32)
        length_arr2   = np.array([row.get("length_mean_m", 0) for row in rows], dtype=np.float32)
        topo_cov2     = np.array([row.get("topology_available_fraction", 0) for row in rows], dtype=np.float32)
        total_frames  = sum(row.get("n_frames", 0) for row in rows)
        lines.append(f"Total frames: {total_frames}")
        lines.append(f"straightness_mean: mean={float(np.mean(straight_arr2)):.3f} median={float(np.median(straight_arr2)):.3f} p10={float(np.percentile(straight_arr2, 10)):.3f}")
        lines.append(f"n_lanes_mean: mean={float(np.mean(n_lanes_arr2)):.2f} median={float(np.median(n_lanes_arr2)):.2f} p90={float(np.percentile(n_lanes_arr2, 90)):.2f}")
        lines.append(f"length_mean_m: mean={float(np.mean(length_arr2)):.1f} median={float(np.median(length_arr2)):.1f}")
        lines.append(f"topology_available_fraction: mean={float(np.mean(topo_cov2)):.3f}")
    if split_manifest:
        lines.append("")
        lines.append("Split manifest:")
        for split_name, payload in split_manifest.items():
            lines.append(
                f"  {split_name}: "
                f"segments={payload.get('n_segments', 0)} "
                f"frames={payload.get('n_frames', 0)} "
                f"buckets={payload.get('bucket_counts', {})}"
            )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_dual_hist(x1, x2, label1, label2, title, xlabel, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    max_val = int(max(x1.max() if len(x1) else 0, x2.max() if len(x2) else 0)) + 2
    bins = range(0, max_val + 1)
    ax.hist(x1, bins=bins, alpha=0.6, color="#3a78b8", label=label1)
    ax.hist(x2, bins=bins, alpha=0.6, color="#d4832f", label=label2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Segments")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    group_dir = Path(args.group_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(group_dir / "segment_summary.jsonl")
    split_manifest_path = group_dir / "split_manifest.json"
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8")) if split_manifest_path.exists() else {}

    curvature_counts = Counter(row["curvature_bucket"] for row in rows)
    topology_counts = Counter(row["topology_bucket"] for row in rows)
    combined_counts = Counter(row["combined_bucket"] for row in rows)

    save_bar(curvature_counts, "Segments Per Curvature Bucket", "Curvature bucket", out_dir / "curvature_buckets.png")
    save_bar(topology_counts, "Segments Per Topology Bucket", "Topology bucket", out_dir / "topology_buckets.png")
    combined_counts = Counter(dict(sorted(combined_counts.items(), key=lambda kv: -kv[1])))
    save_bar(combined_counts, "Segments Per Combined Bucket", "Combined bucket", out_dir / "combined_buckets.png", rotate=True)

    curv_fw = Counter()
    topo_fw = Counter()
    for row in rows:
        nf = row.get("n_frames", 1)
        curv_fw[row["curvature_bucket"]] += nf
        topo_fw[row["topology_bucket"]] += nf
    save_bar(curv_fw, "Segments Per Curvature Bucket (frame-weighted)", "Curvature bucket", out_dir / "curvature_buckets_fw.png")
    save_bar(topo_fw, "Segments Per Topology Bucket (frame-weighted)", "Topology bucket", out_dir / "topology_buckets_fw.png")

    min_radius = np.array([row["min_radius_m_p10"] for row in rows], dtype=np.float32)
    max_turn = np.array([row["max_total_turning_deg"] for row in rows], dtype=np.float32)
    if len(rows):
        save_hist(np.clip(min_radius, 0, 500), bins=min(20, max(5, len(rows))), title="Segment min_radius_m_p10 (clipped at 500 m)", xlabel="metres", out_path=out_dir / "min_radius_hist.png", xlim=(0, 500))
        save_hist(max_turn, bins=min(20, max(5, len(rows))), title="Segment max_total_turning_deg", xlabel="degrees", out_path=out_dir / "max_turn_hist.png")
        save_scatter(min_radius, max_turn, [row["topology_bucket"] for row in rows], out_dir / "curvature_topology_scatter.png")

        n_lanes_arr = np.array([row.get("n_lanes_mean", 0) for row in rows], dtype=np.float32)
        length_arr  = np.array([row.get("length_mean_m", 0) for row in rows], dtype=np.float32)
        straight_arr = np.array([row.get("straightness_mean", 0) for row in rows], dtype=np.float32)
        topo_cov_arr = np.array([row.get("topology_available_fraction", 0) for row in rows], dtype=np.float32)
        max_in_arr   = np.array([row.get("max_in_degree", 0) for row in rows], dtype=np.float32)
        max_out_arr  = np.array([row.get("max_out_degree", 0) for row in rows], dtype=np.float32)

        save_hist(n_lanes_arr, bins=20, title="Mean lane count per frame", xlabel="Lanes per frame", out_path=out_dir / "n_lanes_hist.png")
        save_hist(np.clip(length_arr, 0, 80), bins=20, title="Mean lane length (m, clipped at 80 m)", xlabel="Length (m)", out_path=out_dir / "length_hist.png", xlim=(0, 80))
        save_hist(straight_arr, bins=20, title="Mean straightness ratio", xlabel="Straightness (0=squiggle, 1=straight)", out_path=out_dir / "straightness_hist.png", xlim=(0, 1), vlines=[0.3, 0.8])
        save_hist(topo_cov_arr, bins=10, title="Topology coverage (fraction of frames with valid topology)", xlabel="Fraction", out_path=out_dir / "topology_coverage.png", xlim=(0, 1))
        save_dual_hist(max_in_arr, max_out_arr, "max_in_degree", "max_out_degree",
                       "Max topology in/out degree per segment", "Degree",
                       out_dir / "branch_degree_hist.png")

    if split_manifest:
        save_split_heatmap(split_manifest, out_dir / "split_bucket_heatmap.png")

    write_summary(rows, split_manifest, out_dir / "summary.txt")
    print(f"[viz] wrote charts to {out_dir}")


if __name__ == "__main__":
    main()
