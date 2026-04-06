"""
Sample random OpenLane-V2 frames from each grouping category and render:
- front-camera image (raw, no overlay)
- BEV GT lane view

This consumes the outputs from group_openlane_by_geometry.py.
It reads the raw OpenLane-V2 per-frame JSON files directly (no mmdet3d
pipeline required), so it can run without a trained model or config file.

Path format: relative_path in frame_summary.jsonl is stored relative to
data_root (as produced by group_openlane_by_geometry.py).

Example
-------
python tools/sample_group_images.py \
    --group_dir ./openlane_groups_train \
    --data_root ./data/OpenLane-V2 \
    --output_dir ./openlane_groups_train/samples \
    --samples_per_category 8
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample grouped OpenLane frames and render GT overlays.")
    p.add_argument("--group_dir", required=True, help="Directory containing frame_summary.jsonl and segment_summary.jsonl")
    p.add_argument("--data_root", required=True, help="OpenLane-V2 root")
    p.add_argument("--output_dir", required=True, help="Where to write sample images")
    p.add_argument("--category_field", default="combined_bucket",
                   choices=["combined_bucket", "curvature_bucket", "topology_bucket"])
    p.add_argument("--samples_per_category", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--front_camera", default="ring_front_center")
    p.add_argument("--bev_range", type=float, nargs=4, default=[-50, 50, -25, 25],
                   metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
                   help="BEV extent in metres [x_min x_max y_min y_max] (default: -50 50 -25 25)")
    p.add_argument("--max_categories", type=int, default=None, help="Optional cap for quick inspection")
    return p.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------

def build_frame_to_category(segment_rows: List[dict], category_field: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for seg in segment_rows:
        category = seg[category_field]
        for rel_path in seg.get("frame_paths", []):
            mapping[Path(rel_path).as_posix()] = category
    return mapping


def sample_frames_by_category(
    frame_rows: List[dict],
    frame_to_category: Dict[str, str],
    samples_per_category: int,
    seed: int,
) -> Dict[str, List[dict]]:
    rng = random.Random(seed)
    groups: Dict[str, List[dict]] = defaultdict(list)
    for row in frame_rows:
        rel_path = Path(row["relative_path"]).as_posix()
        category = frame_to_category.get(rel_path)
        if category is None:
            continue
        groups[category].append(row)

    sampled: Dict[str, List[dict]] = {}
    for category, rows in sorted(groups.items()):
        rows_copy = list(rows)
        rng.shuffle(rows_copy)
        sampled[category] = rows_copy[: min(samples_per_category, len(rows_copy))]
    return sampled


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

_COLOR_CENTERLINE = (243, 90, 2)   # BGR


def _draw_bev_lanes(
    lanes: List[np.ndarray],
    bev_range: List[float],
    canvas_hw: tuple = (400, 200),
) -> np.ndarray:
    """
    Draw GT centerlines on a BEV canvas.

    Args:
        lanes      : list of (N, 2) ego-frame arrays [x_fwd, y_lat] in metres
        bev_range  : [x_min, x_max, y_min, y_max] in metres
        canvas_hw  : (height, width) of output canvas in pixels

    Returns:
        (H, W, 3) uint8 BGR canvas
    """
    x_min, x_max, y_min, y_max = bev_range
    H, W = canvas_hw
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    for pts in lanes:
        if pts.shape[0] < 2:
            continue
        # x (forward) → row: row 0 = front (x_max), row H-1 = rear (x_min)
        rows = ((x_max - pts[:, 0]) / (x_max - x_min) * (H - 1)).astype(np.int32)
        # y (lateral) → col: col 0 = left (y_min), col W-1 = right (y_max)
        cols = ((pts[:, 1] - y_min) / (y_max - y_min) * (W - 1)).astype(np.int32)
        rows = np.clip(rows, 0, H - 1)
        cols = np.clip(cols, 0, W - 1)
        for i in range(len(rows) - 1):
            cv2.line(canvas, (cols[i], rows[i]), (cols[i + 1], rows[i + 1]),
                     _COLOR_CENTERLINE, thickness=2, lineType=cv2.LINE_AA)
    return canvas


def _load_front_image(data_root: Path, frame_row: dict, front_camera: str) -> Optional[np.ndarray]:
    """
    Load the front-camera image for a frame using the raw per-frame JSON.
    Returns (H, W, 3) uint8 BGR or None on failure.
    """
    rel = frame_row["relative_path"]
    abs_json = (data_root / rel) if not Path(rel).is_absolute() else Path(rel)
    try:
        data = json.loads(abs_json.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[warn] failed to read annotation {abs_json}: {e}")
        return None

    sensor = data.get("sensor", {})
    cam = sensor.get(front_camera)
    if cam is None:
        print(f"[warn] camera '{front_camera}' not found in {abs_json}")
        return None

    img_rel = cam.get("image_path") or cam.get("filename")
    if img_rel is None:
        print(f"[warn] no image_path for camera '{front_camera}' in {abs_json}")
        return None

    img_abs = (data_root / img_rel) if not Path(img_rel).is_absolute() else Path(img_rel)
    img = cv2.imread(str(img_abs))
    if img is None:
        print(f"[warn] cv2.imread failed: {img_abs}")
    return img


def _load_gt_lanes(data_root: Path, frame_row: dict) -> List[np.ndarray]:
    """Return list of (N, 2) [x_fwd, y_lat] ego-frame arrays from the raw JSON."""
    rel = frame_row["relative_path"]
    abs_json = (data_root / rel) if not Path(rel).is_absolute() else Path(rel)
    try:
        data = json.loads(abs_json.read_text(encoding="utf-8"))
    except Exception:
        return []
    lanes = data.get("annotation", {}).get("lane_centerline", [])
    result = []
    for lane in lanes:
        pts = np.asarray(lane.get("points", []), dtype=np.float32)
        if pts.ndim == 2 and pts.shape[0] >= 2 and pts.shape[1] >= 2:
            result.append(pts[:, :2])
    return result


def render_one_sample(
    data_root: Path,
    frame_row: dict,
    front_camera: str,
    bev_range: List[float],
    target_h: int = 480,
) -> Optional[np.ndarray]:
    """
    Render a side-by-side (front image | BEV GT) panel.
    Returns (H, W, 3) uint8 BGR or None if the front image is unavailable.
    """
    front_img = _load_front_image(data_root, frame_row, front_camera)
    gt_lanes  = _load_gt_lanes(data_root, frame_row)

    x_min, x_max, y_min, y_max = bev_range
    bev_aspect = (x_max - x_min) / (y_max - y_min)   # typically 100/50 = 2
    bev_h = target_h
    bev_w = max(1, int(round(bev_h / bev_aspect)))
    bev_canvas = _draw_bev_lanes(gt_lanes, bev_range, canvas_hw=(bev_h, bev_w))

    if front_img is not None:
        scale = target_h / front_img.shape[0]
        front_resized = cv2.resize(front_img, (int(front_img.shape[1] * scale), target_h))
    else:
        # Grey placeholder
        front_resized = np.full((target_h, int(target_h * 16 / 9), 3), 80, dtype=np.uint8)
        cv2.putText(front_resized, "image not found", (20, target_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    # Pad BEV height to match front if needed
    if bev_canvas.shape[0] != target_h:
        bev_canvas = cv2.resize(bev_canvas, (bev_w, target_h))

    panel = np.concatenate([front_resized, bev_canvas], axis=1)
    return panel


def make_contact_sheet(images: List[np.ndarray], cell_gap: int = 8, max_cols: int = 3) -> np.ndarray:
    if not images:
        raise ValueError("No images provided for contact sheet.")
    h = max(img.shape[0] for img in images)
    w = max(img.shape[1] for img in images)
    cols = min(max_cols, len(images))
    rows = (len(images) + cols - 1) // cols
    sheet = np.zeros((rows * h + (rows - 1) * cell_gap,
                      cols * w + (cols - 1) * cell_gap, 3), dtype=np.uint8)
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        y = r * (h + cell_gap)
        x = c * (w + cell_gap)
        sheet[y:y + img.shape[0], x:x + img.shape[1]] = img
    return sheet


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    group_dir = Path(args.group_dir)
    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_rows    = load_jsonl(group_dir / "frame_summary.jsonl")
    segment_rows  = load_jsonl(group_dir / "segment_summary.jsonl")
    frame_to_cat  = build_frame_to_category(segment_rows, args.category_field)
    sampled       = sample_frames_by_category(
        frame_rows=frame_rows,
        frame_to_category=frame_to_cat,
        samples_per_category=args.samples_per_category,
        seed=args.seed,
    )

    manifest = {}
    categories = sorted(sampled.keys())
    if args.max_categories is not None:
        categories = categories[: args.max_categories]

    for category in categories:
        safe_cat = sanitize(category)
        cat_dir  = out_dir / safe_cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        renders = []
        rows = sampled[category]
        print(f"[sample] category={category} n={len(rows)}", flush=True)
        saved_rows = []
        for idx, row in enumerate(rows):
            panel = render_one_sample(data_root, row, args.front_camera, args.bev_range)
            if panel is None:
                continue
            save_path = cat_dir / f"{idx:02d}_{row['segment_id']}_{row['timestamp']}.png"
            cv2.imwrite(str(save_path), panel)
            renders.append(panel)
            saved_rows.append({
                "segment_id":    row["segment_id"],
                "timestamp":     row["timestamp"],
                "relative_path": row["relative_path"],
                "image_path":    str(save_path),
            })

        if renders:
            sheet = make_contact_sheet(renders)
            cv2.imwrite(str(cat_dir / "contact_sheet.png"), sheet)

        manifest[category] = {
            "n_samples": len(saved_rows),
            "samples":   saved_rows,
        }

    (out_dir / "sample_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[done] wrote sampled category images to {out_dir}")


if __name__ == "__main__":
    main()
