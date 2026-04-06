"""
BEV visualization of SD-map query polylines from the training pipeline.

Requires a config whose dataset pipeline loads ``map_query_segs`` and
``map_query_mask`` (e.g. a map-query / SD-map augmented setup). The default
LaneSegNet configs in this repo do not add these keys; running this script
against them will raise KeyError until the corresponding dataset transforms
and model config exist.
"""

import argparse
import os
import os.path as osp
import sys

import cv2
import numpy as np
from mmcv import Config
from mmdet3d.datasets import build_dataset

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))


def normalized_to_metric(seg_xy, pc_range):
    x_min, y_min, _, x_max, y_max, _ = pc_range
    pts = seg_xy.astype(np.float32).copy()
    pts[:, 0] = pts[:, 0] * (x_max - x_min) + x_min
    pts[:, 1] = pts[:, 1] * (y_max - y_min) + y_min
    return pts


def metres_to_bev_pixel(pts_m, pc_range, bev_h, bev_w):
    x_min, y_min, _, x_max, y_max, _ = pc_range
    rows = (bev_h - 1) * (x_max - pts_m[:, 0]) / max(x_max - x_min, 1e-6)
    cols = (bev_w - 1) * (pts_m[:, 1] - y_min) / max(y_max - y_min, 1e-6)
    return np.stack([rows, cols], axis=1)


def draw_polyline(canvas, pts_m, pc_range, bev_h, bev_w, color, thickness=2):
    pix = metres_to_bev_pixel(pts_m[:, :2], pc_range, bev_h, bev_w)
    for idx in range(len(pix) - 1):
        r0 = int(np.clip(pix[idx, 0], 0, bev_h - 1))
        c0 = int(np.clip(pix[idx, 1], 0, bev_w - 1))
        r1 = int(np.clip(pix[idx + 1, 0], 0, bev_h - 1))
        c1 = int(np.clip(pix[idx + 1, 1], 0, bev_w - 1))
        cv2.line(canvas, (c0, r0), (c1, r1), color, thickness)
    return pix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--out_dir", default="work_dirs/mapqueries_vis")
    parser.add_argument("--draw_padding", action="store_true")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.train)

    bev_h = int(cfg.model.lane_head.bev_h)
    bev_w = int(cfg.model.lane_head.bev_w)
    pc_range = list(cfg.model.lane_head.pc_range)

    os.makedirs(args.out_dir, exist_ok=True)

    for sample_idx in range(min(args.limit, len(dataset))):
        sample = dataset[sample_idx]
        segs = sample["map_query_segs"].data.cpu().numpy()
        mask = sample["map_query_mask"].data.cpu().numpy().astype(bool)
        types = sample.get("map_query_types", None)
        if types is not None:
            types = types.data.cpu().numpy()
        meta = sample["img_metas"].data
        if isinstance(meta, dict) and 0 in meta:
            meta = meta[0]

        canvas = np.full((bev_h, bev_w, 3), 24, dtype=np.uint8)
        for row in range(0, bev_h, 20):
            cv2.line(canvas, (0, row), (bev_w - 1, row), (48, 48, 48), 1)
        for col in range(0, bev_w, 20):
            cv2.line(canvas, (col, 0), (col, bev_h - 1), (48, 48, 48), 1)
        cv2.line(canvas, (bev_w // 2, 0), (bev_w // 2, bev_h - 1), (72, 72, 72), 1)
        cv2.putText(canvas, "front", (bev_w // 2 + 4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        cv2.putText(canvas, "rear", (bev_w // 2 + 4, bev_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        valid_count = 0
        for query_idx, (seg, is_pad) in enumerate(zip(segs, mask)):
            if is_pad and not args.draw_padding:
                continue
            pts_m = normalized_to_metric(seg[:, :2], pc_range)
            color = (90, 90, 90) if is_pad else (
                int((37 * query_idx) % 255),
                int((97 * query_idx) % 255),
                int((173 * query_idx) % 255),
            )
            pix = draw_polyline(canvas, pts_m, pc_range, bev_h, bev_w, color, thickness=1 if is_pad else 2)
            if not is_pad:
                valid_count += 1
            mid = pix[len(pix) // 2]
            label = str(query_idx)
            if types is not None and not is_pad:
                label = f"{query_idx}:{int(types[query_idx])}"
            cv2.putText(
                canvas,
                label,
                (int(np.clip(mid[1] + 2, 0, bev_w - 20)), int(np.clip(mid[0] - 2, 8, bev_h - 4))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.28,
                color,
                1,
            )

        header = np.full((28, bev_w, 3), 12, dtype=np.uint8)
        scene = meta.get("scene_token", "unknown")
        sample_token = meta.get("sample_idx", sample_idx)
        cv2.putText(
            header,
            f"sample={sample_idx} scene={scene} token={sample_token} valid_queries={valid_count}",
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
        )
        out = np.concatenate([header, canvas], axis=0)
        out_path = osp.join(args.out_dir, f"map_queries_{sample_idx:03d}.png")
        cv2.imwrite(out_path, out)
        print("saved", out_path)


if __name__ == "__main__":
    main()
