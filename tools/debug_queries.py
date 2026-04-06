"""
debug_queries.py — diagnostic tool for LaneSegNet's DETR query mechanism.

Main outputs:
  1.  query_activation_freq.png   — bar chart: which queries fire most often
  2.  query_spatial_spread.png    — BEV scatter: where each query's midpoints land
  3.  bev_features_pca_NNNN.png   — PCA-coloured BEV feature map per sample
  4.  query_embed_pca.png         — PCA of the 200 learned query embeddings (PC1 vs PC2)
  5.  query_embed_pca13.png       — PC1 vs PC3
  6.  query_embed_pca23.png       — PC2 vs PC3
  7.  query_priors_bev.png        — all 200 prior lanes decoded from ZERO BEV
  8.  query_similarity.png        — 200×200 cosine-similarity matrix of query embeddings
  9.  query_length_dist.png       — distribution of predicted centerline arc lengths
  10. query_prior_stats.png       — prior length / curvature / (x,y) distribution
  11. query_smoothness.png        — 20-panel smoothness report
  12. query_grid_NN.png           — per-query BEV panels tiled 5×4  (--query_grid)
  summary.txt                    — text summary + plot interpretation guide

Layer analysis outputs (--layer_analysis, written to <output_dir>/layers/):
  layer_update_norm.png          — heatmap: ‖q_L − q_{L-1}‖₂ per query per layer
  layer_score_evolution.png      — per-query sigmoid score after each decoder layer
  layer_similarity_grid.png      — inter-query cosine-sim matrix per layer
  layer_attn_entropy.png         — mean LaneAttention weight entropy per decoder layer
  layer_attn_bev_NN.png          — per-layer BEV heatmap of query attention density
  layer_pca_trajectories.png     — query paths through PCA space across layers
  layer_lanes_bev.png            — predicted centerlines per layer (BEV mosaic)
  layer_geo_convergence.png      — mean point displacement between consecutive layers
  layer_activation_crossing.png  — which layer each query first crosses score threshold
  layer_lane_diversity.png       — mean pairwise midpoint distance per layer
  layer_curvature_dist.png       — boxplot of mean radius of curvature per layer

Usage
─────
# Prior analysis only — no dataset needed:
python tools/debug_queries.py \\
    --config   projects/configs/lanesegnet_r50_8x1_24e_olv2_subset_A.py \\
    --checkpoint output/epoch_24.pth \\
    --priors_only [--layer_analysis] [--query_grid] [--output_dir ./debug_out]

# Full analysis over N dataset samples:
python tools/debug_queries.py \\
    --config   projects/configs/lanesegnet_r50_8x1_24e_olv2_subset_A.py \\
    --checkpoint output/epoch_24.pth \\
    [--split val] [--indices_file work_dirs/.../val_indices.json] \\
    [--n_samples 50] [--score_thresh 0.3] [--top_k 5] \\
    [--layer_analysis] [--output_dir ./debug_out]
"""

import argparse
import os
import os.path as osp
import sys
from pathlib import Path

# Add workspace root so 'projects.*' imports resolve
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import math
import json

import cv2
import numpy as np
import torch

import mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset, build_dataloader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Plot interpretation guide (appended to summary.txt)
# ---------------------------------------------------------------------------

_PLOT_GUIDE = """
════════════════════════════════════════════════════════════════════════════════
Plot Guide
════════════════════════════════════════════════════════════════════════════════

── Main outputs ────────────────────────────────────────────────────────────────

query_activation_freq.png
  What    : Bar chart of how many times each query slot fired (score ≥ thresh)
            across all evaluated samples.
  Healthy : Many queries with non-zero, roughly-spread counts.
  Broken  : All mass on 2–3 bars → mode collapse.
            All bars near zero → queries never activate; lower score_thresh.

query_spatial_spread.png
  What    : BEV scatter of predicted centerline midpoints, colour = query index.
  Healthy : Tight colour clusters in distinct BEV regions → spatial specialisation.
  Broken  : Random multicolour scatter → no specialisation yet.

bev_features_pca_NNNN.png
  What    : False-colour BEV feature map; PCA of 256-channel feature at each cell.
  Healthy : Structured blobs aligned with road direction → BEVFormer working.
  Broken  : Flat uniform grey → BEV encoder not yet meaningful.

query_embed_pca.png / query_embed_pca13.png / query_embed_pca23.png
  What    : PCA of the raw 512-dim query_embedding weight matrix (200 × 512).
  Healthy : Clusters or gradients in PCA space → embeddings differentiated.
  Broken  : Tight ball near origin → embeddings near initialisation (early training).

query_priors_bev.png
  What    : Lanes decoded from a fully-zero BEV (no scene content).
            Reveals the learned spatial prior before any camera data is seen.
            Left panel: all queries, brightness ∝ prior score.
            Right panel: score-filtered queries, coloured by query index.
  Healthy : Lanes spread across distinct BEV positions with diverse lateral offsets.
  Broken  : All lanes collapse near y=0 or all scores ≈ 0.5 (mode collapse).

query_similarity.png
  What    : 200×200 cosine-similarity heatmap of learned query embeddings.
  Healthy : Mostly blue off-diagonal (low similarity) → diverse queries.
  Broken  : Uniformly red → all queries look alike, mode collapse risk.

query_prior_stats.png
  What    : 3-panel: arc-length histogram, radius-of-curvature histogram,
            (x, y) point cloud of all prior lane points.
  Healthy : Length peak 20–80 m; points spread across BEV canvas.
  Broken  : Length spike near 0 m; all points collapsed near origin.

query_smoothness.png
  What    : 20-panel smoothness report (4 rows × 5 cols).
            Flags: DEGEN (<2 m), SQUIGGLE (str<0.3), OVERTURN (>180°),
                   ZIGZAG (≥3 sign changes), HAIRPIN (<5 m radius).
  Healthy : Most queries clean; arc 10–80 m; straightness >0.8.
  Broken  : Many flags, especially SQUIGGLE + ZIGZAG → squiggle collapse.

query_length_dist.png
  What    : Distribution of predicted centerline arc lengths across dataset samples.
  Healthy : Single peak at 20–80 m.
  Broken  : Spike near 0 (degenerate) or very long tail (out-of-range lanes).

── Layer analysis outputs (layers/ subdir) ─────────────────────────────────────

layer_update_norm.png
  What    : Heatmap of ‖q_L − q_{L-1}‖₂ for each (layer, query) pair.
            Colour = L2 norm of the embedding update at that layer.
  Healthy : High update norms early, decreasing toward last layer → convergent.
  Broken  : Flat or near-zero → layers dead or residual-dominated.

layer_score_evolution.png
  What    : Sigmoid classification score per query plotted across decoder layers.
            Coloured lines = queries that reach score_thresh; grey = below.
  Healthy : Scores rise steadily across layers.
  Broken  : All flat at 0.5 (untrained cls_head), or spike only at last layer
            (cls_head not coupled to decoder refinement).

layer_similarity_grid.png
  What    : One cosine-sim matrix panel per decoder layer.
  Healthy : Off-diagonal values decrease across layers (queries differentiate).
  Broken  : Uniformly high matrix at all layers → collapse risk.

layer_attn_entropy.png
  What    : Mean LaneAttention weight entropy per decoder layer.
            LaneAttention is deformable: each query/head samples num_points
            locations. Entropy = −Σ w·log(w) over the num_points distribution.
  Computed: Hook on each layer's LaneAttention.attention_weights linear;
            recompute softmax; compute entropy per (query, head); average.
  Healthy : Decreasing entropy → queries focus on fewer sampling points over
            layers (sharpening). Low entropy early = already focused.
  Broken  : Flat / increasing → attention stays diffuse; more training needed.
  Note    : With zero BEV input (priors_only mode) all weights are near-uniform
            → entropy is high and flat. Most meaningful with real scene input.

layer_attn_bev_NN.png  (one file per layer NN)
  What    : BEV density map of where queries are attending for layer NN.
            Each active query's predicted centerline midpoint is splatted as a
            filled circle (radius ~4 % of BEV height) weighted by its score.
  Computed: midpoint = cl_m[layer, query, N_pts//2] (metres), converted to BEV
            pixel.  Score-weighted circles accumulated into a float grid.
            Gaussian blur on the grid (before colormap), sqrt normalisation to
            bring up faint regions, then HOT colourmap.
  Healthy : Bright blobs spread across distinct BEV positions → spatially diverse.
  Broken  : Single central blob → all queries collapse to one location.
            Entirely dark → all queries near zero score (untrained model).

layer_pca_trajectories.png
  What    : Each query's path through 2D PCA space from layer 0 (initial embed)
            to layer L. Dots grow larger/brighter with depth.
  Healthy : Long, diverging trajectories → large updates, queries specialising.
  Broken  : Short paths, all converging to one point → collapse.

layer_lanes_bev.png
  What    : BEV mosaic showing predicted centerlines at each decoder layer.
            Brightness ∝ score at that layer.
  Healthy : Lanes progressively spread and sharpen across layers.
  Broken  : All layers look identical → pts_head ignores decoder refinement.

layer_geo_convergence.png
  What    : Heatmap of mean point displacement between consecutive layers (metres).
  Healthy : High displacement early (large geometry updates), decaying → converges.
  Broken  : Flat/zero → reg_branches disconnected from decoder.

layer_activation_crossing.png
  What    : Histogram of which decoder layer each query first crosses score_thresh.
  Healthy : Mass on early layers (model commits early in the decoder stack).
  Broken  : All mass on final layer or 'never' → cls_head bottlenecked.

layer_lane_diversity.png
  What    : Mean pairwise centerline midpoint distance per layer (metres).
  Healthy : Increasing across layers → queries spreading to distinct positions.
  Broken  : Flat or decreasing → lane collapse (queries predicting same location).

layer_curvature_dist.png
  What    : Boxplot of mean radius of curvature distribution per layer.
  Healthy : Spread increases and median rises across layers → diverse curvatures.
  Broken  : All layers identical → pts_head ignores decoder refinement.
"""


# ---------------------------------------------------------------------------
# Global BEV parameters (set once from config)
# ---------------------------------------------------------------------------

_PC_RANGE = None   # [x_min, y_min, z_min, x_max, y_max, z_max]
_BEV_H    = None   # canvas rows  (maps to x = forward direction)
_BEV_W    = None   # canvas cols  (maps to y = lateral direction)


def _set_bev_params(pc_range, bev_h, bev_w):
    global _PC_RANGE, _BEV_H, _BEV_W
    _PC_RANGE = list(pc_range)
    _BEV_H    = int(bev_h)
    _BEV_W    = int(bev_w)


# ---------------------------------------------------------------------------
# BEV coordinate helpers
# ---------------------------------------------------------------------------

def metres_to_bev_pixel(pts_m: np.ndarray) -> np.ndarray:
    """
    Convert metric (x_fwd, y_lat) → BEV pixel (row, col).

    Physical space (metres):
      x: forward,  range [x_min, x_max]
      y: lateral,  range [y_min, y_max]

    BEV canvas (pixels):
      row 0   = front (+x_max),  row bev_h-1 = rear (x_min)
      col 0   = left  (y_min),   col bev_w-1 = right (y_max)

    Args:
        pts_m: (N, 2) float array [x, y] in metres
    Returns:
        (N, 2) float array [row, col]
    """
    x_min, y_min = _PC_RANGE[0], _PC_RANGE[1]
    x_max, y_max = _PC_RANGE[3], _PC_RANGE[4]
    rows = (_BEV_H - 1) * (x_max - pts_m[:, 0]) / (x_max - x_min)
    cols = (_BEV_W - 1) * (pts_m[:, 1] - y_min) / (y_max - y_min)
    return np.stack([rows, cols], axis=1)


def _draw_bev_lane(canvas: np.ndarray, pts_m: np.ndarray,
                   color: tuple, thickness: int = 1) -> None:
    """Draw a metric (x, y) lane polyline onto a BEV canvas (in-place)."""
    if len(pts_m) < 2:
        return
    try:
        pix = metres_to_bev_pixel(pts_m[:, :2])
    except Exception:
        return
    for i in range(len(pix) - 1):
        r0 = int(np.clip(pix[i,     0], 0, _BEV_H - 1))
        c0 = int(np.clip(pix[i,     1], 0, _BEV_W - 1))
        r1 = int(np.clip(pix[i + 1, 0], 0, _BEV_H - 1))
        c1 = int(np.clip(pix[i + 1, 1], 0, _BEV_W - 1))
        cv2.line(canvas, (c0, r0), (c1, r1), color, thickness)


def _bev_midpoint_pixel(pts_m: np.ndarray) -> tuple:
    """Return BEV (row, col) of the midpoint of a metric lane."""
    mid = pts_m[len(pts_m) // 2][None]
    rc  = metres_to_bev_pixel(mid)[0]
    return float(rc[0]), float(rc[1])


# ---------------------------------------------------------------------------
# Polyline geometry (all coordinates are in metres)
# ---------------------------------------------------------------------------

def _polyline_length_m(pts_m: np.ndarray) -> float:
    """Arc length of a polyline (metres)."""
    return float(np.linalg.norm(np.diff(pts_m, axis=0), axis=1).sum())


def _polyline_chord_length_m(pts_m: np.ndarray) -> float:
    """Straight-line distance from first to last point (metres)."""
    return float(np.linalg.norm(pts_m[-1] - pts_m[0]))


def _polyline_straightness_ratio(pts_m: np.ndarray) -> float:
    """chord / arc in [0, 1]. 1.0 = perfectly straight."""
    arc   = _polyline_length_m(pts_m)
    chord = _polyline_chord_length_m(pts_m)
    if arc < 1e-6:
        return 0.0
    return float(min(chord / arc, 1.0))


def _polyline_total_turning_deg(pts_m: np.ndarray) -> float:
    """Total absolute turning angle along the polyline (degrees)."""
    if len(pts_m) < 3:
        return 0.0
    segs       = pts_m[1:] - pts_m[:-1]
    norms      = np.linalg.norm(segs, axis=1, keepdims=True)
    segs_unit  = np.where(norms > 1e-9, segs / (norms + 1e-12), 0.0)
    dots       = np.clip((segs_unit[:-1] * segs_unit[1:]).sum(axis=1), -1.0, 1.0)
    return float(np.degrees(np.arccos(dots)).sum())


def _polyline_curvature_sign_changes(pts_m: np.ndarray) -> int:
    """Number of times the signed curvature changes sign."""
    if len(pts_m) < 4:
        return 0
    dx    = pts_m[1:, 0] - pts_m[:-1, 0]
    dy    = pts_m[1:, 1] - pts_m[:-1, 1]
    cross = dx[:-1] * dy[1:] - dy[:-1] * dx[1:]
    signs = np.sign(cross)
    signs = signs[np.abs(cross) > 1e-9]
    if len(signs) < 2:
        return 0
    return int(np.sum(signs[:-1] != signs[1:]))


def _polyline_mean_radius_m(pts_m: np.ndarray, min_radius: float = 1e4) -> float:
    """Mean radius of curvature (metres). Uses discrete Frenet formula."""
    if len(pts_m) < 3:
        return min_radius
    dx     = pts_m[1:, 0] - pts_m[:-1, 0]
    dy     = pts_m[1:, 1] - pts_m[:-1, 1]
    d2x    = dx[1:] - dx[:-1]
    d2y    = dy[1:] - dy[:-1]
    tx, ty = dx[:-1], dy[:-1]
    cross  = np.abs(tx * d2y - ty * d2x)
    speed3 = (tx**2 + ty**2) ** 1.5
    valid  = speed3 > 1e-12
    if not valid.any():
        return min_radius
    kappa = cross[valid] / speed3[valid]
    kappa = np.clip(kappa, 1.0 / min_radius, np.inf)
    return float(np.mean(1.0 / kappa))


def _polyline_min_radius_m(pts_m: np.ndarray, min_radius: float = 1e4) -> float:
    """Minimum radius of curvature (metres) = 1 / kappa_max."""
    if len(pts_m) < 3:
        return min_radius
    dx     = pts_m[1:, 0] - pts_m[:-1, 0]
    dy     = pts_m[1:, 1] - pts_m[:-1, 1]
    d2x    = dx[1:] - dx[:-1]
    d2y    = dy[1:] - dy[:-1]
    tx, ty = dx[:-1], dy[:-1]
    cross  = np.abs(tx * d2y - ty * d2x)
    speed3 = (tx**2 + ty**2) ** 1.5
    valid  = speed3 > 1e-12
    if not valid.any():
        return min_radius
    kappa_max = (cross[valid] / speed3[valid]).max()
    if kappa_max < 1.0 / min_radius:
        return min_radius
    return float(1.0 / kappa_max)


# ---------------------------------------------------------------------------
# PCA helper
# ---------------------------------------------------------------------------

def _pca(X: np.ndarray, n_components: int = 3):
    """PCA via numpy SVD. Returns (coords (N, n), var_ratio (n,))."""
    X = X - X.mean(axis=0)
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    n         = int(min(n_components, Vt.shape[0]))
    coords    = X @ Vt[:n].T
    var_ratio = (s[:n] ** 2) / ((s ** 2).sum() + 1e-12)
    return coords, var_ratio


def _plot_query_embed_plane(coords: np.ndarray, var_ratio: np.ndarray,
                             x_idx: int, y_idx: int, out_path: Path,
                             title: str, c=None, cmap: str = "plasma",
                             colorbar_label: str = None) -> None:
    """Scatter one PCA plane for query embeddings."""
    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(coords[:, x_idx], coords[:, y_idx],
                    c=c, cmap=cmap, s=40, zorder=3)
    if colorbar_label is not None:
        plt.colorbar(sc, ax=ax, label=colorbar_label)
    for q in range(coords.shape[0]):
        ax.annotate(str(q), (coords[q, x_idx], coords[q, y_idx]),
                    fontsize=5, alpha=0.55)
    ax.set_xlabel(f"PC{x_idx + 1} ({var_ratio[x_idx]:.1%} variance)")
    ax.set_ylabel(f"PC{y_idx + 1} ({var_ratio[y_idx]:.1%} variance)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[debug] {out_path}")


# ---------------------------------------------------------------------------
# BEV feature PCA image
# ---------------------------------------------------------------------------

def bev_pca_image(bev_feat: np.ndarray) -> np.ndarray:
    """
    Project (C, H, W) BEV feature tensor → H×W×3 uint8 RGB via PCA.

    Structured colour blobs aligned with road geometry → BEVFormer is working.
    Flat uniform grey → features not meaningful (untrained / broken pipeline).
    """
    C, H, W = bev_feat.shape
    flat = bev_feat.reshape(C, -1).T          # (H*W, C)
    flat -= flat.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(flat, full_matrices=False)
    proj = flat @ Vt[:3].T                    # (H*W, 3)
    lo   = proj.min(axis=0, keepdims=True)
    hi   = proj.max(axis=0, keepdims=True)
    proj = (proj - lo) / (hi - lo + 1e-8)
    return (proj.reshape(H, W, 3) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Prior analysis — zero-BEV forward pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def analyze_query_priors(model, out_dir: Path, device: torch.device,
                          score_thresh: float = 0.3,
                          query_grid: bool = False) -> list:
    """
    Decode all N_QUERIES from a zero BEV feature map.

    With bev = zeros, cross-attention has no scene signal; each query's
    output depends only on self-attention between query embeddings and the
    FFN.  This reveals each query's *learned spatial prior* — the lane
    position it defaults to before seeing any camera data.

    Returns a list of text lines for summary.txt.
    """
    head       = model.pts_bbox_head
    bev_h      = head.bev_h
    bev_w      = head.bev_w
    embed_dims = head.embed_dims
    N_QUERIES  = head.num_query

    # bev_feats must be (bs, bev_h*bev_w, embed_dims) to match the shape
    # returned by BEVFormerConstructer (encoder permutes to batch-first).
    bev_zero = torch.zeros(1, bev_h * bev_w, embed_dims,
                           device=device, dtype=torch.float32)

    model.eval()
    outs = head.forward(mlvl_feats=None, bev_feats=bev_zero, img_metas=[{}])

    # Last decoder layer, batch index 0.
    # In eval mode, all_lanes_preds are already converted to metres by
    # the head's coord * (pc_range[3]-pc_range[0]) + pc_range[0] logic.
    cls_logits = outs['all_cls_scores'][-1][0]    # (N_Q, num_classes+1)
    lanes_pred = outs['all_lanes_preds'][-1][0]   # (N_Q, 90) in metres

    scores   = cls_logits[:, 0].sigmoid().cpu().numpy()              # (N_Q,)
    pred_np  = lanes_pred.cpu().numpy().reshape(N_QUERIES, 3, 10, 3) # 3 curves×10pts×xyz
    # Centerline (curve 0), x-forward and y-lateral only
    cl_m = pred_np[:, 0, :, :2]   # (N_Q, 10, 2) metres

    cmap = matplotlib.colormaps["tab20"]

    # --- Left panel: all queries, brightness ∝ score -------------------------
    canvas_all = np.full((_BEV_H, _BEV_W, 3), 20, dtype=np.uint8)
    for q in range(N_QUERIES):
        s    = float(scores[q])
        gray = int(s * 220)
        _draw_bev_lane(canvas_all, cl_m[q], (gray, gray, gray), thickness=1)

    # --- Right panel: high-confidence queries, coloured by index -------------
    canvas_hi = np.full((_BEV_H, _BEV_W, 3), 20, dtype=np.uint8)
    active_qs = np.where(scores >= score_thresh)[0]
    for q in active_qs:
        bgr = (np.array(cmap(q % 20)[:3][::-1]) * 255).astype(int).tolist()
        _draw_bev_lane(canvas_hi, cl_m[q], bgr, thickness=2)
        pix   = metres_to_bev_pixel(cl_m[q])
        mid_r = int(np.clip(pix[len(pix) // 2, 0], 4, _BEV_H - 4))
        mid_c = int(np.clip(pix[len(pix) // 2, 1], 4, _BEV_W - 4))
        cv2.putText(canvas_hi, str(q), (mid_c, mid_r),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (230, 230, 230), 1)

    for canvas in (canvas_all, canvas_hi):
        cv2.line(canvas, (_BEV_W // 2, 0), (_BEV_W // 2, _BEV_H - 1),
                 (60, 60, 60), 1)
        cv2.putText(canvas, "front", (_BEV_W // 2 + 2, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (160, 160, 160), 1)
        cv2.putText(canvas, "rear", (_BEV_W // 2 + 2, _BEV_H - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (160, 160, 160), 1)

    gap       = np.full((_BEV_H, 6, 3), 40, dtype=np.uint8)
    panel     = np.hstack([canvas_all, gap, canvas_hi])
    title_h   = 28
    title_bar = np.zeros((title_h, panel.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_bar,
                f"LEFT: all {N_QUERIES} priors (brightness=score)   "
                f"RIGHT: score>={score_thresh:.2f} (colour=query idx)",
                (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1)
    cv2.imwrite(str(out_dir / "query_priors_bev.png"),
                np.vstack([title_bar, panel]))
    print(f"[debug] {out_dir / 'query_priors_bev.png'}")

    prior_score_line = (
        f"Prior scores — min={scores.min():.3f}  max={scores.max():.3f}  "
        f"mean={scores.mean():.3f}  n>={score_thresh:.2f}: {len(active_qs)}/{N_QUERIES}"
    )
    print(f"  {prior_score_line}")

    if len(active_qs) == 0:
        print("  WARNING: no query has prior score >= threshold.  "
              "Check score_thresh or model training state.")
    elif len(active_qs) == N_QUERIES:
        print("  WARNING: ALL queries have high prior score — "
              "cls_head may output a large positive bias (mode collapse risk).")

    print("\n[debug] === Prior lane statistics ===")
    plot_prior_stats(cl_m, scores, out_dir)

    print("\n[debug] === Prior lane smoothness ===")
    smoothness_lines = plot_query_smoothness(cl_m, scores, out_dir)

    if query_grid:
        print("\n[debug] === Per-query BEV grid ===")
        plot_query_grid(cl_m, scores, out_dir, cols=5, rows=4,
                        score_thresh=score_thresh)

    return [prior_score_line, ""] + smoothness_lines


# ---------------------------------------------------------------------------
# Query similarity matrix
# ---------------------------------------------------------------------------

def query_similarity_matrix(query_embeds: np.ndarray, out_dir: Path) -> None:
    """
    Plot a N_Q×N_Q cosine-similarity heatmap of the learned query embeddings.

    Mostly blue off-diagonal = diverse queries (healthy).
    Uniformly red = all queries look alike (mode collapse risk).
    """
    N      = query_embeds.shape[0]
    norms  = np.linalg.norm(query_embeds, axis=1, keepdims=True)
    normed = query_embeds / (norms + 1e-8)
    sim    = normed @ normed.T

    off_diag = sim[np.triu_indices(N, k=1)]
    print(f"  Cosine similarity (off-diagonal) — "
          f"mean={off_diag.mean():.3f}  max={off_diag.max():.3f}  "
          f"min={off_diag.min():.3f}")

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(sim, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    ax.set_xlabel("Query index")
    ax.set_ylabel("Query index")
    ax.set_title(
        "Query embedding cosine similarity\n"
        "Healthy: mostly blue off-diagonal (diverse).  "
        "Broken: red everywhere (collapse)."
    )
    plt.tight_layout()
    p = out_dir / "query_similarity.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"[debug] {p}")


# ---------------------------------------------------------------------------
# plot_prior_stats
# ---------------------------------------------------------------------------

def plot_prior_stats(cl_m: np.ndarray, scores: np.ndarray,
                     out_dir: Path) -> None:
    """
    Three-panel figure: arc length histogram, radius of curvature histogram,
    and (x, y) point cloud of all prior lane points.

    Healthy: lengths peak at 20–80 m, points spread across BEV canvas.
    Broken:  length spike near 0, all points clustered at origin.
    """
    N_Q, N_PTS, _ = cl_m.shape

    lengths = np.array([_polyline_length_m(cl_m[q])      for q in range(N_Q)])
    radii   = np.array([_polyline_mean_radius_m(cl_m[q]) for q in range(N_Q)])

    all_pts_m  = cl_m.reshape(-1, 2)
    all_scores = np.repeat(scores, N_PTS)

    x_min, x_max = _PC_RANGE[0], _PC_RANGE[3]
    y_min, y_max = _PC_RANGE[1], _PC_RANGE[4]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    ax = axes[0]
    ax.hist(lengths, bins=min(30, N_Q), color="steelblue", edgecolor="none",
            weights=scores, alpha=0.75, label="score-weighted")
    ax.hist(lengths, bins=min(30, N_Q), color="steelblue", edgecolor="none",
            alpha=0.25, label="unweighted")
    for val, style, label in [(np.median(lengths), "orange", f"median={np.median(lengths):.1f}m"),
                               (np.mean(lengths),   "red",    f"mean={np.mean(lengths):.1f}m")]:
        ax.axvline(val, color=style, linestyle="--", linewidth=1.5, label=label)
    ax.set_xlabel("Prior arc length (m)")
    ax.set_ylabel("Count")
    ax.set_title(f"Prior arc lengths  (N={N_Q})\nHealthy: peak 20–80 m.  Broken: spike near 0.")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    radii_clip = np.clip(radii, 0, 2000)
    ax.hist(radii_clip, bins=min(30, N_Q), color="mediumseagreen", edgecolor="none",
            weights=scores, alpha=0.75, label="score-weighted")
    ax.hist(radii_clip, bins=min(30, N_Q), color="mediumseagreen", edgecolor="none",
            alpha=0.25, label="unweighted")
    for val, style, label in [(min(np.median(radii), 2000), "orange",
                                f"median={np.median(radii):.0f}m"),
                               (min(np.mean(radii),   2000), "red",
                                f"mean={np.mean(radii):.0f}m")]:
        ax.axvline(val, color=style, linestyle="--", linewidth=1.5, label=label)
    ax.set_xlabel("Mean radius of curvature (m)  [clipped 2000 m]")
    ax.set_ylabel("Count")
    ax.set_title(f"Prior radius of curvature  (N={N_Q})\nHighway: 300–1000 m.  Urban: 15–30 m.")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    sc = ax.scatter(all_pts_m[:, 1], all_pts_m[:, 0],
                    c=all_scores, cmap="plasma", s=3, alpha=0.5,
                    vmin=0.0, vmax=1.0)
    plt.colorbar(sc, ax=ax, label="Prior score")
    ax.add_patch(plt.Rectangle(
        (y_min, x_min), y_max - y_min, x_max - x_min,
        fill=False, edgecolor="grey", linestyle="--", linewidth=0.8,
    ))
    ax.axhline(0, color="grey", linestyle=":", linewidth=0.6)
    ax.axvline(0, color="grey", linestyle=":", linewidth=0.6)
    ax.set_xlabel("Lateral y (m)  ← left / right →")
    ax.set_ylabel("Forward x (m)")
    ax.set_xlim(y_min - 2, y_max + 2)
    ax.set_ylim(x_min - 2, x_max + 2)
    ax.set_aspect("equal")
    ax.set_title(f"Prior (x, y) point cloud  ({N_Q * N_PTS} pts)\n"
                 "Healthy: spread across BEV.  Broken: collapsed near origin.")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    p = out_dir / "query_prior_stats.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"[debug] {p}")

    near_zero  = int((lengths < 1.0).sum())
    n_straight = int((radii >= 1000).sum())
    print(f"  Prior lengths — min={lengths.min():.1f}m  max={lengths.max():.1f}m  "
          f"mean={lengths.mean():.1f}m  near-zero(<1m): {near_zero}/{N_Q}")
    print(f"  Prior radii   — min={radii.min():.0f}m  max={radii.max():.0f}m  "
          f"mean={radii.mean():.0f}m  straight(>=1000m): {n_straight}/{N_Q}")


# ---------------------------------------------------------------------------
# plot_query_smoothness  (20-panel report)
# ---------------------------------------------------------------------------

def plot_query_smoothness(cl_m: np.ndarray, scores: np.ndarray,
                           out_dir: Path) -> list:
    """
    Twenty-panel smoothness report.  Layout — 4 rows × 5 cols:
      Row 0: length histograms (arc, chord, straightness, arc-vs-chord, gap)
      Row 1: curvature histograms (turning, sign changes, min R, mean R, scatter)
      Row 2: per-query bars (length metrics)
      Row 3: per-query bars (curvature) + flag summary

    Returns a list of text lines suitable for summary.txt.
    """
    N_Q = cl_m.shape[0]
    qs  = np.arange(N_Q)

    arc_lengths   = np.array([_polyline_length_m(cl_m[q])               for q in range(N_Q)])
    chord_lengths = np.array([_polyline_chord_length_m(cl_m[q])         for q in range(N_Q)])
    straightness  = np.array([_polyline_straightness_ratio(cl_m[q])     for q in range(N_Q)])
    turnings      = np.array([_polyline_total_turning_deg(cl_m[q])      for q in range(N_Q)])
    sign_changes  = np.array([_polyline_curvature_sign_changes(cl_m[q]) for q in range(N_Q)])
    min_radii     = np.array([_polyline_min_radius_m(cl_m[q])           for q in range(N_Q)])
    mean_radii    = np.array([_polyline_mean_radius_m(cl_m[q])          for q in range(N_Q)])
    arc_chord_gap = arc_lengths - chord_lengths

    flag_degen    = arc_lengths  <  2.0
    flag_squiggle = straightness <  0.3
    flag_overturn = turnings     > 180.0
    flag_zigzag   = sign_changes >=   3
    flag_hairpin  = min_radii    <  5.0
    flag_counts   = (flag_degen.astype(int) + flag_squiggle.astype(int) +
                     flag_overturn.astype(int) + flag_zigzag.astype(int) +
                     flag_hairpin.astype(int))

    fig, axes = plt.subplots(4, 5, figsize=(28, 22))

    def _hist(ax, data, xlabel, title, valid_band=None, clip=None,
              color="steelblue", bins=None):
        disp = np.clip(data, 0, clip) if clip is not None else data
        b    = bins if bins is not None else min(30, N_Q)
        ax.hist(disp, bins=b, color=color, edgecolor="none",
                weights=scores, alpha=0.75, label="score-weighted")
        ax.hist(disp, bins=b, color=color, edgecolor="none",
                alpha=0.25, label="unweighted")
        med      = np.median(data)
        mu       = np.mean(data)
        disp_med = min(med, clip) if clip is not None else med
        disp_mu  = min(mu,  clip) if clip is not None else mu
        ax.axvline(disp_med, color="orange", linestyle="--", linewidth=1.5,
                   label=f"median={med:.2g}")
        ax.axvline(disp_mu,  color="red",    linestyle="--", linewidth=1.5,
                   label=f"mean={mu:.2g}")
        if valid_band is not None:
            lo, hi  = valid_band
            hi_disp = min(hi, clip) if clip is not None else hi
            ax.axvspan(lo, hi_disp, color="limegreen", alpha=0.08,
                       label="valid range")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    def _bar(ax, data, ylabel, title, clip=None, flag_mask=None):
        disp   = np.clip(data, 0, clip) if clip is not None else data
        colors = plt.cm.plasma(scores)
        bars   = ax.bar(qs, disp, width=1.0, color=colors, edgecolor="none")
        if flag_mask is not None and flag_mask.any():
            for q_i in np.where(flag_mask)[0]:
                bars[q_i].set_edgecolor("red")
                bars[q_i].set_linewidth(0.8)
        ax.set_xlabel("Query index")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
        if clip is not None:
            ax.set_ylim(0, clip * 1.05)

    # Row 0 — length histograms
    _hist(axes[0, 0], arc_lengths,   "Arc length (m)",
          f"Arc length  (N={N_Q})\n10–80 m = healthy.  <2 m = degenerate.",
          valid_band=(5, 120), color="steelblue")
    _hist(axes[0, 1], chord_lengths, "Chord length (m)",
          f"Chord length  (N={N_Q})\nShould track arc for straight lanes.",
          valid_band=(5, 120), color="cornflowerblue")
    _hist(axes[0, 2], straightness,  "Straightness (chord / arc)",
          f"Straightness  (N={N_Q})\n>0.8=straight  0.4–0.8=curved  <0.3=squiggle",
          valid_band=(0.4, 1.0), color="mediumpurple",
          bins=min(30, N_Q))

    ax = axes[0, 3]
    sc = ax.scatter(arc_lengths, chord_lengths, c=scores,
                    cmap="plasma", s=20, alpha=0.8, vmin=0, vmax=1)
    lim = max(arc_lengths.max(), chord_lengths.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, label="chord = arc (straight)")
    plt.colorbar(sc, ax=ax, label="Prior score")
    ax.set_xlabel("Arc length (m)")
    ax.set_ylabel("Chord length (m)")
    ax.set_title(f"Arc vs chord  (N={N_Q})\nPoints on diagonal = straight lanes.")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    _hist(axes[0, 4], arc_chord_gap, "Arc − chord gap (m)",
          f"Arc−chord gap  (N={N_Q})\n>20 m = large deviation from straight.",
          valid_band=(0, 20), color="slategrey",
          clip=float(arc_lengths.max()))

    # Row 1 — curvature histograms
    _hist(axes[1, 0], turnings,     "Total turning angle (°)",
          f"Total turning  (N={N_Q})\n<30°=straight  <90°=healthy  >180°=squiggle",
          valid_band=(0, 90), color="tomato", clip=360)
    _hist(axes[1, 1], sign_changes, "Curvature sign changes",
          f"Sign changes  (N={N_Q})\n0–1=healthy  ≥3=squiggle",
          valid_band=(0, 2), color="darkorange",
          bins=max(1, int(sign_changes.max()) + 1) if len(sign_changes) else 10)
    _hist(axes[1, 2], min_radii,   "Min radius of curvature (m)",
          f"Min radius  (N={N_Q})\n>50 m=valid  <5 m=hairpin",
          valid_band=(5, 500), color="mediumseagreen", clip=500)
    _hist(axes[1, 3], mean_radii,  "Mean radius of curvature (m)",
          f"Mean radius  (N={N_Q})\nComplements min radius.",
          valid_band=(50, 2000), color="seagreen", clip=2000)

    ax = axes[1, 4]
    sc = ax.scatter(straightness, np.clip(turnings, 0, 360), c=scores,
                    cmap="plasma", s=20, alpha=0.8, vmin=0, vmax=1)
    ax.axvline(0.3, color="red",    linestyle="--", linewidth=0.8, label="str=0.3")
    ax.axhline(180, color="orange", linestyle="--", linewidth=0.8, label="180°")
    plt.colorbar(sc, ax=ax, label="Prior score")
    ax.set_xlabel("Straightness ratio")
    ax.set_ylabel("Total turning (°)  [clipped 360°]")
    ax.set_title(f"Turning vs straightness  (N={N_Q})\nBottom-right=healthy  Top-left=squiggle.")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    # Row 2 — per-query bars (length)
    _bar(axes[2, 0], arc_lengths,  "Arc length (m)",    "Per-query arc length\n(red=DEGEN)",    clip=150, flag_mask=flag_degen)
    _bar(axes[2, 1], chord_lengths,"Chord length (m)",  "Per-query chord length",               clip=150, flag_mask=flag_squiggle)
    _bar(axes[2, 2], straightness, "Straightness",      "Per-query straightness\n(red=SQUIGGLE)",          flag_mask=flag_squiggle)
    _bar(axes[2, 3], turnings,     "Total turning (°)", "Per-query turning\n(red=OVERTURN)",    clip=360, flag_mask=flag_overturn)
    _bar(axes[2, 4], sign_changes, "Sign changes",      "Per-query sign changes\n(red=ZIGZAG)",            flag_mask=flag_zigzag)

    # Row 3 — per-query bars (curvature) + summary
    _bar(axes[3, 0], np.clip(min_radii,  0,  500), "Min radius (m)",  "Per-query min radius\n(red=HAIRPIN)", flag_mask=flag_hairpin)
    _bar(axes[3, 1], np.clip(mean_radii, 0, 2000), "Mean radius (m)", "Per-query mean radius")
    _bar(axes[3, 2], flag_counts, "Flag count (0–5)",  "Per-query flag count\n(0=clean  5=all flags)",      flag_mask=flag_counts >= 2)
    _bar(axes[3, 3], scores,      "Prior score",       "Per-query prior score\n(plasma = score reference)")

    ax = axes[3, 4]
    flag_names  = ["DEGEN\n(<2m)", "SQUIGGLE\n(str<0.3)",
                   "OVERTURN\n(>180°)", "ZIGZAG\n(≥3 chg)", "HAIRPIN\n(<5m R)"]
    flag_totals = [flag_degen.sum(), flag_squiggle.sum(),
                   flag_overturn.sum(), flag_zigzag.sum(), flag_hairpin.sum()]
    bar_colors  = ["steelblue", "mediumpurple", "tomato", "darkorange", "mediumseagreen"]
    bars = ax.bar(flag_names, flag_totals, color=bar_colors, edgecolor="none")
    for bar, v in zip(bars, flag_totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(v), ha="center", va="bottom", fontsize=8)
    ax.axhline(N_Q * 0.5, color="grey", linestyle="--", linewidth=0.8,
               label="50% of queries")
    ax.set_ylabel("Queries flagged")
    ax.set_title(f"Flag-type summary  (N={N_Q})\nHow many queries triggered each flag.")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, N_Q + 2)

    plt.tight_layout()
    p = out_dir / "query_smoothness.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"[debug] {p}")

    summary_lines = [
        "smoothness stats:",
        f"  Arc length    — min={arc_lengths.min():.1f}m  max={arc_lengths.max():.1f}m  "
        f"mean={arc_lengths.mean():.1f}m  degenerate(<2m): {int(flag_degen.sum())}/{N_Q}",
        f"  Straightness  — mean={straightness.mean():.2f}  "
        f"squiggles(<0.3): {int(flag_squiggle.sum())}/{N_Q}",
        f"  Total turning — mean={turnings.mean():.1f}°  "
        f"suspect(>180°): {int(flag_overturn.sum())}/{N_Q}",
        f"  Sign changes  — mean={sign_changes.mean():.1f}  "
        f"zigzag(>=3): {int(flag_zigzag.sum())}/{N_Q}",
        f"  Min radius    — mean={min_radii.mean():.0f}m  "
        f"hairpins(<5m): {int(flag_hairpin.sum())}/{N_Q}",
    ]
    for line in summary_lines:
        print(line)
    return summary_lines


# ---------------------------------------------------------------------------
# plot_query_grid  (one BEV panel per query, tiled)
# ---------------------------------------------------------------------------

def plot_query_grid(cl_m: np.ndarray, scores: np.ndarray, out_dir: Path,
                    cols: int = 5, rows: int = 4,
                    score_thresh: float = None) -> None:
    """
    Draw each query's prior centerline on its own BEV panel, tiled in a
    cols×rows grid.  One output file per page of 20 queries (5 cols × 4 rows).

    Each cell shows the BEV canvas + a label strip with:
      Q{idx}  s={score}  arc  chord  straightness  turning  sign_changes  min_R  flags
    Red label background when any smoothness flag is triggered.
    """
    N_Q      = cl_m.shape[0]
    cmap     = matplotlib.colormaps["tab20"]
    per_page = cols * rows

    arc_lengths   = np.array([_polyline_length_m(cl_m[q])               for q in range(N_Q)])
    chord_lengths = np.array([_polyline_chord_length_m(cl_m[q])         for q in range(N_Q)])
    straightness  = np.array([_polyline_straightness_ratio(cl_m[q])     for q in range(N_Q)])
    turnings      = np.array([_polyline_total_turning_deg(cl_m[q])      for q in range(N_Q)])
    sign_changes  = np.array([_polyline_curvature_sign_changes(cl_m[q]) for q in range(N_Q)])
    min_radii     = np.array([_polyline_min_radius_m(cl_m[q])           for q in range(N_Q)])

    scale   = 2
    draw_h  = _BEV_H * scale
    draw_w  = _BEV_W * scale
    gap     = 6
    label_h = 60
    line_h  = 14
    font    = cv2.FONT_HERSHEY_SIMPLEX
    fscale  = 0.38
    page_h  = rows * (draw_h + label_h + gap) + gap
    page_w  = cols * (draw_w + gap) + gap

    for page in range(int(np.ceil(N_Q / per_page))):
        canvas = np.full((page_h, page_w, 3), 15, dtype=np.uint8)
        start  = page * per_page

        for local_idx in range(per_page):
            q = start + local_idx
            if q >= N_Q:
                break
            row_i = local_idx // cols
            col_i = local_idx %  cols
            y0    = gap + row_i * (draw_h + label_h + gap)
            x0    = gap + col_i * (draw_w + gap)
            y_bev = y0 + label_h

            flags = []
            if arc_lengths[q]  <  2.0:  flags.append("DEGEN")
            if straightness[q] <  0.3:  flags.append("SQUIGGLE")
            if turnings[q]     > 180.0: flags.append("OVERTURN")
            if sign_changes[q] >=    3: flags.append("ZIGZAG")
            if min_radii[q]    <  5.0:  flags.append("HAIRPIN")

            canvas[y0: y0 + label_h, x0: x0 + draw_w] = (60, 20, 20) if flags else (30, 30, 30)

            cell = np.full((_BEV_H, _BEV_W, 3), 25, dtype=np.uint8)
            cv2.line(cell, (_BEV_W // 2, 0), (_BEV_W // 2, _BEV_H - 1), (50, 50, 50), 1)

            s = float(scores[q])
            if score_thresh is not None and s < score_thresh:
                color = (70, 70, 70)
            else:
                color = tuple((np.array(cmap(q % 20)[:3][::-1]) * 255).astype(int).tolist())
            _draw_bev_lane(cell, cl_m[q], color, thickness=2)
            cell = cv2.resize(cell, (draw_w, draw_h), interpolation=cv2.INTER_NEAREST)
            canvas[y_bev: y_bev + draw_h, x0: x0 + draw_w] = cell

            txt_color = (255, 120, 120) if flags else (200, 200, 200)
            label_lines = [
                f"Q{q}  s={s:.2f}",
                f"arc={arc_lengths[q]:.0f}m  crd={chord_lengths[q]:.0f}m  "
                f"str={straightness[q]:.2f}",
                f"trn={turnings[q]:.0f}°  sgn={sign_changes[q]}  "
                f"Rm={min_radii[q]:.0f}m",
                " ".join(flags) if flags else "clean",
            ]
            for li, line in enumerate(label_lines):
                cv2.putText(canvas, line,
                            (x0 + 3, y0 + 4 + (li + 1) * line_h - 2),
                            font, fscale, txt_color, 1, cv2.LINE_AA)

        p = out_dir / f"query_grid_{page:02d}.png"
        cv2.imwrite(str(p), canvas)
        end_q = min(start + per_page, N_Q) - 1
        print(f"[debug] {p}  (queries {start}–{end_q})")


# ---------------------------------------------------------------------------
# plot_query_lengths
# ---------------------------------------------------------------------------

def plot_query_lengths(query_lengths: list, out_dir: Path) -> None:
    """
    Two-panel figure:
      Left  — histogram of all predicted lane arc lengths (metres).
      Right — per-query mean arc length ± 1 std.

    Healthy: single peak at 20–80 m.  Broken: spike near 0 (degenerate).
    """
    all_lengths = [l for ql in query_lengths for l in ql]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    ax = axes[0]
    if all_lengths:
        ax.hist(all_lengths, bins=40, color="steelblue", edgecolor="none")
        med = np.median(all_lengths)
        mu  = np.mean(all_lengths)
        ax.axvline(med, color="orange", linestyle="--", linewidth=1.5,
                   label=f"median={med:.1f}m")
        ax.axvline(mu,  color="red",    linestyle="--", linewidth=1.5,
                   label=f"mean={mu:.1f}m")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no active queries", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel("Predicted lane arc length (m)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of predicted centerline arc lengths\n"
                 "Healthy: single peak at 20–80 m.  Broken: spike near 0.")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    active_qs = [q for q in range(len(query_lengths)) if query_lengths[q]]
    if active_qs:
        means = np.array([np.mean(query_lengths[q]) for q in active_qs])
        stds  = np.array([np.std(query_lengths[q])  for q in active_qs])
        ax.bar(active_qs, means, width=1.0, color="steelblue", alpha=0.7)
        ax.errorbar(active_qs, means, yerr=stds, fmt="none",
                    ecolor="black", elinewidth=0.7, capsize=2)
    ax.set_xlabel("Query index")
    ax.set_ylabel("Mean arc length (m)")
    ax.set_title("Per-query mean predicted arc length ± 1 std\n"
                 "Healthy: similar heights.  Broken: bars near 0 (degenerate).")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    p = out_dir / "query_length_dist.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"[debug] {p}")

    if all_lengths:
        near_zero = sum(1 for l in all_lengths if l < 1.0)
        print(f"  Lane lengths — min={min(all_lengths):.1f}m  "
              f"max={max(all_lengths):.1f}m  "
              f"mean={np.mean(all_lengths):.1f}m  "
              f"near-zero(<1m): {near_zero}/{len(all_lengths)}")


# ---------------------------------------------------------------------------
# Per-layer decoder analysis
# ---------------------------------------------------------------------------

@torch.no_grad()
def analyze_decoder_layers(model, out_dir: Path, device: torch.device,
                            score_thresh: float = 0.3) -> list:
    """
    Run one forward pass (zero BEV) with the head in training mode to capture
    all 6 decoder layers' outputs, then produce per-layer diagnostics.

    Training mode is used only for the head so that `all_lanes_preds` and
    `history_states` contain all decoder layers, not just the last one.
    The rest of the model stays in eval mode.

    Outputs (written to out_dir/layers/):
      layer_update_norm.png        — heatmap ‖q_L − q_{L-1}‖₂
      layer_score_evolution.png    — sigmoid score per query per layer
      layer_similarity_grid.png    — inter-query cosine-sim per layer
      layer_attn_entropy.png       — LaneAttention weight entropy per layer
      layer_attn_bev_NN.png        — BEV density map of query attention per layer
      layer_pca_trajectories.png   — query paths through PCA space
      layer_lanes_bev.png          — predicted centerlines per layer (BEV mosaic)
      layer_geo_convergence.png    — mean point displacement between layers
      layer_activation_crossing.png— which layer each query first crosses threshold
      layer_lane_diversity.png     — mean pairwise midpoint distance per layer
      layer_curvature_dist.png     — boxplot of radius of curvature per layer

    Returns a list of summary text lines.
    """
    head       = model.pts_bbox_head
    bev_h      = head.bev_h
    bev_w      = head.bev_w
    embed_dims = head.embed_dims
    N_QUERIES  = head.num_query
    layers_dir = out_dir / "layers"
    layers_dir.mkdir(exist_ok=True)

    # Zero BEV — shape (1, bev_h*bev_w, embed_dims) matches BEVFormerConstructer output
    bev_zero = torch.zeros(1, bev_h * bev_w, embed_dims,
                           device=device, dtype=torch.float32)

    # ── Set up LaneAttention entropy hooks ────────────────────────────────────
    # LaneAttention is deformable: each (query, head) attends to num_points
    # sampling locations.  We hook the attention_weights linear sub-module so
    # we can recompute the softmax distribution and compute its entropy without
    # modifying the forward pass.
    _deform_aw_per_layer = []   # one (bs, N_Q, num_heads, n_levels*n_pts) per layer

    _attn_handles = []
    decoder_layers = head.transformer.decoder.layers
    for _layer in decoder_layers:
        _la = _layer.attentions[1]          # LaneAttention (cross-attn)
        _n_h  = _la.num_heads
        _n_lp = _la.num_levels * _la.num_points

        def _make_aw_hook(n_h, n_lp):
            def _hook(module, input, output):
                # output: (bs, N_Q, n_h * n_lp)  (raw linear logits)
                bs, nq, _ = output.shape
                aw = output.detach().float().view(bs, nq, n_h, n_lp).softmax(-1)
                _deform_aw_per_layer.append(aw.cpu())
            return _hook

        _attn_handles.append(
            _la.attention_weights.register_forward_hook(_make_aw_hook(_n_h, _n_lp))
        )

    # Temporarily set head to training mode to get all-layer outputs.
    # The rest of the model stays eval (no stochastic dropout in backbone/neck).
    try:
        head.train()
        outs = head.forward(mlvl_feats=None, bev_feats=bev_zero, img_metas=[{}])
    finally:
        head.eval()
        for _h in _attn_handles:
            _h.remove()

    # history_states: (nb_dec, bs, N_Q, embed_dims)
    hs = outs['history_states'].detach().float().cpu().numpy()[:, 0]  # (L, N_Q, d)
    L, N_Q, d = hs.shape

    # all_cls_scores: (nb_dec, bs, N_Q, num_cls+1)  — already sigmoid'd below
    scores_arr = (outs['all_cls_scores'].detach().float()
                  [:, 0, :, 0].sigmoid().cpu().numpy())   # (L, N_Q)

    # all_lanes_preds: (nb_dec, bs, N_Q, 90) — first 30 = centerline (metres)
    lanes_np = (outs['all_lanes_preds'].detach().float()
                .cpu().numpy()[:, 0, :, :30])              # (L, N_Q, 30)
    cl_m = lanes_np.reshape(L, N_Q, 10, 3)[:, :, :, :2]   # (L, N_Q, 10, 2) metres XY

    # "Layer 0" = initial content query embedding (second half of 512-dim weight)
    q0 = (head.query_embedding.weight.detach()
          .float().cpu().numpy()[:, embed_dims:])           # (N_Q, d)
    qs_all = np.concatenate([q0[None], hs], axis=0)        # (L+1, N_Q, d)

    # ── Update norms ‖q_L − q_{L-1}‖₂ ────────────────────────────────────────
    update_norms = np.linalg.norm(np.diff(qs_all, axis=0), axis=-1)  # (L, N_Q)

    def _cosine_sim_matrix(X):
        n  = np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8
        Xn = X / n
        return Xn @ Xn.T

    # ── Plot 1: Update norm heatmap ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, N_Q // 5), max(3, L // 2 + 1)))
    im = ax.imshow(update_norms, aspect="auto", cmap="viridis",
                   interpolation="nearest")
    ax.set_xlabel("Query index")
    ax.set_ylabel("Decoder layer")
    ax.set_yticks(range(L))
    ax.set_yticklabels([f"L{i + 1}" for i in range(L)])
    ax.set_title(
        "Update norm per layer per query  ‖q_L − q_{L−1}‖₂\n"
        "High early + decaying → healthy refinement.  "
        "Flat/zero → layer dead or residual-dominated."
    )
    plt.colorbar(im, ax=ax, label="L2 norm of update")
    plt.tight_layout()
    p = layers_dir / "layer_update_norm.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"[debug] {p}")

    # ── Plot 2: Score evolution ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    cmap20  = matplotlib.colormaps["tab20"]
    active  = np.where(scores_arr.max(axis=0) >= score_thresh)[0]
    for qi in active:
        ax.plot(range(1, L + 1), scores_arr[:, qi],
                color=cmap20(qi % 20), alpha=0.7, linewidth=1.2)
    for qi in range(N_Q):
        if qi not in active:
            ax.plot(range(1, L + 1), scores_arr[:, qi],
                    color="grey", alpha=0.1, linewidth=0.6)
    ax.axhline(score_thresh, color="red", linestyle="--", linewidth=1,
               label=f"thresh={score_thresh:.2f}")
    ax.set_xlabel("Decoder layer")
    ax.set_ylabel("Sigmoid score")
    ax.set_xticks(range(1, L + 1))
    ax.set_xticklabels([f"L{i}" for i in range(1, L + 1)])
    ax.set_title(
        "Score evolution per query across decoder layers\n"
        "Healthy: scores rise steadily.  "
        "Broken: all flat at 0.5, or spike only at last layer."
    )
    ax.legend(loc="upper left", fontsize=7)
    plt.tight_layout()
    p = layers_dir / "layer_score_evolution.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"[debug] {p}")

    # ── Plot 3: Inter-query similarity per layer ───────────────────────────────
    n_cols = min(L, 4)
    n_rows = math.ceil(L / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.5, n_rows * 3.2))
    axes_flat = np.array(axes).flatten() if L > 1 else [axes]
    for li in range(L):
        sim = _cosine_sim_matrix(hs[li])
        off = float(np.mean(np.abs(sim - np.eye(N_Q))))
        axes_flat[li].imshow(sim, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        axes_flat[li].set_title(f"L{li + 1}  mean_off_diag={off:.3f}", fontsize=8)
        axes_flat[li].axis("off")
    for li in range(L, len(axes_flat)):
        axes_flat[li].axis("off")
    fig.suptitle(
        "Inter-query cosine similarity per layer\n"
        "Healthy: off-diagonal decreases (queries differentiate).\n"
        "Broken: uniformly high (mode collapse risk).",
        fontsize=9,
    )
    plt.tight_layout()
    p = layers_dir / "layer_similarity_grid.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"[debug] {p}")

    # ── Plot 4: LaneAttention entropy + BEV density maps ─────────────────────
    if _deform_aw_per_layer:
        def _entropy(w_np):
            """Entropy of distribution over last axis (nats)."""
            w = np.clip(w_np, 1e-9, 1.0)
            return -(w * np.log(w)).sum(axis=-1)

        entropy_by_layer = []
        for li, aw_t in enumerate(_deform_aw_per_layer):
            # aw_t: (bs, N_Q, num_heads, n_lp)  — CPU torch tensor from hook
            aw_np = aw_t[0].numpy()                # (N_Q, num_heads, n_lp)
            ent   = _entropy(aw_np)                # (N_Q, num_heads)
            entropy_by_layer.append(float(np.mean(ent)))

            # BEV density map: scatter score-weighted query midpoints.
            # Use filled circles (radius ~ 3 % of BEV height) rather than
            # single pixels so the sparse scatter is visible on a small canvas.
            bev_acc = np.zeros((_BEV_H, _BEV_W), dtype=np.float32)
            dot_r   = max(2, int(round(_BEV_H * 0.04)))   # ~4 px on 100-row canvas
            mid_idx = cl_m.shape[2] // 2
            for qi in range(N_Q):
                s = float(scores_arr[li, qi])
                if s < score_thresh * 0.4:
                    continue
                mid_m = cl_m[li, qi, mid_idx]          # (2,) metres [x_fwd, y_lat]
                pix   = metres_to_bev_pixel(mid_m[None])  # (1, 2)
                r = int(np.clip(pix[0, 0], 0, _BEV_H - 1))
                c = int(np.clip(pix[0, 1], 0, _BEV_W - 1))
                cv2.circle(bev_acc, (c, r), dot_r, s, thickness=-1)

            # Gaussian blur on the accumulation grid (before colormap) so that
            # overlapping dots blend smoothly.  Blur AFTER colormap (old code)
            # has no effect on the mostly-black background.
            ksize = max(3, dot_r * 4 + 1) | 1          # odd kernel, at least 3
            bev_acc = cv2.GaussianBlur(bev_acc, (ksize, ksize), dot_r * 0.8)

            # Sqrt normalization: brings up low-density regions so faint dots
            # are not swallowed by the dark end of the HOT colourmap.
            bev_acc -= bev_acc.min()
            bev_max  = bev_acc.max()
            if bev_max > 1e-8:
                bev_acc  = np.sqrt(bev_acc / bev_max)
            bev_uint = (bev_acc * 255).astype(np.uint8)
            heat = cv2.applyColorMap(bev_uint, cv2.COLORMAP_HOT)
            cv2.putText(heat, f"L{li + 1}", (3, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            p = layers_dir / f"layer_attn_bev_{li + 1:02d}.png"
            cv2.imwrite(str(p), heat)
            print(f"[debug] {p}")

        # Entropy line plot
        if entropy_by_layer:
            fig, ax = plt.subplots(figsize=(7, 3))
            xs = list(range(1, len(entropy_by_layer) + 1))
            ax.plot(xs, entropy_by_layer, "o-", color="steelblue", linewidth=2)
            ax.set_xlabel("Decoder layer")
            ax.set_ylabel("Mean attention entropy (nats)")
            ax.set_xticks(xs)
            ax.set_xticklabels([f"L{x}" for x in xs])
            ax.set_title(
                "LaneAttention weight entropy per layer\n"
                "Decreasing → queries focus on fewer sampling points (sharpening).\n"
                "Flat/increasing → attention stays diffuse (may need more training)."
            )
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            p = layers_dir / "layer_attn_entropy.png"
            plt.savefig(p, dpi=150)
            plt.close()
            print(f"[debug] {p}")
    else:
        entropy_by_layer = []

    # ── Plot 6: PCA trajectories ───────────────────────────────────────────────
    all_q = qs_all.reshape(-1, d)   # ((L+1)*N_Q, d)
    if all_q.shape[0] >= 3:
        coords2d, var2d = _pca(all_q, n_components=2)
        coords2d = coords2d.reshape(L + 1, N_Q, 2)
        fig, ax = plt.subplots(figsize=(9, 7))
        for qi in range(min(N_Q, 40)):    # cap at 40 for readability
            traj  = coords2d[:, qi]
            color = cmap20(qi % 20)
            ax.plot(traj[:, 0], traj[:, 1], "-", color=color,
                    alpha=0.5, linewidth=0.9)
            for li in range(L + 1):
                sz    = 20 + li * 12
                alpha = 0.4 + 0.6 * li / max(L, 1)
                ax.scatter(traj[li, 0], traj[li, 1],
                           s=sz, color=color, alpha=alpha, zorder=3)
            ax.annotate(f"q{qi}", (traj[0, 0], traj[0, 1]),
                        fontsize=4, alpha=0.5)
        ax.set_xlabel(f"PC1 ({var2d[0]:.1%} var)")
        ax.set_ylabel(f"PC2 ({var2d[1]:.1%} var)")
        ax.set_title(
            "Query PCA trajectories across decoder layers\n"
            "Each line = one query; dots grow larger/brighter with depth.\n"
            "Long diverging paths → healthy updates.  "
            "Short paths → small updates."
        )
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        p = layers_dir / "layer_pca_trajectories.png"
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"[debug] {p}")

    # ── Plot 7: BEV lanes mosaic per layer ────────────────────────────────────
    label_h    = 18
    cell_h     = _BEV_H + label_h
    cell_w     = _BEV_W
    n_cols_bev = min(L, 4)
    n_rows_bev = math.ceil(L / n_cols_bev)
    mosaic     = np.full((n_rows_bev * cell_h, n_cols_bev * cell_w, 3),
                         10, dtype=np.uint8)
    for li in range(L):
        row_i, col_i = divmod(li, n_cols_bev)
        y0 = row_i * cell_h
        x0 = col_i * cell_w
        strip = np.full((label_h, cell_w, 3), 35, dtype=np.uint8)
        n_active_li = int((scores_arr[li] >= score_thresh).sum())
        cv2.putText(strip, f"L{li + 1}  active={n_active_li}", (3, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        mosaic[y0:y0 + label_h, x0:x0 + cell_w] = strip
        canvas = np.full((_BEV_H, _BEV_W, 3), 20, dtype=np.uint8)
        cv2.line(canvas, (_BEV_W // 2, 0), (_BEV_W // 2, _BEV_H - 1),
                 (50, 50, 50), 1)
        s_max = float(scores_arr[li].max()) + 1e-8
        for qi in range(N_Q):
            s = float(scores_arr[li, qi])
            if s < score_thresh * 0.4:
                continue
            rgba  = cmap20(qi % 20)
            dim   = max(0.25, s / s_max)
            color = (int(rgba[2] * 255 * dim),
                     int(rgba[1] * 255 * dim),
                     int(rgba[0] * 255 * dim))
            _draw_bev_lane(canvas, cl_m[li, qi], color, thickness=1)
        mosaic[y0 + label_h:y0 + cell_h, x0:x0 + cell_w] = canvas
    cv2.imwrite(str(layers_dir / "layer_lanes_bev.png"), mosaic)
    print(f"[debug] {layers_dir / 'layer_lanes_bev.png'}")

    # ── Plot 8: Geometric convergence ─────────────────────────────────────────
    geo_conv = None
    if L > 1:
        # mean per-point displacement between consecutive layers: (L-1, N_Q)
        geo_conv = np.linalg.norm(
            np.diff(cl_m, axis=0), axis=-1
        ).mean(axis=-1)   # (L-1, N_Q) → wait: diff on axis=0 gives (L-1,N_Q,10,2)
        # Recompute correctly
        cl_diff  = np.diff(cl_m, axis=0)                         # (L-1, N_Q, 10, 2)
        geo_conv = np.linalg.norm(cl_diff, axis=-1).mean(axis=-1)  # (L-1, N_Q)

        fig, ax = plt.subplots(
            figsize=(max(8, N_Q // 5), max(3, (L - 1) // 2 + 1)))
        im = ax.imshow(geo_conv, aspect="auto", cmap="plasma",
                       interpolation="nearest")
        ax.set_xlabel("Query index")
        ax.set_ylabel("Layer transition")
        ax.set_yticks(range(L - 1))
        ax.set_yticklabels([f"L{i + 1}→L{i + 2}" for i in range(L - 1)])
        ax.set_title(
            "Geometric convergence: mean point displacement between layers (m)\n"
            "High early + decaying → progressive refinement.  "
            "Flat → reg_branches disconnected from decoder."
        )
        plt.colorbar(im, ax=ax, label="Mean displacement (m)")
        plt.tight_layout()
        p = layers_dir / "layer_geo_convergence.png"
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"[debug] {p}")

    # ── Plot 9: Activation threshold crossing histogram ───────────────────────
    crossing_layer = []
    for qi in range(N_Q):
        crossed = np.where(scores_arr[:, qi] >= score_thresh)[0]
        crossing_layer.append(int(crossed[0]) + 1 if len(crossed) > 0 else 0)
    crossing_arr = np.array(crossing_layer)
    never_count  = int((crossing_arr == 0).sum())
    bar_vals     = [int((crossing_arr == li).sum()) for li in range(1, L + 1)]
    if never_count > 0:
        bar_vals.append(never_count)
    xtick_labels = [f"L{i}" for i in range(1, L + 1)]
    if never_count > 0:
        xtick_labels.append("never")
    bar_colors = ["steelblue"] * L + (["salmon"] if never_count > 0 else [])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(bar_vals)), bar_vals, color=bar_colors, alpha=0.85)
    ax.set_xticks(range(len(xtick_labels)))
    ax.set_xticklabels(xtick_labels)
    ax.set_xlabel("First layer where score ≥ threshold")
    ax.set_ylabel("Number of queries")
    ax.set_title(
        f"Activation crossing layer  (score_thresh={score_thresh:.2f})\n"
        f"Healthy: mass on early layers.  "
        f"Mass on L{L} or 'never' → cls_head bottlenecked."
    )
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    p = layers_dir / "layer_activation_crossing.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"[debug] {p}")

    # ── Plot 10: Lane diversity per layer ─────────────────────────────────────
    diversity_per_layer = []
    mid_idx = cl_m.shape[2] // 2
    for li in range(L):
        mids   = cl_m[li, :, mid_idx, :]          # (N_Q, 2) metres
        dists  = np.linalg.norm(
            mids[:, None, :] - mids[None, :, :], axis=-1)   # (N_Q, N_Q)
        triu   = dists[np.triu_indices(N_Q, k=1)]
        diversity_per_layer.append(float(np.mean(triu)))

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(range(1, L + 1), diversity_per_layer,
            "o-", color="darkorange", linewidth=2)
    ax.set_xlabel("Decoder layer")
    ax.set_ylabel("Mean pairwise midpoint distance (m)")
    ax.set_xticks(range(1, L + 1))
    ax.set_xticklabels([f"L{i}" for i in range(1, L + 1)])
    ax.set_title(
        "Lane diversity per layer (mean pairwise midpoint distance)\n"
        "Increasing → queries spreading to distinct positions.  "
        "Flat/decreasing → lane collapse."
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = layers_dir / "layer_lane_diversity.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"[debug] {p}")

    # ── Plot 11: Curvature distribution per layer ─────────────────────────────
    curv_data = []
    for li in range(L):
        radii = [_polyline_mean_radius_m(cl_m[li, qi]) for qi in range(N_Q)]
        curv_data.append(np.array(radii))

    fig, ax = plt.subplots(figsize=(max(7, L * 1.5), 4))
    ax.boxplot(curv_data, positions=range(1, L + 1), widths=0.6,
               patch_artist=True,
               boxprops=dict(facecolor="lightblue", alpha=0.7),
               medianprops=dict(color="navy", linewidth=2),
               flierprops=dict(marker=".", markersize=3, alpha=0.4))
    ax.set_xlabel("Decoder layer")
    ax.set_ylabel("Mean radius of curvature (m)")
    ax.set_xticks(range(1, L + 1))
    ax.set_xticklabels([f"L{i}" for i in range(1, L + 1)])
    ax.set_title(
        "Curvature distribution per layer\n"
        "Increasing spread + higher median → diverse curvatures developing.\n"
        "All layers identical → pts_head ignores decoder refinement."
    )
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    p = layers_dir / "layer_curvature_dist.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"[debug] {p}")

    # ── Summary text ──────────────────────────────────────────────────────────
    mean_norm_per_layer = update_norms.mean(axis=1)      # (L,)
    mean_sim_per_layer  = [
        float(np.mean(np.abs(_cosine_sim_matrix(hs[li]) - np.eye(N_Q))))
        for li in range(L)
    ]
    final_active = int((scores_arr[-1] >= score_thresh).sum())

    summary = [
        "── Layer analysis ──────────────────────────────────",
        f"  n_layers     : {L}",
        f"  N_Q          : {N_Q}",
        f"  active @L{L}  : {final_active} queries  (score>={score_thresh:.2f})",
        "  first-cross  : "
        + "  ".join(f"L{li + 1}={int((crossing_arr == li + 1).sum())}"
                    for li in range(L))
        + (f"  never={never_count}" if never_count > 0 else ""),
        "  update norms (mean per layer): "
        + "  ".join(f"L{li + 1}={v:.3f}"
                    for li, v in enumerate(mean_norm_per_layer)),
        "  mean |off-diag sim| per layer: "
        + "  ".join(f"L{li + 1}={v:.3f}"
                    for li, v in enumerate(mean_sim_per_layer)),
        "  lane diversity (m): "
        + "  ".join(f"L{li + 1}={v:.1f}"
                    for li, v in enumerate(diversity_per_layer)),
        "  curvature median (m): "
        + "  ".join(f"L{li + 1}={float(np.median(curv_data[li])):.0f}"
                    for li in range(L)),
    ]
    if geo_conv is not None:
        summary.append(
            "  geo conv (m): "
            + "  ".join(f"L{i + 1}→L{i + 2}={float(geo_conv[i].mean()):.2f}"
                        for i in range(L - 1))
        )
    if entropy_by_layer:
        ent_dec = (len(entropy_by_layer) > 1 and
                   entropy_by_layer[-1] <= entropy_by_layer[0])
        summary.append(
            "  attn entropy (nats): "
            + "  ".join(f"L{li + 1}={e:.3f}"
                        for li, e in enumerate(entropy_by_layer))
            + (f"  {'decreasing ✓' if ent_dec else 'NOT decreasing (diffuse attn)'}")
        )
    norms_ok   = mean_norm_per_layer[0] >= mean_norm_per_layer[-1]
    sim_ok     = (len(mean_sim_per_layer) > 1 and
                  mean_sim_per_layer[0] >= mean_sim_per_layer[-1])
    div_ok     = (len(diversity_per_layer) > 1 and
                  diversity_per_layer[-1] >= diversity_per_layer[0])
    entropy_ok = (len(entropy_by_layer) > 1 and
                  entropy_by_layer[-1] <= entropy_by_layer[0])
    entropy_interp = (
        "decreasing ✓ (queries sharpening focus)"  if entropy_ok
        else "NOT decreasing — attention stays diffuse"
        if entropy_by_layer else "n/a (no attn hooks captured)"
    )
    summary += [
        "  interpretation:",
        f"    norms      {'decreasing ✓' if norms_ok else 'NOT decreasing — layers may not converge'}",
        f"    similarity {'decreasing ✓ (queries differentiate)' if sim_ok else 'NOT decreasing — check for collapse'}",
        f"    diversity  {'increasing ✓ (queries spreading)' if div_ok else 'NOT increasing — check for lane collapse'}",
        f"    attn entropy {entropy_interp}",
        "",
    ]
    for line in summary:
        print(f"  {line}" if line and not line.startswith("──") else line)
    return summary


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@torch.no_grad()
def run(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ---- Load config --------------------------------------------------------
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None

    # Resolve BEV grid dimensions — configs use 'bev_h_' / 'bev_w_' convention
    bev_h = getattr(cfg, 'bev_h_', None) or getattr(cfg, 'bev_h', 100)
    bev_w = getattr(cfg, 'bev_w_', None) or getattr(cfg, 'bev_w', 200)
    pc_range = list(cfg.point_cloud_range)
    _set_bev_params(pc_range, bev_h, bev_w)
    print(f"[debug] BEV grid: {bev_h}×{bev_w}  pc_range: {pc_range}")

    # ---- Build model --------------------------------------------------------
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        from mmcv.runner import wrap_fp16_model
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = model.to(device)
    model.eval()
    print(f"[debug] Loaded checkpoint: {args.checkpoint}")

    # ---- Detect model variant and map_prior_only mode -----------------------
    model_type = cfg.model.get('type', 'LaneSegNet')
    is_map_prior_model = model_type in ('LaneSegNetMapPrior', 'LaneSegNetMapGraphPrior')
    map_prior_only = getattr(model, 'map_prior_only', False)

    # Inspect loaded model attributes — no need to re-read the checkpoint file.
    has_map_enc = (
        (hasattr(model, 'map_prior_encoder')  and model.map_prior_encoder  is not None) or
        (hasattr(model, 'map_graph_encoder')  and model.map_graph_encoder  is not None)
    )
    has_backbone = hasattr(model, 'img_backbone') and model.img_backbone is not None

    if is_map_prior_model:
        print(f"\n[debug] Model type: {model_type}")
        if map_prior_only:
            print(
                "[debug] map_prior_only=True detected.\n"
                "  In this mode the model receives a map prior raster instead of\n"
                "  camera BEV features.  The zero-BEV prior analysis below uses\n"
                "  a zero tensor as a stand-in for the map prior.  Results show the\n"
                "  model's spatial priors before any map prior signal is seen, which\n"
                "  is less informative than running with real map prior input.\n"
                "  Use the dataset loop (omit --priors_only) for analysis with real\n"
                "  map prior rasters from actual frames."
            )
    elif has_map_enc and not has_backbone:
        print(
            "\n[debug] Model has map encoder weights but no image backbone.\n"
            "  If the forward pass fails, ensure --config matches the checkpoint."
        )

    N_QUERIES = model.pts_bbox_head.num_query
    # Full 512-dim embeddings (query_pos || query) capture both positional
    # and content information — use all dims for PCA / similarity.
    query_embeds = (model.pts_bbox_head.query_embedding.weight
                    .detach().cpu().numpy())
    print(f"[debug] Query embedding shape: {query_embeds.shape}")  # (200, 512)

    # ---- Prior analysis (no dataset required) -------------------------------
    print("\n[debug] === Query prior analysis (zero BEV) ===")
    prior_summary_lines = analyze_query_priors(
        model, out_dir, device,
        score_thresh=args.score_thresh,
        query_grid=args.query_grid,
    )

    print("\n[debug] === Query embedding similarity ===")
    query_similarity_matrix(query_embeds, out_dir)

    # ---- Layer analysis (optional, no dataset required) ---------------------
    layer_summary_lines = []
    if args.layer_analysis:
        mode_label = "zero map-prior raster" if map_prior_only else "zero BEV"
        print(f"\n[debug] === Decoder layer analysis ({mode_label}) ===")
        layer_summary_lines = analyze_decoder_layers(
            model, out_dir, device, score_thresh=args.score_thresh)

    if args.priors_only:
        e3d, var = _pca(query_embeds, n_components=3)
        for xi, yi, suffix in [(0, 1, ""), (0, 2, "13"), (1, 2, "23")]:
            _plot_query_embed_plane(
                e3d, var, xi, yi,
                out_path=out_dir / f"query_embed_pca{suffix}.png",
                title=(f"Query embedding PCA (PC{xi+1} vs PC{yi+1}, "
                       f"colour = query index)"),
                c=range(N_QUERIES), cmap="tab20",
            )

        layer_mode = "+layer_analysis" if layer_summary_lines else ""
        map_note   = f"  [{model_type}, map_prior_only={map_prior_only}]" if is_map_prior_model else ""
        lines = [
            f"checkpoint  : {args.checkpoint}",
            f"config      : {args.config}",
            f"model_type  : {model_type}{map_note}",
            f"mode        : priors_only{layer_mode}",
            f"score_thresh: {args.score_thresh}",
            "",
        ] + prior_summary_lines + ([""] + layer_summary_lines if layer_summary_lines else [])
        with open(out_dir / "summary.txt", "w") as f:
            f.write("\n".join(lines) + "\n")
            f.write(_PLOT_GUIDE)
        print(f"\n[debug] Outputs saved to: {out_dir}")
        return

    # ---- Dataset loop -------------------------------------------------------
    # Select split: prefer args.split, fall back to config test split
    from mmdet.datasets import replace_ImageToTensor
    split_cfg = getattr(cfg.data, args.split, cfg.data.test)
    split_cfg.test_mode = True
    if args.data_root:
        split_cfg.data_root = args.data_root
    split_cfg.pop('samples_per_gpu', None)
    pipeline = split_cfg.get('pipeline', [])
    split_cfg['pipeline'] = replace_ImageToTensor(pipeline)

    dataset     = build_dataset(split_cfg)

    # Optionally restrict to a saved subset (e.g. val_indices.json from training)
    indices_file = None
    if args.indices_file:
        indices_file = Path(args.indices_file)
    else:
        # Auto-detect from checkpoint directory (same convention as training)
        ckpt_dir = Path(args.checkpoint).resolve().parent
        for candidate in [ckpt_dir / "val_indices.json",
                          ckpt_dir / f"{args.split}_indices.json"]:
            if candidate.exists():
                indices_file = candidate
                break

    if indices_file is not None and indices_file.exists():
        with open(indices_file) as f:
            indices = json.load(f)
        from torch.utils.data import Subset
        dataset = Subset(dataset, indices)
        print(f"[debug] Using indices from: {indices_file} ({len(dataset)} samples)")
    else:
        print(f"[debug] Using full {args.split} split ({len(dataset)} samples)")

    data_loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=0,
        dist=False, shuffle=False,
    )
    n_to_process = min(len(dataset), args.n_samples)

    activation_count = np.zeros(N_QUERIES, dtype=np.int32)
    midpoints        = [[] for _ in range(N_QUERIES)]
    query_lengths    = [[] for _ in range(N_QUERIES)]
    bev_vis_saved    = 0

    for sample_idx, data in enumerate(data_loader):
        if sample_idx >= args.n_samples:
            break

        # Unpack mmdet DataContainer format
        img       = data['img'][0].data[0].to(device)   # (bs, N_cam, C, H, W)
        img_metas = data['img_metas'][0].data[0]         # list[dict]

        img_feats = model.extract_feat(img=img, img_metas=img_metas)
        # bev_feats: (bs, bev_h*bev_w, embed_dims)  — encoder returns batch-first
        bev_feats = model.bev_constructor(img_feats, img_metas, prev_bev=None)
        outs      = model.pts_bbox_head(img_feats, bev_feats, img_metas)

        # Last decoder layer, batch 0.  Coordinates already in metres (eval mode).
        cls_logits = outs['all_cls_scores'][-1][0]    # (N_Q, num_classes+1)
        lanes_pred = outs['all_lanes_preds'][-1][0]   # (N_Q, 90) in metres

        scores  = cls_logits[:, 0].sigmoid().cpu().numpy()
        pred_np = lanes_pred.cpu().numpy().reshape(N_QUERIES, 3, 10, 3)
        cl_m    = pred_np[:, 0, :, :2]   # (N_Q, 10, 2) metres — centerline XY

        # Decide which queries to record this sample
        if args.top_k > 0:
            record_mask = np.zeros(N_QUERIES, dtype=bool)
            record_mask[np.argsort(scores)[::-1][:args.top_k]] = True
        else:
            record_mask = scores >= args.score_thresh

        activation_count += record_mask.astype(np.int32)

        for q in range(N_QUERIES):
            if record_mask[q]:
                try:
                    r, c = _bev_midpoint_pixel(cl_m[q])
                    midpoints[q].append((r, c))
                    query_lengths[q].append(_polyline_length_m(cl_m[q]))
                except Exception:
                    pass

        # BEV PCA image for first n_bev_vis samples
        if bev_vis_saved < args.n_bev_vis:
            # bev_feats[0]: (bev_h*bev_w, embed_dims) — batch index 0
            bev_np   = bev_feats[0].detach().cpu().float().numpy()
            bev_grid = bev_np.reshape(bev_h, bev_w, -1).transpose(2, 0, 1)
            pca_rgb  = bev_pca_image(bev_grid)   # (bev_h, bev_w, 3) RGB
            cv2.imwrite(
                str(out_dir / f"bev_features_pca_{sample_idx:04d}.png"),
                cv2.cvtColor(pca_rgb, cv2.COLOR_RGB2BGR),
            )
            bev_vis_saved += 1

        if (sample_idx + 1) % 10 == 0:
            pct = 100.0 * (scores >= args.score_thresh).mean()
            print(f"  sample {sample_idx + 1}/{n_to_process}  "
                  f"active: {record_mask.sum()}  "
                  f"score [{scores.min():.3f}, {scores.max():.3f}]  "
                  f"pct>={args.score_thresh:.2f}: {pct:.0f}%")

    n_total = min(sample_idx + 1, args.n_samples)

    # ---- Plot 1: Query activation frequency ---------------------------------
    fig, ax = plt.subplots(figsize=(18, 4))
    colors = ["tomato" if c == 0 else "steelblue" for c in activation_count]
    ax.bar(range(N_QUERIES), activation_count, width=1.0, color=colors)
    ax.axhline(n_total * 0.5, color="orange", linestyle="--", linewidth=1.0,
               label="50% of samples")
    ax.set_xlabel("Query index")
    ax.set_ylabel(f"Activation count  (score >= {args.score_thresh})")
    ax.set_title(
        f"Query activation frequency — {n_total} samples\n"
        "Red = never active.  "
        "Healthy: spread across many queries.  "
        "Broken: all mass on 2–3 bars (collapse) or all red (nothing fires)."
    )
    ax.legend()
    plt.tight_layout()
    p = out_dir / "query_activation_freq.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"[debug] {p}")

    # ---- Plot 2: Spatial spread in BEV --------------------------------------
    canvas = np.full((_BEV_H, _BEV_W, 3), 30, dtype=np.uint8)
    for r in range(0, _BEV_H, 20):
        cv2.line(canvas, (0, r), (_BEV_W - 1, r), (55, 55, 55), 1)
    for c in range(0, _BEV_W, 20):
        cv2.line(canvas, (c, 0), (c, _BEV_H - 1), (55, 55, 55), 1)
    cv2.line(canvas, (_BEV_W // 2, 0), (_BEV_W // 2, _BEV_H - 1), (80, 80, 80), 1)
    cmap = matplotlib.colormaps["tab20"]
    for q in range(N_QUERIES):
        if not midpoints[q]:
            continue
        bgr = (np.array(cmap(q % 20)[:3][::-1]) * 255).astype(int).tolist()
        for rr, cc in midpoints[q]:
            cv2.circle(canvas,
                       (int(np.clip(cc, 0, _BEV_W - 1)),
                        int(np.clip(rr, 0, _BEV_H - 1))),
                       2, bgr, -1)
    cv2.putText(canvas, "front", (_BEV_W // 2 + 2, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.putText(canvas, "rear",  (_BEV_W // 2 + 2, _BEV_H - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    p = out_dir / "query_spatial_spread.png"
    cv2.imwrite(str(p), canvas)
    print(f"[debug] {p}")

    # ---- Plot 3: Embedding PCA (colour = activation count) ------------------
    e3d, var = _pca(query_embeds, n_components=3)
    for xi, yi, suffix in [(0, 1, ""), (0, 2, "13"), (1, 2, "23")]:
        _plot_query_embed_plane(
            e3d, var, xi, yi,
            out_path=out_dir / f"query_embed_pca{suffix}.png",
            title=(f"Query embedding PCA (PC{xi+1} vs PC{yi+1}, "
                   f"colour = activation count)\n"
                   "Healthy: bright clusters separate from dim.  "
                   "Broken: random scatter, uniform brightness."),
            c=activation_count, cmap="plasma",
            colorbar_label="Activation count",
        )

    # ---- Plot 4: Length distribution ----------------------------------------
    print("\n[debug] === Query length distribution ===")
    plot_query_lengths(query_lengths, out_dir)

    # ---- Text summary -------------------------------------------------------
    n_dead   = int((activation_count == 0).sum())
    n_always = int((activation_count == n_total).sum())
    top5     = list(np.argsort(activation_count)[::-1][:5])
    top5_share = (activation_count[top5].sum() /
                  max(activation_count.sum(), 1))
    all_lens = [l for ql in query_lengths for l in ql]
    len_summary = (
        f"min={min(all_lens):.1f}m  max={max(all_lens):.1f}m  "
        f"mean={np.mean(all_lens):.1f}m  "
        f"near-zero(<1m)={sum(1 for l in all_lens if l < 1.0)}/{len(all_lens)}"
    ) if all_lens else "no active queries"

    collapse_warn = "  ← >80% = likely mode collapse" if top5_share > 0.8 else ""
    lines = [
        f"checkpoint  : {args.checkpoint}",
        f"config      : {args.config}",
        f"split       : {args.split}",
        f"n_samples   : {n_total}",
        f"score_thresh: {args.score_thresh}",
        f"top_k       : {args.top_k}",
        "",
        f"dead queries  (never active): {n_dead}/{N_QUERIES}",
        f"always active (every sample): {n_always}/{N_QUERIES}",
        f"top-5 queries: {top5}  ({top5_share:.1%} of all activations){collapse_warn}",
        f"lane lengths : {len_summary}",
        "",
    ] + prior_summary_lines + ([""] + layer_summary_lines if layer_summary_lines else [])

    with open(out_dir / "summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
        f.write(_PLOT_GUIDE)
    print(f"\n[debug] All outputs saved to: {out_dir}")
    print(f"[debug] {out_dir / 'summary.txt'}")


def parse_args():
    p = argparse.ArgumentParser(description="LaneSegNet query debug / visualisation tool")
    p.add_argument("--config",      required=True,
                   help="mmdet config file path")
    p.add_argument("--checkpoint",  required=True,
                   help="model checkpoint (.pth)")
    p.add_argument("--data_root",   default=None,
                   help="dataset root directory (overrides config data_root)")
    p.add_argument("--output_dir",  default="./debug_out",
                   help="directory for all output images and summary.txt")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--priors_only", action="store_true",
                   help="only run zero-BEV prior analysis (no dataset needed)")
    p.add_argument("--layer_analysis", action="store_true",
                   help="run per-layer decoder analysis; outputs to <output_dir>/layers/")
    p.add_argument("--split",       default="test",
                   choices=["train", "val", "test"],
                   help="dataset split for the full analysis loop (default: test)")
    p.add_argument("--indices_file", default=None,
                   help="JSON list of dataset indices to use (auto-detected from "
                        "checkpoint dir if omitted)")
    p.add_argument("--n_samples",   type=int, default=50,
                   help="max dataset samples for the full analysis loop")
    p.add_argument("--score_thresh", type=float, default=0.3,
                   help="classification score threshold for 'active' queries")
    p.add_argument("--top_k",       type=int, default=0,
                   help="if >0, record top-k queries per sample instead of thresh")
    p.add_argument("--n_bev_vis",   type=int, default=5,
                   help="number of BEV PCA images to save from the dataset loop")
    p.add_argument("--query_grid",  action="store_true",
                   help="also generate per-query BEV grid panels (query_grid_NN.png)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
