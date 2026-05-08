"""
03_figures.py
=============
Figures for the **novice single-image** workflow.

Reads ``output/pm_full.png``, ``output/pnp_results.json``, and landmarks under
``landmarks/CBCT_3D`` / ``landmarks/PM_2D``. Generates overlays for the
**active** scenario only (A or B), as recorded by ``02_pnp.py``.

Usage
-----
    python3 03_figures.py

Dependencies
------------
    pip install numpy opencv-python matplotlib trimesh
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import trimesh

BASE = Path(__file__).parent

MESH1_PATH = BASE / "meshes" / "mesh_restorations.obj"
MESH2_PATH = BASE / "meshes" / "mesh_bone.obj"

LM_3D_DIR = BASE / "landmarks" / "CBCT_3D"
LM_2D_DIR = BASE / "landmarks" / "PM_2D"

META_JSON = BASE / "output" / "case_metadata.json"
RES_JSON = BASE / "output" / "pnp_results.json"
FIG_DIR = BASE / "output" / "figures"

GREEN = (0, 210, 80)
BLUE = (220, 80, 0)


def project_points(pts3: np.ndarray, K, rvec, tvec, dist) -> np.ndarray:
    proj, _ = cv2.projectPoints(pts3.astype(np.float64), rvec, tvec, K, dist)
    return proj.reshape(-1, 2)


def camera_world_position(rvec, tvec) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    return (-R.T @ tvec).flatten()


def silhouette_edges(
    verts: np.ndarray, faces: np.ndarray, cam: np.ndarray
) -> list[tuple]:
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    centers = (v0 + v1 + v2) / 3.0
    dot = (normals * (cam - centers)).sum(axis=1)
    front = dot > 0

    edge_to_tris: dict = {}
    for fi, face in enumerate(faces):
        for a, b in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]:
            key = (min(a, b), max(a, b))
            edge_to_tris.setdefault(key, []).append(fi)

    sil = []
    for edge, tris in edge_to_tris.items():
        if len(tris) == 1:
            if front[tris[0]]:
                sil.append(edge)
        elif front[tris[0]] != front[tris[1]]:
            sil.append(edge)
    return sil


def draw_silhouette(
    img: np.ndarray,
    pts2d: np.ndarray,
    edges: list,
    color: tuple,
    alpha: float,
    thickness: int = 1,
) -> np.ndarray:
    h, w = img.shape[:2]
    overlay = img.copy()
    for a, b in edges:
        pa = tuple(pts2d[a].astype(int))
        pb = tuple(pts2d[b].astype(int))
        if not (
            0 <= pa[0] < w
            and 0 <= pa[1] < h
            and 0 <= pb[0] < w
            and 0 <= pb[1] < h
        ):
            continue
        cv2.line(overlay, pa, pb, color, thickness, cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def load_fcsv_labels(directory: Path) -> dict[str, list[float]]:
    result: dict = {}
    for fcsv in sorted(directory.glob("*.fcsv")):
        with open(fcsv, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                parts = line.split(",")
                if len(parts) < 12:
                    continue
                lbl = parts[11].strip()
                if lbl:
                    try:
                        result[lbl] = [
                            float(parts[1]),
                            float(parts[2]),
                            float(parts[3]),
                        ]
                    except ValueError:
                        pass
    return result


def _draw_landmarks_on_ax(
    ax, img_bgr, pts2_manual, proj_pts, labels, per_pt, title,
    fontsize_annot=7, ms_pt=8, ms_proj=12,
):
    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    for lbl, p2, pp, err in zip(labels, pts2_manual, proj_pts, per_pt):
        ax.plot(*p2, "o", color="#00CFFF", ms=ms_pt, mew=1.5, mec="black")
        ax.plot(*pp, "+", color="tomato", ms=ms_proj, mew=2)
        ax.annotate(
            f"{lbl}\n{err:.1f}px",
            p2,
            textcoords="offset points",
            xytext=(7, 7),
            fontsize=fontsize_annot,
            color="white",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55),
        )
    rmse_val = float(np.sqrt(np.mean(np.array(per_pt) ** 2)))
    handles = [
        mpatches.Patch(color="#00CFFF", label="Manual 2D landmarks"),
        mpatches.Patch(color="tomato", label="Reprojected 3D landmarks"),
    ]
    ax.set_title(
        f"{title}  |  RMSE = {rmse_val:.2f} px  ({len(labels)} landmarks)",
        fontsize=11,
        fontweight="bold",
    )
    ax.axis("off")
    return rmse_val, handles


def figure_landmarks(img_bgr, pts2_manual, proj_pts, labels, per_pt, title, out_path):
    fig, ax = plt.subplots(figsize=(14, 10))
    _, handles = _draw_landmarks_on_ax(
        ax, img_bgr, pts2_manual, proj_pts, labels, per_pt, title
    )
    ax.legend(handles=handles, loc="upper right", fontsize=9)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path.name}")


def compute_zoom_box(img_bgr, pts_list, pad_px=150):
    all_pts = np.vstack(pts_list)
    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)
    h, w = img_bgr.shape[:2]
    x1 = max(0, int(x_min) - pad_px)
    y1 = max(0, int(y_min) - pad_px)
    x2 = min(w, int(x_max) + pad_px)
    y2 = min(h, int(y_max) + pad_px)
    return x1, y1, x2, y2


def figure_landmarks_zoomed(
    img_bgr, pts2_manual, proj_pts, labels, per_pt, title, out_path, zoom_box,
):
    x1, y1, x2, y2 = zoom_box
    fig, ax = plt.subplots(figsize=(14, 10))
    _, handles = _draw_landmarks_on_ax(
        ax,
        img_bgr,
        pts2_manual,
        proj_pts,
        labels,
        per_pt,
        title + " [zoomed]",
        fontsize_annot=10,
        ms_pt=12,
        ms_proj=18,
    )
    ax.set_xlim(x1, x2)
    ax.set_ylim(y2, y1)
    ax.legend(handles=handles, loc="best", fontsize=10)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path.name}")


def expand_zoom_box(zoom_box, img_bgr, factor=1.0):
    if factor == 1.0:
        return zoom_box
    x1, y1, x2, y2 = zoom_box
    h, w = img_bgr.shape[:2]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    half_w = (x2 - x1) / 2.0 * factor
    half_h = (y2 - y1) / 2.0 * factor
    return (
        max(0, int(cx - half_w)),
        max(0, int(cy - half_h)),
        min(w, int(cx + half_w)),
        min(h, int(cy + half_h)),
    )


def save_zoomed_crop(img_bgr, zoom_box, out_path):
    x1, y1, x2, y2 = zoom_box
    crop = img_bgr[y1:y2, x1:x2]
    cv2.imwrite(str(out_path), crop)
    print(f"  → {out_path.name}")


def main():
    print("=" * 60)
    print("03_figures.py — single active scenario")
    print("=" * 60)

    if not RES_JSON.exists():
        print(f"⚠  Missing {RES_JSON.name} — run 02_pnp.py first.")
        return
    if not META_JSON.exists():
        print(f"⚠  Missing {META_JSON.name}")
        return

    with open(META_JSON, encoding="utf-8") as fh:
        meta = json.load(fh)
    with open(RES_JSON, encoding="utf-8") as fh:
        res = json.load(fh)

    block = res.get("scenario_block")
    sc_data = res.get(block) if block else None
    if not sc_data or not sc_data.get("converged"):
        print("⚠  PnP results missing or did not converge.")
        return

    png_path = BASE / meta["png_path"]
    if not png_path.exists():
        print(f"⚠  Image not found: {png_path}")
        return

    img_bgr = cv2.imread(str(png_path))
    if img_bgr is None:
        print(f"⚠  Could not read {png_path}")
        return

    lm2d = load_fcsv_labels(LM_2D_DIR)
    lm3d = load_fcsv_labels(LM_3D_DIR)
    labels = res["labels"]
    coord = res.get("lm_2d_coord_system", "mm")
    pix = float(res.get("lm_2d_pixel_mm", 1.0))

    pts3 = np.array([lm3d[k] for k in labels if k in lm3d], dtype=np.float64)
    pts2 = np.array([lm2d[k][:2] for k in labels if k in lm2d], dtype=np.float64)
    if coord == "mm" and len(pts2):
        pts2 = pts2 / pix

    ffd_ref = meta.get("FFD_mm")
    ffd_str = f"{ffd_ref:.0f} mm" if ffd_ref else "n/a"

    sc_tag = block.split("_")[-1].upper() if block else "?"

    K = np.array(sc_data["K_matrix"], dtype=np.float64)
    rvec = np.array(sc_data["rvec"], dtype=np.float64).reshape(3, 1)
    tvec = np.array(sc_data["tvec"], dtype=np.float64).reshape(3, 1)
    dist = np.array(sc_data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)

    print(f"\nScenario {sc_tag}  |  RMSE = {sc_data['RMSE_px']:.2f} px")
    print(f"  FFD (metadata) : {ffd_str}")

    per_pt = list(
        sc_data.get("per_point_px", {k: 0.0 for k in labels}).values()
    )
    proj = project_points(pts3, K, rvec, tvec, dist)
    zoom_box = compute_zoom_box(img_bgr, [pts2, proj])

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    series = meta.get("series_description") or meta.get("input_filename", "")

    title = f"Scenario {sc_tag} — {series}"
    figure_landmarks(
        img_bgr,
        pts2,
        proj,
        labels,
        per_pt,
        title,
        FIG_DIR / f"overlay_landmarks_{sc_tag}.png",
    )
    figure_landmarks_zoomed(
        img_bgr,
        pts2,
        proj,
        labels,
        per_pt,
        title,
        FIG_DIR / f"overlay_landmarks_{sc_tag}_zoomed.png",
        zoom_box,
    )

    print("\nLoading meshes…")
    m1 = trimesh.load(str(MESH1_PATH), force="mesh")
    v1 = np.array(m1.vertices, dtype=np.float64)
    f1 = np.array(m1.faces, dtype=np.int64)
    m2 = trimesh.load(str(MESH2_PATH), force="mesh")
    v2 = np.array(m2.vertices, dtype=np.float64)
    f2 = np.array(m2.faces, dtype=np.int64)

    cam = camera_world_position(rvec, tvec)
    pts1_2d = project_points(v1, K, rvec, tvec, dist)
    sil1 = silhouette_edges(v1, f1, cam)
    pts2_2d = project_points(v2, K, rvec, tvec, dist)
    sil2 = silhouette_edges(v2, f2, cam)
    print(f"  Silhouette edges : restorations {len(sil1):,}  |  bone {len(sil2):,}")

    mesh_zoom_box = expand_zoom_box(zoom_box, img_bgr, factor=1.3)

    r2 = draw_silhouette(img_bgr.copy(), pts1_2d, sil1, GREEN, alpha=0.80)
    p2 = FIG_DIR / f"mesh_silhouette_restorations_{sc_tag}.png"
    cv2.imwrite(str(p2), r2)
    print(f"  → {p2.name}")
    save_zoomed_crop(
        r2, mesh_zoom_box, FIG_DIR / f"mesh_silhouette_restorations_{sc_tag}_zoomed.png"
    )

    r3 = draw_silhouette(img_bgr.copy(), pts2_2d, sil2, BLUE, alpha=0.80)
    p3 = FIG_DIR / f"mesh_silhouette_bone_{sc_tag}.png"
    cv2.imwrite(str(p3), r3)
    print(f"  → {p3.name}")
    save_zoomed_crop(
        r3, mesh_zoom_box, FIG_DIR / f"mesh_silhouette_bone_{sc_tag}_zoomed.png"
    )

    r4 = draw_silhouette(img_bgr.copy(), pts1_2d, sil1, GREEN, alpha=0.25)
    r4 = draw_silhouette(r4, pts2_2d, sil2, BLUE, alpha=0.25)
    p4 = FIG_DIR / f"mesh_combined_{sc_tag}.png"
    cv2.imwrite(str(p4), r4)
    print(f"  → {p4.name}")
    save_zoomed_crop(r4, zoom_box, FIG_DIR / f"mesh_combined_{sc_tag}_zoomed.png")

    print(f"\n✔ Figures saved under {FIG_DIR.relative_to(BASE)}")


if __name__ == "__main__":
    main()
