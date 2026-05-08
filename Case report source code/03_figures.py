"""
03_figures.py
=============
Step 3 — Publication-quality figure generation.

For each PM image (img1, img2), produces:
  - overlay_landmarks_{scenario}.png   : manually annotated 2D landmarks
                                         + reprojected 3D CBCT landmarks
  - mesh_silhouette_green_{scenario}.png : dental restoration silhouette (green)
  - mesh_silhouette_blue_{scenario}.png  : bone/osseous structure silhouette (blue)
  - mesh_combined_{scenario}.png         : combined overlay (25% opacity each)

The scenario used for figures is configurable (see SCENARIO constant below).

Usage
-----
    python3 03_figures.py

Dependencies
------------
    pip install numpy opencv-python matplotlib trimesh Pillow pydicom
    pip install pylibjpeg pylibjpeg-libjpeg
"""

import json
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import trimesh

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent

MESH1_PATH = BASE / "meshes" / "mesh_restorations.obj"   # dental restorations (green)
MESH2_PATH = BASE / "meshes" / "mesh_bone.obj"           # osseous structures  (blue)

LM_3D_DIR = BASE / "landmarks" / "AM_3D"
LM_2D_DIR = BASE / "landmarks"

DICOM_DIR  = BASE.parent / "Data" / "PM-2D-Anon" / "UNNAMED"
DICOM_IMG1 = DICOM_DIR / "00010001"
DICOM_IMG2 = DICOM_DIR / "00020001"

META_JSON = BASE / "output" / "case_metadata.json"
RES_IMG1  = BASE / "output" / "img1" / "pnp_results.json"
RES_IMG2  = BASE / "output" / "img2" / "pnp_results.json"

# Scenarios to generate figures for (both A and B by default)
SCENARIOS = ["scenario_A", "scenario_B"]

# Same as 02_pnp.py — must be consistent
LM_2D_COORD_SYSTEM = "mm"    # "mm" (DICOM in Slicer) or "px" (JPEG/PNG)
LM_2D_PIXEL_MM     = 0.148   # Detector pixel pitch (mm/px)

# BGR colours (OpenCV convention)
GREEN = (0,  210,  80)
BLUE  = (220,  80,   0)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_dicom_image(dicom_path: Path) -> np.ndarray:
    """
    Load a CR DICOM and return a BGR uint8 array (OpenCV-compatible).
    Applies native DICOM window/level settings.
    """
    ds  = pydicom.dcmread(str(dicom_path))
    arr = ds.pixel_array.astype(np.float64)
    slope     = float(getattr(ds, "RescaleSlope",     1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    wc = float(ds.WindowCenter)
    ww = float(ds.WindowWidth)
    lo, hi = wc - ww / 2, wc + ww / 2
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo) * 255.0

    invert = (getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
              == "MONOCHROME1")
    if invert:
        arr = 255.0 - arr

    gray = arr.astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def project_points(pts3: np.ndarray, K, rvec, tvec, dist) -> np.ndarray:
    """Project 3D world points onto the image plane."""
    proj, _ = cv2.projectPoints(pts3.astype(np.float64), rvec, tvec, K, dist)
    return proj.reshape(-1, 2)


def camera_world_position(rvec, tvec) -> np.ndarray:
    """Compute camera centre in world (LPS) coordinates."""
    R, _ = cv2.Rodrigues(rvec)
    return (-R.T @ tvec).flatten()


def silhouette_edges(verts: np.ndarray, faces: np.ndarray,
                     cam: np.ndarray) -> list[tuple]:
    """
    Compute silhouette edges of a triangle mesh as seen from camera position.

    A silhouette edge is shared by one front-facing and one back-facing
    triangle, or belongs to a boundary (single triangle).
    """
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    normals  = np.cross(v1 - v0, v2 - v0)
    centers  = (v0 + v1 + v2) / 3.0
    dot      = (normals * (cam - centers)).sum(axis=1)
    front    = dot > 0

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


def draw_silhouette(img: np.ndarray, pts2d: np.ndarray,
                    edges: list, color: tuple,
                    alpha: float, thickness: int = 1) -> np.ndarray:
    """Overlay silhouette edges on a copy of img with transparency."""
    h, w = img.shape[:2]
    overlay = img.copy()
    for a, b in edges:
        pa = tuple(pts2d[a].astype(int))
        pb = tuple(pts2d[b].astype(int))
        if not (0 <= pa[0] < w and 0 <= pa[1] < h and
                0 <= pb[0] < w and 0 <= pb[1] < h):
            continue
        cv2.line(overlay, pa, pb, color, thickness, cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def load_fcsv_labels(directory: Path) -> dict[str, list[float]]:
    """Load all FCSV files in directory, keyed by the label column."""
    result: dict = {}
    for fcsv in sorted(directory.glob("*.fcsv")):
        with open(fcsv) as fh:
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
                        result[lbl] = [float(parts[1]),
                                       float(parts[2]),
                                       float(parts[3])]
                    except ValueError:
                        pass
    return result


def _draw_landmarks_on_ax(ax, img_bgr, pts2_manual, proj_pts, labels, per_pt,
                          title, fontsize_annot=7, ms_pt=8, ms_proj=12):
    """
    Internal helper: draw landmark overlay on a pre-created Axes.
    Returns (rmse_val, handles) for legend/title use.
    """
    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    for lbl, p2, pp, err in zip(labels, pts2_manual, proj_pts, per_pt):
        ax.plot(*p2, "o", color="#00CFFF", ms=ms_pt, mew=1.5, mec="black")
        ax.plot(*pp, "+", color="tomato",  ms=ms_proj, mew=2)
        ax.annotate(f"{lbl}\n{err:.1f}px", p2,
                    textcoords="offset points", xytext=(7, 7),
                    fontsize=fontsize_annot, color="white", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55))

    rmse_val = float(np.sqrt(np.mean(np.array(per_pt) ** 2)))
    handles  = [
        mpatches.Patch(color="#00CFFF", label="Manual 2D landmarks"),
        mpatches.Patch(color="tomato",  label="Reprojected 3D CBCT landmark"),
    ]
    ax.set_title(f"{title}  |  RMSE = {rmse_val:.2f} px  ({len(labels)} landmarks)",
                 fontsize=11, fontweight="bold")
    ax.axis("off")
    return rmse_val, handles


def figure_landmarks(img_bgr, pts2_manual, proj_pts, labels,
                     per_pt, title, out_path):
    """
    Create and save a full-image landmark overlay figure showing:
      - cyan/orange circles : manually annotated 2D landmarks
      - red crosses         : reprojected 3D CBCT landmarks
      - yellow lines        : reprojection error vectors
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    rmse_val, handles = _draw_landmarks_on_ax(
        ax, img_bgr, pts2_manual, proj_pts, labels, per_pt, title)
    ax.legend(handles=handles, loc="upper right", fontsize=9)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path.name}")


def compute_zoom_box(img_bgr, pts_list, pad_px=150):
    """
    Compute the crop bounding box (x1, y1, x2, y2) that encompasses all
    point arrays in *pts_list* with *pad_px* pixels of margin on each side.
    Coordinates are clamped to the image dimensions.
    """
    all_pts = np.vstack(pts_list)
    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)
    h, w = img_bgr.shape[:2]
    x1 = max(0, int(x_min) - pad_px)
    y1 = max(0, int(y_min) - pad_px)
    x2 = min(w, int(x_max) + pad_px)
    y2 = min(h, int(y_max) + pad_px)
    return x1, y1, x2, y2


def figure_landmarks_zoomed(img_bgr, pts2_manual, proj_pts, labels,
                            per_pt, title, out_path, zoom_box):
    """
    Create and save a zoomed landmark overlay figure restricted to *zoom_box*
    (x1, y1, x2, y2).  Annotations and markers are scaled up for legibility.
    """
    x1, y1, x2, y2 = zoom_box
    fig, ax = plt.subplots(figsize=(14, 10))
    _, handles = _draw_landmarks_on_ax(
        ax, img_bgr, pts2_manual, proj_pts, labels, per_pt,
        title + " [zoomed]", fontsize_annot=10, ms_pt=12, ms_proj=18)
    ax.set_xlim(x1, x2)
    ax.set_ylim(y2, y1)   # y-axis inverted in image space
    ax.legend(handles=handles, loc="best", fontsize=10)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path.name}")


def expand_zoom_box(zoom_box, img_bgr, factor=1.0):
    """
    Expand (or shrink) a zoom_box around its centre by *factor*.
    factor > 1  → wider view (less zoomed in).
    Coordinates are clamped to the image dimensions.
    """
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
    """
    Crop an OpenCV BGR image to *zoom_box* (x1, y1, x2, y2) and save it.
    Used to produce zoomed versions of pre-rendered mesh overlay images.
    """
    x1, y1, x2, y2 = zoom_box
    crop = img_bgr[y1:y2, x1:x2]
    cv2.imwrite(str(out_path), crop)
    print(f"  → {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# PER-IMAGE PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process(img_label: str, dicom_path: Path, res_path: Path,
            lm2d_dir: Path,
            mesh_v1, mesh_f1, mesh_v2, mesh_f2):
    """Generate all publication figures for one PM image (both scenarios)."""
    print(f"\n{'='*60}")
    print(f"Image: {img_label}")
    print(f"{'='*60}")

    if not res_path.exists():
        print(f"  ⚠  {res_path} not found — run 02_pnp.py first.")
        return

    with open(res_path) as fh:
        res = json.load(fh)

    if res.get("status") in ("landmarks_missing", "insufficient_landmarks"):
        print(f"  ⚠  Landmarks missing for {img_label}.")
        return

    # Load image and landmarks once (shared across scenarios)
    img_bgr = load_dicom_image(dicom_path)
    out_dir = BASE / "output" / img_label
    out_dir.mkdir(parents=True, exist_ok=True)

    lm2d: dict = load_fcsv_labels(lm2d_dir) if lm2d_dir.exists() else {}
    lm3d: dict = load_fcsv_labels(LM_3D_DIR)

    labels = res["labels"]
    pts3   = np.array([lm3d[k]     for k in labels if k in lm3d], dtype=np.float64)
    pts2   = np.array([lm2d[k][:2] for k in labels if k in lm2d], dtype=np.float64)
    if LM_2D_COORD_SYSTEM == "mm" and len(pts2):
        pts2 = pts2 / LM_2D_PIXEL_MM   # mm → pixels

    ffd_ref = res["FFD_mm_DICOM"]
    series  = res["series_description"]

    # ── Loop over scenarios ───────────────────────────────────────────────────
    for scenario in SCENARIOS:
        sc = res.get(scenario)
        if not sc or not sc.get("converged"):
            print(f"  ⚠  Scenario '{scenario}' did not converge — skipped.")
            continue

        sc_tag = scenario.split("_")[1].upper()

        K    = np.array(sc["K_matrix"],    dtype=np.float64)
        rvec = np.array(sc["rvec"],        dtype=np.float64).reshape(3, 1)
        tvec = np.array(sc["tvec"],        dtype=np.float64).reshape(3, 1)
        dist = np.array(sc["dist_coeffs"], dtype=np.float64).reshape(-1, 1)

        print(f"\n  — Scenario {sc_tag} —")
        print(f"  RMSE = {sc['RMSE_px']:.2f} px ({sc['RMSE_mm']:.3f} mm)  |  "
              f"f = {sc['focal_px']:.0f} px = {sc['focal_mm']:.0f} mm  |  "
              f"FFD DICOM = {ffd_ref:.0f} mm")

        per_pt = list(sc.get("per_point_px", {k: 0.0 for k in labels}).values())
        proj   = project_points(pts3, K, rvec, tvec, dist)

        # Shared zoom box (landmark bounding box + padding), consistent across all figures
        zoom_box = compute_zoom_box(img_bgr, [pts2, proj])

        # Figure 1: Landmark overlay (full + zoomed)
        title    = f"Scenario {sc_tag} — {img_label} — {series}"
        out_lm   = out_dir / f"overlay_landmarks_{sc_tag}.png"
        out_zoom = out_dir / f"overlay_landmarks_{sc_tag}_zoomed.png"
        figure_landmarks(img_bgr, pts2, proj, labels, per_pt, title, out_lm)
        figure_landmarks_zoomed(img_bgr, pts2, proj, labels, per_pt, title, out_zoom, zoom_box)

        # Silhouette projections
        cam = camera_world_position(rvec, tvec)

        print(f"  Projecting mesh 1 (restorations)…")
        pts1_2d = project_points(mesh_v1, K, rvec, tvec, dist)
        sil1    = silhouette_edges(mesh_v1, mesh_f1, cam)
        print(f"    {len(sil1):,} silhouette edges")

        print(f"  Projecting mesh 2 (bone)…")
        pts2_2d = project_points(mesh_v2, K, rvec, tvec, dist)
        sil2    = silhouette_edges(mesh_v2, mesh_f2, cam)
        print(f"    {len(sil2):,} silhouette edges")

        # Slightly wider zoom for silhouette-only figures (no landmark labels to anchor)
        mesh_zoom_box = expand_zoom_box(zoom_box, img_bgr, factor=1.3)

        # Figure 2: green silhouette (restorations, 80%) — full + zoomed
        r2 = draw_silhouette(img_bgr.copy(), pts1_2d, sil1, GREEN, alpha=0.80)
        p2 = out_dir / f"mesh_silhouette_restorations_{sc_tag}.png"
        cv2.imwrite(str(p2), r2)
        print(f"  → {p2.name}")
        save_zoomed_crop(r2, mesh_zoom_box, out_dir / f"mesh_silhouette_restorations_{sc_tag}_zoomed.png")

        # Figure 3: blue silhouette (bone, 80%) — full + zoomed
        r3 = draw_silhouette(img_bgr.copy(), pts2_2d, sil2, BLUE, alpha=0.80)
        p3 = out_dir / f"mesh_silhouette_bone_{sc_tag}.png"
        cv2.imwrite(str(p3), r3)
        print(f"  → {p3.name}")
        save_zoomed_crop(r3, mesh_zoom_box, out_dir / f"mesh_silhouette_bone_{sc_tag}_zoomed.png")

        # Figure 4: combined overlay (25% + 25%) — full + zoomed
        r4 = draw_silhouette(img_bgr.copy(), pts1_2d, sil1, GREEN, alpha=0.25)
        r4 = draw_silhouette(r4,             pts2_2d, sil2, BLUE,  alpha=0.25)
        p4 = out_dir / f"mesh_combined_{sc_tag}.png"
        cv2.imwrite(str(p4), r4)
        print(f"  → {p4.name}")
        save_zoomed_crop(r4, zoom_box, out_dir / f"mesh_combined_{sc_tag}_zoomed.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"03_figures.py — Figure generation (Scenarios: {', '.join(SCENARIOS)})")
    print("=" * 60)

    # ── Load meshes ───────────────────────────────────────────────────────────
    print("\nLoading meshes…")
    m1 = trimesh.load(str(MESH1_PATH), force="mesh")
    v1 = np.array(m1.vertices, dtype=np.float64)
    f1 = np.array(m1.faces,    dtype=np.int64)
    print(f"  Mesh 1 (restorations) : {len(v1):,} vertices  {len(f1):,} triangles")

    m2 = trimesh.load(str(MESH2_PATH), force="mesh")
    v2 = np.array(m2.vertices, dtype=np.float64)
    f2 = np.array(m2.faces,    dtype=np.int64)
    print(f"  Mesh 2 (bone)         : {len(v2):,} vertices  {len(f2):,} triangles")

    # ── Process each image ────────────────────────────────────────────────────
    lm2d_img1 = LM_2D_DIR / "PM_2D_img1"
    lm2d_img2 = LM_2D_DIR / "PM_2D_img2"

    process("img1", DICOM_IMG1, RES_IMG1, lm2d_img1, v1, f1, v2, f2)
    process("img2", DICOM_IMG2, RES_IMG2, lm2d_img2, v1, f1, v2, f2)

    print("\n✔ Done — figures saved in Workflow_v1/output/img1/ and img2/")


if __name__ == "__main__":
    main()
