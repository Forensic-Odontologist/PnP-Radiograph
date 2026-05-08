"""
02_pnp.py
=========
Step 2 — PnP registration for both PM radiographic images.

For each image (img1, img2), runs:
  - Scenario A: fixed focal length derived from DICOM acquisition parameters
                (DistanceSourceToReferencePoint = 650 mm, pixel pitch = 0.148 mm/px
                 → f = 4392 px). This is the calibrated scenario.
  - Scenario B: free focal length estimation (uncalibrated scenario).

Landmarks are loaded from:
  - landmarks/AM_3D/       → 3D FCSV from 3D Slicer (CBCT, LPS, mm)
  - landmarks/PM_2D_img1/  → 2D FCSV from 3D Slicer (Image 1, LPS, mm)
  - landmarks/PM_2D_img2/  → 2D FCSV from 3D Slicer (Image 2, LPS, mm)

Outputs
-------
    output/img1/pnp_results.json
    output/img2/pnp_results.json

Usage
-----
    python3 02_pnp.py

Dependencies
------------
    pip install numpy opencv-python
"""

import json
from pathlib import Path

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent

LM_3D_DIR  = BASE / "landmarks" / "AM_3D"
LM_2D_IMG1 = BASE / "landmarks" / "PM_2D_img1"
LM_2D_IMG2 = BASE / "landmarks" / "PM_2D_img2"

META_JSON  = BASE / "output" / "case_metadata.json"

# ─────────────────────────────────────────────────────────────────────────────
# 2D LANDMARK COORDINATE SYSTEM
#
# When 3D Slicer loads a CR DICOM with ImagerPixelSpacing = 0.148 mm/px,
# its internal coordinate system uses physical units (mm). FCSV files
# exported from Slicer therefore contain coordinates in mm, not pixels.
#
# LM_2D_COORD_SYSTEM:
#   "mm"  → coordinates in mm (DICOM loaded natively in Slicer) — DEFAULT
#   "px"  → coordinates already in pixels (e.g. JPEG with 1 mm/px spacing)
#
# In "mm" mode, coordinates are divided by LM_2D_PIXEL_MM before PnP.
# ─────────────────────────────────────────────────────────────────────────────
LM_2D_COORD_SYSTEM = "mm"    # "mm" (DICOM in Slicer) or "px" (JPEG/PNG)
LM_2D_PIXEL_MM     = 0.148   # Detector pixel pitch (mm/px)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — FCSV loading
# ─────────────────────────────────────────────────────────────────────────────

def load_fcsv_dir(directory: Path) -> dict[str, list[float]]:
    """
    Merge all *.fcsv files in a directory into a single dict.

    Returns
    -------
    dict mapping label (str) → [x, y, z] (float, in the coordinate
    system of the FCSV file — mm for DICOM-based FCSV from Slicer).
    """
    merged: dict[str, list[float]] = {}
    for fcsv in sorted(directory.glob("*.fcsv")):
        with open(fcsv) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                parts = line.split(",")
                if len(parts) < 12:
                    continue
                label = parts[11].strip()
                if not label:
                    continue
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                except ValueError:
                    continue
                if label not in merged:
                    merged[label] = [x, y, z]
    return merged


def build_arrays(lm3d: dict, lm2d: dict,
                 coord_system: str = "px",
                 pixel_mm: float = 0.148):
    """
    Build numpy arrays for PnP from landmark dicts.

    Parameters
    ----------
    lm3d         : {label: [X, Y, Z]} — 3D points in LPS mm (from CBCT)
    lm2d         : {label: [x, y, z]} — 2D points (z=0, from PM radiograph)
    coord_system : "mm" → divide 2D coords by pixel_mm to get pixels
                   "px" → 2D coords already in pixels
    pixel_mm     : detector pixel pitch (mm/px), used when coord_system=="mm"

    Returns
    -------
    pts3  : (N,3) float64 — 3D world points
    pts2  : (N,2) float64 — 2D image points in pixels
    common: list of matched landmark labels
    """
    common = sorted(set(lm3d) & set(lm2d))
    if len(common) < 4:
        raise ValueError(f"Fewer than 4 common landmarks: {common}")
    pts3 = np.array([lm3d[k]      for k in common], dtype=np.float64)
    pts2 = np.array([lm2d[k][:2]  for k in common], dtype=np.float64)
    if coord_system == "mm":
        pts2 = pts2 / pixel_mm   # mm → pixels
    return pts3, pts2, common


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — PnP solvers
# ─────────────────────────────────────────────────────────────────────────────

def reprojection_error(pts3, pts2, rvec, tvec, K, dist):
    """Compute per-point reprojection errors and overall RMSE (pixels)."""
    proj, _ = cv2.projectPoints(pts3, rvec, tvec, K, dist)
    errors  = np.linalg.norm(pts2 - proj.reshape(-1, 2), axis=1)
    return float(np.sqrt(np.mean(errors ** 2))), errors.tolist()


def run_scenario_A(pts3, pts2, f_px: float, cx: float, cy: float):
    """
    Scenario A — calibrated PnP with fixed focal length.

    The intrinsic matrix K is built from DICOM acquisition parameters:
      f = DistanceSourceToReferencePoint / ImagerPixelSpacing
    Distortion coefficients are assumed zero (flat-panel detector).
    """
    K    = np.array([[f_px, 0, cx], [0, f_px, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1))
    ok, rvec, tvec = cv2.solvePnP(pts3, pts2, K, dist,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
    return ok, rvec, tvec, K, dist


def run_scenario_B(pts3, pts2, img_w: int, img_h: int):
    """
    Scenario B — uncalibrated PnP with free focal length estimation.

    Initialises the focal length as the diagonal of the image (heuristic),
    then refines it via cv2.calibrateCamera with fixed principal point,
    fixed aspect ratio, no tangential distortion and no radial distortion.

    This is the 'unknown camera' scenario, representative of real-world
    forensic cases where ante-mortem data are transmitted as screenshots
    or plain image files stripped of acquisition metadata (EXIF/DICOM),
    making the original camera geometry irrecoverable. This situation
    commonly arises on the ante-mortem side (e.g. social-media screenshots,
    scanned photographs, or exported images from patient management software).
    """
    cx, cy  = img_w / 2.0, img_h / 2.0
    f_init  = float(np.sqrt(img_w ** 2 + img_h ** 2))
    K_init  = np.array([[f_init, 0, cx], [0, f_init, cy], [0, 0, 1]],
                        dtype=np.float64)
    dist    = np.zeros((4, 1))

    ok, rvec, tvec = cv2.solvePnP(pts3, pts2, K_init, dist,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return False, None, None, None, None

    obj_f32 = pts3.astype(np.float32).reshape(-1, 1, 3)
    img_f32 = pts2.astype(np.float32).reshape(-1, 1, 2)
    flags   = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT |
               cv2.CALIB_FIX_ASPECT_RATIO    | cv2.CALIB_ZERO_TANGENT_DIST   |
               cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
    _ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        [obj_f32], [img_f32], (img_w, img_h), K_init.copy(), dist,
        flags=flags)
    return True, rvecs[0], tvecs[0], K, dist


def scenario_summary(label: str, ok: bool,
                     rvec, tvec, K, dist,
                     pts3, pts2, common: list,
                     pixel_mm: float,
                     ffd_mm_ref: float | None = None) -> dict:
    """Build a result dictionary for one PnP scenario."""
    if not ok:
        return {"converged": False}

    R_mat, _  = cv2.Rodrigues(rvec)
    angles    = cv2.RQDecomp3x3(R_mat)[0]
    rmse, per = reprojection_error(pts3, pts2, rvec, tvec, K, dist)
    f_px      = float(K[0, 0])
    f_mm      = f_px * pixel_mm

    print(f"  [{label}] Converged | RMSE = {rmse:.2f} px | "
          f"f = {f_px:.0f} px → FFD ≈ {f_mm:.0f} mm", end="")
    if ffd_mm_ref:
        err_pct = abs(f_mm - ffd_mm_ref) / ffd_mm_ref * 100
        print(f" | deviation vs DICOM = {err_pct:.1f}%", end="")
    print()

    return {
        "converged"      : True,
        "focal_px"       : round(f_px, 2),
        "focal_mm"       : round(f_mm, 1),
        "RMSE_px"        : round(rmse, 4),
        "RMSE_mm"        : round(rmse * pixel_mm, 4),
        "rotation_deg"   : {ax: round(float(a), 2) for ax, a in zip("XYZ", angles)},
        "translation_mm" : {ax: round(float(t), 2) for ax, t in zip("XYZ", tvec.flatten())},
        "rvec"           : rvec.flatten().tolist(),
        "tvec"           : tvec.flatten().tolist(),
        "K_matrix"       : K.tolist(),
        "dist_coeffs"    : dist.flatten().tolist(),
        "per_point_px"   : {k: round(e, 4) for k, e in zip(common, per)},
        "per_point_mm"   : {k: round(e * pixel_mm, 4) for k, e in zip(common, per)},
    }


# ─────────────────────────────────────────────────────────────────────────────
# PER-IMAGE PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_image(img_label: str, lm2d_dir: Path,
                  lm3d: dict, meta_img: dict) -> dict:
    """Run Scenarios A and B for a single PM image."""
    print(f"\n{'='*60}")
    print(f"Image : {img_label}  —  {meta_img['series_description']}")
    print(f"{'='*60}")

    if not lm2d_dir.exists() or not any(lm2d_dir.glob("*.fcsv")):
        print(f"  ⚠  No FCSV found in {lm2d_dir}")
        print("  → Annotate landmarks in 3D Slicer, then re-run this script.")
        return {"status": "landmarks_missing"}

    lm2d = load_fcsv_dir(lm2d_dir)
    print(f"  2D landmarks loaded : {len(lm2d)} points")

    try:
        pts3, pts2, common = build_arrays(lm3d, lm2d,
                                          coord_system=LM_2D_COORD_SYSTEM,
                                          pixel_mm=LM_2D_PIXEL_MM)
    except ValueError as exc:
        print(f"  ⚠  {exc}")
        return {"status": "insufficient_landmarks"}

    print(f"  Common landmarks ({len(common)}) : {common}")

    img_w   = meta_img["columns"]
    img_h   = meta_img["rows"]
    cx, cy  = img_w / 2.0, img_h / 2.0
    f_A     = meta_img["focal_A_px"]
    pix_mm  = meta_img["pixel_size_mm"]
    ffd_ref = meta_img["FFD_mm"]

    results = {
        "image_label"        : img_label,
        "series_description" : meta_img["series_description"],
        "image_size_px"      : [img_w, img_h],
        "n_landmarks"        : len(common),
        "labels"             : common,
        "pixel_size_mm"      : pix_mm,
        "FFD_mm_DICOM"       : ffd_ref,
        "focal_A_px"         : f_A,
    }

    # ── Scenario A ────────────────────────────────────────────────────────────
    print(f"\n  Scenario A (fixed f = {f_A:.0f} px, FFD = {ffd_ref:.0f} mm)…")
    ok_A, rv_A, tv_A, K_A, d_A = run_scenario_A(pts3, pts2, f_A, cx, cy)
    sc_A = scenario_summary("A", ok_A, rv_A, tv_A, K_A, d_A,
                             pts3, pts2, common, pix_mm, ffd_ref)
    if ok_A:
        print("  Per-landmark reprojection errors:")
        for lbl, err_px in sc_A["per_point_px"].items():
            flag = "  ⚠" if err_px > 10 else ""
            print(f"    {lbl:22s}  {err_px:.2f} px  "
                  f"({err_px * pix_mm:.2f} mm){flag}")
    results["scenario_A"] = sc_A

    # ── Scenario B ────────────────────────────────────────────────────────────
    print(f"\n  Scenario B (free focal length estimation)…")
    ok_B, rv_B, tv_B, K_B, d_B = run_scenario_B(pts3, pts2, img_w, img_h)
    sc_B = scenario_summary("B", ok_B, rv_B, tv_B, K_B, d_B,
                             pts3, pts2, common, pix_mm, ffd_ref)
    if ok_B:
        print("  Per-landmark reprojection errors:")
        for lbl, err_px in sc_B["per_point_px"].items():
            flag = "  ⚠" if err_px > 10 else ""
            print(f"    {lbl:22s}  {err_px:.2f} px  "
                  f"({err_px * pix_mm:.2f} mm){flag}")
    results["scenario_B"] = sc_B

    # ── A vs B comparison ─────────────────────────────────────────────────────
    if ok_A and ok_B:
        print(f"\n  ── Comparison ───────────────────────────")
        print(f"  RMSE A (calibrated) : {sc_A['RMSE_px']:.2f} px "
              f"({sc_A['RMSE_mm']:.3f} mm)")
        print(f"  RMSE B (estimated)  : {sc_B['RMSE_px']:.2f} px "
              f"({sc_B['RMSE_mm']:.3f} mm)")
        f_B_mm = sc_B["focal_mm"]
        dev    = abs(f_B_mm - ffd_ref) / ffd_ref * 100
        print(f"  FFD estimated (B)   : {f_B_mm:.0f} mm  "
              f"vs DICOM {ffd_ref:.0f} mm — deviation {dev:.1f}%")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("02_pnp.py — PnP Registration (Scenarios A and B)")
    print("=" * 60)

    if not META_JSON.exists():
        print("⚠  case_metadata.json not found. Run 01_extract_dicom.py first.")
        return
    with open(META_JSON) as fh:
        meta = json.load(fh)
    meta_img1, meta_img2 = meta["images"][0], meta["images"][1]

    if not any(LM_3D_DIR.glob("*.fcsv")):
        print(f"⚠  No 3D FCSV files found in {LM_3D_DIR}")
        return
    lm3d = load_fcsv_dir(LM_3D_DIR)
    print(f"3D landmarks : {len(lm3d)} points loaded from {LM_3D_DIR.name}/")

    # Image 1
    res1 = process_image("img1", LM_2D_IMG1, lm3d, meta_img1)
    out1 = BASE / "output" / "img1" / "pnp_results.json"
    out1.parent.mkdir(parents=True, exist_ok=True)
    with open(out1, "w") as fh:
        json.dump(res1, fh, indent=2, ensure_ascii=False)
    print(f"\n✔ Results img1 → {out1}")

    # Image 2
    res2 = process_image("img2", LM_2D_IMG2, lm3d, meta_img2)
    out2 = BASE / "output" / "img2" / "pnp_results.json"
    out2.parent.mkdir(parents=True, exist_ok=True)
    with open(out2, "w") as fh:
        json.dump(res2, fh, indent=2, ensure_ascii=False)
    print(f"✔ Results img2 → {out2}")

    print("\nNext step: python3 03_figures.py")


if __name__ == "__main__":
    main()
