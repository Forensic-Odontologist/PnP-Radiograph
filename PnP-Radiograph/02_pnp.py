"""
02_pnp.py
=========
Perspective-*n*-Point alignment for **one** plain radiograph (novice workflow).

Reads ``output/case_metadata.json`` produced by ``01_prepare_image.py`` and runs **one**
scenario:

  - **Scenario A** — fixed focal length from DICOM-derived geometry (default after step 1).
  - **Scenario B** — free focal-length estimation — only if ``pnp_scenario`` is set to ``B``
    in the JSON (optional comparison; step 1 normally writes ``A`` only).

Landmarks (3D Slicer FCSV):

  - ``landmarks/3D_Landmarks/`` — 3D points (volume / LPS, mm)
  - ``landmarks/2D_Landmarks/`` — 2D points on the radiograph

Output: ``output/pnp_results.json``

Usage
-----
    python3 02_pnp.py

Dependencies
------------
    pip install numpy opencv-python
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).parent

LM_3D_DIR = BASE / "landmarks" / "3D_Landmarks"
LM_2D_DIR = BASE / "landmarks" / "2D_Landmarks"
META_JSON = BASE / "output" / "case_metadata.json"
OUT_JSON = BASE / "output" / "pnp_results.json"


def load_fcsv_dir(directory: Path) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for fcsv in sorted(directory.glob("*.fcsv")):
        with open(fcsv, encoding="utf-8") as fh:
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


def build_arrays(
    lm3d: dict,
    lm2d: dict,
    coord_system: str = "mm",
    pixel_mm: float = 1.0,
):
    common = sorted(set(lm3d) & set(lm2d))
    if len(common) < 4:
        raise ValueError(f"Fewer than 4 common landmarks: {common}")
    pts3 = np.array([lm3d[k] for k in common], dtype=np.float64)
    pts2 = np.array([lm2d[k][:2] for k in common], dtype=np.float64)
    if coord_system == "mm":
        pts2 = pts2 / pixel_mm
    return pts3, pts2, common


def reprojection_error(pts3, pts2, rvec, tvec, K, dist):
    proj, _ = cv2.projectPoints(pts3, rvec, tvec, K, dist)
    errors = np.linalg.norm(pts2 - proj.reshape(-1, 2), axis=1)
    return float(np.sqrt(np.mean(errors ** 2))), errors.tolist()


def run_scenario_A(pts3, pts2, f_px: float, cx: float, cy: float):
    K = np.array([[f_px, 0, cx], [0, f_px, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1))
    ok, rvec, tvec = cv2.solvePnP(
        pts3, pts2, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    return ok, rvec, tvec, K, dist


def run_scenario_B(pts3, pts2, img_w: int, img_h: int):
    cx, cy = img_w / 2.0, img_h / 2.0
    f_init = float(np.sqrt(img_w ** 2 + img_h ** 2))
    K_init = np.array(
        [[f_init, 0, cx], [0, f_init, cy], [0, 0, 1]], dtype=np.float64
    )
    dist = np.zeros((4, 1))

    ok, rvec, tvec = cv2.solvePnP(
        pts3, pts2, K_init, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return False, None, None, None, None

    obj_f32 = pts3.astype(np.float32).reshape(-1, 1, 3)
    img_f32 = pts2.astype(np.float32).reshape(-1, 1, 2)
    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_FIX_PRINCIPAL_POINT
        | cv2.CALIB_FIX_ASPECT_RATIO
        | cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_FIX_K1
        | cv2.CALIB_FIX_K2
        | cv2.CALIB_FIX_K3
    )
    _ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        [obj_f32], [img_f32], (img_w, img_h), K_init.copy(), dist, flags=flags
    )
    return True, rvecs[0], tvecs[0], K, dist


def scenario_summary(
    label: str,
    ok: bool,
    rvec,
    tvec,
    K,
    dist,
    pts3,
    pts2,
    common: list,
    pixel_mm: float,
    ffd_mm_ref: float | None = None,
) -> dict:
    if not ok:
        return {"converged": False}

    R_mat, _ = cv2.Rodrigues(rvec)
    angles = cv2.RQDecomp3x3(R_mat)[0]
    rmse, per = reprojection_error(pts3, pts2, rvec, tvec, K, dist)
    f_px = float(K[0, 0])
    f_mm = f_px * pixel_mm

    print(
        f"  [{label}] Converged | RMSE = {rmse:.2f} px | "
        f"f ≈ {f_px:.0f} px → source–detector ≈ {f_mm:.0f} mm",
        end="",
    )
    if ffd_mm_ref:
        err_pct = abs(f_mm - ffd_mm_ref) / ffd_mm_ref * 100
        print(f" | vs metadata FFD = {err_pct:.1f}% diff", end="")
    print()

    return {
        "converged": True,
        "focal_px": round(f_px, 2),
        "focal_mm": round(f_mm, 1),
        "RMSE_px": round(rmse, 4),
        "RMSE_mm": round(rmse * pixel_mm, 4),
        "rotation_deg": {ax: round(float(a), 2) for ax, a in zip("XYZ", angles)},
        "translation_mm": {
            ax: round(float(t), 2) for ax, t in zip("XYZ", tvec.flatten())
        },
        "rvec": rvec.flatten().tolist(),
        "tvec": tvec.flatten().tolist(),
        "K_matrix": K.tolist(),
        "dist_coeffs": dist.flatten().tolist(),
        "per_point_px": {k: round(e, 4) for k, e in zip(common, per)},
        "per_point_mm": {k: round(e * pixel_mm, 4) for k, e in zip(common, per)},
    }


def main():
    print("=" * 60)
    print("02_pnp.py — single-image PnP (scenario from metadata)")
    print("=" * 60)

    if not META_JSON.exists():
        print(f"⚠  Missing {META_JSON.name}. Run 01_prepare_image.py first.")
        return

    with open(META_JSON, encoding="utf-8") as fh:
        meta = json.load(fh)

    scenario = meta.get("pnp_scenario", "B").upper()
    img_w = meta["columns"]
    img_h = meta["rows"]
    cx, cy = img_w / 2.0, img_h / 2.0
    pix_mm = float(meta["lm_2d_pixel_mm"])
    coord = meta.get("lm_2d_coord_system", "mm")
    ffd_ref = meta.get("FFD_mm")

    if not LM_3D_DIR.exists() or not any(LM_3D_DIR.glob("*.fcsv")):
        print(f"⚠  No 3D FCSV in {LM_3D_DIR}")
        return
    if not LM_2D_DIR.exists() or not any(LM_2D_DIR.glob("*.fcsv")):
        print(f"⚠  No 2D FCSV in {LM_2D_DIR}")
        return

    lm3d = load_fcsv_dir(LM_3D_DIR)
    lm2d = load_fcsv_dir(LM_2D_DIR)
    print(f"3D landmarks : {len(lm3d)}  |  2D landmarks : {len(lm2d)}")

    try:
        pts3, pts2, common = build_arrays(
            lm3d, lm2d, coord_system=coord, pixel_mm=pix_mm
        )
    except ValueError as exc:
        print(f"⚠  {exc}")
        return

    print(f"Matched pairs : {len(common)}  →  {common}")

    block_key = f"scenario_{scenario}"
    results: dict = {
        "active_scenario": scenario,
        "scenario_block": block_key,
        "labels": common,
        "n_landmarks": len(common),
        "image_size_px": [img_w, img_h],
        "pixel_size_mm": float(meta.get("pixel_size_mm", pix_mm)),
        "lm_2d_coord_system": coord,
        "lm_2d_pixel_mm": pix_mm,
    }

    if scenario == "A":
        f_a = meta.get("focal_A_px")
        if f_a is None:
            print("⚠  focal_A_px missing in metadata — cannot run scenario A.")
            return
        f_a = float(f_a)
        print(f"\nScenario A — fixed f = {f_a:.0f} px (from metadata)…")
        ok, rv, tv, K, d = run_scenario_A(pts3, pts2, f_a, cx, cy)
        block = scenario_summary(
            "A",
            ok,
            rv,
            tv,
            K,
            d,
            pts3,
            pts2,
            common,
            pix_mm,
            float(ffd_ref) if ffd_ref else None,
        )
        results[block_key] = block
        if ok:
            for lbl, err_px in block["per_point_px"].items():
                flag = "  ⚠" if err_px > 10 else ""
                print(
                    f"    {lbl:22s}  {err_px:.2f} px  "
                    f"({err_px * pix_mm:.2f} mm){flag}"
                )
    else:
        print("\nScenario B — estimated focal length …")
        ok, rv, tv, K, d = run_scenario_B(pts3, pts2, img_w, img_h)
        block = scenario_summary(
            "B", ok, rv, tv, K, d, pts3, pts2, common, pix_mm, None
        )
        results[block_key] = block
        if ok:
            for lbl, err_px in block["per_point_px"].items():
                flag = "  ⚠" if err_px > 10 else ""
                print(
                    f"    {lbl:22s}  {err_px:.2f} px  "
                    f"({err_px * pix_mm:.2f} mm){flag}"
                )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    print(f"\n✔ Saved → {OUT_JSON.relative_to(BASE)}")
    print("Next: python3 03_figures.py")


if __name__ == "__main__":
    main()
