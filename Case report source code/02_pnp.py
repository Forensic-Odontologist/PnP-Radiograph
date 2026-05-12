"""
02_pnp.py
=========
Step 2 — PnP registration for both PM radiographs, **per intra-operator session**.

For each session folder (Session1 … Session3), loads **matched** landmark sets:
  - landmarks/3D_Landmarks/{Session*}     → 3D FCSV (CBCT, LPS, mm)
  - landmarks/2D_Landmarks_PM1/{Session*} → 2D FCSV for image 1 (PM1)
  - landmarks/2D_Landmarks_PM2/{Session*} → 2D FCSV for image 2 (PM2)

Each session runs Scenarios A and B independently

Outputs
-------
    output/{Session*}/img1/pnp_results.json
    output/{Session*}/img2/pnp_results.json
    output/pnp_all_sessions.json   — merged summary for downstream reporting

Usage
-----
    python3 02_pnp.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent

LM_ROOT = BASE / "landmarks"
LM_3D_ROOT = LM_ROOT / "3D_Landmarks"
LM_PM1_ROOT = LM_ROOT / "2D_Landmarks_PM1"
LM_PM2_ROOT = LM_ROOT / "2D_Landmarks_PM2"

# Homologous sessions (2D PM1, 2D PM2, 3D must share the same folder name)
SESSION_NAMES = ("Session1", "Session2", "Session3")

META_JSON = BASE / "output" / "case_metadata.json"

# ─────────────────────────────────────────────────────────────────────────────
LM_2D_COORD_SYSTEM = "mm"
LM_2D_PIXEL_MM = 0.148


def load_fcsv_dir(directory: Path) -> dict[str, list[float]]:
    """Merge all *.fcsv in *directory* into label → [x, y, z]."""
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
    coord_system: str = "px",
    pixel_mm: float = 0.148,
):
    s3, s2 = set(lm3d), set(lm2d)
    common = sorted(s3 & s2)
    if len(common) < 4:
        only3 = sorted(s3 - s2)
        only2 = sorted(s2 - s3)
        raise ValueError(
            f"Fewer than 4 common landmarks ({len(common)}): {common}\n"
            f"  Labels only in 3D ({len(only3)}): {only3}\n"
            f"  Labels only in 2D ({len(only2)}): {only2}\n"
            f"  → Vérifier orthographe et **casse** des labels FCSV (ex. 44_Apex vs 44_apex)."
        )
    pts3 = np.array([lm3d[k] for k in common], dtype=np.float64)
    pts2 = np.array([lm2d[k][:2] for k in common], dtype=np.float64)
    if coord_system == "mm":
        pts2 = pts2 / pixel_mm
    return pts3, pts2, common


def reprojection_error(pts3, pts2, rvec, tvec, K, dist):
    proj, _ = cv2.projectPoints(pts3, rvec, tvec, K, dist)
    errors = np.linalg.norm(pts2 - proj.reshape(-1, 2), axis=1)
    return float(np.sqrt(np.mean(errors**2))), errors.tolist()


def solve_pnp_with_K(pts3, pts2, K, dist):
    """
    OpenCV 4.13+ : SOLVEPNP_ITERATIVE s'appuie sur un DLT qui exige **≥ 6** points.
    Avec 4 ou 5 correspondances, utiliser EPNP puis raffiner en itératif.
    """
    n = len(pts3)
    dist = np.asarray(dist, dtype=np.float64)
    if n >= 6:
        return cv2.solvePnP(pts3, pts2, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    ok, rvec, tvec = cv2.solvePnP(pts3, pts2, K, dist, flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        return ok, rvec, tvec
    return cv2.solvePnP(
        pts3,
        pts2,
        K,
        dist,
        rvec,
        tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )


def run_scenario_A(pts3, pts2, f_px: float, cx: float, cy: float):
    K = np.array([[f_px, 0, cx], [0, f_px, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1))
    return solve_pnp_with_K(pts3, pts2, K, dist) + (K, dist)


def run_scenario_B(pts3, pts2, img_w: int, img_h: int):
    cx, cy = img_w / 2.0, img_h / 2.0
    f_init = float(np.sqrt(img_w**2 + img_h**2))
    K_init = np.array(
        [[f_init, 0, cx], [0, f_init, cy], [0, 0, 1]], dtype=np.float64
    )
    dist = np.zeros((4, 1))

    ok, rvec, tvec = solve_pnp_with_K(pts3, pts2, K_init, dist)[:3]
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
        f"f = {f_px:.0f} px → FFD ≈ {f_mm:.0f} mm",
        end="",
    )
    if ffd_mm_ref:
        err_pct = abs(f_mm - ffd_mm_ref) / ffd_mm_ref * 100
        print(f" | deviation vs DICOM = {err_pct:.1f}%", end="")
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


def session_landmark_dirs(session: str) -> tuple[Path, Path, Path]:
    return (
        LM_3D_ROOT / session,
        LM_PM1_ROOT / session,
        LM_PM2_ROOT / session,
    )


def session_ready(session: str) -> bool:
    d3, d1, d2 = session_landmark_dirs(session)
    return (
        d3.exists()
        and d1.exists()
        and d2.exists()
        and any(d3.glob("*.fcsv"))
        and any(d1.glob("*.fcsv"))
        and any(d2.glob("*.fcsv"))
    )


def process_image(
    img_label: str,
    lm2d_dir: Path,
    lm3d: dict,
    meta_img: dict,
    session: str,
) -> dict:
    print(f"\n{'='*60}")
    print(f"{session} | {img_label}  —  {meta_img['series_description']}")
    print(f"{'='*60}")

    if not lm2d_dir.exists() or not any(lm2d_dir.glob("*.fcsv")):
        print(f"  ⚠  No FCSV found in {lm2d_dir}")
        return {"status": "landmarks_missing", "session": session}

    lm2d = load_fcsv_dir(lm2d_dir)
    print(f"  2D landmarks loaded : {len(lm2d)} points")

    try:
        pts3, pts2, common = build_arrays(
            lm3d,
            lm2d,
            coord_system=LM_2D_COORD_SYSTEM,
            pixel_mm=LM_2D_PIXEL_MM,
        )
    except ValueError as exc:
        print(f"  ⚠  {exc}")
        return {"status": "insufficient_landmarks", "session": session}

    print(f"  Common landmarks ({len(common)}) : {common}")
    if len(common) < 6:
        print(
            "  Note : avec < 6 correspondances, OpenCV n'accepte pas un PnP purement "
            "itératif (init. DLT) ; ce script utilise EPNP puis raffinement itératif."
        )

    img_w = meta_img["columns"]
    img_h = meta_img["rows"]
    cx, cy = img_w / 2.0, img_h / 2.0
    f_A = meta_img["focal_A_px"]
    pix_mm = meta_img["pixel_size_mm"]
    ffd_ref = meta_img["FFD_mm"]

    results = {
        "session": session,
        "image_label": img_label,
        "series_description": meta_img["series_description"],
        "image_size_px": [img_w, img_h],
        "n_landmarks": len(common),
        "labels": common,
        "pixel_size_mm": pix_mm,
        "FFD_mm_DICOM": ffd_ref,
        "focal_A_px": f_A,
    }

    print(f"\n  Scenario A (fixed f = {f_A:.0f} px, FFD = {ffd_ref:.0f} mm)…")
    ok_A, rv_A, tv_A, K_A, d_A = run_scenario_A(pts3, pts2, f_A, cx, cy)
    sc_A = scenario_summary(
        "A", ok_A, rv_A, tv_A, K_A, d_A, pts3, pts2, common, pix_mm, ffd_ref
    )
    if ok_A:
        print("  Per-landmark reprojection errors:")
        for lbl, err_px in sc_A["per_point_px"].items():
            flag = "  ⚠" if err_px > 10 else ""
            print(
                f"    {lbl:22s}  {err_px:.2f} px  "
                f"({err_px * pix_mm:.2f} mm){flag}"
            )
    results["scenario_A"] = sc_A

    print(f"\n  Scenario B (free focal length estimation)…")
    ok_B, rv_B, tv_B, K_B, d_B = run_scenario_B(pts3, pts2, img_w, img_h)
    sc_B = scenario_summary(
        "B", ok_B, rv_B, tv_B, K_B, d_B, pts3, pts2, common, pix_mm, ffd_ref
    )
    if ok_B:
        print("  Per-landmark reprojection errors:")
        for lbl, err_px in sc_B["per_point_px"].items():
            flag = "  ⚠" if err_px > 10 else ""
            print(
                f"    {lbl:22s}  {err_px:.2f} px  "
                f"({err_px * pix_mm:.2f} mm){flag}"
            )
    results["scenario_B"] = sc_B

    if ok_A and ok_B:
        print(f"\n  ── Comparison ───────────────────────────")
        print(
            f"  RMSE A (calibrated) : {sc_A['RMSE_px']:.2f} px "
            f"({sc_A['RMSE_mm']:.3f} mm)"
        )
        print(
            f"  RMSE B (estimated)  : {sc_B['RMSE_px']:.2f} px "
            f"({sc_B['RMSE_mm']:.3f} mm)"
        )
        f_B_mm = sc_B["focal_mm"]
        dev = abs(f_B_mm - ffd_ref) / ffd_ref * 100
        print(
            f"  FFD estimated (B)   : {f_B_mm:.0f} mm  "
            f"vs DICOM {ffd_ref:.0f} mm — deviation {dev:.1f}%"
        )

    return results


def main():
    print("=" * 60)
    print("02_pnp.py — PnP (paired sessions: PM1/PM2/3D homologous folders)")
    print("=" * 60)

    if not META_JSON.exists():
        print("⚠  case_metadata.json not found. Run 01_extract_dicom.py first.")
        return
    with open(META_JSON, encoding="utf-8") as fh:
        meta = json.load(fh)
    meta_img1, meta_img2 = meta["images"][0], meta["images"][1]

    merged: dict[str, dict] = {"sessions": {}, "session_order": []}

    for session in SESSION_NAMES:
        if not session_ready(session):
            print(f"\n⚠  Skipping {session}: missing FCSV in one of")
            print(f"    {LM_3D_ROOT / session}")
            print(f"    {LM_PM1_ROOT / session}")
            print(f"    {LM_PM2_ROOT / session}")
            continue

        d3, d_pm1, d_pm2 = session_landmark_dirs(session)
        lm3d = load_fcsv_dir(d3)
        print(f"\n{'#'*60}\n# {session}: 3D landmarks = {len(lm3d)} from {d3.name}/\n{'#'*60}")

        merged["session_order"].append(session)
        merged["sessions"][session] = {}

        res1 = process_image("img1", d_pm1, lm3d, meta_img1, session)
        out1 = BASE / "output" / session / "img1" / "pnp_results.json"
        out1.parent.mkdir(parents=True, exist_ok=True)
        with open(out1, "w", encoding="utf-8") as fh:
            json.dump(res1, fh, indent=2, ensure_ascii=False)
        print(f"\n✔ {session} → {out1}")
        merged["sessions"][session]["img1"] = res1

        res2 = process_image("img2", d_pm2, lm3d, meta_img2, session)
        out2 = BASE / "output" / session / "img2" / "pnp_results.json"
        out2.parent.mkdir(parents=True, exist_ok=True)
        with open(out2, "w", encoding="utf-8") as fh:
            json.dump(res2, fh, indent=2, ensure_ascii=False)
        print(f"✔ {session} → {out2}")
        merged["sessions"][session]["img2"] = res2

    summary_path = BASE / "output" / "pnp_all_sessions.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2, ensure_ascii=False)
    print(f"\n✔ Merged summary → {summary_path}")
    print("\nNext: python3 03_figures.py  then  python3 04_article_outputs.py")


if __name__ == "__main__":
    main()
