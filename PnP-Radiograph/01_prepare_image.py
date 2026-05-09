"""
01_prepare_image.py
===================
Single-entry preparation for the novice workflow.

Place **exactly one** DICOM radiograph in ``data/input/`` (see extensions below).
The pipeline always uses **Scenario A** (calibrated PnP: fixed focal length from
DICOM geometry, with defaults if some tags are missing).

Raster formats (JPG, PNG, TIFF, …) are **not** supported in this version.

Writes:

  - ``output/pm_full.png`` — 8-bit image used by steps 2–3
  - ``output/case_metadata.json`` — geometry + landmark hints for PnP

Usage
-----
    python3 01_prepare_image.py

Dependencies
------------
    pip install pydicom numpy opencv-python
    pip install pylibjpeg pylibjpeg-libjpeg   # if JPEG-lossless DICOM
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pydicom

BASE = Path(__file__).parent
INPUT_DIR = BASE / "data" / "input"
OUT_DIR = BASE / "output"
PNG_PATH = OUT_DIR / "pm_full.png"
META_PATH = OUT_DIR / "case_metadata.json"

# Explicitly rejected suffixes (users must supply native DICOM).
UNSUPPORTED_RASTER_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
    ".gif",
    ".webp",
}

DEFAULT_PIXEL_MM = 0.148
DEFAULT_FFD_MM = 650.0


def _list_input_files() -> list[Path]:
    if not INPUT_DIR.is_dir():
        return []
    files: list[Path] = []
    for p in INPUT_DIR.iterdir():
        if p.is_file() and not p.name.startswith("."):
            files.append(p)
    return sorted(files)


def _reject_if_raster_extension(path: Path) -> str | None:
    suf = path.suffix.lower()
    if suf in UNSUPPORTED_RASTER_SUFFIXES:
        return (
            f"Raster files ({suf}) are not supported. "
            "Export or convert the radiograph to DICOM before placing it here."
        )
    return None


def _first_scalar_float(val) -> float | None:
    if val is None:
        return None
    if isinstance(val, (list, tuple)) or (
        hasattr(val, "__iter__") and not isinstance(val, (str, bytes))
    ):
        try:
            return float(next(iter(val)))
        except (StopIteration, TypeError, ValueError):
            return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def apply_window(
    pixel_array: np.ndarray,
    center: float,
    width: float,
    invert: bool = False,
) -> np.ndarray:
    lo = center - width / 2.0
    hi = center + width / 2.0
    clipped = np.clip(pixel_array, lo, hi)
    normalised = (clipped - lo) / (hi - lo) * 255.0
    if invert:
        normalised = 255.0 - normalised
    return normalised.astype(np.uint8)


def dicom_to_raw(ds: pydicom.Dataset) -> np.ndarray:
    arr = ds.pixel_array.astype(np.float64)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    inter = float(getattr(ds, "RescaleIntercept", 0.0))
    return arr * slope + inter


def _ffd_mm_from_dataset(ds: pydicom.Dataset) -> float | None:
    for attr in ("DistanceSourceToDetector", "DistanceSourceToPatient"):
        v = getattr(ds, attr, None)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    for elem in ds.iterall():
        if elem.keyword in (
            "DistanceSourceToDetector",
            "DistanceSourceToPatient",
            "DistanceSourceToReferencePoint",
        ) and elem.value is not None:
            try:
                return float(elem.value)
            except (TypeError, ValueError):
                pass
    return None


def prepare_dicom(path: Path) -> dict:
    ds = pydicom.dcmread(str(path))
    raw = dicom_to_raw(ds)
    rows = int(ds.Rows)
    cols = int(ds.Columns)

    pixel_spacing = getattr(
        ds,
        "ImagerPixelSpacing",
        getattr(ds, "PixelSpacing", [DEFAULT_PIXEL_MM, DEFAULT_PIXEL_MM]),
    )
    ps = float(pixel_spacing[0])

    wc = _first_scalar_float(getattr(ds, "WindowCenter", None))
    ww = _first_scalar_float(getattr(ds, "WindowWidth", None))
    if wc is None or ww is None:
        wc = float(np.mean(raw))
        span = float(np.max(raw) - np.min(raw))
        ww = span if span > 1e-6 else 1.0

    invert = (
        getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1"
    )
    gray_u8 = apply_window(raw, wc, ww, invert)
    bgr = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(PNG_PATH), bgr)

    ffd = _ffd_mm_from_dataset(ds)
    if ffd is None:
        ffd = DEFAULT_FFD_MM
        ffd_note = "missing_tag_used_default"
    else:
        ffd_note = "from_dicom"

    focal_a_px = ffd / ps

    return {
        "input_kind": "dicom",
        "input_filename": path.name,
        "pnp_scenario": "A",
        "series_description": str(getattr(ds, "SeriesDescription", "")),
        "rows": rows,
        "columns": cols,
        "png_path": str(PNG_PATH.relative_to(BASE)),
        "pixel_size_mm": ps,
        "FFD_mm": ffd,
        "FFD_source": ffd_note,
        "focal_A_px": focal_a_px,
        "principal_point_px": [cols / 2.0, rows / 2.0],
        "lm_2d_coord_system": "mm",
        "lm_2d_pixel_mm": ps,
        "window_center": wc,
        "window_width": ww,
    }


def main():
    print("=" * 60)
    print("01_prepare_image.py — single DICOM radiograph → PNG + metadata")
    print("=" * 60)

    files = _list_input_files()
    if not files:
        print(f"\n⚠  No file found in {INPUT_DIR}")
        print("    Add exactly one DICOM radiograph and re-run.")
        return
    if len(files) > 1:
        print(f"\n⚠  Multiple files in {INPUT_DIR}:")
        for f in files:
            print(f"    - {f.name}")
        print("    Keep only ONE radiograph.")
        return

    path = files[0]
    err = _reject_if_raster_extension(path)
    if err:
        print(f"\n⚠  {err}")
        return

    print(f"\nInput : {path.name}")

    try:
        pydicom.dcmread(path, stop_before_pixels=True)
    except Exception as exc:
        print(
            f"\n⚠  Not a readable DICOM file: {exc}\n"
            "    Use a native DICOM object (e.g. .dcm, or PACS export without "
            "a consumer extension like .jpg)."
        )
        return

    try:
        meta = prepare_dicom(path)
    except Exception as exc:
        print(f"\n⚠  Failed to decode DICOM pixels: {exc}")
        return

    envelope = {
        "workflow": "Github2-novice-single-image-dicom-only",
        "readme_hint": (
            "DICOM input only; Scenario A. Landmarks: landmarks/3D_Landmarks and "
            "landmarks/2D_Landmarks."
        ),
        **meta,
    }

    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(META_PATH, "w", encoding="utf-8") as fh:
        json.dump(envelope, fh, indent=2, ensure_ascii=False)

    print(f"\n  PNG written    : {PNG_PATH.relative_to(BASE)}")
    print(f"  Metadata       : {META_PATH.relative_to(BASE)}")
    print(f"  PnP scenario   : A (DICOM geometry — calibrated focal length)")
    print(f"\nNext: python3 02_pnp.py")


if __name__ == "__main__":
    main()
