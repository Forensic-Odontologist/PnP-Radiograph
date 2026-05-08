"""
01_extract_dicom.py
===================
Step 1 — Post-mortem DICOM image extraction and metadata summary.

Reads the two PM DICOM files (Image 1: X_Cervical, Image 2: X_Thorax),
applies native DICOM windowing, exports 16-bit PNG images, and generates
a case_metadata.json file containing acquisition parameters for both AM
and PM devices.

Usage
-----
    python3 01_extract_dicom.py

Dependencies
------------
    pip install pydicom numpy Pillow pylibjpeg pylibjpeg-libjpeg
"""

import json
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE   = Path(__file__).parent
DATA   = BASE.parent / "Data"
DICOM  = DATA / "PM-2D-Anon" / "UNNAMED"

# PM DICOM files — confirmed mapping
DICOM_IMG1 = DICOM / "00010001"   # X_Cervical   (frontal skull view)
DICOM_IMG2 = DICOM / "00020001"   # X_Thorax a.p. (frontal thorax/skull view)

# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRIC PARAMETERS
# Source: DICOM Structured Report (Dose SR)
#   Tag DistanceSourceToReferencePoint = 650 mm
#   Tag ImagerPixelSpacing = [0.148, 0.148] mm/px
#
# NOTE: DistanceSourceToReferencePoint is the source-to-patient surface
# distance, NOT the actual Source-Image Distance (SID). The true SID is
# estimated to be ~850–1100 mm (patient thickness + air gap not included).
# This tag is used here as the best available DICOM approximation for
# Scenario A (calibrated PnP). See Discussion section of the manuscript.
# ─────────────────────────────────────────────────────────────────────────────
PIXEL_SIZE_MM  = 0.148          # Detector pixel pitch (mm/px)
FFD_MM         = 650.0          # Source-to-reference-point distance (mm)
FOCAL_A_PX     = FFD_MM / PIXEL_SIZE_MM   # Equivalent focal length (px)


def apply_window(pixel_array: np.ndarray,
                 center: float, width: float,
                 invert: bool = False) -> np.ndarray:
    """Apply linear DICOM windowing and normalise to [0, 65535]."""
    lo = center - width / 2.0
    hi = center + width / 2.0
    clipped = np.clip(pixel_array, lo, hi)
    normalised = (clipped - lo) / (hi - lo) * 65535.0
    if invert:
        normalised = 65535.0 - normalised
    return normalised.astype(np.uint16)


def dicom_to_raw(ds: pydicom.Dataset) -> np.ndarray:
    """Convert DICOM pixel array to raw values (HU or manufacturer-scaled)."""
    arr   = ds.pixel_array.astype(np.float64)
    slope = float(getattr(ds, "RescaleSlope",     1.0))
    inter = float(getattr(ds, "RescaleIntercept", 0.0))
    return arr * slope + inter


def extract_image(dicom_path: Path, out_dir: Path, img_label: str) -> dict:
    """
    Read a DICOM CR file, apply native windowing, export a 16-bit PNG,
    and return a metadata dict for the case_metadata.json file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"PM_{img_label}_windowed.png"

    print(f"\n--- {img_label}: {dicom_path.name} ---")
    ds  = pydicom.dcmread(str(dicom_path))
    raw = dicom_to_raw(ds)

    rows = int(ds.Rows)
    cols = int(ds.Columns)
    print(f"  Dimensions     : {cols} × {rows} px")

    # Pixel spacing
    pixel_spacing = getattr(ds, "ImagerPixelSpacing",
                    getattr(ds, "PixelSpacing", [PIXEL_SIZE_MM, PIXEL_SIZE_MM]))
    ps = float(pixel_spacing[0])
    print(f"  Pixel spacing  : {ps} mm/px")

    # DICOM windowing
    wc = float(ds.WindowCenter)
    ww = float(ds.WindowWidth)
    invert = (getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
              == "MONOCHROME1")
    print(f"  Window         : C={wc}, W={ww}, invert={invert}")

    windowed = apply_window(raw, wc, ww, invert)
    Image.fromarray(windowed, mode="I;16").save(str(out_png))
    print(f"  Exported       : {out_png.relative_to(BASE)}")

    return {
        "label"                   : img_label,
        "dicom_file"              : dicom_path.name,
        "modality"                : str(getattr(ds, "Modality",         "CR")),
        "series_description"      : str(getattr(ds, "SeriesDescription", "")),
        "rows"                    : rows,
        "columns"                 : cols,
        "imager_pixel_spacing_mm" : [float(x) for x in pixel_spacing],
        "pixel_size_mm"           : PIXEL_SIZE_MM,
        "FFD_mm"                  : FFD_MM,
        "focal_A_px"              : FOCAL_A_PX,
        "principal_point_px"      : [cols / 2.0, rows / 2.0],
        "window_center"           : wc,
        "window_width"            : ww,
        "png_path"                : str(out_png.relative_to(BASE)),
    }


def main():
    print("=" * 60)
    print("01_extract_dicom.py — DICOM extraction & metadata")
    print("=" * 60)

    out_root = BASE / "output"

    # ── Case context ─────────────────────────────────────────────────────────
    metadata = {
        "case_context": (
            "Retrospective forensic identification. Investigators had access "
            "only to two post-mortem (PM) 2D radiographs and an ante-mortem "
            "(AM) dental CBCT. No PM 3D scan was available."
        ),
        "AM_device": {
            "manufacturer" : "Planmeca",
            "model"        : "ProMax",
            "modality"     : "CBCT",
            "voxel_size_mm": 0.2,
            "FOV_mm"       : 80,
            "kvp"          : 90,
            "mA"           : 13,
            "software"     : "Romexis 6.2.1.19",
        },
        "PM_device": {
            "manufacturer" : "Siemens",
            "model"        : "MOBILETT Elara Max",
            "detector"     : "Fluorospot Compact FD",
            "modality"     : "CR",
            "pixel_size_mm": PIXEL_SIZE_MM,
            "FFD_mm"       : FFD_MM,
            "note": (
                "Mobile (portable) radiography unit. FFD derived from "
                "the DICOM tag 'DistanceSourceToReferencePoint' in the "
                "Radiation Dose Structured Report (RDDSR). This tag "
                "represents the source-to-patient surface distance, not "
                "the actual Source-Image Distance (SID)."
            ),
        },
        "AM_to_PM_interval_months": 40,
    }

    # ── Image extraction ─────────────────────────────────────────────────────
    img1_meta = extract_image(DICOM_IMG1, out_root / "img1", "img1")
    img2_meta = extract_image(DICOM_IMG2, out_root / "img2", "img2")
    metadata["images"] = [img1_meta, img2_meta]

    # ── Save metadata ─────────────────────────────────────────────────────────
    json_path = out_root / "case_metadata.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Metadata saved  : {json_path.relative_to(BASE)}")
    print(f"Image 1         : {img1_meta['columns']} × {img1_meta['rows']} px"
          f"  —  {img1_meta['series_description']}")
    print(f"Image 2         : {img2_meta['columns']} × {img2_meta['rows']} px"
          f"  —  {img2_meta['series_description']}")
    print(f"Focal A (DICOM) : {FOCAL_A_PX:.0f} px  "
          f"({FFD_MM:.0f} mm / {PIXEL_SIZE_MM} mm·px⁻¹)")
    print(f"\nNext step: python3 02_pnp.py")


if __name__ == "__main__":
    main()
