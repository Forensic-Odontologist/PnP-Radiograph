# Novice single-image workflow (CBCT × one plain radiograph)

This folder is a **simplified** companion to `Github/`. It targets users who need:

- **One** plain radiograph supplied as **DICOM only** (no JPG/PNG/TIFF pipeline in this version).
- **Scenario A** — fixed focal length from **DICOM** geometry (tags when present, documented defaults otherwise).

Scripts do **not** embed case-specific device names or manuscript context; defaults are generic and editable in `01_prepare_image.py`.

---

## Installation

```bash
cd PnP-Radiograph
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install `pylibjpeg` + `pylibjpeg-libjpeg` if your DICOM uses JPEG-lossless transfer syntax.

---

## Layout

```
PnP-Radiograph/
├── 01_prepare_image.py    # Step 1 — DICOM → PNG + metadata (Scenario A)
├── 02_pnp.py              # Step 2 — PnP for the active scenario (default A)
├── 03_figures.py          # Step 3 — overlays & mesh silhouettes
├── data/input/            # YOU: exactly ONE DICOM radiograph
├── landmarks/
│   ├── CBCT_3D/           # 3D FCSV landmarks (CBCT / LPS, mm)
│   └── PM_2D/             # 2D FCSV landmarks on the radiograph
├── meshes/
│   ├── mesh_restorations.obj
│   └── mesh_bone.obj
└── output/                # generated (ignored by git)
    ├── pm_full.png
    ├── case_metadata.json
    ├── pnp_results.json
    └── figures/
```

---

## What you provide

1. **One DICOM** in `data/input/` (see `data/input/README.md`).
2. **Landmarks** exported from 3D Slicer as `.fcsv`:
   - Same **labels** in `CBCT_3D` and `PM_2D` (≥ four pairs).
   - Annotate the CR in Slicer; `case_metadata.json` stores `lm_2d_pixel_mm` from DICOM `ImagerPixelSpacing` / `PixelSpacing`.
3. **Meshes** aligned with the CBCT frame (`mesh_restorations.obj`, `mesh_bone.obj`).

---

## Commands

Run from `PnP-Radiograph`:

```bash
python3 01_prepare_image.py
python3 02_pnp.py
python3 03_figures.py
```

Step 1 writes **Scenario A** metadata. Step 2 solves the scenario specified in `case_metadata.json` (default **A**). Step 3 writes PNGs under `output/figures/`.

---

## Metadata defaults (DICOM)

If distance tags are missing, `01_prepare_image.py` uses:

- `DEFAULT_PIXEL_MM = 0.148`
- `DEFAULT_FFD_MM = 650.0`

Adjust these constants for your hardware **before** publishing results, or edit `output/case_metadata.json` and re-run steps 2–3.

---

## Overriding the scenario (advanced)

The stock pipeline uses **Scenario A**. To run **Scenario B** (free focal length) for comparison — e.g. sensitivity analysis — set `"pnp_scenario": "B"` in `output/case_metadata.json` after step 1, then run steps 2–3 again. This is optional and **not** the default novice path.

