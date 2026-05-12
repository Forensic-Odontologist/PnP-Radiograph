# Novice single-image workflow (CBCT × one plain radiograph)

This folder is a **simplified** workflow for PnP Radiograph alignment. It targets users who need:

- **One** plain radiograph supplied as **DICOM only** (no JPG/PNG/TIFF pipeline in this version).
- **Scenario A** — fixed focal length from **DICOM** geometry (tags when present, documented defaults otherwise).

Scripts do **not** embed case-specific device names or manuscript context; defaults are generic and editable in `01_prepare_image.py`.

<img width="1382" height="489" alt="Illustration-Github" src="https://github.com/user-attachments/assets/5257c56b-08c4-4ba2-94fe-48d20cc64e8b" />

<img width="1020" height="652" alt="mesh_silhouette_01_Segmentation_A_zoomed" src="https://github.com/user-attachments/assets/c3ac3357-54e3-41fc-95b7-38a73b828a00" />

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
│   ├── 3D_Landmarks/      # 3D FCSV landmarks (volume / LPS, mm)
│   └── 2D_Landmarks/      # 2D FCSV landmarks on the radiograph
├── meshes/                # One or more *.obj (any filenames)
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
   - Same **labels** in `3D_Landmarks` and `2D_Landmarks` (≥ four pairs).
   - Annotate the CR in Slicer; `case_metadata.json` stores `lm_2d_pixel_mm` from DICOM `ImagerPixelSpacing` / `PixelSpacing`.
3. **Meshes** as Wavefront **OBJ** in `meshes/`, aligned with the same 3D frame as the landmarks:
   - **Any number of files** with **any names** ending in `.obj` — all are loaded (sorted alphabetically).
   - **One mesh** → silhouette drawn in **blue** by default.
   - **Several meshes** → distinct colours (blue, green, orange, …) in order.

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
