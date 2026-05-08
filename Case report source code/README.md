# PnP-Based 2D/3D Radiographic Superimposition

---

## Overview

This repository provides a fully open-source Python workflow for 3D/2D
Perspective-n-Point (PnP) registration of a dental CBCT model onto plain
radiographs — applied to a forensic identification case where only
ante-mortem (AM) CBCT and post-mortem (PM) 2D radiographs were available.

All landmark annotation was performed in [3D Slicer](https://www.slicer.org)
(free, open-source). PnP computation and figure generation use standard
scientific Python libraries (OpenCV, NumPy, trimesh, matplotlib, pydicom).

**This work extends** [SmilePnP](https://github.com/Forensic-Odontologist/SmilePnP)
— a previously published Blender add-on for PnP-based smile photograph
superimposition — to the radiographic domain, replacing the photographic
modality with DICOM-formatted plain radiographs and the IOS dental scan
with a CBCT-derived 3D mesh.

---

## Workflow

```
Workflow_v1/
├── 01_extract_dicom.py   # Extract PM DICOM images + acquisition metadata
├── 02_pnp.py             # PnP registration (Scenario A: calibrated, B: uncalibrated)
├── 03_figures.py         # Publication-quality figure generation
│
├── landmarks/
│   ├── AM_3D/            # 3D CBCT fiducials (FCSV, LPS mm, from 3D Slicer)
│   ├── PM_2D_img1/       # 2D fiducials for Image 1 (FCSV, LPS mm)
│   └── PM_2D_img2/       # 2D fiducials for Image 2 (FCSV, LPS mm)
│
├── meshes/
│   ├── mesh_restorations.obj   # Dental restorations (3D Slicer segmentation) -> No available on Github
│   └── mesh_bone.obj           # Mandibular/osseous structures -> No available on Github
│
└── output/
    ├── case_metadata.json
    ├── img1/  pnp_results.json + figures
    └── img2/  pnp_results.json + figures
```

---

## Key Results

| Image | Landmarks | RMSE — Scenario A | RMSE — Scenario B |
|---|---|---|---|
| Image 1 (X_Cervical) | 11 | **3.53 px (0.52 mm)** | 3.17 px (0.47 mm) |
| Image 2 (X_Thorax)   |  6 | **1.97 px (0.29 mm)** | 1.88 px (0.28 mm) |

**Scenario A** uses the focal length derived from the DICOM tag
`DistanceSourceToReferencePoint` (650 mm) and detector pixel pitch
(0.148 mm/px → f = 4 392 px).

**Scenario B** estimates the focal length freely. It converges to
geometrically consistent poses but systematically overestimates the
source-image distance (+60–72%), consistent with the near-coplanar
geometry of mandibular landmarks in AP frontal projection.

---

## Quick Start

### 1. Install dependencies

```bash
pip install pydicom numpy Pillow pylibjpeg pylibjpeg-libjpeg \
            opencv-python trimesh matplotlib python-docx
```

### 2. Prepare data

Place your DICOM files in `Data/PM-2D-Anon/UNNAMED/` and your CBCT in
`Data/AM-CBCT-Anon/`. Edit the paths at the top of each script if needed.

### 3. Annotate landmarks in 3D Slicer

- Load the Dental CBCT and place 3D fiducials → export FCSV to `landmarks/AM_3D/`
- Load each PM DICOM and place 2D fiducials → export FCSV to
  `landmarks/PM_2D_img1/` and `landmarks/PM_2D_img2/`
- Export CBCT or segmentations in .obj file and place it in meshes/ subfolder. 

Landmark naming convention: `FDI_Descriptor`
(e.g., `35_Apex`, `45_Crown`, `44_Abut`, `II_Midpoint`)

### 4. Run the workflow

```bash
cd Workflow_v1/
python3 01_extract_dicom.py   # Extract images and metadata
python3 02_pnp.py             # Compute PnP (Scenarios A and B)
python3 03_figures.py         # Generate figures
```

---

## Landmark Coordinate System

3D Slicer exports FCSV files in the **LPS** (Left–Posterior–Superior)
coordinate system. When a CR DICOM is loaded with its native
`ImagerPixelSpacing` tag (e.g., 0.148 mm/px), 2D fiducial coordinates
are in **millimetres**, not pixels. The scripts convert automatically
(`coords_px = coords_mm / pixel_pitch`).

---

## Notes on DICOM Geometry

The focal length used in **Scenario A** is derived from the DICOM tag
`DistanceSourceToReferencePoint` (Radiation Dose SR), which represents
the source-to-patient surface distance — **not** the actual
Source-Image Distance (SID). The true SID is typically 100–150 cm for
portable chest radiography; when available, the tag
`DistanceSourceToDetector` (0018,1110) should be preferred.

---

## Related Work

- **SmilePnP** (smile photographs + Blender):
  https://github.com/Forensic-Odontologist/SmilePnP
- **3D Slicer**: https://www.slicer.org

## Citation

If you use this code, please cite it.