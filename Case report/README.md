# PnP-Based 2D/3D Radiographic Superimposition

---

## Overview

This repository provides the fully open-source Python workflow for 3D/2D
Perspective-n-Point (PnP) registration of a dental CBCT model onto plain
radiographs — applied to a forensic identification case where only
ante-mortem (AM) CBCT and post-mortem (PM) 2D radiographs were available.

All landmark annotation was performed in [3D Slicer](https://www.slicer.org)
(free, open-source). PnP computation and figure generation use standard
scientific Python libraries (OpenCV, NumPy, trimesh, matplotlib, pydicom).

**This work echoes** [SmilePnP](https://github.com/Forensic-Odontologist/SmilePnP)
— a previously published Blender add-on for PnP-based smile photograph
superimposition — to the radiographic domain, replacing the photographic
modality with DICOM-formatted plain radiographs and the IOS dental scan
with a CBCT-derived 3D mesh.

---

## Landmark layout

```
Workflow/
├── 01_extract_dicom.py
├── 02_pnp.py                 # PnP per session (k–k pairing)
├── 03_figures.py             # Figures per session
│
├── landmarks/
│   ├── 3D_Landmarks/Session1/ … Session3/    # 3D FCSV (same landmarks across sessions)
│   ├── 2D_Landmarks_PM1/Session1/ … Session3/
│   └── 2D_Landmarks_PM2/Session1/ … Session3/
│
├── meshes/ 
└── output/
    ├── case_metadata.json
    ├── pnp_all_sessions.json
    ├── Session*/img1|img2/pnp_results.json
```

**Pairing rule:** for session `Sessionk`, the script uses only  
`3D_Landmarks/Sessionk` + `2D_Landmarks_PM1/Sessionk` + `2D_Landmarks_PM2/Sessionk`.  
FCSV landmarks must be **identically named** between 2D and 3D within a given session.

---

## Related Work

- **SmilePnP** (smile photographs + Blender):
  https://github.com/Forensic-Odontologist/SmilePnP
- **3D Slicer**: https://www.slicer.org

## Citation

If you use this code, please cite it.
