# SARA Validation Data Guidelines

> [!IMPORTANT]
> **Strict Data Compliance Rules**

This directory is designated exclusively for **validation-only** and **evaluation benchmark** data.

### Permitted Data Rules:

1. **Analysis & Pipeline Inputs (Allowed):**
   - Sentinel-1 SAR imagery
   - Sentinel-2 Optical imagery (when cloud-free/suitable)
   - Copernicus DEM elevation rasters
   - Pre-event OpenStreetMap (OSM) vector data
   - Permitted open flood training datasets (e.g., KuroSiwo, Sen1Floods11 subject to license verification)

2. **Validation-Only Material (Strictly Isolated):**
   - Copernicus EMS / EMSR927 damage and activation maps
   - UNOSAT published flood/damage vectors
   - Post-event OpenStreetMap edits or manually digitized post-disaster labels
   - Any external reference flood boundaries prohibited from analysis by hackathon/challenge rules

### Operating Rules:
- **Isolation:** Never import or load files from `validation/` inside the automated inference, feature selection, or training pipelines in `backend/`, `ml/`, or `geospatial/`.
- **Benchmarking Only:** Validation datasets may only be used by standalone evaluation scripts to compute metric accuracy (IoU, Dice, Precision, Recall) against SARA model predictions.
- **Labeling:** Mark all validation reports and benchmark outputs explicitly as ground-truth evaluation metrics.
