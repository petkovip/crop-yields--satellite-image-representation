# Geo-SimCLR — Self-supervised satellite features for U.S. corn-yield prediction

Final project for **AMTH 5520 / CPSC 4520 — Deep Learning Theory and Applications** (Yale, Spring 2026). Author: **Ivan Petkov**. Instructor: **Prof. Smita Krishnaswamy**.

For the full write-up (motivation, related work, methods, results, limitations, references) see [`final-report/AMTH_5520_DL_geo-simclr_neurips_2026.pdf`](final-report/AMTH_5520_DL_geo-simclr_neurips_2026.pdf). For the recorded talk, see [`final-report/presentation.pdf`](final-report/presentation.pdf).

## TL;DR

**Problem.** County-level corn-yield prediction from Sentinel-2 satellite imagery across seven U.S. Corn Belt states (IA, IL, IN, MN, NE, OH, SD), 2017–2022.

**Method.** Pre-train a ResNet-18 encoder with a SimCLR-style contrastive objective whose **positive pairs are the same field at adjacent biweeks** (Geo-SimCLR), then freeze the encoder and run linear / tree-based yield probes. Compared against six hand-crafted NDVI baselines, three frozen-encoder baselines (DINOv2, Prithvi-EO 2.0, random-init ResNet-18), and a supervised end-to-end ConvLSTM.

**Headline result.** Geo-SimCLR is the strongest single encoder on **both** generalisation tests:

| Encoder | Time-split $R^2$ (train ≤ 2020, test 2022) | County-split $R^2$ (held-out 20 % counties) |
|---|---:|---:|
| **Geo-SimCLR ResNet-18 (this work)** | **+0.669** | **+0.660** |
| DINOv2-base | +0.661 | +0.602 |
| Prithvi-EO 2.0 300M | +0.550 | +0.519 |
| Random-init ResNet-18 | +0.558 | +0.583 |
| Supervised ConvLSTM (end-to-end) | +0.551 | — |
| Best NDVI baseline (state + year + biweekly NDVI, LightGBM) | +0.429 | +0.651 |

A diagnostic in `notebooks-vF/06_visualize.ipynb` documents a partial dimensional-collapse limitation in the learned representation and proposes concrete fixes for future work.

## Repository layout

```
project-geospatial/
├── README.md                  ← you are here
├── requirements.txt           ← Python dependencies
├── PROJECT_STATUS.md          ← high-level status / decision log (development notes)
├── Satellite Data Notes.md    ← initial scoping notes
│
├── final-report/              ← all submission artefacts
│   ├── AMTH_5520_DL_geo-simclr_neurips_2026.pdf   ← final report (PDF)
│   ├── geo-simclr_neurips_2026.tex                ← LaTeX source (NeurIPS style)
│   ├── neurips_2026.sty                           ← style file
│   ├── references.bib                             ← BibTeX
│   ├── presentation.pdf                           ← 5-minute Beamer slides (PDF)
│   ├── presentation.tex                           ← LaTeX source for the slides
│   ├── charts/                                    ← every figure in the report (PNG, dpi 200)
│   │   ├── fig01_pipeline.png                     ← Geo-SimCLR architecture schematic
│   │   ├── fig02_augmentation_examples.png        ← real Sentinel-2 augmentations
│   │   ├── fig03_data_summary.png                 ← scope and coverage
│   │   ├── fig04_ndvi_phenology.png               ← NDVI motivation
│   │   ├── fig05_encoder_comparison.png           ← headline R² bar chart
│   │   ├── fig06_training_curves.png              ← SSL training trajectory
│   │   ├── fig07_mphate.png                       ← M-PHATE 2-panel
│   │   ├── fig08_collapse.png                     ← collapse diagnostic
│   │   ├── fig09_gradcam.png                      ← GradCAM interpretability
│   │   └── figA1..A3*.png                         ← appendix figures
│   └── tables/                                    ← every LaTeX table in the report
│       ├── tab01_encoder_headline.tex
│       ├── tab02_baselines.tex
│       ├── tab03_effective_rank.tex
│       ├── tabA1_full_grid.tex
│       └── tabA2_hyperparameters.tex
│
├── notebooks-vF/              ← submission version of the notebooks (run end-to-end)
│   ├── 02_acquire_data.ipynb
│   ├── 03_baselines_data.ipynb
│   ├── 04_geosimclr.ipynb
│   ├── 05_extract_and_probe.ipynb
│   ├── 06_visualize.ipynb
│   └── 09_results.ipynb
│
├── readings/                  ← reference papers used for related-work / methods
│
├── results/                   ← canonical results (small parquets + figures from nb 03/05/06)
│   ├── tables/                ← e.g. corn_yield_stage1.parquet, baseline_yield_results.parquet, effective_rank.parquet
│   └── charts/                ← raw figure outputs of the notebooks (the report copies live in final-report/charts/)
│
├── data/    (NOT on GitHub — see "Large artefacts" below)
│   ├── raw/cropnet/
│   │   ├── Sentinel-2 Imagery/data/{AG,NDVI}/...        ← biweekly multispectral HDF5s (~258 GB at full 7-state scope)
│   │   ├── USDA Crop Dataset/...                       ← USDA NASS yield + condition CSVs
│   │   └── WRF-HRRR Computed Dataset/...               ← weather covariates (NetCDF)
│   └── processed/
│       ├── county_index.parquet                        ← (state, fips, year, biweek) → h5_path index
│       ├── crd_lookup.parquet                          ← FIPS → Crop Reporting District lookup
│       ├── phen_stage_lookup.parquet                   ← phenological stage per (state, week)
│       ├── biweekly_ndvi.parquet                       ← per (fips, year, biweek) NDVI
│       ├── baseline_features.parquet                   ← peak / growing-season aggregates
│       ├── yield_county_year.parquet                   ← USDA corn yields (county-year)
│       ├── yield_soybean_county_year.parquet           ← USDA soybean yields (kept for future ablations)
│       └── features/
│           ├── feature_index.parquet                   ← per-tile (state, fips, year, biweek, grid_idx)
│           ├── geo_simclr_resnet18.npy                 ← per-tile Geo-SimCLR features (n × 512)
│           ├── dinov2_base.npy                         ← per-tile DINOv2 features (n × 768)
│           ├── prithvi_eo_300m.npy                     ← per-tile Prithvi features (n × 1024)
│           ├── random_resnet18.npy                     ← per-tile random-init ResNet-18 features
│           ├── per_county_year/<encoder>.parquet       ← (fips, year, feat_*) aggregates used by probes
│           └── per_county_year_biweek/<encoder>.parquet← (state, fips, year, biweek, feat_*) for visualisations
│
└── models/  (NOT on GitHub — see "Large artefacts" below)
    ├── geo_simclr_resnet18/
    │   ├── geo_simclr_resnet18_best.pt                 ← best-loss encoder + projector (~43 MB), used downstream
    │   ├── band_stats.json                             ← per-band normalisation statistics
    │   ├── training_log.parquet                        ← per-epoch loss / pos-sim / ‖h‖ / lr
    │   └── checkpoints/                                ← per-epoch full state (~130 MB each × 50 epochs ≈ 6.5 GB)
    └── convlstm/
        └── convlstm_best.pt                            ← best-val supervised ConvLSTM weights (~46 MB)
```

### Large artefacts (`data/` and `models/`)

The `data/` tree is roughly **260 GB** at the full seven-state scope (the raw CropNet Sentinel-2 imagery dominates) and the `models/` tree is roughly **6.5 GB** (mostly the per-epoch SSL checkpoints). Both are too large to host on GitHub, so they are gitignored. Two ways to obtain them:

1. **Direct copy.** I will share a OneDrive link with the instructor that contains the full `data/` and `models/` folders verbatim. Drop them at the repo root (preserving the layout above) and every notebook will run as-is. Available **on request** for anyone else.
2. **Reproduce from scratch.** The notebooks regenerate the full pipeline deterministically from public sources (CropNet HuggingFace + USDA NASS open data). Run them in order: `02 → 03 → 04 → 05 → 06 → 09`. The first part of `02_acquire_data.ipynb` covers a three-state pilot (~78 GB) for users with limited storage; the second part expands to seven states (~260 GB).

## Notebook walk-through (run order)

| Notebook | Purpose | Inputs | Outputs | Compute |
|---|---|---|---|---|
| `02_acquire_data.ipynb` | Download CropNet Sentinel-2 imagery, USDA crop yields, weather; build the `(state, fips, year, biweek, grid_idx)` tile index and the CRD lookup. The first half covers a 3-state pilot scope (IA / IL / IN, ~78 GB). The second half expands to the full seven-state scope (+~180 GB). | HuggingFace + USDA NASS APIs | `data/processed/county_index.parquet`, `data/processed/yield_county_year.parquet`, `data/processed/crd_lookup.parquet`, `data/processed/phen_stage_lookup.parquet`, `data/raw/cropnet/...` | CPU. Dominated by download bandwidth. |
| `03_baselines_data.ipynb` | Compute biweekly NDVI per (fips, year), build the six hand-crafted NDVI baselines (peak, growing-season summary, biweekly series, with / without state + year), and benchmark them with Ridge / LightGBM under both splits. Establishes the non-DL bar to beat. | Raw NDVI HDF5s, USDA yields | `data/processed/biweekly_ndvi.parquet`, `data/processed/baseline_features.parquet`, `results/tables/baseline_yield_results.parquet` | CPU. ~10–15 min on a laptop. |
| `04_geosimclr.ipynb` | Train the headline Geo-SimCLR encoder: ResNet-18 + projection head, NT-Xent on temporal-positive pairs (same field at biweek $t \pm 1$), 50 epochs at batch 1024 with AMP and DataParallel. Saves the best-loss encoder, a per-epoch training log, and per-band statistics. | `county_index.parquet`, raw AG HDF5s, `crd_lookup.parquet` | `models/geo_simclr_resnet18/geo_simclr_resnet18_best.pt`, `models/geo_simclr_resnet18/training_log.parquet`, `models/geo_simclr_resnet18/band_stats.json` | **GPU required for production** (~5 h on 2 GPUs). DEBUG path runs a 2-epoch CPU smoke test. |
| `05_extract_and_probe.ipynb` | Extract frozen features for **four encoders** (Geo-SimCLR, DINOv2-base, Prithvi-EO 2.0, random-init ResNet-18), aggregate per (county, year), run Ridge + LightGBM probes with bootstrap CIs under time and county splits, and train the supervised ConvLSTM comparator end-to-end. Produces the headline results table consumed by `09`. | Geo-SimCLR weights, raw AG HDF5s, USDA yields, NDVI baselines | `data/processed/features/<encoder>.npy`, `data/processed/features/per_county_year/<encoder>.parquet`, `models/convlstm/convlstm_best.pt`, `results/tables/corn_yield_stage1.parquet` | GPU strongly recommended (~3–4 h on 2 GPUs); CPU smoke test available via DEBUG. |
| `06_visualize.ipynb` | Diagnose the Geo-SimCLR representation: SVD spectrum and effective rank, M-PHATE 2-panel on the year-averaged county-biweek tensor, UMAP facets (state / CRD / yield quintile), per-county PCA biweek trajectories, GradCAM in agriculture-false-colour, plain PHATE for sanity. Surfaces the partial-collapse limitation. | `feature_index.parquet`, Geo-SimCLR features + weights | `results/tables/effective_rank.parquet`, `results/charts/{singular_spectrum, m_phate_collapse, umap_facets, pca_county_trajectories, gradcam_examples, phate_collapse}.png` | CPU-only, ~10–20 min. |
| `09_results.ipynb` | Single source of truth for every figure and LaTeX table in the report: re-renders `fig01..fig09 + figA1..figA3` into `final-report/charts/` and `tab01..tab03 + tabA1..tabA2` into `final-report/tables/`. Re-running this notebook regenerates the report assets verbatim from the cached parquet files. | All artefacts produced by the notebooks above | `final-report/charts/*.png`, `final-report/tables/*.tex` | CPU-only. |

## Setup

```bash
# Create a fresh Python 3.11 environment, then:
pip install -r requirements.txt
```

A HuggingFace account + token will speed up the CropNet downloads in `02_acquire_data.ipynb` (`huggingface-cli login` once before running the cell). Without a token, the package falls back to anonymous access, which is slower.

## Pointers

- For results, the headline numbers, the narrative, and the limitations discussion → see [`final-report/AMTH_5520_DL_geo-simclr_neurips_2026.pdf`](final-report/AMTH_5520_DL_geo-simclr_neurips_2026.pdf).
- For the 5-minute talk → [`final-report/presentation.pdf`](final-report/presentation.pdf).
- For implementation details and per-step rationale → the notebooks in `notebooks-vF/` (each starts with a *purpose / inputs / outputs / runtime* block).
- For the underlying dataset → CropNet (Lin et al., KDD 2024).
