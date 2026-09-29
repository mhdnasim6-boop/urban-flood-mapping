# Urban flood mapping from Sentinel-1 intensity and coherence

MTech thesis project. Maps **flooded open areas** and **flooded urban areas** from Sentinel-1 SAR by
fusing dual-polarisation (VV/VH) backscatter intensity and InSAR coherence with physics-guided change
features and terrain/water context. A U-Net is trained with an imbalance-aware loss and gives
per-pixel uncertainty through Monte-Carlo dropout. Accuracy is measured on three flood events whose
reference maps were drawn by hand from optical imagery and never used for training.

## Research gap addressed

From the review by Zhao et al. (2025, IEEE GRSM) and the UrbanSARFloods benchmark (Zhao et al., 2024):

1. **Manual thresholds.** 17 of 19 thresholding studies set thresholds by hand, so they transfer poorly
   between events. In this data the event-wide coherence drop of stable pixels ranges from about 0 to 0.44,
   so no single coherence threshold fits every event.
2. **No model uncertainty.** About 90% of methods output only a flooded/not-flooded map.
3. **Single-event validation.** Most methods are tuned and tested on one flood.
4. **Urban flood detection remains weak.** The benchmark authors report that weighted cross-entropy and
   transfer learning do not overcome the extreme class imbalance (urban flood pixels are well under 1% of labelled pixels).

Contribution: physics-guided features (event-normalised coherence drop, intensity change, cross-pol
ratio, urban flooding index) plus terrain/water priors (HAND, slope, JRC water occurrence), focal + Dice
loss with class-aware sampling, MC-dropout uncertainty, and a controlled ablation against five baselines,
evaluated on independent manually labelled events.

## Input channels

| # | Channel | Meaning |
|---|---|---|
| 1-4 | `coh_pre_vh`, `coh_pre_vv`, `coh_co_vh`, `coh_co_vv` | InSAR coherence, pre-event and co-event pairs |
| 5-8 | `int_pre_vh`, `int_pre_vv`, `int_post_vh`, `int_post_vv` | Backscatter intensity (dB), pre and post event |
| 9 | `d_int_vv` | VV intensity change (dB): strong drop = open water, rise = double bounce |
| 10 | `d_int_vh` | VH intensity change (dB): separates flooded vegetation |
| 11-12 | `d_coh_vv_norm`, `d_coh_vh_norm` | Coherence drop minus the event-wide drop of stable pixels |
| 13 | `xpol_post` | VV - VH after the event: double bounce vs volume scattering |
| 14 | `ufi_vv` | Urban flooding index (Zhang et al., 2021) in dB: intensity rise and coherence loss |
| 15 | `hand` | log(1 + height above nearest drainage) from GLO-30 HAND |
| 16 | `slope` | Terrain slope (deg) from Copernicus GLO-30 DEM |
| 17 | `water_occ` | JRC Global Surface Water occurrence 1984-2021 |

## Methods compared

| ID | Method |
|---|---|
| B1 | Intensity change detection: Otsu water threshold + 3 dB decrease (open floods only) |
| B2 | B1 + fixed coherence-drop threshold 0.3 in stable pixels |
| B3 | Rule-based decision tree on intensity + event-normalised coherence (after Natsuaki & Hirose, 2018) |
| B4 | Random forest on the same 17 per-pixel inputs |
| B5 | U-Net, 8 raw bands, weighted cross-entropy: the dataset authors' setup |
| A | U-Net, 8 raw bands, focal + Dice + class-aware sampling |
| B | A + physics features (14 channels) |
| C | A + HAND / slope / water (11 channels) |
| **D** | **Proposed: all 17 channels** (also evaluated with 20 MC-dropout passes) |
| E | D without coherence |
| F | D without VH |

Comparisons: B5 vs A = training recipe; A vs B = physics features; A vs C = terrain/water context;
A vs D = everything added; D vs E = value of coherence; D vs F = value of VH.
All U-Nets share architecture (ResNet-34 encoder, ImageNet weights), optimiser and schedule.

## Data notes (checked against the data, not taken from the paper)

- **Band order.** The GeoTIFFs store `[coh_pre_VH, coh_pre_VV, coh_co_VH, coh_co_VV, int_pre_VH,
  int_pre_VV, int_post_VH, int_post_VV]`. That differs from the order given in the paper text.
  Verified on the Jubba reference: flooded-open pixels drop about 8-11 dB only in bands 7-8, and
  flooded-urban pixels lose coherence from 0.61/0.79 (bands 1-2) to 0.21/0.25 (bands 3-4).
- **Training labels are semi-automatic.** They were made with an intensity-change threshold, a fixed 0.3
  coherence threshold and the WSF2019 settlement mask. So B2 mirrors the labelling rule, and built-up maps
  are deliberately **not** used as inputs, because that would leak the labels. Final accuracy comes only from the
  manually labelled test events (Weihui, Nova Kakhovka, Jubba).
- **Missing test chips.** 997 of the 2,029 published `20231201_Jubba_2` test chips are empty files on
  Hugging Face. They are skipped, and Jubba_2 is evaluated on the remaining 1,032 chips.
- **Storage.** Intensity is stored in 0.2 dB steps and coherence in 1/254 steps (uint8). Both are far
  finer than speckle, and this lets the 38 GB archive fit in Kaggle's 20 GB output limit.

## How to run

### 1. Local check (laptop)
```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/smoke_test.py --work /tmp/ufm_smoke
```
Downloads ~40 test chips and runs every step for 2 epochs. The metrics are meaningless; it only
proves the pipeline works.

### 2. Push to GitHub
Create an empty repository `urban-flood-mapping` on github.com, then:
```bash
git remote add origin https://github.com/mhdnasim6-boop/urban-flood-mapping.git
git push -u origin main
```

### 3. Kaggle (in this order)
Import the notebooks from `kaggle/` (*Create > New Notebook > File > Import Notebook*), set
`GITHUB_REPO` in the first cell, then *Save Version > Save & Run All*:

| Notebook | Accelerator | Output | Time (approx.) |
|---|---|---|---|
| `01_prepare_trainval.ipynb` | None | `ufm-trainval` (~17 GB) | 1-2 h |
| `02_prepare_test.ipynb` | None | `ufm-test` (~4 GB) | 0.5-1 h |
| `03_train_evaluate.ipynb` | GPU | `runs/`, `report/` | 6-8 h |

For notebook 03, add the outputs of 01 and 02 as inputs (*Add Input > Your Work*). The report
contains `comparison.md`, `f1_bars.png` and `maps_<event>.png` for the presentation.

## Layout
```
configs/default.yaml            shared settings
configs/experiments/*.yaml      one file per experiment (B5, A-F)
src/ufm/                        library: bands, features, data, model, losses, metrics, baselines, aux layers
scripts/prepare_train.py        stream + compress train/val archive
scripts/prepare_test.py         download manually labelled test events
scripts/build_aux.py            HAND, slope, JRC water per chip
scripts/compute_stats.py        normalisation statistics
scripts/train.py                train one experiment
scripts/evaluate.py             test metrics + maps (+ MC-dropout uncertainty)
scripts/run_baselines.py        B1-B4
scripts/summarize.py            comparison table and figures
scripts/smoke_test.py           end-to-end local check
kaggle/                         notebooks that run the scripts on Kaggle
```

## References
- Zhao, J., Li, M., Li, Y., Matgen, P., Chini, M. (2025). Urban flood mapping using satellite synthetic
  aperture radar data: A review of characteristics, approaches, and datasets. *IEEE GRSM*, 13(1), 237-268.
- Zhao, J., et al. (2024). UrbanSARFloods: Sentinel-1 SLC-based benchmark dataset for urban and open-area
  flood mapping. *CVPR Workshops (EarthVision)*. Data: CC-BY-4.0, huggingface.co/datasets/S1Floodbenchmark/UrbanSARFloods_v1
- Chini, M., et al. (2019). Sentinel-1 InSAR coherence to detect floodwater in urban areas: Houston and
  Hurricane Harvey as a test case. *Remote Sensing*, 11(2), 107.
- Pelich, R., et al. (2022). Mapping floods in urban areas from dual-polarization InSAR coherence data.
  *IEEE GRSL*, 19, 1-5.
- Pulvirenti, L., Chini, M., Pierdicca, N., Boni, G. (2019). Flood detection in urban areas: analysis of
  time series of coherence data in stable scatterers. *IGARSS*.
- Zhang, H., et al. (2021). An urban flooding index for unsupervised inundated urban area detection using
  Sentinel-1 polarimetric SAR images. *Remote Sensing*, 13(22), 4511.
- Natsuaki, R., Hirose, A. (2018). L-band SAR interferometric analysis for flood detection in urban area:
  a case study in 2015 Joso flood, Japan. *IGARSS*.
- Nobre, A. D., et al. (2011). Height Above the Nearest Drainage: a hydrologically relevant new terrain
  model. *Journal of Hydrology*, 404, 13-29.
- Pekel, J.-F., et al. (2016). High-resolution mapping of global surface water and its long-term changes.
  *Nature*, 540, 418-422.
