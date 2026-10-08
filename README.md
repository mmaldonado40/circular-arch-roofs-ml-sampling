# Sampling Strategies for Machine Learning-Based Wind Load Prediction of Circular Arch Roofs: A Comparative Study of Accuracy and Data Efficiency

**Misael Maldonado-Fajardo, Humberto Yáñez-Godoy, Jaime M. Horta-Rangel, L. Francisco Pérez-Moreno, Enrique Rico-García, and Iván F. Arjona-Catzim**

Code, realized designs, processed CFD datasets, accepted results and trained models accompanying the manuscript submitted to *Machine Learning with Applications* (2026). The manuscript is **submitted and unpublished**, with no assigned DOI. This repository supports saved-result verification, independent ML refits, figure generation and inference with accepted final models.

## Title and citation

Maldonado-Fajardo, M., Yáñez-Godoy, H., Horta-Rangel, J. M., Pérez-Moreno, L. F., Rico-García, E., and Arjona-Catzim, I. F. (2026). *Sampling Strategies for Machine Learning-Based Wind Load Prediction of Circular Arch Roofs: A Comparative Study of Accuracy and Data Efficiency*. Manuscript submitted to *Machine Learning with Applications*; unpublished.

~~~bibtex
@unpublished{MaldonadoFajardo2026CircularArchRoofs,
  author = {Maldonado-Fajardo, Misael and Y{\'a}{\~n}ez-Godoy, Humberto and
            Horta-Rangel, Jaime M. and P{\'e}rez-Moreno, L. Francisco and
            Rico-Garc{\'i}a, Enrique and Arjona-Catzim, Iv{\'a}n F.},
  title  = {Sampling Strategies for Machine Learning-Based Wind Load Prediction of Circular Arch Roofs: A Comparative Study of Accuracy and Data Efficiency},
  year   = {2026},
  note   = {Manuscript submitted to Machine Learning with Applications; unpublished}
}
~~~

[CITATION.cff](CITATION.cff) supplies the machine-readable preferred citation. Source code: [MIT](LICENSE). Original data, results, figures, trained models and scalers: [CC BY 4.0](LICENSE-DATA.md).

## Overview

The study examines how experimental design and training-set size affect prediction of wind-pressure coefficients on circular arched roofs. Each CFD configuration supplies five geometry/wind inputs and 315 pressure coefficients. Six algorithms are compared across nine realized designs: artificial neural networks (ANN), support vector regression (SVR), k-nearest neighbors (KNN), decision trees (DT), random forests (RF), and gradient boosting (GBoost). ANN/SVR convergence is evaluated on ten retained subset sizes with three recorded seeds per size.

The accepted package includes the ANN corrections and the RF correction completed on 5 October 2026. Saved predictions allow the ML analyses to be checked without model training or CFD simulation.

## Repository structure

~~~text
circular-arch-roofs-ml-sampling/
├── README.md                         # Scientific scope and workflow
├── CITATION.cff                      # Manuscript citation
├── LICENSE                           # MIT source-code license
├── LICENSE-DATA.md                   # CC BY 4.0 scope and exclusions
├── requirements.txt                  # Pinned direct dependencies
├── config.json                       # Accepted parameters, IDs, folds and seeds
├── reproduce.py                      # Plan or independent refits
├── verify.py                         # Saved-result checks and table exports
├── code/
│   ├── evaluation.py                 # Accepted six-algorithm evaluation core
│   └── figures.py                    # F5–F8 rendering; F1–F4 preservation
├── data/
│   ├── README.md                     # Variable and dataset dictionary
│   ├── All_dataset.csv               # Authoritative 221-case master table
│   ├── doe_input_quantization.xlsx   # Design-construction provenance
│   └── designs/                      # 18 realized design/subset CSV files
├── results/
│   ├── predictions.npz               # Test/OOF arrays for 112 logical fits
│   ├── reference.json                # Metrics, comparisons and corrected timings
│   ├── verification.json             # Recorded package-verification evidence
│   ├── model_verification.json       # Recorded source-model verification
│   ├── SHA256SUMS.txt                # Versioned package-file hashes
│   └── figures/                      # 23 retained PNG panels
└── models/
    ├── README.md                     # Download, integrity and inference
    ├── index.json                    # Logical fits, canonical paths and hashes
    ├── index.csv                     # Compact lookup table
    ├── query.py                      # Load models and predict; never fit
    ├── verify_artifacts.py           # ZIP/extracted-file SHA-256 verification
    ├── release.json                  # Asset checksum, size and counts
    ├── ARTIFACT_SHA256SUMS.txt        # All 110 distinct binary-file hashes
    └── ARCHIVE_README.md             # Extraction instructions and model license
~~~

The Release ZIP adds indexed binaries under models/principal/ and models/convergence/. Identical scalers share canonical indexed paths. Weights are excluded from Git history. Optional commands write separate generated_tables/, generated_figures/ or new_runs/ directories.

## Requirements

Use Python 3.12 and [requirements.txt](requirements.txt): NumPy 2.2.6, SciPy 1.16.1, scikit-learn 1.7.1, joblib 1.5.1, TensorFlow 2.20.0, Keras 3.11.3, threadpoolctl 3.6.0 and Matplotlib 3.10.6. Accepted runs used Windows/Python 3.12.14; environments are recorded in config.json. An independent Windows/Python 3.12.14 installation and the workflow commands were checked as recorded in results/verification.json.

Install in a dedicated environment from the repository root. On Windows:

~~~powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip check
~~~

On Linux/macOS use .venv/bin/python instead. In all commands below, **python means the environment's interpreter**: replace it with .venv/Scripts/python.exe or .venv/bin/python as appropriate. The supplied verification and inference commands need no GPU.

## Reproduction workflow

### 1. Experimental designs and datasets

Start with data/All_dataset.csv, data/designs/ and the [data dictionary](data/README.md). These are the realized matrices used by the accepted evaluations, rather than matrices regenerated from an assumed LHS seed. Configuration maps jobs to matrices and ordered development/test IDs. Verification checks matrix values against the master and retained hashes. The quantization workbook documents construction; its alternative mapped ratios must not replace authoritative master inputs.

### 2. Reference result verification

~~~sh
python -B verify.py
~~~

Expected output includes status PASS_ML, 112 unique evaluations, 224 test/OOF partitions, 54 principal selections, 60 convergence selections, 90 correlations, 18 bootstrap comparisons and ten sampling diagnostics. The command checks saved predictions, metrics, IDs, folds, ANN epochs, index consistency and results/SHA256SUMS.txt. It performs no training or model inference and writes no files unless an output option is supplied.

### 3. ML model configuration

~~~sh
python -B reproduce.py --plan
python -B reproduce.py --algorithm RF --dataset CCD43 --plan
~~~

The full plan ends with **PLAN ONLY: no fitting 112 unique evaluations**. It checks data and job membership through the accepted core; filters select recorded jobs without changing their settings. Hyperparameters, selected candidates, seeds, folds, ANN epoch choices and historical search evidence are in config.json. No new search is needed.

### 4. Training and evaluation

For an intentional independent refit, supported commands are:

~~~sh
python -B reproduce.py --execute
python -B reproduce.py --algorithm RF --dataset CCD43 --execute
~~~

**--execute performs training**; omit it for saved-result verification and plans. Refits apply the accepted CV, scaling, selection, refit and test procedure and write configuration, fold/final artifacts, predictions and metrics into a new new_runs/<run-id>/ directory. Accepted artifacts are preserved. Full refits were not performed for the distribution checks; identical ANN weights across platforms are not guaranteed.

### 5. Results and figures

~~~sh
python -B verify.py --output-dir generated_tables
python -B code/figures.py --output generated_figures
python -B code/figures.py --check
~~~

Choose new output directories. Table export produces metrics.csv, case_R2.csv, local_performance.csv, convergence.csv, correlations.json, bootstrap.json, sampling.json and an export-specific verification.json. Figure export produces 23 PNG panels: 19 regenerated F5–F8 panels and four preserved F1–F4 graphics. --check compares regenerated pixels to retained panels; the expected result in the verified rendering environment is PASS_PIXEL_EXACT. Fonts/rendering can change pixels across environments even when numerical inputs agree.

### 6. Trained model download and inference

Download [Trained Models v1.0](https://github.com/mmaldonado40/circular-arch-roofs-ml-sampling/releases/tag/models-v1.0), asset [trained-models-v1.0.zip](https://github.com/mmaldonado40/circular-arch-roofs-ml-sampling/releases/download/models-v1.0/trained-models-v1.0.zip), outside Git. While the repository is private, downloads require an authorized GitHub account. With an authenticated GitHub CLI, from the repository root:

~~~sh
gh release download models-v1.0 --repo mmaldonado40/circular-arch-roofs-ml-sampling --pattern trained-models-v1.0.zip --dir ../mlwa-model-download
python -B models/verify_artifacts.py --archive ../mlwa-model-download/trained-models-v1.0.zip
python -m zipfile -e ../mlwa-model-download/trained-models-v1.0.zip .
python -B models/verify_artifacts.py
python -B models/query.py --list
~~~

Extract into the root containing config.json; no renaming is needed. Verification checks asset size/SHA-256, every model/scaler hash and all 112 logical mappings. [models/release.json](models/release.json) records size and checksum. The ZIP contains 92 models and 18 shared scaler files, representing 92 distinct model–scaler pairs. --list lists all 112 logical fits.

After step 5 creates generated_tables/, predict the 20 recorded test configurations with the corrected ANN/LHS126 final model:

~~~sh
python -B models/query.py --id r003 --output generated_tables/ANN_LHS126_test_predictions.csv
~~~

For custom inputs the CSV must provide R/D, B/D, D, h and Angle; optional ID values label rows:

~~~sh
python -B models/query.py --id r003 --input inputs.csv --output generated_tables/ANN_LHS126_custom_predictions.csv
~~~

Outputs are ID and Cp_001–Cp_315 in original dimensionless Cp units. The output parent must exist and the file must be new. Loading checks hashes, training IDs and estimator parameters; prediction applies fitted X scaling and inverse Y scaling without fit. See [models/README.md](models/README.md).

## Experimental configuration

| Input | Meaning | Unit | Declared range |
|---|---|---|---|
| R/D | Roof rise / span | dimensionless | 0.10–0.50 |
| B/D | Longitudinal roof length / span | dimensionless | 1.00–3.00 |
| D | Span | m | 10–100 |
| h | Eave height | m | 5–25 |
| Angle | Wind direction: normal to ridge at 0°, parallel at 90° | degrees | 0–90 |

R denotes roof rise. Outputs represent 15 longitudinal planes × 21 points; local analyses use planes 1, 8 and 15. The master has 221 unique geometries: 201 development cases and the common 20 held-out cases T_1–T_20. All outputs of one configuration stay in the same partition.

| Principal family | Base | + 16-case half factorial (FFD) | + 32-case full factorial (2k) |
|---|---:|---:|---:|
| Central composite design (CCD) | 43 | 59 | 75 |
| Retained reduced LHS (R-LHS; configuration label LHS43) | 43 | 59 | 75 |
| Full LHS | 126 | 142 | 158 |

Nine principal matrices plus nine additional subsets give 18 CSV matrices. Convergence uses N = 13, 25, 38, 50, 63, 76, 88, 101, 113 and 126. N126 reuses LHS126.csv, also labelled CONV_H10 in configuration. Six algorithms × nine designs give 54 principal selections. ANN/SVR × ten sizes × three seeds give 60 convergence selections. Two fits are shared, leaving 112 logical fits. Twenty redundant final SVR pairs map to canonical artifacts, leaving 92 distinct pairs; logical seed/fold/OOF records stay separate.

Shuffled five-fold CV uses recorded seeds. Principal selections use seed 42; convergence uses 42, 43 and 44. Fold seeds are job seed + zero-based fold index + 1. MinMax scalers are fitted within training partitions. ANN epoch selection uses an inner 20% training validation split and patience 15, followed by refitting; median selected CV epochs define final refit duration. The common test set is evaluated after refitting. Corrected RF parameters and its CCD59 candidate selection remain in config.json.

ANN uses Dense/ReLU, Dropout, a linear 315-output layer and Adam/MSE. SVR/GBoost use MultiOutputRegressor; DT/RF/KNN support multiple outputs directly. All accepted hyperparameters are specified per job in config.json. Execution records use two threads; RF/KNN estimators and multioutput wrappers use n_jobs=1 in the accepted core.

## Evaluation and results

[results/reference.json](results/reference.json) records pooled R², mean output-wise R², MAE, MSE, RMSE, pooled RSR with population standard deviation (ddof=0), and case-level R² on original Cp values. Test/OOF arrays in results/predictions.npz are keyed by logical fit ID and partition. Verification recalculates these analyses from saved arrays.

Other analyses include local performance; 90 Pearson correlations with Fisher intervals and Holm correction across 90 tests; 18 paired percentile-bootstrap RMSE comparisons (10,000 draws, seed 20260908, CFD configuration as resampling unit); ANN/SVR convergence means and sample standard deviations across three seeds; and normalized-input sampling diagnostics (minimum pairwise distance and squared centered discrepancy).

F5 compares SVR predictions with CFD coefficients; F6 relates wind angle to ANN/SVR case-level R²; F7–F8 show pooled R²/MSE convergence. F1–F3 preserve geometry, measurement and domain schematics; F4 preserves the historical design projection. Reference panels and corrected RF timing records are retained.

## Reproducibility

Ordered IDs, folds, seeds, parameters, epochs, environments, provenance identifiers and hashes are supplied. Realized matrices are authoritative: historical LHS generation seeds and complete per-design tuning ledgers are unavailable, some matrices were assembled manually and convergence subsets are not all nested. Historical sklearn exploration scaled full development data before SearchCV; accepted evaluation fits scalers within training partitions. These procedures are recorded separately.

The test set had already been examined historically; it is not a new external validation. Times are observations rather than controlled hardware benchmarks. ANN refits can vary across platforms. Full independent refits and Linux/macOS execution have not been tested. Independent F1–F4 generators were not recovered; supplied graphics are preserved.

CFD validation belongs to Maldonado-Fajardo et al. (2025), [*Optimizing CFD Configuration for Accurate Prediction of Wind Pressure Coefficients on Circular Arched Roof Structures*](https://doi.org/10.9734/cjast/2025/v44i124646), and is outside this reproduction scope. Aggregate provenance is retained; restricted third-party experimental measurements are not distributed. Historical source/run paths are provenance identifiers, not operative dependencies: execution uses this repository's data, configuration and downloaded models.

## Trained models

The [models-v1.0 Release](https://github.com/mmaldonado40/circular-arch-roofs-ml-sampling/releases/tag/models-v1.0)
contains trained-models-v1.0.zip (731,877,651 bytes; approximately 731.9 MB),
with 92 models and 18 shared scaler files covering all 112 logical fits.
Follow [the download and inference workflow](#6-trained-model-download-and-inference)
or [models/README.md](models/README.md). Checksums and distribution-validation
evidence are recorded in [models/release.json](models/release.json).

The distribution was checked in an isolated Windows/Python 3.12.9 environment
with all eight pinned direct dependencies: complete ZIP/extraction hashes and
logical coverage, saved-result verification, the no-fit plan, eight table exports,
23 figures (19 regenerated panels pixel identical), representative inference for
all six algorithms and a custom-input check. No training or CFD was performed.

## License

Source code is licensed under [MIT](LICENSE). Original datasets, results, figures, trained models and fitted scalers are licensed under [CC BY 4.0](LICENSE-DATA.md), including commercial reuse with attribution and the other license conditions. This covers the GitHub Release binaries. Third-party material and the manuscript are excluded from these grants.
