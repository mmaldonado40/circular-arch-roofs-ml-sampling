# Accepted trained models

[Trained Models v1.0](https://github.com/mmaldonado40/circular-arch-roofs-ml-sampling/releases/tag/models-v1.0) provides accepted final models and scalers in trained-models-v1.0.zip. Original binaries are licensed under [CC BY 4.0](../LICENSE-DATA.md), including commercial reuse; source code is MIT licensed.

## Index and binary sharing

index.json preserves 112 logical fits, the 54 principal and 60 convergence selections, parameters, environments, provenance and SHA-256 hashes. Two fits occur in both analyses. Ten groups of three byte-identical final SVR pairs share canonical artifacts, leaving 92 distinct pairs. Logical seeds, folds and OOF predictions stay separate: identical final fits need not have identical OOF predictions.

The archive contains 92 model files (38 ANN, 18 SVR and nine each of DT, GBoost, KNN and RF) and 18 shared scaler files. Each scaler file contains both fitted X and Y scalers and ordered fit IDs. Sharing is based on SHA-256 equality, not an assumption about training partitions. Every indexed path is usable by query.py; no renaming or duplicate files are needed.

index.csv is the compact logical-fit lookup. Its included field and index.json's included_binary_files/artifact_bytes refer to binaries in the versioned Git package (zero). release.json describes separately distributed binaries, size and checksum. Weights never enter Git history.

## Download, extraction and integrity

Use the Release page or an authenticated GitHub CLI. From the repository root, keep the ZIP outside Git and extract into the root containing config.json:

~~~sh
gh release download models-v1.0 --repo mmaldonado40/circular-arch-roofs-ml-sampling --pattern trained-models-v1.0.zip --dir ../mlwa-model-download
python -B models/verify_artifacts.py --archive ../mlwa-model-download/trained-models-v1.0.zip
python -m zipfile -e ../mlwa-model-download/trained-models-v1.0.zip .
python -B models/verify_artifacts.py
python -B models/query.py --list
~~~

An authorized account is required while the repository is private. The standard-library verifier checks size/SHA-256 against release.json, each binary against ARTIFACT_SHA256SUMS.txt and index.json, and all 112 logical mappings. Without --archive it checks extracted files. Neither mode loads or fits models.

The ZIP contains indexed models/... binaries, the manifest and ARCHIVE_README.md with extraction/license instructions. It contains no datasets, fold models, manuscript or superseded models.

## Inference

Install [the pinned requirements](../requirements.txt) and use that environment's interpreter. Create the table directory through saved-result export, then predict with the corrected ANN/LHS126 model:

~~~sh
python -B verify.py --output-dir generated_tables
python -B models/query.py --id r003 --output generated_tables/ANN_LHS126_test_predictions.csv
~~~

Without --input, inference uses the 20 recorded test configurations. Custom CSV inputs must provide R/D, B/D, D, h and Angle; optional ID values label rows:

~~~sh
python -B models/query.py --id r003 --input inputs.csv --output generated_tables/ANN_LHS126_custom_predictions.csv
~~~

The output parent must exist and the file must be new. Output columns are ID and the 315 original-scale Cp coefficients. Loading checks hashes, scaler fit IDs and estimator parameters. Prediction uses stored scaling without fit.

An optional exhaustive inference check of all logical fits, scaler extrema, parameters and recorded test predictions is available:

~~~sh
python -B models/query.py --verify
~~~

This performs inference, not training. The source-delivery check is recorded in ../results/model_verification.json. Distribution checks use complete hash/coverage checks and representative inference for all six algorithms; ANN numerical identity across environments has the documented tolerance and platform limitations.
