# Trained Models v1.0 — extraction and license

This archive contains the accepted final models and fitted input/output scalers
for the circular arched roof wind-pressure study. Its paths are relative to the
root of `circular-arch-roofs-ml-sampling`; extract it there, alongside `config.json`.
It provides 92 distinct model–scaler pairs for all 112 logical fits. Identical
scalers share canonical paths in `models/index.json`: 92 model files and 18 scaler
files are sufficient. No fold models or superseded fits are included.

From the repository root, with the ZIP stored outside the Git working tree:

```sh
python -B models/verify_artifacts.py --archive ../mlwa-model-download/trained-models-v1.0.zip
python -m zipfile -e ../mlwa-model-download/trained-models-v1.0.zip .
python -B models/verify_artifacts.py
python -B models/query.py --list
```

`ARTIFACT_SHA256SUMS.txt` covers every binary file. The verifier also checks the
archive checksum recorded in `models/release.json` and all logical mappings.
See `models/README.md` for inference commands. Model files remain excluded from
conventional Git history by `.gitignore`.

The original trained models and scalers are licensed under **CC BY 4.0**, including
commercial reuse, by authorization of the author. Attribution: Misael Maldonado
Fajardo (2026), trained models accompanying *Sampling Strategies for Machine
Learning-Based Wind Load Prediction of Circular Arch Roofs: A Comparative Study
of Accuracy and Data Efficiency*.

License: https://creativecommons.org/licenses/by/4.0/
Legal text: https://creativecommons.org/licenses/by/4.0/legalcode
Scope and exclusions: `LICENSE-DATA.md` in the repository root. Third-party
material and the manuscript are excluded. The MIT license applies to source code.
