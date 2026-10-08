# Wind-pressure surrogate modelling on circular arched roofs

Preliminary research package for the study submitted to *Machine Learning with Applications*. It compares ANN, SVR, KNN, DT, RF and gradient boosting on nine retained experimental designs, with ANN/SVR convergence on ten retained subset sizes. The accepted results include the ANN corrections and the RF correction completed on 5 October 2026. This package has not been published.

- **data/**: one master table, 18 realized design/subset tables, quantization workbook and variable dictionary.
- **code/**: accepted evaluation functions and portable plotting functions.
- **config.json**: parameters, seeds, ordered IDs, folds, epoch choices, environments and provenance.
- **results/**: test/OOF predictions, reference metrics and corrected RF timings, verification evidence and 23 reference figures.
- **models/**: indices of 112 logical fits / 92 distinct model–scaler pairs. Binary weights are not included.

## Use

Python 3.12.14 was recorded for the accepted runs. From this directory:

~~~sh
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -B verify.py
.venv/Scripts/python.exe -B reproduce.py --plan
.venv/Scripts/python.exe -B code/figures.py --check
~~~

On Linux/macOS use .venv/bin/python. These commands check saved results and the refit plan without training. A new isolated Windows/Python 3.12.14 environment successfully installed requirements.txt and passed imports, saved-result verification, the no-fit plan, and table/figure exports. See results/verification.json. Full refits and Linux/macOS execution were not tested.

Optional exports to new folders:

~~~sh
python -B verify.py --output-dir generated_tables
python -B code/figures.py --output generated_figures
python -B models/query.py --list
~~~

To perform new fits, only when intended:

~~~sh
python -B reproduce.py --execute
python -B reproduce.py --algorithm RF --dataset CCD43 --execute
~~~

These commands preserve the accepted procedure and write separate new_runs/ outputs; they do not tune new hyperparameters or replace reference results. All 54 principal and 60 convergence evaluations map to 112 unique logical fits.

## Scope and limits

Numerical checks use saved predictions in the original Cp scale. Five folds split CFD configurations; the 315 outputs of one configuration stay together. The same 20 held-out cases are used across algorithms. ANN epoch selection uses an inner training split, followed by refitting. Reference timings are recorded observations, not controlled hardware comparisons.

The realized matrices are authoritative. Historical LHS generation seeds and a complete tuning ledger are unavailable; some designs were built manually and the convergence subsets are not all nested. ANN refits can vary across platforms. The test set was already examined historically, so it is not a new external validation.

CFD validation was developed in prior research: Maldonado-Fajardo et al. (2025), [Optimizing CFD Configuration for Accurate Prediction of Wind Pressure Coefficients on Circular Arched Roof Structures](https://doi.org/10.9734/cjast/2025/v44i124646). It is outside this repository’s reproduction objectives. This package focuses on retained designs, processed CFD datasets for ML, the six algorithms, accepted training/evaluation configurations, convergence, and article metrics/figures. No CFD validation data or reconstruction is required; aggregate prior-study provenance is retained in results/reference.json.

F5–F8 can be rendered from saved data. F1–F3 schematics and the historical F4 projection are preserved graphics; their independent generators have not been recovered. Source/run paths in JSON are relative historical provenance identifiers, not files required by this package.

The associated manuscript is submitted and unpublished; its current citation is given below. A future model-download location has not been assigned. Trained binaries, manuscript, reviewer responses and private audit records are excluded.

## Citation

Please cite the associated manuscript using the preferred citation in [CITATION.cff](CITATION.cff):

Misael Maldonado-Fajardo, Humberto Yáñez-Godoy, Jaime M. Horta-Rangel, L. Francisco Pérez-Moreno, Enrique Rico-García, and Iván F. Arjona-Catzim (2026). *Sampling Strategies for Machine Learning-Based Wind Load Prediction of Circular Arch Roofs: A Comparative Study of Accuracy and Data Efficiency*. Manuscript submitted to *Machine Learning with Applications*; unpublished. No DOI has been assigned.

## License

Source code is licensed under the MIT License (see `LICENSE`).

Original research datasets, results, and figures are licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0; see `LICENSE-DATA.md`).

Third-party materials are excluded from these grants. The manuscript and separately deposited trained model binaries are not covered by this repository's licenses.
