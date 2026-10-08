# Accepted model index

This draft includes metadata only. No trained model/scaler binaries are included or uploaded to GitHub; no public download link is available yet.

index.json preserves 112 accepted logical fits, their parameters, environments, provenance and SHA-256 hashes. The 54 principal and 60 convergence selections share two fits. Ten groups of three identical final SVR model–scaler pairs map to canonical_artifact_id, leaving 92 distinct binary pairs. Their logical seed/fold/OOF records remain separate.

index.csv lists each logical ID, algorithm, training count, seed, analyses and the relative model/scaler paths expected by a future deposit. Each pair consists of model.keras (ANN) or model.joblib and scalers.joblib. Input/output column order is recorded once in index.json.

~~~sh
python -B models/query.py --list
~~~

Inference and model verification require the future authorized binaries at the indexed relative paths. The current query command gives an explicit missing-binaries error if inference is requested. The prior model verification in ../results/model_verification.json is reused evidence from the accepted source delivery, not inference rerun on this draft.

Hash equality of deterministic SVR final fits does not imply equality of their out-of-fold predictions. Binary deduplication is an index mapping; no original artifacts were modified. Licensing and a future distribution location for separately deposited model binaries remain pending; those binaries are not covered by this repository's licenses.
