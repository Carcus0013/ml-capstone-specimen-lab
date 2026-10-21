# Progress log

*Last updated: 2026-10-21. This file records the project's state so that work can resume without repeating completed steps.*

## Current phase
Phase E: building the Streamlit application. Phases A–D (inspect, design, implement, execute) are complete for all three tracks.

## Completed
| Milestone | Evidence |
|---|---|
| Official guidelines located and read (7 pages) | `docs/official_requirements.md`. The project-root copy was 0 bytes at first; the full copy came from `~/Downloads`, and the user later restored an identical copy (same MD5) |
| Prior work reviewed | `C:\Users\HP\ML_Casestudy\review1_regression.ipynb` and `review1_classification.ipynb` (Review 1). Their leakage audit and engineered-feature decisions were reused. The originals are untouched |
| Datasets audited, mapping justified | `reports/results/dataset_audit.{json,csv}`, `data/README.md` |
| Shared package | `src/` (config, loading, audit, feature engineering, preprocessing, model specs, evaluation, plots) |
| Regression notebook executed | `notebooks/regression.ipynb`: 0 error cells. Selected Polynomial Regression (deg 3): test R² 0.99866, CV R² 0.99829 |
| Classification notebook executed | `notebooks/classification.ipynb`: 0 error cells. 7/10 models perfect on test; selected Logistic Regression (C=100) |
| Clustering notebook executed | `notebooks/clustering.ipynb`: 0 error cells. k = 4; K-Means silhouette 0.574 (all rows); Ward agglomerative 0.569 (10k sample); ARI between methods 0.979 |
| Model artefacts | `models/{regression,classification,clustering}/` (all tuned pipelines + metadata JSON, `selection.json`, `feature_schema.json`) |
| Git | Local repo on `main` with milestone commits (`git log --oneline`) |

## Commands that reproduce the results
```bash
python -m src.audit_datasets
jupyter nbconvert --to notebook --execute --inplace notebooks/regression.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/classification.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/clustering.ipynb
```
Measured runtimes on this machine (16 logical CPUs, 15 GB RAM): regression ≈ 95 s, classification ≈ 145 s, clustering ≈ 55 s.

## Defects found and fixed
1. `regression.ipynb` cell that transformed with only the `preprocess` step (it skipped feature engineering). Fixed by using `pipeline[:-1]`.
2. scikit-learn 1.7.2 `KNeighborsClassifier(metric="manhattan", weights="uniform")` fails with string labels. Fixed at the root by encoding the target 0/1 (`config.CLF_TARGET_ENCODING`).
3. The classification selection rule's log-loss tie-breaker picked 1-NN, whose probabilities are degenerate (always 0 or 1). It was replaced by a one-standard-deviation tie plus a simplicity rule, and the change is documented in the notebook.

## Known issues
- On Windows, joblib's loky resource tracker prints harmless `KeyError` lines to the host terminal after parallel grid searches. These do not appear in notebook outputs.

## Outstanding
- Streamlit app (`app/`), tests (`tests/`), README, methodology, viva questions, checklist, CI workflow.
- Public deployment: needs the user to create or authorise a GitHub repository and sign in to Streamlit Community Cloud.

## Local-only helpers (not committed)
`.dev/` holds the notebook builder scripts (`build_*.py`, `nbtools.py`). Rebuild a notebook with `python .dev/build_<track>.py` and execute it with `python .dev/nbtools.py exec <track>`.

## Next action
Build `app/app.py` and its pages using the saved pipelines and result files.
