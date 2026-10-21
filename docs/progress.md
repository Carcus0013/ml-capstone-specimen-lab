# Progress log

*Last updated: 2026-10-21. This file records the project's state so that work can resume without repeating completed steps.*

## Current phase
Complete. Deployment is verified. What remains are team-only tasks (see Outstanding).

## Completed
| Milestone | Evidence |
|---|---|
| Official guidelines located and read (7 pages) | `docs/official_requirements.md`. The project-root copy was 0 bytes at first; the user restored an identical copy (same MD5 as `~/Downloads`) |
| Prior work reviewed and reused | `C:\Users\HP\ML_Casestudy\review1_*.ipynb`: the leakage audit and engineered-feature ideas were reused. The originals are untouched |
| Datasets audited, mapping justified | `reports/results/dataset_audit.{json,csv}`, `data/README.md` |
| Shared package | `src/` |
| Notebooks executed (0 error cells) | regression (≈ 95 s), classification (≈ 145 s), clustering (≈ 60 s) |
| Models and results | `models/`, `reports/results/`, `reports/model_comparison/summary.md` |
| Streamlit app | `app/`: 7 pages, checked with AppTest and manually in a browser (desktop and 375 px mobile); peak memory ≈ 305 MB |
| Tests | `python -m pytest`: 55 passed |
| Docs | README, methodology, viva questions (41), checklist, requirements traceability |
| CI | `.github/workflows/tests.yml` (runs once pushed) |
| Platform research | Streamlit Community Cloud chosen (free; 690 MB–2.7 GB; sleeps after 12 h). Hugging Face Docker Spaces now need a paid plan; Render free spins down after 15 min |

## Key results
- **Regression:** Polynomial Regression (degree 3), test R² 0.99866, RMSE 0.0201, MAE 0.0151; 5-fold CV R² 0.99829 ± 0.00035.
- **Classification:** 7 of 10 models perfect on the 1,625-row test set. The selected model is Logistic Regression (C = 100): F1 = ROC-AUC = 1.000, 0 poisonous predicted edible.
- **Clustering:** k = 4. K-Means silhouette 0.574 (all rows); Ward agglomerative 0.569 on a 10k sample; ARI between the two methods 0.979; bootstrap stability ARI 0.999.

## Defects found and fixed
1. A regression notebook cell transformed with only the `preprocess` step, skipping feature engineering. Fixed by using `pipeline[:-1]`.
2. scikit-learn 1.7.2 `KNeighborsClassifier(metric="manhattan", weights="uniform")` fails with string labels. Fixed at the root by encoding the target as 0/1.
3. The classification selection tie-breaker (CV log-loss) favoured 1-NN, whose probabilities are degenerate. It was replaced by a one-standard-deviation tie plus a simplicity rule, documented in the notebook.
4. Notebook outputs exposed local Windows paths (`sys.executable` and `WindowsPath` echoes). Removed, and all notebooks were re-executed.
5. App: a Streamlit numeric-type mismatch (int `step` with float bounds), and a wrong "actual score" comparison. Both fixed and covered by tests.

## Known issues
- On Windows, joblib's loky resource tracker prints harmless `KeyError` lines to the host terminal after parallel searches. They do not appear in notebook outputs.
- One classification re-run hung in the Gradient Boosting grid search for 30 minutes while a large `pip install` and the clustering notebook were running at the same time. The same notebook had completed in about 145 s three times before, so it was re-run on its own; see the latest entry below.

## Deployment (verified 2026-10-21)
- **Repository:** https://github.com/Carcus0013/ml-capstone-specimen-lab. Created by Claude in the built-in browser at the user's request and pushed from `main` with milestone history.
- **App:** https://ml-capstone-specimen-lab.streamlit.app on Streamlit Community Cloud, Python 3.13.16, installed from `app/requirements.txt`. The build log is clean apart from the expected "two requirements files" warning.
- **Public access:** an anonymous request completes Streamlit's cookie redirect and returns HTTP 200; `/_stcore/health` returns `ok`.
- **Live tests:**
  - All 7 pages render without errors.
  - Regression: test banana 1 is predicted at 2.71 against a measured 2.73.
  - Classification: the poisonous test mushroom is classified as poisonous.
  - Clustering: 48° N, 10° E, 50 kg, year 1850 is assigned to "Historic observed falls".
  - CSV upload: 2 rows assigned, 2 rejected with reasons.
- **Hosting limits:** the app sleeps after 12 hours without traffic, and any visitor can wake it. Memory is 690 MB to 2.7 GB, and the app's measured peak is about 305 MB locally. Pushes to `main` redeploy automatically.

## Outstanding (team only)
1. Rewrite every ✍️ DRAFT notebook cell in your own words (guideline 7.5), then re-run the notebook and commit.
2. Fill in the track owners in the README (guideline 7.4).

## Local-only helpers (not committed)
`.dev/` holds the notebook builder scripts. Rebuild with `python .dev/build_<track>.py` and execute with `python .dev/nbtools.py exec <track>`.

## Next action
Team: rewrite the DRAFT cells and add the track owners, then `git push` (the app redeploys automatically).
