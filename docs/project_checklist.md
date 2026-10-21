# Project checklist and compliance audit

**PASS** means implemented *and* verified by execution or inspection, with the evidence named.
**BLOCKED** means it needs an external action. **TEAM** means only the student team can do it.
The status here reflects the last verification run, recorded in `docs/progress.md`.

## Definition of done

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | Official guidelines inspected | PASS | 7-page PDF read in full; `docs/official_requirements.md` |
| 2 | Requirement traceability document | PASS | `docs/official_requirements.md` |
| 3 | All three assigned datasets inspected | PASS | `reports/results/dataset_audit.{json,csv}` |
| 4 | Dataset-to-track mapping justified | PASS | `data/README.md` § "Why each dataset was given to each track" |
| 5 | Original datasets preserved | PASS | `tests/test_data.py::test_raw_files_unchanged` (MD5); root originals untouched; `.gitattributes` disables EOL conversion |
| 6 | EDA completed | PASS | §6 of each notebook; `reports/figures/*/eda_*.png` |
| 7 | Data preprocessing implemented | PASS | `src/preprocessing.py`; notebook cleaning sections |
| 8 | Leakage-safe pipelines | PASS | `tests/test_preprocessing.py` (scaler / encoder fitted on train only) |
| 9 | All 10 regression algorithms | PASS | `reports/results/regression_results.csv` (10 rows); `tests/test_models.py` |
| 10 | All 10 classification algorithms | PASS | `reports/results/classification_results.csv` (10 rows) |
| 11 | Both clustering algorithms | PASS | `reports/results/clustering_results.csv` |
| 12 | All mandatory metrics | PASS | R²/RMSE/MAE; Accuracy/Precision/Recall/F1/ROC-AUC + confusion matrices; Silhouette/DB/CH, all checked by tests |
| 13 | All mandatory visualisations | PASS | `tests/test_models.py::test_required_figures_exist` |
| 14 | Hyperparameter tuning | PASS | `reports/results/hyperparameter_results.json` (9 regressors, 10 classifiers) |
| 15 | Cross-validation | PASS | `regression_cv.json` (top 2, 5-fold R²); `classification_cv.json` (stratified 5-fold); clustering bootstrap stability |
| 16 | Model comparison results saved | PASS | `reports/results/*.csv`, `reports/model_comparison/summary.md` |
| 17 | Selected pipelines saved | PASS | `models/*/selection.json`; reload reproduces test scores (tests) |
| 18 | Regression notebook completed and executed | PASS | 0 error cells, executed in order (`test_notebooks_fully_executed_without_errors`) |
| 19 | Classification notebook completed and executed | PASS | same test |
| 20 | Clustering notebook completed and executed | PASS | same test |
| 21 | README completed | PASS | `README.md`, including the verified live URL |
| 22 | requirements.txt | PASS | `requirements.txt` (pinned) and `app/requirements.txt`; fresh-venv install verified |
| 23 | Viva questions | PASS | `docs/viva_questions.md` (41 Q&A, dataset-specific) |
| 24 | Automated tests created and executed | PASS | 55 passed (`python -m pytest`) |
| 25 | Interactive GUI implemented | PASS | `app/` (7 pages) |
| 26 | Regression GUI tested | PASS | AppTest submits the form, and the prediction is within 0.1 of the held-out banana's measured score; checked manually in a browser |
| 27 | Classification GUI tested | PASS | AppTest predicts edible and poisonous test mushrooms correctly |
| 28 | Clustering interface tested | PASS | AppTest assigns a meteorite; CSV validation unit-tested; map checked in a browser |
| 29 | Model comparison interface tested | PASS | AppTest renders without exceptions; checked in a browser |
| 30 | Data explorer tested | PASS | AppTest renders without exceptions; checked in a browser |
| 31 | Deployment configuration completed | PASS | `app/requirements.txt`, `.streamlit/config.toml`, relative paths, startup check |
| 32 | Public deployment attempted | PASS | Deployed on Streamlit Community Cloud from `Carcus0013/ml-capstone-specimen-lab@main`; build log clean (Python 3.13.16) |
| 33 | Public URL verified | PASS | https://ml-capstone-specimen-lab.streamlit.app loads without signing in; all 7 pages render without errors |
| 34 | Deployed prediction workflows tested | PASS | Live checks: regression 2.71 vs measured 2.73; poisonous mushroom correctly classified; a meteorite assigned to a cluster; CSV upload with 2 assigned and 2 rejected |
| 35 | GitHub and deployment instructions documented | PASS | `README.md` § Deployment |
| 36 | Requirement-by-requirement audit | PASS | table below |
| 37 | Progress file updated | PASS | `docs/progress.md` |
| 38 | Remaining blockers reported | PASS | `docs/progress.md` § Outstanding |

## Official rubric audit

| Rubric item | Requirement | Status | Where |
|---|---|---|---|
| R1-A1 | Shape, dtypes, missing values, target distribution | PASS | §5 of every notebook |
| R1-A2 | Distribution plots for each feature, correlation heatmap, target distribution, ≥ 2 feature-target scatter plots | PASS | regression §6.1–6.5 (4 scatter plots); classification §6 (Cramér's V as the categorical analogue of a correlation heatmap) |
| R1-A3 | Written observation per major plot | PASS (**TEAM** must rewrite) | ✍️ DRAFT cells; guideline 7.5 forbids AI-written interpretation, so the team must rewrite them |
| R1-B1 | Missing values justified; duplicates and outliers checked | PASS | §7 of every notebook |
| R1-B2 | Encoding, scaler fitted on train only, stratified split | PASS | stratified for classification; regression's continuous target explained in its §8 |
| R1-B3 | ≥ 1 engineered feature with justification | PASS | `firmness_squared`, `stalk_ring_asymmetry`, unit-sphere coordinates and `log10_mass` |
| R1-C1 | 10 regressors, no errors | PASS | regression §11 |
| R1-C2 | Summary table of R², RMSE, MAE ranked by R² | PASS | regression §12 |
| R1-C3 | GridSearchCV / RandomizedSearchCV on ≥ 2 models, improvement reported | PASS | 9 models tuned; §13 table |
| R1-C4 | Residual plot, predicted vs actual, tree feature importance | PASS | regression §15 and §11.6–11.8 |
| R1-D1/D2 | Part A trained; Accuracy, weighted F1, confusion matrix, preliminary table | PASS | classification §11 and §11.6 |
| R2-A1 | Part B trained on the same data | PASS | classification §12 |
| R2-A2 | One table of all 10 classifiers × 5 metrics | PASS | classification §13 |
| R2-A3 | Best model justified, tuned, ≥ 1 metric improved | PASS | §16: Logistic Regression, test F1 0.99938 → 1.0, CV log-loss 0.0077 → 0.0006 |
| R2-B1 | Both clustering algorithms fitted | PASS | clustering §9–10 |
| R2-B2 | Silhouette, DB and CH for all; elbow; dendrogram | PASS | clustering §9, §10.3, §11 |
| R2-B3 | PCA scatter for every algorithm; t-SNE for ≥ 1 | PASS | clustering §12 (t-SNE for both) |
| R2-C1 | Clean notebooks, modular functions, comments | PASS | `src/` package; notebooks run top to bottom |
| R2-C2 | README with dataset description, problem, results table, setup, how to run | PASS | `README.md` |
| R2-C3 | Meaningful commit history; requirements.txt | PASS | milestone commits on https://github.com/Carcus0013/ml-capstone-specimen-lab |
| Bonus +1 | Working GUI | PASS | `app/`, tests and browser check |
| Bonus +1 | Public deployment | PASS | https://ml-capstone-specimen-lab.streamlit.app (verified) |
| 7.1 | `random_state=42`; scalers fitted on train only | PASS | `src/config.py`; tests |
| 7.2 | Consistent split; CV for the 2 best per track; summary tables | PASS | see items 15 and 16 |
| 7.3 | Titles, labels, legends, tight layout, colour-blind palettes | PASS | `src/visualization.py` (seaborn "colorblind"); app uses the Okabe-Ito palette |
| 7.4 | One owner per track | **TEAM** | README § Team and track ownership |
| 7.5 | Cite external code; AI use cited in README; own interpretation | PASS (disclosure) / **TEAM** (rewrite DRAFT cells) | README § 14 |
| 8 | Repository structure (README, requirements, data/, notebooks/, models/, app/) | PASS | `tests` check the layout through the file paths they use |
