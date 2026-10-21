# Official requirements traceability

**Source document:** `23CSE301_ML_26_27_Capstone_Guidelines.pdf`, 7 pages, *Machine Learning –
Capstone Project: Guidelines, Algorithm List & Evaluation Rubrics*, Academic Year 2026-27.

The copy in the project root was 0 bytes when work started. The full 184,017-byte file was
found in `C:\Users\HP\Downloads\` and `C:\Users\HP\ML_Casestudy\` and read from there. The user
then restored it to the project root, and its MD5 checksum (`2b3a9688…1333`) matches the
Downloads copy. The PDF is not committed to the public repository (it is listed in
`.gitignore`) because it is the instructor's document. Page references below are to that PDF.

Status values: **PASS** means implemented *and* verified by execution; **BLOCKED** means it
cannot be completed without an external action; **TEAM** means only the student team can do it.
The final status of each item is recorded in [`project_checklist.md`](project_checklist.md).

## 1. Overview and tooling (p.1)

| ID | Requirement | Implementation | Verification |
|---|---|---|---|
| G1 | End-to-end pipeline for Regression, Classification and Clustering | `notebooks/`, `src/` | All three notebooks execute top to bottom |
| G2 | Python 3, scikit-learn, Pandas, NumPy, Matplotlib, Seaborn | `requirements.txt` | Imports in notebooks; pinned versions |
| G3 | Submission: Jupyter notebook(s) + GitHub repository | `notebooks/*.ipynb`, git history | Executed notebooks committed and pushed to https://github.com/Carcus0013/ml-capstone-specimen-lab |
| G4 | Team of 3; each member owns ≥ 1 track (7.4) | — | **TEAM**: assign owners in README |

## 2. Regression track (p.1–2, §3.1; rubric §5 C1–C4)

| ID | Requirement | Implementation | Verification |
|---|---|---|---|
| R1 | 10 algorithms trained on the same preprocessed data, same held-out test set | `src/model_training.regression_specs`, `notebooks/regression.ipynb` §6 | 10 rows in `reports/results/regression_results.csv` |
| R2 | Linear: interpret coefficients | regression notebook §6.1 | coefficient table printed |
| R3 | Ridge: tune alpha | grid `model__alpha` | best alpha in `hyperparameter_results.json` |
| R4 | Lasso: observe sparsity | grid + zero-coefficient count | notebook output |
| R5 | ElasticNet: tune l1_ratio | grid alpha × l1_ratio | best params saved |
| R6 | Polynomial: PolynomialFeatures + LinearRegression; compare degrees | degree ∈ {1,2,3} | degree comparison table |
| R7 | Decision Tree: tune max_depth; feature importance | grid + importance plot | figure saved |
| R8 | Random Forest: tune n_estimators | RandomizedSearchCV | best params saved |
| R9 | Gradient Boosting: tune learning_rate | RandomizedSearchCV | best params saved |
| R10 | SVR: scale first; tune C and kernel | StandardScaler in pipeline; grid | best params saved |
| R11 | KNN: tune k; discuss scaling | grid + scaled vs unscaled experiment | notebook output |
| R12 | Metrics R², RMSE, MAE for every model | `src/evaluation.regression_metrics` | results CSV columns |
| R13 | 5-fold CV R² for the two best models | `src/model_training.cross_validate` | `regression_cv.json` |
| R14 (C2) | Single summary table ranked by R² | regression notebook §7 | DataFrame displayed |
| R15 (C3) | GridSearchCV/RandomizedSearchCV on ≥ 2 models; best params and improvement reported | all tunable models searched | baseline vs tuned table |
| R16 (C4) | Residual + predicted-vs-actual for best model; feature importance for a tree model | `src/visualization` | PNGs in `reports/figures/regression/` |

## 3. Classification track (p.2, §3.2; rubric §5 D1–D2, §6 A1–A3)

| ID | Requirement | Implementation | Verification |
|---|---|---|---|
| C1 | Part A: Logistic Regression, KNN, Gaussian NB, Decision Tree, SVC | `classification_specs()[:5]` | results CSV |
| C2 | Part B: Random Forest, AdaBoost, Gradient Boosting, Bagging (DT base), MLP | `classification_specs()[5:]` | results CSV |
| C3 | Logistic: coefficients / odds ratios | notebook §6.1 | odds-ratio table |
| C4 | KNN: tune k; discuss distance metrics | grid over k, weights, metric | notebook output |
| C5 | Gaussian NB: conditional independence discussion | notebook §6.3 | markdown + evidence |
| C6 | Decision Tree: tune max_depth; visualise tree | grid + `plot_tree` | figure saved |
| C7 | SVC: tune C and kernel; scale | grid; scaler in pipeline | best params saved |
| C8 | Random Forest: feature importance; tune n_estimators | grid + importance plot | figure saved |
| C9 | AdaBoost: tune n_estimators and learning_rate | grid | best params saved |
| C10 | Gradient Boosting (sklearn GBM) | grid | best params saved |
| C11 | Bagging with Decision Tree base estimator | `BaggingClassifier(estimator=DecisionTreeClassifier())` | notebook output |
| C12 | MLP: tune hidden_layer_sizes and activation | grid | best params saved |
| C13 | Accuracy, Precision, Recall, weighted F1, confusion matrix, ROC-AUC (OvR if multiclass) | `src/evaluation.classification_metrics` | results CSV + confusion-matrix figure |
| C14 (A2) | Single consolidated table of all 10 algorithms | notebook §8 | `classification_results.csv` |
| C15 (A3) | Best model justified; tuning applied; improvement in ≥ 1 metric documented | notebook §9–10 | tuning table |
| C16 (B2) | Stratified train/test split | `train_test(..., stratify=True)` | class proportions printed |

## 4. Clustering track (p.2–3, §3.3; rubric §6 B1–B3)

| ID | Requirement | Implementation | Verification |
|---|---|---|---|
| K1 | K-Means with an elbow curve (inertia vs k) | clustering notebook §5 | `elbow_curve.png` |
| K2 | Agglomerative clustering with a dendrogram; compare linkages | notebook §6 | `dendrogram.png`, linkage table |
| K3 | Silhouette, Davies-Bouldin, Calinski-Harabasz for all algorithms | `src/evaluation.clustering_metrics` | `clustering_results.csv` |
| K4 | PCA 2-D scatter coloured by cluster for every algorithm (mandatory) | notebook §7 | two PCA figures |
| K5 | t-SNE for at least one algorithm (encouraged; rubric B3 requires it) | notebook §7 | t-SNE figure |
| K6 | Ground-truth labels only for after-the-fact interpretation | `fall`/`recclass` cross-tabs after fitting | notebook §8 |

## 5. Deliverables (p.3, §4) and repository structure (p.7, §8)

| ID | Requirement | Implementation | Verification |
|---|---|---|---|
| D1 | Clean, fully run notebooks with visible outputs | `notebooks/` | executed by `nbclient`; error-cell count = 0 |
| D2 | GitHub repo with meaningful commit history (not a single bulk upload) | milestone commits | `git log` |
| D3 | Comparative results tables | `reports/results/*.csv` | files exist |
| D4 | All required visualisations | `reports/figures/` | test checks files exist |
| D5 | README: dataset description, problem statement, results table, setup, how to run | `README.md` | review |
| D6 | Consolidated 10-algorithm classification comparison | `classification_results.csv` | 10 rows |
| D7 | Elbow, dendrogram, PCA plots | `reports/figures/clustering/` | files exist |
| D8 | GUI / deployed application (bonus) | `app/` | local test + public URL |
| S1 | Root has README.md, requirements.txt, data/, notebooks/, models/ (.pkl/.joblib via joblib), app/ | repository layout | `tests/test_project_structure.py` |

## 6. General guidelines (p.6–7, §7)

| ID | Requirement | Implementation | Verification |
|---|---|---|---|
| P1 | One notebook per track | 3 notebooks | files exist |
| P2 | Markdown cells explain each major code block | notebooks | review |
| P3 | `random_state=42` wherever applicable | `src/config.RANDOM_STATE` | grep + review |
| P4 | Scalers/encoders fitted on the training set only | sklearn `Pipeline` + `ColumnTransformer` | `tests/test_preprocessing.py` leakage test |
| P5 | Consistent 80:20 split across all algorithms in a track | `src/preprocessing.train_test` | identical test indices |
| P6 | Cross-validation for at least the two best models **in each track** | regression: 5-fold CV R²; classification: stratified 5-fold CV; clustering: see note | result JSON files |
| P7 | Results in summary tables, not scattered prints | DataFrames in notebooks | review |
| P8 | Every plot: title, labelled axes, legend where applicable; tight layout | `src/visualization` | review |
| P9 | Colour-blind-friendly palettes | seaborn `colorblind` | review |
| P10 | Cite code taken from external sources | README "References" | review |
| P11 | Analysis, interpretation and feature-engineering decisions must be the team's own | see conflict below | **TEAM** |
| P12 | If generative AI is used, cite it in the README | README "AI-assistance disclosure" | review |
| P13 | Commit at every meaningful milestone | git history | `git log` |

## 7. Bonus (p.6)

| ID | Requirement | Implementation | Verification |
|---|---|---|---|
| B1 | +1 interactive GUI that accepts user inputs and returns predictions from the trained model | Streamlit app in `app/` | `tests/test_app.py` (AppTest) + manual browser test |
| B2 | +1 public deployment via a public URL | Streamlit Community Cloud | https://ml-capstone-specimen-lab.streamlit.app, verified on the live site |

## 8. Conflicts between the task prompt and the official guidelines

1. **AI-written analysis vs guideline 7.5.** The task prompt asks for interpretation of results.
   The guidelines say generative AI "may be used for code scaffolding but *not* for analysis
   or interpretation", and that each team's analysis must be original. **The official rule
   wins.** Every interpretive Markdown cell in the notebooks is therefore marked
   `✍️ DRAFT – rewrite in your own words`. Those cells contain only observations taken
   directly from computed outputs, so the team can check each claim against the numbers
   and then rewrite it themselves before submission. The README discloses the AI assistance.
2. **Stratified split for regression (rubric B2).** Stratification is defined for class
   labels. Classification uses `stratify=y`. Regression uses a plain random 80:20 split with
   `random_state=42`. The notebook shows that the train and test target distributions match.
   Stratifying on binned `quality_score` would be possible, but it is not required by the
   track definition.
3. **"Cross-validation for the two best models in each track" for clustering.** Supervised
   cross-validation does not apply to unsupervised clustering, because there is no target to
   score held-out folds against. Instead, the clustering notebook checks *stability*: it
   refits K-Means on bootstrap resamples and measures agreement with the Adjusted Rand Index.
   It also reports the internal indices required by the rubric.
4. **Review split.** The guidelines evaluate Regression and Classification Part A in Review 1,
   and Classification Part B and Clustering in Review 2. This repository holds the *final*
   (Review 2) state, with all ten classifiers in one notebook, as D6 requires.
