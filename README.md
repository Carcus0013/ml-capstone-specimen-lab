# 23CSE301 Machine Learning Capstone: Banana, Mushroom and Meteorite

An end-to-end machine-learning project covering the three tracks required by the
23CSE301 capstone guidelines (Academic Year 2026-27). Each track has its own dataset:
**regression** on banana quality, **classification** of edible vs poisonous mushrooms, and
**clustering** of NASA's meteorite landings. Each track has a fully executed notebook,
a shared and tested Python package, saved model pipelines, and an interactive Streamlit app.

- **Live app:** https://ml-capstone-specimen-lab.streamlit.app (Streamlit Community Cloud; verified 2026-10-21, see [Deployment](#deployment))
- **Repository:** https://github.com/Carcus0013/ml-capstone-specimen-lab
- **Notebooks:** [`notebooks/regression.ipynb`](notebooks/regression.ipynb), [`notebooks/classification.ipynb`](notebooks/classification.ipynb), [`notebooks/clustering.ipynb`](notebooks/clustering.ipynb)

---

## 1. Problem statements

| Track | Question | Dataset | Target |
|---|---|---|---|
| Regression | How good is this banana? Predict its continuous quality score from its physical and growing measurements. | `banana_quality_dataset.csv` (1,000 × 16) | `quality_score` |
| Classification | Is this mushroom edible or poisonous, given 21 visible and olfactory attributes? | `mushrooms.csv` (8,124 × 23) | `class` (e/p) |
| Clustering | Without labels, do recorded meteorites form natural groups by location, mass and year? | `meteorite-landings.csv` (45,716 × 10) | none (unsupervised) |

## 2. Datasets and mapping

Full details, column dictionaries and cleaning rules are in [`data/README.md`](data/README.md). The audit
is in `reports/results/dataset_audit.json` and is regenerated with `python -m src.audit_datasets`.

- **Banana → regression.** It is the only dataset with a continuous outcome. `quality_category` is
  `quality_score` cut at 1.50 / 2.50 / 3.50, so it is excluded as **target leakage**. The source is not
  documented in the assignment files, and the data appears synthetic (uniform features, near-deterministic target).
- **Mushrooms → classification.** All 22 features are one-letter categorical codes, and the binary label is
  nearly balanced (51.8% edible / 48.2% poisonous). The source is the UCI Machine Learning Repository *Mushroom*
  dataset (1987), drawn from *The Audubon Society Field Guide to North American Mushrooms* (1981).
  `stalk-root` codes 2,480 missing values as `?`, and `veil-type` is constant.
- **Meteorites → clustering.** No column is a defensible supervised target. `fall` is 1,107 Fell vs 44,609
  Found, and `recclass` has 466 classes; both are used only **after** clustering, for interpretation. The source
  is NASA's *Meteorite Landings* open-data extract (The Meteoritical Society). After removing missing,
  placeholder (0, 0) and impossible records, **31,911** rows remain.

The original files in the project root are never modified. The canonical copies are in `data/raw/`, and
a test checks that their MD5 checksums still match the originals.

## 3. Algorithms implemented

| Regression (10) | Classification (10) | Clustering (2) |
|---|---|---|
| Linear, Ridge, Lasso, ElasticNet, Polynomial (PolynomialFeatures + LinearRegression), Decision Tree, Random Forest, Gradient Boosting, SVR, KNN | **Part A:** Logistic Regression, KNN, Gaussian NB, Decision Tree, SVC · **Part B:** Random Forest, AdaBoost, Gradient Boosting, Bagging (Decision Tree base), MLP | K-Means (elbow + 3 indices), Agglomerative (4 linkages compared, dendrogram) |

## 4. Methodology

- **Split first, fit later.** One 80:20 split per track (`random_state=42`), stratified for the mushrooms. Every model in a track is scored on the same held-out rows.
- **No leakage.** Imputers, one-hot encoders, scalers and polynomial expansion live inside scikit-learn `Pipeline` / `ColumnTransformer` objects, so they are fitted on training rows only, including inside every cross-validation fold. The tests check this directly by comparing scaler means and encoder levels with the training data.
- **Feature engineering** (row-wise, so leakage-free by construction; implemented in `src/feature_engineering.py`):
  - `firmness_squared` (banana): a training-set residual diagnostic showed an inverted-U effect of firmness. It raised 5-fold CV R² of a linear model from 0.9832 to 0.9958.
  - `stalk_ring_asymmetry` (mushroom): the number of stalk attributes that differ above vs below the ring. When both differ, 95.5% of training mushrooms are poisonous.
  - Unit-sphere coordinates `geo_x, geo_y, geo_z` and `log10_mass` (meteorite): longitude wrap-around is removed, and the 60-tonne mass skew is tamed.
- **Tuning.** `GridSearchCV` / `RandomizedSearchCV` with 5-fold CV on the training set for 9 of 10 regressors (Linear Regression has no hyperparameters) and all 10 classifiers. Baseline and tuned scores are both reported.
- **Cross-validation.** 5-fold CV R² for the two best regressors. Stratified 5-fold CV (F1, accuracy, ROC-AUC, log-loss) for every classifier tied at the top. For clustering, supervised CV does not apply, so stability is measured instead: bootstrap refits compared by adjusted Rand index.
- **Metrics.** Regression: R², RMSE, MAE. Classification: Accuracy, weighted Precision / Recall / F1, ROC-AUC, confusion matrices, and the count of poisonous mushrooms predicted edible. Clustering: Silhouette (higher is better), Davies-Bouldin (lower is better), Calinski-Harabasz (higher is better).

Longer explanations are in [`docs/methodology.md`](docs/methodology.md).

## 5. Results

All numbers below are copied from [`reports/model_comparison/summary.md`](reports/model_comparison/summary.md),
which `python -m src.summarize_results` generates from `reports/results/`.

### Regression: held-out test set (200 bananas), ranked by R²

| Rank | Model | Test R² | RMSE | MAE | 5-fold CV R² (train) |
|---:|---|---:|---:|---:|---:|
| 1 | **Polynomial Regression** (degree 3) | **0.9987** | **0.0201** | **0.0151** | 0.9983 |
| 2 | Linear Regression | 0.9959 | 0.0352 | 0.0261 | n/a |
| 3 | Ridge Regression | 0.9959 | 0.0352 | 0.0261 | 0.9958 |
| 4 | ElasticNet Regression | 0.9959 | 0.0353 | 0.0261 | 0.9959 |
| 5 | Lasso Regression | 0.9958 | 0.0356 | 0.0263 | 0.9959 |
| 6 | Support Vector Regressor | 0.9956 | 0.0364 | 0.0237 | 0.9954 |
| 7 | Gradient Boosting Regressor | 0.9936 | 0.0439 | 0.0346 | 0.9927 |
| 8 | Random Forest Regressor | 0.9581 | 0.1123 | 0.0913 | 0.9614 |
| 9 | Decision Tree Regressor | 0.8950 | 0.1778 | 0.1454 | 0.9085 |
| 10 | K-Nearest Neighbors Regressor | 0.8329 | 0.2243 | 0.1839 | 0.8349 |

### Classification: held-out test set (1,625 mushrooms)

| Rank | Model | Accuracy | Precision | Recall | F1 (weighted) | ROC-AUC | Poisonous → edible |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | **Logistic Regression** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0** |
| 2 | K-Nearest Neighbors | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |
| 3 | Support Vector Machine (SVC) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |
| 4 | Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |
| 5 | AdaBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |
| 6 | Bagging (Decision Tree) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |
| 7 | MLP Neural Network | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |
| 8 | Gradient Boosting | 0.9988 | 0.9988 | 0.9988 | 0.9988 | 0.9999 | 2 |
| 9 | Decision Tree | 0.9988 | 0.9988 | 0.9988 | 0.9988 | 0.9990 | 2 |
| 10 | Gaussian Naive Bayes | 0.9932 | 0.9933 | 0.9932 | 0.9932 | 0.9992 | 2 |

### Clustering: cleaned meteorite records, k = 4

| Algorithm | Evaluated on | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|---|---|---:|---:|---:|
| K-Means | all 31,911 rows | 0.5741 | 0.9236 | 18,424 |
| K-Means | 10,000-row sample | 0.5747 | 0.9200 | 5,522 |
| Agglomerative (Ward) | same 10,000-row sample | 0.5693 | 0.9673 | 5,213 |

The two algorithms agree closely (adjusted Rand index 0.979), and K-Means is stable under bootstrap refits
(ARI 0.999). The four K-Means clusters correspond to Antarctic search finds (71%, median mass 14 g),
Africa and Asia finds (19%), Americas finds (6%), and historic heavy meteorites (3%, median year 1887). In
that last group, 52.4% were seen falling, against 3.3% overall; the `fall` label was used only after fitting.

## 6. Selected models and why

| Track | Selected model | Selection rule |
|---|---|---|
| Regression | Polynomial Regression, degree 3 | Of the two best by test R², it has the higher mean 5-fold CV R² (0.99829 ± 0.00035 vs 0.99582 ± 0.00044), a gap of more than one standard deviation |
| Classification | Logistic Regression, C = 100 | Safety first (0 poisonous predicted edible) and best test F1. The CV F1 of all 7 safe, perfect models lies within one standard deviation, so the **simplest** (smallest) model was chosen. Tuning improved it from 1 poisonous-as-edible error to 0 and from CV log-loss 0.0077 to 0.0006 |
| Clustering | K-Means, k = 4 | Silhouette and Davies-Bouldin both favour k = 4 (Calinski-Harabasz favours 3). Supported by the elbow, the dendrogram gaps and the stability check. It is the only algorithm that can assign *new* meteorites |

## 7. Installation

Python 3.13 is recommended (the results were produced with 3.13.9; `scikit-learn` must be **1.7.2** to load the saved models).

```bash
git clone https://github.com/Carcus0013/ml-capstone-specimen-lab.git
cd ml-capstone-specimen-lab
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

No secrets or API keys are needed; `.env.example` documents this.

## 8. Running the notebooks

Each notebook runs top to bottom and regenerates its figures (`reports/figures/`), results (`reports/results/`)
and models (`models/`). Measured runtimes on a 16-thread laptop: regression ≈ 95 s, classification ≈ 145 s,
clustering ≈ 55 s.

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/regression.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/classification.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/clustering.ipynb
python -m src.audit_datasets
python -m src.summarize_results
```

## 9. Running the app

```bash
streamlit run app/app.py
```

Then open http://localhost:8501. The app has seven pages:
- **Overview:** the three questions and their selected models.
- **Banana quality:** a regression form and prediction, with all ten models compared.
- **Mushroom safety:** classification with probabilities and log-odds contributions.
- **Meteorite clusters:** a cluster map, assignment of a single meteorite, CSV upload, and the agglomerative clusters shown retrospectively.
- **Model comparison**, **Data explorer** and **About**.

The app only loads saved pipelines and never retrains anything. See [`app/README.md`](app/README.md).

## 10. Tests

```bash
python -m pytest
```

There are 55 tests covering:
- the raw-data checksums and dataset schemas,
- leakage (scaler and encoder fitted on training rows only),
- unseen categories and missing values,
- the saved models reproducing their recorded test scores,
- the notebooks having been executed top to bottom without error cells,
- the required figures,
- input and CSV validation,
- Streamlit smoke tests that submit each form and check the real prediction.

GitHub Actions runs the same suite on every push (`.github/workflows/tests.yml`).

## Deployment

The target is **Streamlit Community Cloud**, chosen after checking the current platform documentation:
- It is free, deploys directly from GitHub, provides 690 MB to 2.7 GB of memory, and puts an app to sleep after 12 hours without traffic. Any visitor can wake it with one click.
- Hugging Face Spaces was ruled out because new Docker Spaces on free CPU now require a paid plan, and its Streamlit SDK has been retired.
- Render's free tier would work, but it spins down after 15 minutes and takes about a minute to start again.

The app's measured peak memory locally is about 305 MB, and the saved models total 6.9 MB.

**Deployed at https://ml-capstone-specimen-lab.streamlit.app** (repository `Carcus0013/ml-capstone-specimen-lab`, branch `main`, entry point `app/app.py`, Python 3.13).

Verified on 2026-10-21:
- **Public access:** the URL loads without signing in.
- **Build log:** clean. Dependencies were installed from `app/requirements.txt` on Python 3.13.16. Streamlit's one warning is that the repository has two requirements files, which is intended.
- **All seven pages** render without errors.
- **Banana prediction:** test banana 1 is predicted at 2.71 against a measured 2.73, identical to the local result.
- **Mushroom classification:** the held-out poisonous test mushroom is classified as poisonous.
- **New meteorite:** 48° N, 10° E, 50 kg, year 1850 is assigned to "Historic observed falls".
- **CSV upload:** a 4-row file gave 2 rows assigned and 2 rejected, each with a reason.

To redeploy from scratch:
1. Push this repository to GitHub (public).
2. At https://share.streamlit.io, choose **Create app → Deploy a public app from GitHub**, pick the repository and the `main` branch, and set the main file path to `app/app.py`.
3. Under **Advanced settings**, choose **Python 3.13**. Community Cloud installs `app/requirements.txt`, because it looks next to the entrypoint first.
4. Deploy, wait for the build, then open the URL and test one prediction on each page.

## 11. Repository structure

```
├── README.md                 this file
├── requirements.txt          full environment (notebooks, tests, app)
├── data/raw/                 the three assigned datasets (read-only)
├── data/README.md            dataset documentation and cleaning rules
├── notebooks/                regression / classification / clustering (executed)
├── src/                      shared code: config, loading, audit, feature engineering,
│                             preprocessing, model specs, evaluation, plots
├── models/<track>/           saved .joblib pipelines + metadata, selection.json, feature_schema.json
├── reports/figures/<track>/  every figure produced by the notebooks
├── reports/results/          machine-readable metrics (CSV / JSON)
├── reports/model_comparison/ generated summary tables
├── app/                      Streamlit app (app.py, core/, views/, assets/, requirements.txt)
├── tests/                    pytest suite
├── docs/                     requirements traceability, methodology, viva questions, checklist, progress log
└── .github/workflows/        CI
```

## 12. Limitations

- The banana data appears synthetic, and the mushroom data is almost separable (odour alone gives about 98.5% accuracy). The near-perfect scores reflect the data more than real-world skill. **The mushroom model must never be used to judge real mushrooms.**
- The meteorite clusters mirror *where meteorites are searched for* (Antarctic ice, hot deserts), not where they fall. Feature weighting (three coordinates for location) is a modelling choice.
- Agglomerative clustering was fitted on a 10,000-row random sample, because the full distance matrix would need about 4.1 GB.
- Hyperparameters were tuned on the same CV folds used for the reported CV means, which makes those means slightly optimistic. The untouched test sets are the unbiased check.

## 13. Team and track ownership

Guideline 7.4 asks each member to own at least one track. *To be completed by the team:*

| Track | Owner |
|---|---|
| Regression | _name_ |
| Classification | _name_ |
| Clustering | _name_ |

## 14. Academic integrity and AI-assistance disclosure

This repository was produced with substantial help from an AI assistant, **Anthropic's Claude (Claude Code)**,
working on the team's assigned datasets, the official guidelines and the team's earlier Review 1 notebooks
(whose leakage audit and feature-engineering ideas were reused). The assistant wrote:
- the `src/` package and the notebook code,
- the Streamlit app and the tests,
- the documentation,
- first drafts of the notebook commentary.

It also ran the notebooks and tests. Every metric reported here comes from that execution, not from estimates.

Guideline 7.5 permits generative AI for **code scaffolding but not for analysis or interpretation**. For that
reason, every interpretive Markdown cell in the notebooks is marked **✍️ DRAFT – rewrite in your own words**.
The team must check each observation against the printed outputs and rewrite it before submission, and each
owner must be able to explain every line of their track in the viva. No code was copied from other external
sources; library usage follows the official scikit-learn, pandas, SciPy and Streamlit documentation.

## 15. References

- Schlimmer, J. (1987). *Mushroom* [Dataset]. UCI Machine Learning Repository.
- NASA Open Data Portal. *Meteorite Landings* (data from The Meteoritical Society's Meteoritical Bulletin Database).
- Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12, 2825–2830.
- Streamlit Community Cloud documentation, https://docs.streamlit.io/deploy/streamlit-community-cloud
