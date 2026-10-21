# Methodology

This document explains *why* the pipeline is built the way it is. The *what* (code) is in `src/` and the
notebooks, and the measured results are in `reports/`.

## 1. Workflow per track

```
raw CSV (data/raw, read-only)
  → audit (shape, dtypes, missing, sentinels, duplicates, constants, identifiers)
  → target / leakage decision
  → EDA with written observations
  → cleaning rules (logged)
  → split 80:20 (stratified for classification)          ← nothing is fitted before this
  → Pipeline[ feature engineering → ColumnTransformer(impute, encode, scale) → model ]
  → baseline fit → GridSearch/RandomizedSearch on 5-fold CV of TRAIN → test-set scoring
  → comparison table, CV of the best models, selection rule, diagnostics
  → joblib pipelines + JSON metadata → Streamlit app
```
Clustering follows the same shape, but it has no split and no labels during fitting.

## 2. Leakage prevention

| Risk | Where it could happen | How it is prevented |
|---|---|---|
| Target leakage | `quality_category` is the binned target | Excluded; evidence in notebook §5.1 and `tests/test_data.py` |
| Preprocessing leakage | Scaler/encoder fitted on all rows | Every learned step sits inside a `Pipeline`; tests compare the scaler mean and encoder levels with the training rows |
| Tuning leakage | Picking hyperparameters by test score | Tuning uses only CV on the training set; the test set scores the finished models once |
| Feature-engineering leakage | Features computed from cross-row statistics | All engineered features are row-wise arithmetic, and the residual diagnostic used training rows only |
| Label leakage in clustering | Using `fall`/`recclass` to form clusters | They are excluded from the features and used only after fitting |

## 3. Preprocessing decisions

**Missing values.**
- Banana: no missing values.
- Mushrooms: `?` in `stalk-root` is kept as a category, because its missingness is predictive (71.0% poisonous vs 38.2%).
- Meteorites: rows missing mass, year or location are removed. These are the clustering features themselves, so imputing them would create artificial points.
- Median / most-frequent imputers are still placed in the supervised pipelines as safeguards against incomplete inputs in the app. They are fitted on training rows only.

**Duplicates.**
- Banana and mushrooms have none.
- Meteorites: 2,561 rows are identical once name and id are ignored. They are distinct specimens recovered at the same logged site, so they are kept.

**Outliers.**
- Banana: one low target value (0.92) is retained. It lies inside the dataset's `Unripe` band, and its inputs explain it.
- Meteorites: extreme masses are real, so they are log-transformed rather than deleted. Impossible values (year 2101 or 2501, the 0/0 placeholder, a longitude of 354°, 0 g) are removed.

**Encoding.**
- Nominal codes are one-hot encoded with `handle_unknown="ignore"`, so an unseen level becomes all zeros instead of crashing.
- For regression, `drop="first"` avoids the exact collinearity of a full dummy set in least squares.

**Scaling.** `StandardScaler` is applied inside the pipelines. KNN, SVM, MLP and the penalised linear models depend on distances or coefficient sizes. Tree models are invariant to monotone scaling, so for them scaling is harmless.

**Constant and identifier columns.** `veil-type` (constant), `sample_id`, `id` and `name` (identifiers) and `GeoLocation` (a duplicate of two columns) are dropped.

## 4. Evaluation

- **Same split for every model in a track.** `src/preprocessing.train_test` (80:20, `random_state=42`).
- **Ranking.** Regression is ranked by test R², as rubric C2 requires. Classification is ranked by weighted F1, then ROC-AUC, then the number of poisonous mushrooms predicted edible.
- **Selection.** Selection is separate from ranking and uses training-only evidence (CV). When the CV difference is smaller than one standard deviation, the simpler model is preferred. This same rule is used in both supervised tracks.
- **Classification safety metric.** Calling a poisonous mushroom edible is the dangerous error, so it is counted separately for every model.
- **Clustering.** Internal indices are computed for k = 2 to 10. Linkages are compared, and cluster sizes are checked as well, because single linkage scores the best silhouette while producing a useless 9,997 / 1 / 1 / 1 split. Stability is measured with bootstrap adjusted Rand index. External labels are used only after fitting.

## 5. Why some results look "too good"

- **Banana:** the features are uniform within round-number bounds, and the target is an almost deterministic function of four inputs. The data is very likely synthetic. Ten models reaching R² ≥ 0.83, and the linear family reaching 0.996, reflects this.
- **Mushrooms:** odour alone gives about 98.5% accuracy. Seven of ten models are perfect on the test set; the "rule of three" says zero errors in 1,625 cases is still consistent with an error rate of up to about 0.2%.
- **Meteorites:** the clusters reflect *collection effort* (Antarctic expeditions, desert collecting) rather than anything physical about where meteorites fall.

## 6. Computational choices

- Random Forest and Gradient Boosting regressors use `RandomizedSearchCV` (20 and 25 configurations) instead of exhaustive grids. All other models use full grids.
- Agglomerative clustering uses a documented random sample of 10,000 rows. The full pairwise-distance matrix would need about 4.1 GB. K-Means is evaluated on the same sample as well, for a like-for-like comparison.
- t-SNE runs on 4,000 points; PCA runs on all rows.
- Measured notebook runtimes (16 threads): about 95 s, 145 s and 55 s. The app's peak memory is about 305 MB.

## 7. Model persistence and the app

Each saved `.joblib` file is a **complete pipeline**: feature engineering → imputation → encoding → scaling → model.
Next to each one is a JSON file with its test metrics, tuned parameters and library versions. `selection.json` names
the selected model, and `feature_schema.json` holds the training ranges and levels that the app uses to validate inputs.
The app therefore sends raw inputs and gets exactly the training-time transformation. There is one
preprocessing implementation, used by the notebooks, the saved models and the app.

Ordinary `AgglomerativeClustering` has no `predict` method. The app shows its clusters only retrospectively and
assigns new meteorites with K-Means alone.
