# Data

The three datasets assigned by the instructor live in `data/raw/`. They are **read-only**:
no code in this project writes to them. Byte-for-byte copies of the originals delivered with
the assignment (identical MD5 checksums, verified when the folder was created) remain in the
project root and in the team's `ML_Casestudy` folder as backups.

| File | Rows × columns | Track | Target / use |
|---|---|---|---|
| `raw/banana_quality_dataset.csv` | 1,000 × 16 | Regression | `quality_score` (continuous) |
| `raw/mushrooms.csv` | 8,124 × 23 | Classification | `class` (e = edible, p = poisonous) |
| `raw/meteorite-landings.csv` | 45,716 × 10 | Clustering | no target; `fall` and `recclass` only for after-the-fact interpretation |

All numbers in this file come from `python -m src.audit_datasets`, which writes the full audit
to `reports/results/dataset_audit.json` (dataset level) and `reports/results/dataset_audit.csv`
(one row per column).

## Why each dataset was given to each track

The mapping was decided after inspecting the files, not from their names.

* **Banana → regression.** It is the only dataset with a continuous outcome that a regression
  model can sensibly predict (`quality_score`, 0.92–3.89, skew 0.01). Its label
  `quality_category` is just `quality_score` cut at 1.50 / 2.50 / 3.50, so it is excluded as
  target leakage.
* **Mushrooms → classification.** Every column is a one-letter category code, so there is no
  continuous outcome for regression. The binary `class` label is nearly balanced
  (4,208 edible / 3,916 poisonous; 51.8% / 48.2%).
* **Meteorites → clustering.** No column is a defensible supervised target. `mass` is
  continuous, but predicting it from year and location answers no clear question. `fall` is
  extremely imbalanced (1,107 Fell vs 44,609 Found), and `recclass` has 466 levels. The data
  does have clear structure in location, mass and year, which is what unsupervised clustering
  is for. `fall` and `recclass` are used **only after fitting**, to interpret the clusters.

## Dataset notes

### banana_quality_dataset.csv
* **Source:** the files supplied with the assignment do not document it. Uniform feature
  distributions with round-number bounds, plus an almost deterministic relationship with the
  target, suggest it is **synthetic**. Treat the results as an exercise in pipeline
  correctness, not as agronomy.
* **Columns:** `sample_id` (identifier), `variety` (8 levels), `region` (8 levels),
  `quality_score` (target), `quality_category` (binned target: leakage),
  `ripeness_index`, `ripeness_category` (binned ripeness_index), `sugar_content_brix`,
  `firmness_kgf`, `length_cm`, `weight_g`, `harvest_date` (2023-09-01 to 2023-10-31),
  `tree_age_years`, `altitude_m`, `rainfall_mm`, `soil_nitrogen_ppm`.
* **Quality:** 0 missing cells, 0 duplicate rows (also 0 when `sample_id` is ignored), no
  sentinel codes.
* **Excluded from the model:** `sample_id` (identifier), `quality_category` (leakage),
  `ripeness_category` (redundant copy of `ripeness_index`), and `harvest_date` (a single
  61-day season; correlation of day-of-season with the target is −0.013).

### mushrooms.csv
* **Source:** the UCI Machine Learning Repository *Mushroom* dataset (records drawn from *The
  Audubon Society Field Guide to North American Mushrooms*, 1981; donated in 1987). The
  attribute codes are expanded into readable labels in `src/config.py`
  (`MUSHROOM_CODE_LABELS`).
* **Quality:** 0 `NaN` cells, **but `stalk-root` uses `?` for 2,480 rows (30.53%)**. The
  missingness is informative: 70.97% of those rows are poisonous, against 38.20% when
  `stalk-root` is recorded. `?` is therefore kept as its own category, not imputed.
  `veil-type` is constant (always `p`) and is dropped. There are 0 duplicate rows and no
  identical feature vectors with conflicting labels.
* **Warning:** this is a historical teaching dataset. Do not use any model trained on it to
  decide whether a real mushroom is safe to eat.

### meteorite-landings.csv
* **Source:** NASA's *Meteorite Landings* open-data extract (compiled from The Meteoritical
  Society's Meteoritical Bulletin Database), as supplied with the assignment.
* **Columns:** `name`, `id` (identifiers), `nametype` (Valid / Relict), `recclass` (466
  classes), `mass` (grams), `fall` (Fell / Found), `year`, `reclat`, `reclong`, and
  `GeoLocation`, which repeats `reclat`/`reclong` as a string.
* **Quality problems found:** 131 missing masses, 288 missing years, 7,315 missing
  coordinates, **6,214 rows with the placeholder location (0, 0)**, impossible years
  (2101 and 2501 are in the future; 301 and 601 predate the oldest witnessed fall in the data,
  Nogata in 860), one longitude of 354.47° (Meridiani Planum, a meteorite found *on Mars*),
  and 19 masses of 0 g.
* **Cleaning rules** (applied in order by `src.preprocessing.clean_meteorites`):

  | Rule | Rows removed | Rows remaining |
  |---|---:|---:|
  | mass missing | 131 | 45,585 |
  | year missing | 274 | 45,311 |
  | coordinates missing | 7,195 | 38,116 |
  | placeholder coordinates (0, 0) | 6,186 | 31,930 |
  | year outside 860–2016 | 1 | 31,929 |
  | longitude outside [−180, 180] | 0 (already removed by an earlier rule) | 31,929 |
  | mass equal to 0 g | 18 | **31,911** |

* **Duplicates are kept.** 2,561 rows are identical once `name` and `id` are ignored. They are
  separate, named specimens recovered at the same recorded site (common for Antarctic
  expeditions, which log one coordinate per search area), not repeated records.
* After cleaning, **69.25% of meteorites come from Antarctica** (latitude < −60°). This
  sampling bias is a property of where meteorites are *searched for*, and it shapes the
  clusters.

## data/processed/

Reserved for derived files. The pipelines clean and transform data in memory, so nothing has
to be stored here to reproduce the results.
