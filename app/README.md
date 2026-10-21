# Streamlit app

An interactive front end for the three trained models. It **only loads** the pipelines and result
files that the notebooks saved; it never retrains, and it shows no hand-typed scores.

## Run locally

From the repository root (so that `src/`, `models/` and `reports/` are found):

```bash
pip install -r app/requirements.txt     # or the full ../requirements.txt
streamlit run app/app.py
```

## Pages

| Page | What it does | Uses |
|---|---|---|
| Overview | The three questions, selected models, held-out scores, and how the work was done | `models/*/selection.json`, results CSVs, dataset audit |
| Banana quality | Validated form, then a prediction with the selected regression pipeline, the predictions of all ten models, and the position of the prediction in the score distribution | `models/regression/*.joblib`, `feature_schema.json` |
| Mushroom safety | Validated form, then predicted class, probabilities, log-odds contributions (logistic regression) and all ten models' answers | `models/classification/*.joblib` |
| Meteorite clusters | Cluster map and profiles, assignment of a new meteorite with K-Means, a validated CSV upload with downloadable results, and retrospective agglomerative clusters | `models/clustering/*` |
| Model comparison | Tables and charts for every algorithm, the effect of tuning, CV results and saved figures | `reports/results`, `reports/figures` |
| Data explorer | Audit summary, column statistics, interactive distributions and EDA figures | `data/raw`, `reports/results/dataset_audit.json` |
| About | Method, model provenance (versions, save time), AI disclosure and limitations | model metadata JSON |

## Structure

```
app/
├── app.py            entry point: page config, navigation, startup check for missing artefacts
├── core/
│   ├── logic.py      pure, unit-tested validation and prediction helpers (no Streamlit import)
│   ├── data.py       cached loading of models (st.cache_resource) and tables (st.cache_data)
│   ├── theme.py      colours, Plotly template, small HTML helpers
│   └── nav.py        page registry for cross-links
├── views/            one module per page, each with render()
├── assets/style.css  typography and layout
└── requirements.txt  runtime-only dependencies (used by Streamlit Community Cloud)
```

## Reliability measures

- **Startup check.** If any required model or result file is missing, the app shows which files are missing and stops; it never shows invented output.
- **Input validation.** Numbers must be finite and within the training range plus a 5% margin. Values outside the training range trigger an extrapolation warning. Categories must be levels seen in training.
- **CSV upload.** Files are limited to 5 MB and 5,000 rows. They are parsed only with `pandas.read_csv` and never executed. The schema and value ranges are checked row by row, and rejected rows are listed with a reason.
- **Unseen categories** are encoded as all zeros (`handle_unknown="ignore"`), and **missing numeric values** are imputed inside the pipeline.
- **Error display.** Raw stack traces are hidden from users (`client.showErrorDetails = "none"` in `.streamlit/config.toml`), and predictions are wrapped so that failures show a plain message.
- **Paths** are resolved relative to the files, so the app runs unchanged in the cloud. A test checks that no absolute paths appear in `app/` or `src/`.

## Measured footprint

- **Peak memory:** about 305 MB with every page visited and all 22 models cached (Windows, Python 3.13).
- **Model files:** 6.9 MB in total.
- **Agglomerative clustering:** has no `predict` method in scikit-learn, so new meteorites are assigned with K-Means only, and the agglomerative clusters are shown for the training sample.
