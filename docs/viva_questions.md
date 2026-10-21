# Viva preparation

Answers are short and technically precise, and refer to *this* project's measured results. Read each one
against the notebook output and put it in your own words; the examiner will expect you to explain, not recite.

---

## A. Dataset-specific decisions

**Q1. Why is `quality_score` the regression target, and why was `quality_category` dropped?**
`quality_score` is the only continuous outcome. Grouping it by `quality_category` gives contiguous, non-overlapping
ranges with exact cut points at 1.50, 2.50 and 3.50, so the category is the target in disguise. Using it as a
feature would be **target leakage**: near-perfect training scores that fail on any new banana whose category is unknown.

**Q2. Why was `ripeness_category` dropped when it is not leakage?**
It is a binned copy of `ripeness_index`, a legitimate predictor we already keep. It adds redundancy and nothing new.

**Q3. Firmness has r = −0.07 with quality. Why is it in the model, and squared?**
Pearson's r only measures *linear* association. The training-set residuals of a linear model form an inverted
parabola against firmness (about −0.09 at both ends, +0.09 in the middle), so the effect is non-linear. Adding
`firmness²` lets a linear model represent the peak, and 5-fold CV R² rose from 0.9832 to 0.9958.

**Q4. Why keep `?` in `stalk-root` instead of imputing it?**
The missingness is informative: 71.0% of `?` rows are poisonous, against 38.2% otherwise. Dropping those rows
would discard 30.5% of the data and bias the class balance. Imputing them would erase the signal. One-hot
encoding `?` as its own level keeps the information, and because it learns nothing from the data, it cannot leak.

**Q5. Why is the mushroom accuracy 100%? Is something wrong?**
The dataset is almost separable. Eight of nine odour levels belong entirely to one class, and odour alone gives
about 98.5% CV accuracy. There are no duplicate rows and no conflicting labels, and the split was made before
any fitting, so this is a property of the data, not a leak. Zero errors on 1,625 cases still allows an error
rate of up to about 0.2% (rule of three: 3/1625).

**Q6. Which classification error matters most, and how did you account for it?**
Predicting a poisonous mushroom as edible. We count it for every model ("poisonous predicted edible") and made
it the first criterion of the selection rule. Gradient Boosting, the tuned Decision Tree and Gaussian NB each make 2 such errors.

**Q7. Why is the meteorite data used for clustering rather than regression or classification?**
- `mass` is continuous, but predicting it from location and year answers no meaningful question.
- `fall` is 1,107 vs 44,609, extremely imbalanced, and `recclass` has 466 levels.
- The data does show clear unlabelled structure by location, mass and era, which is what clustering looks for.

**Q8. Why convert latitude/longitude to x, y, z?**
Longitude wraps around: −179.9° and +179.9° are neighbours on Earth but 359.8 units apart as numbers. On the
unit sphere, straight-line distance increases with true distance (the notebook's example is about 8 km), so
K-Means and Ward measure real proximity.

**Q9. Why log10 of mass?**
Mass ranges from 0.01 g to 60 tonnes, with skewness about 77. In Euclidean distance a few giants would dominate.
After log10 the skewness is about 0.66.

**Q10. What do the four meteorite clusters mean?**
- Antarctic search finds: 71%, median 14 g.
- Africa and Asia (hot-desert) finds: 19%.
- Americas finds: 6%.
- Historic heavy meteorites: 3%, median 6.4 kg, median year 1887, of which 52.4% were seen falling against 3.3% overall.

The `fall` label was hidden during fitting, so its concentration in one cluster is independent evidence that the
grouping is meaningful. Overall, the clusters describe where people search for meteorites.

## B. Algorithms

**Q11. Linear vs Ridge vs Lasso vs ElasticNet?**
- **OLS** minimises the residual sum of squares (RSS).
- **Ridge** adds α‖β‖², which shrinks coefficients smoothly and stabilises them under collinearity.
- **Lasso** adds α‖β‖₁, which can set coefficients exactly to zero (feature selection).
- **ElasticNet** mixes the two penalties through `l1_ratio`.

Here the features are almost uncorrelated (max |r| = 0.07), so regularisation barely helps: all four score R² ≈ 0.996.

**Q12. Why did Lasso and ElasticNet score R² ≈ −0.007 before tuning?**
The default α = 1.0 is so large, relative to a target with standard deviation 0.54, that every coefficient
becomes zero and the model predicts the training mean. CV chose α = 0.0003 / 0.0001 instead. As α grows,
the number of zero coefficients rises from 0 to 21 of 24.

**Q13. What is polynomial regression, and how did you stop the features exploding?**
It is linear regression on `PolynomialFeatures` (powers and products of the inputs). The expansion was applied
only to the 10 numeric columns, and the degree was capped at 3, giving 24, 79 and 299 columns for degrees 1 to 3.
CV chose degree 3. Its train R² (0.9995) is close to its test R² (0.9987), which suggests it is not over-fitting badly.

**Q14. How does a decision tree split, and why did trees do worse here?**
At each node it picks the split that most reduces impurity: variance for regression, Gini or entropy for
classification. Trees approximate smooth functions with steps, so on a smooth target they trail linear models.
The single regression tree reached R² 0.895, the forest 0.958, and boosting 0.994.

**Q15. Bagging vs Random Forest vs Boosting?**
- **Bagging** trains trees on bootstrap samples and averages them, which reduces variance.
- **Random Forest** also samples features at each split, which decorrelates the trees.
- **Boosting** (AdaBoost, Gradient Boosting) trains trees *sequentially*, each one correcting the previous errors, which reduces bias.

In this project, Bagging became perfect once it used 50% feature subsampling. That is the Random-Forest idea.

**Q16. SVM / SVR: what do C, kernel and epsilon do?**
- **C** trades margin width against training errors: a large C gives a narrow margin and fewer violations.
- **The kernel** sets the shape of the decision boundary: linear, or RBF for non-linear boundaries.
- **Epsilon (SVR)** is the width of the tube inside which errors are ignored.

The SVR tuned itself to a *linear* kernel (0.963 → 0.9956), which is evidence of a near-linear relationship.

**Q17. Why must features be scaled for KNN and SVM?**
Both depend on distances. Unscaled, `rainfall_mm` (SD ≈ 564) dominates `firmness_kgf` (SD ≈ 1.3). Measured
result: KNN's CV R² is 0.835 with scaling and **−0.200** without.

**Q18. Logistic regression: what is an odds ratio?**
The model is linear in the log-odds, log(p/(1−p)) = b₀ + Σbⱼxⱼ. exp(bⱼ) multiplies the odds when feature j goes
up by 1, here when a one-hot level is present. Foul odour (odds ×≈604) and green spore print push towards
poisonous; no odour pushes strongly towards edible.

**Q19. What is the Naive Bayes assumption, and is it true here?**
Features are assumed independent *given the class*, so P(x|y) = Π P(xⱼ|y). It is violated here: Cramér's V is 0.49
between the stalk-surface pair and 0.63 between the stalk-colour pair. Gaussian NB also models 0/1 columns as
Gaussians, which explains why `var_smoothing` mattered so much (F1 0.970 → 0.993).

**Q20. KNN distance metrics for one-hot data?**
For pure 0/1 vectors, Manhattan distance equals 2 × (number of differing attributes), and Euclidean distance is the
square root of that same count, so both rank neighbours the same way. Small differences come from the one numeric
engineered feature and from distance weighting.

**Q21. MLP: what was tuned, and how do you know it converged?**
We tuned `hidden_layer_sizes` and `activation` (plus `alpha`). The final 64→32 ReLU network stopped after
23 of 1,000 epochs, with no `ConvergenceWarning` and a smoothly falling loss curve.

**Q22. K-Means: objective and assumptions?**
It minimises inertia, the sum of squared distances to the nearest centroid, by alternating two steps: assign each
point, then update each centroid. It assumes compact, roughly spherical clusters of similar spread, it needs k in
advance, and it is sensitive to scaling and initialisation (hence `n_init=10`).

**Q23. Agglomerative clustering: what does linkage mean?**
Linkage is how the distance between two clusters is defined:
- **single:** the closest pair of points;
- **complete:** the farthest pair;
- **average:** the mean over all pairs;
- **Ward:** the increase in within-cluster variance caused by merging.

Here single, complete and average linkage isolated a few outliers (for example sizes 9,997 / 1 / 1 / 1), and only
Ward gave balanced clusters.

**Q24. Why can't agglomerative clustering predict new points?**
It builds a hierarchy over the given points and defines no rule for where a new point belongs, so scikit-learn
provides no `predict`. K-Means has centroids, so a new point goes to the nearest one. The app therefore assigns
new meteorites with K-Means only.

## C. Evaluation, tuning and validation

**Q25. R², RMSE, MAE?**
- **R²** = 1 − SS_res / SS_tot: the share of variance explained (it can be negative).
- **RMSE** = √(mean squared error): in target units, and it penalises large errors more.
- **MAE** = mean absolute error: in target units, and more robust to outliers.

Selected model: R² 0.9987, RMSE 0.020, MAE 0.015 on a 0.9–3.9 scale.

**Q26. Precision, recall, F1, weighted average?**
- **Precision** = TP / (TP + FP); **recall** = TP / (TP + FN); **F1** is their harmonic mean.
- **"Weighted"** averages the per-class scores weighted by class support.

**Q27. ROC-AUC, binary vs multiclass?**
AUC is the probability that a random positive gets a higher score than a random negative. It is computed from
probabilities or decision scores, not from hard labels. Ours is binary, using P(poisonous). For multiclass you
would use One-vs-Rest: compute the AUC per class against the rest, then average.

**Q28. Silhouette, Davies-Bouldin, Calinski-Harabasz: which direction is better?**
- **Silhouette** ranges from −1 to 1, and higher is better.
- **Davies-Bouldin** is 0 or more, and lower is better.
- **Calinski-Harabasz** is 0 or more, and higher is better. It grows with n, so it can only be compared on the same rows.

**Q29. Why cross-validation, and why only on training data?**
One test split gives one noisy estimate. k-fold CV averages over k different validation folds. Keeping the
test set out of all decisions means its score stays an unbiased final check. The preprocessing is refitted
inside each fold because it is part of the pipeline.

**Q30. Is the CV estimate biased in any way?**
Slightly optimistic, yes: the hyperparameters were chosen using the same folds. Nested CV would remove that bias.
The untouched test score is the unbiased check, and it agrees (for example 0.99829 CV vs 0.99866 test).

**Q31. GridSearchCV vs RandomizedSearchCV?**
Grid search tries every combination. Randomised search samples a fixed number of them, which is cheaper for large
grids. We used randomised search for the Random Forest and Gradient Boosting regressors.

**Q32. Did tuning always help?**
No. The tuned decision tree classifier got *worse* on the test set (1.000 → 0.9988, with 2 dangerous errors).
Several depths tied in CV, `GridSearchCV` breaks ties by grid order, and the shallower tree it picked happened to
miss two test mushrooms. Tuning optimises the CV estimate, not the test score.

**Q33. How did you choose k?**
Each of the three indices was computed for k = 2 to 10. Silhouette (0.575) and Davies-Bouldin (0.924) both pick
k = 4; Calinski-Harabasz picks 3; the elbow bends at 3–4; and silhouette collapses at k = 5 (0.327). The dendrogram's
largest merge gaps also give four branches, and bootstrap refits reproduce the partition (ARI 0.999).

**Q34. What are PCA and t-SNE, and what can't they show?**
- **PCA** is a linear projection onto the directions of maximum variance. Here two components keep 69.2% of the variance.
- **t-SNE** is a non-linear embedding that preserves local neighbourhoods. Distances between clusters and cluster sizes in a t-SNE plot are not meaningful.

Both are 2-D pictures of 5-D data, so overlap in the plot does not prove overlap in the data.

## D. Concepts

**Q35. Bias vs variance, over- vs under-fitting?**
- **Bias** is error from overly simple assumptions (under-fitting). **Variance** is sensitivity to the particular training sample (over-fitting).
- Evidence from this project: KNN regression scores R² 1.000 on its training rows but 0.833 on the test set (high variance), while the linear models have train ≈ test ≈ 0.996 (low variance).

**Q36. How do you select a model when several are tied?**
Prefer the one with better training-only evidence (CV). If the CV difference is within one standard deviation,
prefer the simpler model. That rule chose Logistic Regression (0.006 MB, interpretable) over 1-NN (5.9 MB, which
memorises the training data).

**Q37. What is data leakage, and give two examples you prevented.**
Leakage is information from outside the training data reaching the model during training. Two examples: (1) target
leakage through `quality_category`; (2) preprocessing leakage, prevented by fitting scalers and encoders inside
the pipeline on training rows only, which a unit test checks.

## E. Engineering and deployment

**Q38. How does the app avoid inconsistent preprocessing?**
It loads the **whole fitted pipeline** (feature engineering, imputation, encoding, scaling and model) from
`models/`. The same object was evaluated in the notebook, so there is only one implementation.

**Q39. How are models saved, and what are the risks?**
`joblib.dump(pipeline)` is used, with a JSON file recording library versions. Pickle files run code when they are
loaded, so only the project's own files are ever loaded, and uploaded files are parsed as CSV data only. The
scikit-learn version must match (1.7.2), which is why it is pinned.

**Q40. What is the deployment architecture?**
- A GitHub repository connects to Streamlit Community Cloud, which installs `app/requirements.txt` and runs `app/app.py`.
- The models and result files ship inside the repository (about 7 MB), so the app needs nothing from a personal computer.
- Models are cached once per process. Peak memory is about 305 MB, against Community Cloud's 690 MB minimum.
- The app sleeps after 12 hours without traffic; any visitor can wake it.

**Q41. How is user input validated?**
- **Regression inputs** must be numeric and within the training range plus a 5% margin. Values just outside the training range trigger an extrapolation warning.
- **Categories** must be levels seen in training.
- **CSV uploads** are checked for required columns, a maximum of 5,000 rows and 5 MB, numeric values, latitude/longitude ranges, mass > 0, a valid year and no (0, 0) placeholder. Invalid rows are reported with a reason instead of crashing the app.
