# Ames House Prices — Predictive Analysis
### Applied Quantitative Studies · Week 3: Multiple Linear Regression (OLS)

---

## Overview

This project applies the Week 3 lecture methodology to the **Ames, Iowa housing dataset** (1,460 transactions, 79 features). The goal is to predict **SalePrice** (USD) using Ordinary Least Squares regression, following a strict five-part structure: descriptive setup → problem framing → model building → honest evaluation → final prediction.

**Hard constraints (methodological discipline):**
- Only `pandas`, `numpy`, `matplotlib`, `scikit-learn` — no deep learning, no tree models, no regularized regression
- Model: `sklearn.linear_model.LinearRegression` (pure OLS)
- Evaluation: 5-fold KFold cross-validation (`random_state=42`)
- Metrics: MAE, RMSE, Adjusted R²
- All preprocessing (imputation, encoding) runs **inside** each CV fold via `Pipeline` + `ColumnTransformer` — no data leakage between folds

**Entry point:** `ames_regression_analysis.py`  
**Data:** `train.csv`, `data_description.txt`

---

## Repository Structure

```
.
├── ames_regression_analysis.py   # Main analysis script (single file, ~700 lines)
├── train.csv                     # Ames housing data (1,460 rows x 81 columns)
├── data_description.txt          # Variable codebook
├── 01_target_distribution.png    # SalePrice histogram
├── 02_top_predictor_scatter.png  # OverallQual vs SalePrice
├── 03_residual_plot.png          # Residuals vs predicted (final model)
├── 04_predicted_vs_actual.png    # Predicted vs actual with 45-degree line
├── 05_model_comparison.png       # CV MAE bar chart across M0-M5
└── 06_model_comparison_r2.png    # CV Adjusted R² bar chart across M0-M5
```

---

## Part 1 — Descriptive Setup

**Target variable:** `SalePrice` (USD), continuous

| Statistic | Value |
|-----------|-------|
| Mean | $180,921 |
| Median | $163,000 |
| Std Dev | $79,443 |
| Min | $34,900 |
| Max | $755,000 |
| Skewness | 1.883 |

**Distribution diagnosis:** Skewness of 1.88 signals a strongly right-skewed distribution. A long upper tail of luxury properties (max $755,000) pulls the mean $17,921 above the median. This mirrors any salary/income distribution: a small number of extreme values inflate the arithmetic mean away from the typical case.

**Benchmark implication (slide 9 logic):**
- MAE benchmark → predict the **median** ($163,000). The median is robust to outliers; it minimises expected absolute error under skew.
- RMSE benchmark → predict the **mean** ($180,921). The mean minimises expected squared error in expectation.

**Plot generated:** `01_target_distribution.png`

---

## Part 2 — Framing the Prediction Problem

| Item | Value |
|------|-------|
| Prediction target | `SalePrice` (USD) — continuous outcome |
| Unit of observation | One residential property transaction in Ames, Iowa |
| Features available | 79 descriptors (physical attributes + transaction metadata) |

### Leakage Assessment

Most Ames features describe the physical house and are available before a sale closes. However, four variables describe the **transaction itself** and are logically contemporaneous with the sale price:

| Variable | Description | Risk |
|----------|-------------|------|
| `MoSold` | Month of sale | Market-timing information not available at listing |
| `YrSold` | Year of sale | Market-timing information not available at listing |
| `SaleType` | Type of deed (conventional, court officer, new construction…) | Not known until transaction completes |
| `SaleCondition` | Transaction circumstance (normal, foreclosure, family sale…) | Not known until transaction completes |

**Decision:** Flagged but not excluded. These variables are excluded from the primary feature sets (M1–M5). In a real listing-time model they would be hard exclusions.

### Naive Benchmark (M0)

| Metric | Value |
|--------|-------|
| Benchmark MAE (predict median $163,000) | $55,534 |
| Benchmark RMSE (predict mean $180,921) | $79,415 |

Any model that cannot beat $55,534 MAE is no better than saying "every house costs $163,000."

**Plot generated:** `02_top_predictor_scatter.png`

---

## Part 3 — Candidate Models

Six models are built in a strict simple → complex progression. Every feature added has a one-sentence theoretical justification. The pipeline structure ensures all imputation and encoding happen inside each CV fold.

### Pipeline Architecture

```
Input DataFrame
      |
      v
ColumnTransformer
  ├── NumericTransformer:   SimpleImputer(strategy="median")
  └── CategoricTransformer: SimpleImputer(fill_value="Missing")
                             --> OneHotEncoder(drop="first", handle_unknown="ignore")
      |
      v
LinearRegression (OLS)
```

### Feature Engineering (pre-computed on full frame, no leakage risk)

The following derived columns are computed once from raw integer columns (no missing values):

| Derived Feature | Formula | Curvature / Interaction Hypothesis |
|----------------|---------|-------------------------------------|
| `GrLivArea_sq` | `GrLivArea²` | Diminishing returns: each extra sq ft adds less value in a very large home than a small one |
| `YearBuilt_sq` | `YearBuilt²` | Non-linear age effect: very new and very old homes behave differently from mid-century stock |
| `LotArea_sq` | `LotArea²` | Large-lot premium flattens: buyers value moderate lots but do not pay proportionally for sprawling acreage |
| `Qual_x_GrLiv` | `OverallQual × GrLivArea` | Quality premium is amplified in larger homes; a big high-quality property is worth more than the additive sum |
| `NridgHt_x_GrLiv` | `(Neighborhood=="NridgHt") × GrLivArea` | In Northridge Heights (top premium neighbourhood) each extra sq ft commands a higher per-unit premium than the city average |
| `YrBlt_x_Cond` | `YearBuilt × OverallCond` | For older homes, physical condition is a stronger value driver; a well-maintained vintage property recovers premium disproportionately |

**Rejected interactions (with reason):**

| Rejected Interaction | Reason |
|---------------------|--------|
| `GarageCars × OverallQual` | Collinear with `Qual_x_GrLiv`; quality already fully captured; adds negligible new information |
| `FullBath × GrLivArea` | Bathroom count correlates strongly with `GrLivArea` alone; the interaction adds very little signal over the main effect |
| `MSZoning × LotArea` | Too many sparse cells after one-hot encoding; zoning effects better captured by `Neighborhood` |

---

### M0 — Naive Benchmark

**Hypothesis:** No model; predict a constant for every house.

| Constant used | Purpose |
|---------------|---------|
| Median ($163,000) | Minimises MAE for skewed distributions |
| Mean ($180,921) | Minimises RMSE in expectation |

This is the "does the model add any value?" threshold (slide 17). Adj R² = 0 by definition.

---

### M1 — Single Strongest Predictor

**Hypothesis:** `OverallQual` (1–10 expert quality score) is the single best proxy for price. It is a composite capturing build quality, finishes, and materials in one interpretable number. Analogue of "Experience" in the lecture's salary example (slide 22).

| # | Feature | Type | Justification |
|---|---------|------|---------------|
| 1 | `OverallQual` | Numeric | 1–10 expert quality score; the primary value driver |

**Total raw features:** 1 numeric, 0 categorical  
**Approximate model size after encoding:** 2 parameters (intercept + 1 coefficient)

---

### M2 — Core Numeric Features (Linear Baseline)

**Hypothesis:** Price is a linear function of the most physically meaningful numeric attributes. No categorical information yet — pure OLS on continuous/ordinal features.

| # | Feature | Type | Justification |
|---|---------|------|---------------|
| 1 | `OverallQual` | Numeric | 1–10 expert quality score — primary value driver |
| 2 | `GrLivArea` | Numeric | Above-ground living area (sq ft) — size is the main per-unit price determinant |
| 3 | `TotalBsmtSF` | Numeric | Total basement area (sq ft) — usable space below grade, valued independently of above-ground size |
| 4 | `GarageCars` | Numeric | Garage capacity (number of cars) — proxy for garage size and storage amenity |
| 5 | `YearBuilt` | Numeric | Construction year — newer homes command a premium via modern systems, insulation, and design |
| 6 | `LotArea` | Numeric | Lot size (sq ft) — land is a distinct value component separate from the structure |
| 7 | `FullBath` | Numeric | Number of full bathrooms — amenity count directly valued by buyers |
| 8 | `TotRmsAbvGrd` | Numeric | Total rooms above grade — general size and utility proxy |

**Missing value handling:** `SimpleImputer(strategy="median")` inside pipeline  
**Total raw features:** 8 numeric, 0 categorical  
**Approximate model size after encoding:** 9 parameters

---

### M3 — Core Numeric + Categorical Features

**Hypothesis:** Location and ordinal quality ratings encode information that a purely numeric model cannot capture. Neighborhood fixed effects alone can shift predicted price by more than $50,000.

**Carries forward all 8 features from M2, plus:**

| # | Feature | Type | Categories (after drop="first") | Justification |
|---|---------|------|---------------------------------|---------------|
| 9 | `Neighborhood` | Categorical | 24 dummies (25 neighborhoods, drop Blmngtn) | Physical location within Ames — strongest location fixed effect; NoRidge/NridgHt homes sell for ~2x IDOTRR homes |
| 10 | `MSZoning` | Categorical | 4 dummies (5 zones, drop C) | Zoning class (RL, RM, FV, RH, C…) — regulatory land-use classification signals permitted density and use |
| 11 | `KitchenQual` | Categorical | 3 dummies (4 levels Ex/Gd/TA/Fa, drop Ex) | Kitchen quality — kitchens are a primary buyer decision driver; Ex kitchens are the reference category |
| 12 | `ExterQual` | Categorical | 3 dummies (4 levels Ex/Gd/TA/Fa, drop Ex) | Exterior material quality — first-impression signal that strongly correlates with overall value |

**Missing value handling:**
- Numeric: `SimpleImputer(strategy="median")`
- Categorical: `SimpleImputer(strategy="constant", fill_value="Missing")` → `OneHotEncoder(drop="first", handle_unknown="ignore")`

**Total raw features:** 8 numeric, 4 categorical  
**Approximate model size after encoding:** ~43 parameters (8 numeric + ~34 dummies + intercept)

---

### M4 — Core Numeric + Categorical + Nonlinear (Squared) Terms

**Hypothesis:** Three physical features exhibit theoretically motivated curvature. Adding their squared terms allows the OLS model to fit non-linear relationships while remaining an ordinary linear regression in parameter space (slide 30).

**Carries forward all features from M3, plus:**

| # | Feature | Type | Curvature Hypothesis |
|---|---------|------|----------------------|
| 13 | `GrLivArea_sq` | Numeric (engineered) | Diminishing returns to size: each additional sq ft is worth less in a 4,000 sq ft home than a 1,000 sq ft home |
| 14 | `YearBuilt_sq` | Numeric (engineered) | Non-monotone age effect: very old (pre-1920) and very new (post-2000) homes behave differently from mid-century stock |
| 15 | `LotArea_sq` | Numeric (engineered) | Large-lot premium flattens: the price-per-sq-ft of land decreases as lot size becomes very large |

**Total raw features:** 11 numeric, 4 categorical  
**Approximate model size after encoding:** ~46 parameters

> **CV result note:** M4's MAE ($21,249) is *worse* than M3's ($20,728). The squared terms added variance without sufficient bias reduction on held-out folds. The curvature hypotheses were theoretically motivated but did not translate into out-of-sample gain with this feature set — a textbook example of why complexity must be measured by CV, not assumed from theory (slide 17 discipline).

---

### M5 — Full Model: Core + Categorical + Nonlinear + Interactions

**Hypothesis:** Three theory-driven interaction terms capture conditional effects that additive models miss. Quality matters more in larger homes; the size gradient differs by neighbourhood; age and condition jointly determine value for older properties (slide 31).

**Carries forward all features from M4, plus:**

| # | Feature | Type | Interaction Hypothesis |
|---|---------|------|------------------------|
| 16 | `Qual_x_GrLiv` | Numeric (engineered) | `OverallQual × GrLivArea`: quality premium is amplified in larger homes — a big, high-quality property is worth more than the additive sum of quality + size effects |
| 17 | `NridgHt_x_GrLiv` | Numeric (engineered) | `(NridgHt dummy) × GrLivArea`: in Northridge Heights, each extra sq ft commands a higher per-unit premium than the city average; the neighbourhood modifies the size gradient |
| 18 | `YrBlt_x_Cond` | Numeric (engineered) | `YearBuilt × OverallCond`: for older homes, physical condition has a stronger effect on price; a well-maintained vintage property recovers value, a neglected one loses it disproportionately |

> **Collinearity note:** `NridgHt_flag` (the raw binary indicator for Northridge Heights) is **not** added separately because the `Neighborhood` one-hot encoding already produces an identical `Neighborhood_NridgHt` dummy. Adding both would create perfect multicollinearity, arbitrarily splitting one coefficient across two identical columns. Only the interaction term `NridgHt_x_GrLiv` is new information.

**Total raw features:** 14 numeric, 4 categorical  
**Approximate model size after encoding:** ~49 parameters

---

### Complete Feature Map Across All Models

The table below shows exactly which variables are active in each model (checkmark = included):

| Feature | M0 | M1 | M2 | M3 | M4 | M5 |
|---------|----|----|----|----|----|----|
| *(constant — median/mean)* | Y | | | | | |
| `OverallQual` | | Y | Y | Y | Y | Y |
| `GrLivArea` | | | Y | Y | Y | Y |
| `TotalBsmtSF` | | | Y | Y | Y | Y |
| `GarageCars` | | | Y | Y | Y | Y |
| `YearBuilt` | | | Y | Y | Y | Y |
| `LotArea` | | | Y | Y | Y | Y |
| `FullBath` | | | Y | Y | Y | Y |
| `TotRmsAbvGrd` | | | Y | Y | Y | Y |
| `Neighborhood` (24 dummies) | | | | Y | Y | Y |
| `MSZoning` (4 dummies) | | | | Y | Y | Y |
| `KitchenQual` (3 dummies) | | | | Y | Y | Y |
| `ExterQual` (3 dummies) | | | | Y | Y | Y |
| `GrLivArea_sq` | | | | | Y | Y |
| `YearBuilt_sq` | | | | | Y | Y |
| `LotArea_sq` | | | | | Y | Y |
| `Qual_x_GrLiv` | | | | | | Y |
| `NridgHt_x_GrLiv` | | | | | | Y |
| `YrBlt_x_Cond` | | | | | | Y |
| **Total raw features** | 0 | 1 | 8 | 8+4 | 11+4 | 14+4 |
| **Approx. parameters (post-encoding)** | 1 | 2 | 9 | ~43 | ~46 | ~49 |

---

## Part 4 — Honest Evaluation (5-Fold Cross-Validation)

### Evaluation Protocol

```
Full dataset (1,460 rows)
        |
   KFold(n_splits=5, shuffle=True, random_state=42)
        |
   For each fold:
     ├── Train on 1,168 rows  -->  pipeline.fit(X_train, y_train)
     │      (imputation medians learned on train only)
     │      (OHE categories learned on train only)
     └── Validate on 292 rows -->  pipeline.predict(X_val)
           MAE, RMSE, Adj R² computed on held-out fold
   
   Report: mean across 5 folds
```

This ensures **no information from the validation fold contaminates preprocessing** — a direct implementation of the slide 17 "honest evaluation" principle.

### Helper Functions

**`adjusted_r2(r2, n, p)`**
```
Adj R² = 1 - (1 - R²) × (n - 1) / (n - p - 1)
```
Penalises adding predictors that do not improve fit. Returns NaN if degrees of freedom are exhausted.

**`kfold_evaluate(pipeline, X_df, y_arr)`**  
Runs 5-fold CV, computes MAE / RMSE / Adj R² per fold, returns the mean across folds. Extracts `p` (number of coefficients) directly from `pipeline.named_steps["reg"].coef_` after each fold fit.

**`m0_kfold(y_arr)`**  
Fold-aware benchmark: predicts the training-fold median (for MAE) and training-fold mean (for RMSE) on each validation fold. Adj R² = 0 by definition.

### CV Results

| Model | CV MAE | CV RMSE | CV Adj R² |
|-------|--------|---------|-----------|
| M0 — Benchmark | $55,656 | $79,245 | 0.000 |
| M1 — Single predictor | $33,814 | $48,717 | 0.618 |
| M2 — Core numeric | $24,307 | $39,029 | 0.731 |
| M3 — +Categorical | $20,728 | $34,884 | **0.753** |
| M4 — +Nonlinear | $21,249 | $40,144 | 0.627 |
| M5 — +Interactions | **$19,345** | $36,628 | 0.685 |

### Slide 17 Sanity Check

```
Benchmark MAE : $55,656
Winner MAE    : $19,345
Improvement   : 65.2%  -->  PASS
```

The best model beats the naive benchmark by over $36,000 MAE. It is substantively useful.

### Slide 34 Complexity Check (1% rule)

The complexity check asks: is there a simpler model whose CV MAE is within 1% of the best?

```
Threshold (best MAE × 1.01): $19,539

M1 -- Single predictor     MAE=$33,814  [outside]
M2 -- Core numeric         MAE=$24,307  [outside]
M3 -- +Categorical         MAE=$20,728  [outside]
M4 -- +Nonlinear           MAE=$21,249  [outside]
M5 -- +Interactions        MAE=$19,345  [within 1%]  <-- SELECTED
```

No simpler model comes within 1% of M5. **M5 is both the MAE winner and the complexity-adjusted winner.**

### Notable Pattern: M4 Regression

M4 (+nonlinear) is *worse* than M3 (+categorical) by $521 MAE, and its Adj R² drops from 0.753 to 0.627. This is a textbook illustration of the slide 34 / slide 17 lesson: theoretically motivated features do not automatically earn their keep out-of-sample. The squared terms added variance without enough bias reduction on this dataset. M5's interaction terms, by contrast, recovered the loss and improved further — the conditional effects captured by interactions carried genuine out-of-sample signal.

**Plots generated:** `05_model_comparison.png`, `06_model_comparison_r2.png`

---

## Part 5 — Final Model and Prediction

### Selected Model: M5 — +Interactions

| Metric | Value |
|--------|-------|
| CV MAE | $19,345 |
| CV RMSE | $36,628 |
| CV Adj R² | 0.685 |

The model is refit on all 1,460 observations to use the maximum available information for the final coefficient estimates.

### Top 25 Coefficients (sorted by absolute magnitude)

| Rank | Feature | Coefficient |
|------|---------|-------------|
| 1 | `Neighborhood_StoneBr` | +$52,904 |
| 2 | `Neighborhood_NoRidge` | +$49,938 |
| 3 | `Neighborhood_NridgHt` | -$35,439 |
| 4 | `KitchenQual_Fa` | -$35,344 |
| 5 | `MSZoning_RL` | +$32,320 |
| 6 | `MSZoning_RH` | +$30,845 |
| 7 | `KitchenQual_TA` | -$30,659 |
| 8 | `MSZoning_FV` | +$30,481 |
| 9 | `Neighborhood_Veenker` | +$29,406 |
| 10 | `Neighborhood_Crawfor` | +$28,427 |
| 11 | `KitchenQual_Gd` | -$27,462 |
| 12 | `MSZoning_RM` | +$27,403 |
| 13 | `OverallQual` | -$23,328 |
| 14 | `Neighborhood_ClearCr` | +$17,381 |
| 15 | `Neighborhood_Somerst` | +$14,678 |
| 16 | `ExterQual_Gd` | -$13,880 |
| 17 | `Neighborhood_Timber` | +$12,170 |
| 18 | `Neighborhood_Blueste` | -$11,868 |
| 19 | `ExterQual_TA` | -$11,668 |
| 20 | `ExterQual_Fa` | -$10,399 |
| 21 | `Neighborhood_Edwards` | -$10,304 |
| 22 | `Neighborhood_MeadowV` | -$9,611 |
| 23 | `GarageCars` | +$9,531 |
| 24 | `Neighborhood_OldTown` | -$8,626 |
| 25 | `YearBuilt` | -$8,032 |

> **Intercept:** $7,556,914 (absorbs the large squared and interaction term offsets — not meaningful in isolation)

**Interpreting the negative `OverallQual` coefficient:** Once `Qual_x_GrLiv` (the quality × size interaction) is in the model, `OverallQual`'s main-effect coefficient (-$23,328) cannot be read in isolation. The total quality effect is `β_qual + β_interaction × GrLivArea`, which is large and positive across all realistic house sizes. The apparent negative main effect is a standard consequence of including an interaction term — it represents the marginal effect of quality when GrLivArea = 0, which is a meaningless extrapolation.

**Interpreting the negative `Neighborhood_NridgHt` coefficient:** NridgHt homes command a large premium, but that premium is captured jointly through the negative main-effect coefficient (baseline shift when GrLivArea = 0, again not meaningful alone) and the positive `NridgHt_x_GrLiv` interaction term. Together they mean: being in NridgHt makes each sq ft more valuable, not that it makes the house cheaper.

### Sample Prediction

Profile: a reasonable mid-market house based on dataset medians.

| Feature | Value | Notes |
|---------|-------|-------|
| `OverallQual` | 6 | Slightly above average; dataset median = 6 |
| `GrLivArea` | 1,400 sq ft | Dataset median ~1,464 |
| `TotalBsmtSF` | 800 sq ft | — |
| `GarageCars` | 2 | Most common value |
| `YearBuilt` | 1985 | Mid-range vintage |
| `LotArea` | 9,000 sq ft | Dataset median ~9,478 |
| `FullBath` | 2 | — |
| `TotRmsAbvGrd` | 7 | — |
| `Neighborhood` | NAmes | Largest neighbourhood (225 houses, 15% of data) |
| `MSZoning` | RL | Most common zone (79% of data) |
| `KitchenQual` | TA | Typical/Average — modal category |
| `ExterQual` | TA | Typical/Average |
| `OverallCond` | 5 | Average condition |

```
Point estimate    : $159,559
Reported as range : $140,214  to  $178,905
                    (+-$19,345 CV MAE)
```

The prediction is reported as a range, not a point estimate. The ±CV MAE interval reflects the model's average absolute error on held-out data — the honest measure of uncertainty (slide 45/47 discipline).

**Plots generated:** `03_residual_plot.png`, `04_predicted_vs_actual.png`

---

## Output Files

| File | Description |
|------|-------------|
| `01_target_distribution.png` | SalePrice histogram with mean (red dashed) and median (orange solid) lines — diagnoses right skew |
| `02_top_predictor_scatter.png` | OverallQual (1–10) vs SalePrice scatter — analogue of slide 21 Experience vs Salary |
| `03_residual_plot.png` | Residuals vs predicted for the final model (M5, in-sample refit) — fan shape visible at high prices indicates heteroskedasticity |
| `04_predicted_vs_actual.png` | Predicted vs actual scatter for M5 with 45-degree perfect-prediction line — in-sample refit |
| `05_model_comparison.png` | Bar chart of CV MAE across M0–M5; red bar = selected model; analogue of slide 43 |
| `06_model_comparison_r2.png` | Bar chart of CV Adjusted R² across M0–M5; highlights that M3 maximises Adj R² while M5 minimises MAE |

---

## Key Findings

**Winner:** M5 (+interactions) with CV MAE of $19,345 — a 65.2% improvement over the naive benchmark.

**Complexity earns its keep in M5 but not in M4.** The squared terms in M4 raised MAE by $521 versus M3. The interaction terms in M5 then reduced MAE by $1,383 versus M3, more than recovering the loss. Theory-motivated features require CV validation — the slide 34 discipline prevents overfitting to theoretical priors.

**Features doing the most predictive work:**
1. **Neighborhood** — location fixed effects shift prices by up to $52,000; irreplaceable in any OLS house price model
2. **Qual_x_GrLiv** (quality × size interaction) — the key nonlinear signal; quality premium scales with size
3. **KitchenQual** — $27,000–$35,000 shifts per quality tier; kitchens are a primary buyer decision point
4. **MSZoning** — $27,000–$32,000 fixed effects by zoning class
5. **GarageCars** — $9,531 per additional car capacity; consistent marginal amenity value

**Caveats:**
- Coefficient signs for `OverallQual` and `Neighborhood_NridgHt` appear negative in isolation due to the interaction terms absorbing the slope — always interpret interaction-model coefficients jointly
- The residual plot shows heteroskedasticity (fan shape at high prices); OLS standard errors are technically biased for luxury homes
- Plots 03 and 04 use the in-sample full-data refit — the CV MAE ($19,345) is the honest out-of-sample number

---

## How to Run

```bash
cd "path/to/Applied Quant STudies Housing"
python ames_regression_analysis.py
```

All six PNG files are saved to the working directory. Runtime is approximately 20–30 seconds (dominated by the 5-fold CV loop across 6 models).

**Dependencies:**
```
pandas
numpy
matplotlib
scikit-learn >= 1.2   (uses sparse_output=False in OneHotEncoder)
```
