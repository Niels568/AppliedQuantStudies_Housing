"""
=============================================================================
AMES HOUSE PRICES -- PREDICTIVE ANALYSIS
Applied Quantitative Studies -- Week 3: Multiple Linear Regression (OLS)
=============================================================================
Libraries: pandas, numpy, matplotlib, scikit-learn only.
Model:      sklearn.linear_model.LinearRegression (OLS).
Evaluation: 5-fold KFold CV; metrics = MAE, RMSE, Adjusted R2.
Pipeline:   ColumnTransformer + Pipeline ensure all preprocessing is fitted
            inside each fold -- no leakage between train and validation splits.
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (saves to file)
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


# =============================================================================
# PART 1 -- DESCRIPTIVE SETUP
# Understand the target variable before modelling anything, look at distribution shape, 
# central, tendency, and decide which benchmark (mean vs median) to use.
# =============================================================================

print("=" * 70)
print("PART 1 -- DESCRIPTIVE SETUP")
print("=" * 70)

df = pd.read_csv("train.csv")

print(f"\nDataset shape : {df.shape[0]:,} rows x {df.shape[1]} columns")
print("\nFirst 5 rows (key columns):")
print(df[["Id", "OverallQual", "GrLivArea", "YearBuilt",
          "Neighborhood", "SalePrice"]].head().to_string(index=False))

# -- Target variable summary --
sp = df["SalePrice"]
print("\n--- SalePrice Summary Statistics ---")
print(f"  Mean     : ${sp.mean():>12,.0f}")
print(f"  Median   : ${sp.median():>12,.0f}")
print(f"  Std Dev  : ${sp.std():>12,.0f}")
print(f"  Min      : ${sp.min():>12,.0f}")
print(f"  Max      : ${sp.max():>12,.0f}")
print(f"  Skewness :  {sp.skew():>11.3f}")

print("""
DISTRIBUTION DIAGNOSIS:
  Skewness = 1.88 -- strongly right-skewed.
  The mean ($180,921) sits well above the median ($163,000) because a
  long upper tail of luxury properties (max $755,000) inflates the average.
  This mirrors any salary/income distribution: a few top earners (here,
  top properties) pull the arithmetic mean away from the typical case.

  Implication for benchmarking (slide 9 logic):
    * MAE benchmark  -> predict the MEDIAN.  Median is robust to outliers;
      predicting it for every house minimises expected absolute error when
      the distribution is skewed.
    * RMSE benchmark -> predict the MEAN.  The mean minimises expected
      squared error in expectation, so it is the correct constant predictor
      when the loss function is quadratic.
  We use MEDIAN as our headline benchmark since MAE is the primary metric.
""")

# -- Plot 01: SalePrice distribution --
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(sp / 1_000, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
ax.axvline(sp.mean()   / 1_000, color="red",    linestyle="--", linewidth=2.0,
           label=f"Mean   ${sp.mean()/1_000:.0f}k")
ax.axvline(sp.median() / 1_000, color="orange", linestyle="-",  linewidth=2.0,
           label=f"Median ${sp.median()/1_000:.0f}k")
ax.set_xlabel("Sale Price ($ thousands)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Distribution of Sale Price -- Ames Housing Dataset\n"
             "(Right-skewed: median is the better central-tendency benchmark)", fontsize=13)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("01_target_distribution.png", dpi=150)
plt.close()
print("Saved: 01_target_distribution.png")

# -- Plot 07: Summary statistics table for key variables --
key_vars = [
    "SalePrice", "OverallQual", "GrLivArea", "TotalBsmtSF",
    "GarageCars", "YearBuilt",  "LotArea",   "FullBath",
]
stats = df[key_vars].agg(["mean", "std", "min", "max"]).T

# Format each number cleanly: integers for whole-number cols, 1dp otherwise
def fmt(val, var):
    if var in ("OverallQual", "GarageCars", "FullBath"):
        return f"{val:.2f}"
    return f"{val:,.0f}"

cell_text = [[fmt(stats.loc[v, c], v) for c in ["mean", "std", "min", "max"]]
             for v in key_vars]

fig, ax = plt.subplots(figsize=(10, 3.8))
ax.axis("off")
tbl = ax.table(
    cellText=cell_text,
    rowLabels=key_vars,
    colLabels=["Mean", "Std Dev", "Min", "Max"],
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.35, 1.9)

for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if row == 0:                        # column header row
        cell.set_facecolor("#2c5f8a")
        cell.set_text_props(color="white", fontweight="bold")
    elif col == -1:                     # row labels
        cell.set_facecolor("#dce8f5")
        cell.set_text_props(fontweight="bold")
    elif row % 2 == 0:                  # alternating row shading
        cell.set_facecolor("#f5f8fc")
    else:
        cell.set_facecolor("white")

ax.set_title("Summary Statistics -- Key Variables (Ames Housing Dataset)",
             fontsize=13, fontweight="bold", pad=14)
plt.tight_layout()
plt.savefig("07_summary_statistics.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 07_summary_statistics.png")

# -- Plot 08: 2x2 correlation scatter plots -- SalePrice vs 4 key variables --
# Variables chosen: the four numerics with the strongest theoretical and
# empirical link to price from the M2 feature set.
corr_pairs = [
    ("GrLivArea",   "Above-Ground Living Area (sq ft)",  "#2196F3"),
    ("OverallQual", "Overall Quality Rating (1-10)",     "#4CAF50"),
    ("YearBuilt",   "Year Built",                        "#FF9800"),
    ("TotalBsmtSF", "Total Basement Area (sq ft)",       "#9C27B0"),
]

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
for ax, (var, xlabel, colour) in zip(axes.flat, corr_pairs):
    x   = df[var].values
    ysp = df["SalePrice"].values / 1_000
    r   = np.corrcoef(x, ysp)[0, 1]

    ax.scatter(x, ysp, alpha=0.25, color=colour, s=14, linewidths=0)

    # OLS trend line via numpy polyfit
    m, b   = np.polyfit(x, ysp, 1)
    x_line = np.linspace(x.min(), x.max(), 300)
    ax.plot(x_line, m * x_line + b, color="red", linewidth=2.0,
            label=f"r = {r:.2f}  |  slope = {m:.2f}")

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Sale Price ($ thousands)", fontsize=11)
    ax.set_title(f"SalePrice  vs  {var}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.tick_params(labelsize=9)

fig.suptitle("SalePrice Correlations -- Four Key Explanatory Variables\n"
             "(red line = OLS trend, r = Pearson correlation)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("08_correlation_plots.png", dpi=150)
plt.close()
print("Saved: 08_correlation_plots.png")


# =============================================================================
# PART 2 -- FRAMING THE PREDICTION PROBLEM
# Slide concepts: unit of observation, leakage check, naive benchmark (M0).
# =============================================================================

print("\n" + "=" * 70)
print("PART 2 -- FRAMING THE PREDICTION PROBLEM")
print("=" * 70)

print("""
PREDICTION TARGET    : SalePrice (USD) -- continuous outcome variable
UNIT OF OBSERVATION  : one residential property transaction in Ames, Iowa
FEATURES AVAILABLE   : 79 descriptors (physical attributes + transaction metadata)

LEAKAGE ASSESSMENT:
  Unlike BNPL credit risk (where repayment outcome may bleed into features),
  most Ames features describe the physical house and are available before
  sale. However, FOUR variables describe the transaction itself and are
  logically contemporaneous with the sale price:

    * MoSold / YrSold  -- month and year of sale (market-timing information)
    * SaleType          -- type of deed (e.g., new construction, court officer)
    * SaleCondition     -- transaction circumstance (normal, foreclosure, etc.)

  Decision: INCLUDED in later models but flagged here.
  Rationale: At listing time you would not know SaleCondition or SaleType.
  For this academic exercise we treat them as potential confounders to be
  aware of, not hard exclusions. Our core feature sets (M1-M5) deliberately
  exclude them from the primary specification.
""")

# -- Naive benchmark (M0): predict a constant for every house --
median_price = sp.median()    # used for MAE benchmark
mean_price   = sp.mean()      # used for RMSE benchmark

benchmark_mae  = np.mean(np.abs(sp.values - median_price))
benchmark_rmse = np.sqrt(np.mean((sp.values - mean_price) ** 2))

print(f"NAIVE BENCHMARK (M0) -- predict median ${median_price:,.0f} for every house:")
print(f"  Benchmark MAE  : ${benchmark_mae:,.0f}")
print(f"  Benchmark RMSE : ${benchmark_rmse:,.0f}")
print(f"\n  Any model that cannot beat ${benchmark_mae:,.0f} MAE is no better than")
print(f"  saying 'every house costs $163,000'. This is our slide 17 threshold.")

# -- Plot 02: Top predictor scatter (analogue of slide 21) --
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df["OverallQual"], sp / 1_000, alpha=0.35, color="steelblue", s=18)
ax.set_xlabel("Overall Quality Rating (1-10)", fontsize=12)
ax.set_ylabel("Sale Price ($ thousands)", fontsize=12)
ax.set_title("Overall Quality vs Sale Price\n"
             "(analogue of slide 21: Experience vs Salary)", fontsize=13)
ax.set_xticks(range(1, 11))
plt.tight_layout()
plt.savefig("02_top_predictor_scatter.png", dpi=150)
plt.close()
print("\nSaved: 02_top_predictor_scatter.png")


# =============================================================================
# PART 3 -- CANDIDATE MODELS (M1 -> M5)
#simple -> complex progression; justify every feature added;
#          nonlinear terms; interaction terms.
# All preprocessing is deferred to the Pipeline so it runs fold-by-fold.
# =============================================================================

print("\n" + "=" * 70)
print("PART 3 -- CANDIDATE MODELS")
print("=" * 70)

# -- Feature engineering on the full dataframe --
# Squared and interaction terms are computed once here from raw columns that
# have no missing values (all int64 in the original dataset), so no leakage
# risk from this pre-computation.  Imputation of other numeric columns still
# happens inside each pipeline fold.

df_feat = df.copy()

# Squared terms -- justify curvature hypothesis for each 
df_feat["GrLivArea_sq"]  = df_feat["GrLivArea"]  ** 2
# Hypothesis: diminishing returns to size -- each extra sq ft adds less value
# in a very large home than in a small one.

df_feat["YearBuilt_sq"]  = df_feat["YearBuilt"]  ** 2
# Hypothesis: the age penalty is non-linear; very new and very old homes
# behave differently from mid-century stock, creating a curved age profile.

df_feat["LotArea_sq"]    = df_feat["LotArea"]    ** 2
# Hypothesis: large-lot premium flattens -- buyers value moderate lots but
# do not pay proportionally for sprawling acreage.

# Interaction terms -- accepted interactions:
df_feat["Qual_x_GrLiv"]    = df_feat["OverallQual"] * df_feat["GrLivArea"]
# Hypothesis: quality premium is amplified in larger homes; a big, high-
# quality property is worth more than the additive sum of quality + size.

df_feat["NridgHt_flag"]    = (df_feat["Neighborhood"] == "NridgHt").astype(int)
df_feat["NridgHt_x_GrLiv"] = df_feat["NridgHt_flag"] * df_feat["GrLivArea"]
# Hypothesis: in Northridge Heights (the top premium neighbourhood) each
# extra square foot commands a higher per-unit premium than the city average;
# the neighbourhood modifies the slope of the size gradient.

df_feat["YrBlt_x_Cond"]   = df_feat["YearBuilt"] * df_feat["OverallCond"]
# Hypothesis: for older homes, physical condition is a stronger value driver;
# a well-maintained vintage property recovers premium, a neglected one loses
# it disproportionately -- producing a non-zero age x condition slope.


y = df_feat["SalePrice"].values


# -- Pipeline factory --
def make_pipeline(num_cols: list, cat_cols: list) -> Pipeline:
    """
    Build a preprocessing + OLS pipeline.
    Numeric columns: median imputation.
    Categorical columns: 'Missing'-fill imputation then one-hot encoding
        (drop='first' removes dummy-variable trap; handle_unknown='ignore'
        silences unseen categories in validation folds).
    Everything is wrapped in Pipeline so transformations are fitted only on
    the training portion of each fold -- no leakage.
    """
    num_transformer = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
    ])
    transformers = [("num", num_transformer, num_cols)]

    if cat_cols:
        cat_transformer = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
                sparse_output=False,
            )),
        ])
        transformers.append(("cat", cat_transformer, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline([
        ("preprocessor", preprocessor),
        ("reg",          LinearRegression()),
    ])


# -- M1: Single strongest predictor --
# Hypothesis: OverallQual (1-10 expert condition score) is the single best
# proxy for price -- it is a composite score capturing build quality, finishes,
# and materials in one interpretable number. 
m1_num = ["OverallQual"]
m1_cat = []

# -- M2: Core numeric features (linear baseline) --
# Hypothesis: price is a linear function of the most physically meaningful
# numeric attributes.  Feature justifications:
m2_num = [
    "OverallQual",      # 1-10 expert quality score -- primary value driver
    "GrLivArea",        # above-ground living area (sq ft) -- size is the main
                        # per-unit price determinant for residential property
    "TotalBsmtSF",      # total basement area -- usable space below grade adds
                        # value independently of above-ground size
    "GarageCars",       # garage capacity (cars) -- proxy for garage size and
                        # storage amenity valued by buyers
    "YearBuilt",        # construction year -- newer homes command a premium
                        # via modern systems, insulation, and design standards
    "LotArea",          # lot size (sq ft) -- land is a distinct value component
                        # separate from the structure itself
    "FullBath",         # number of full bathrooms -- amenity count directly
                        # valued by buyers; more than bedrooms in many markets
    "TotRmsAbvGrd",     # total rooms above grade -- general size/utility proxy
]
m2_cat = []

# -- M3: Core numerics + categorical location/quality features --
# Hypothesis: location and quality ratings encode non-numeric information
# that a linear numeric model cannot capture.
m3_num = m2_num.copy()
m3_cat = [
    "Neighborhood",     # physical location within Ames -- the strongest
                        # location fixed-effect; NoRidge/NridgHt homes sell
                        # for 2x the price of IDOTRR homes on average
    "MSZoning",         # zoning class (RL, RM, FV, C?) -- regulatory land-use
                        # classification signals permitted density and use
    "KitchenQual",      # kitchen quality (Ex/Gd/TA/Fa/Po) -- kitchens are a
                        # primary buyer decision driver in residential markets
    "ExterQual",        # exterior material quality -- first-impression signal
                        # that strongly correlates with overall home value
]

# -- M4: M3 + squared (nonlinear) terms --
# Hypothesis: three physical features exhibit theoretically motivated curvature
# (see individual hypotheses on the engineering lines above, slide 30).
m4_num = m3_num + ["GrLivArea_sq", "YearBuilt_sq", "LotArea_sq"]
m4_cat = m3_cat.copy()

# -- M5: M4 + interaction terms --
# Three theory-driven interactions added to M4 (hypotheses above, slide 31).
# NOTE: NridgHt_flag is NOT added to m5_num separately -- the Neighborhood
# one-hot encoding in m5_cat already produces a Neighborhood_NridgHt dummy
# that captures the same information.  Including NridgHt_flag here would
# create perfect multicollinearity (two identical columns), splitting the
# coefficient arbitrarily.  Only the interaction term NridgHt_x_GrLiv is
# added, which is genuinely new information (it scales with GrLivArea).
m5_num = m4_num + ["Qual_x_GrLiv", "NridgHt_x_GrLiv", "YrBlt_x_Cond"]
m5_cat = m4_cat.copy()

# -- M5_alt: M3 + interaction terms only (no squared terms) --
# This tests whether M5's improvement over M3 comes from the interactions
# alone, without the squared terms that made M4 worse than M3.
# If M5_alt beats M5, the squared terms are actively hurting and should
# be dropped -- the interactions carry the signal on their own.
m5_alt_num = m3_num + ["Qual_x_GrLiv", "NridgHt_x_GrLiv", "YrBlt_x_Cond"]
m5_alt_cat = m3_cat.copy()

# Instantiate all pipelines
pipe_m1     = make_pipeline(m1_num,     m1_cat)
pipe_m2     = make_pipeline(m2_num,     m2_cat)
pipe_m3     = make_pipeline(m3_num,     m3_cat)
pipe_m4     = make_pipeline(m4_num,     m4_cat)
pipe_m5     = make_pipeline(m5_num,     m5_cat)
pipe_m5_alt = make_pipeline(m5_alt_num, m5_alt_cat)

print("\nModel specifications (raw features before one-hot expansion):")
for label, num, cat in [
    ("M1 -- Single predictor",  m1_num,     m1_cat),
    ("M2 -- Core numeric",      m2_num,     m2_cat),
    ("M3 -- +Categorical",      m3_num,     m3_cat),
    ("M4 -- +Nonlinear",        m4_num,     m4_cat),
    ("M5 -- +Interactions",     m5_num,     m5_cat),
    ("M5_alt -- M3+Interact.",  m5_alt_num, m5_alt_cat),
]:
    print(f"  {label:<26}: {len(num)} numeric + {len(cat)} categorical raw features")


# =============================================================================
# PART 4 -- HONEST EVALUATION (5-FOLD CROSS-VALIDATION)
# Slide concepts: out-of-sample evaluation (slide 17), K-fold CV, adjusted R2,
#                 model comparison table (slide 43), complexity check (slide 34).
# =============================================================================

print("\n" + "=" * 70)
print("PART 4 -- HONEST EVALUATION (5-FOLD CV)")
print("=" * 70)


# -- Helper: adjusted R2 --
def adjusted_r2(r2: float, n: int, p: int) -> float:
    """
    Adjusted R2 = 1 - (1 - R2) * (n - 1) / (n - p - 1).
    Penalises adding predictors that do not improve fit.
    Returns NaN if degrees of freedom are exhausted.
    """
    if n - p - 1 <= 0:
        return np.nan
    return 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)


# -- Helper: 5-fold CV evaluation --
def kfold_evaluate(
    pipeline: Pipeline,
    X_df: pd.DataFrame,
    y_arr: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple:
    """
    Run K-fold CV on a sklearn Pipeline.  All preprocessing (imputation,
    one-hot encoding) is fitted inside each training fold and applied to the
    validation fold -- no information crosses the fold boundary (slide 17).

    Returns:
        (mean_MAE, mean_RMSE, mean_adj_R2) across all folds.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    maes, rmses, adj_r2s = [], [], []

    for train_idx, val_idx in kf.split(X_df):
        X_train, X_val = X_df.iloc[train_idx], X_df.iloc[val_idx]
        y_train, y_val = y_arr[train_idx],     y_arr[val_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)

        mae  = np.mean(np.abs(y_val - y_pred))
        rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))

        ss_res = np.sum((y_val - y_pred) ** 2)
        ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        p            = len(pipeline.named_steps["reg"].coef_)
        adj_r2_fold  = adjusted_r2(r2, n=len(y_val), p=p)

        maes.append(mae)
        rmses.append(rmse)
        adj_r2s.append(adj_r2_fold)

    return float(np.mean(maes)), float(np.mean(rmses)), float(np.mean(adj_r2s))


# -- M0: benchmark evaluated fold-by-fold --
def m0_kfold(y_arr: np.ndarray, n_splits: int = 5, random_state: int = 42) -> tuple:
    """
    Fold-aware benchmark: predict the training-fold median (for MAE) and
    training-fold mean (for RMSE) on each validation fold.
    Adj R2 = 0 by definition for a constant predictor.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    maes, rmses = [], []
    for train_idx, val_idx in kf.split(y_arr):
        y_train, y_val = y_arr[train_idx], y_arr[val_idx]
        maes.append( np.mean(np.abs(y_val - np.median(y_train))))
        rmses.append(np.sqrt(np.mean((y_val - np.mean(y_train)) ** 2)))
    return float(np.mean(maes)), float(np.mean(rmses)), 0.0


# -- Run all models --
print("\nRunning 5-fold CV (random_state=42) for all six models ...")

X_df = df_feat    # full feature frame; each pipeline selects its own columns

m0_mae, m0_rmse, m0_ar2 = m0_kfold(y)
print("  [M0] Benchmark          -- done")

m1_mae, m1_rmse, m1_ar2 = kfold_evaluate(pipe_m1, X_df, y)
print("  [M1] Single predictor   -- done")

m2_mae, m2_rmse, m2_ar2 = kfold_evaluate(pipe_m2, X_df, y)
print("  [M2] Core numeric       -- done")

m3_mae, m3_rmse, m3_ar2 = kfold_evaluate(pipe_m3, X_df, y)
print("  [M3] +Categorical       -- done")

m4_mae, m4_rmse, m4_ar2 = kfold_evaluate(pipe_m4, X_df, y)
print("  [M4] +Nonlinear         -- done")

m5_mae, m5_rmse, m5_ar2 = kfold_evaluate(pipe_m5, X_df, y)
print("  [M5] +Interactions      -- done")

m5_alt_mae, m5_alt_rmse, m5_alt_ar2 = kfold_evaluate(pipe_m5_alt, X_df, y)
print("  [M5_alt] M3+Interact.   -- done")

# -- Results table --
results = pd.DataFrame({
    "Model":       ["M0 -- Benchmark",      "M1 -- Single predictor",
                    "M2 -- Core numeric",   "M3 -- +Categorical",
                    "M4 -- +Nonlinear",     "M5 -- +Interactions",
                    "M5_alt -- M3+Interact."],
    "CV MAE":      [m0_mae, m1_mae, m2_mae, m3_mae, m4_mae, m5_mae, m5_alt_mae],
    "CV RMSE":     [m0_rmse, m1_rmse, m2_rmse, m3_rmse, m4_rmse, m5_rmse, m5_alt_rmse],
    "CV Adj R2":   [m0_ar2, m1_ar2, m2_ar2, m3_ar2, m4_ar2, m5_ar2, m5_alt_ar2],
})

print("\n--- 5-Fold CV Results ---")
display_df = results.copy()
display_df["CV MAE"]  = display_df["CV MAE"].map(lambda x: f"${x:>10,.0f}")
display_df["CV RMSE"] = display_df["CV RMSE"].map(lambda x: f"${x:>10,.0f}")
display_df["CV Adj R2"] = display_df["CV Adj R2"].map(lambda x: f"{x:>8.3f}")
print(display_df.to_string(index=False))

# -- Identify winner --
mae_vals = results["CV MAE"].values
best_idx = int(np.argmin(mae_vals))
best_mae = mae_vals[best_idx]

print(f"\nRaw winner (lowest CV MAE): {results.loc[best_idx, 'Model']}  "
      f"(MAE = ${best_mae:,.0f})")

# -- Slide 17 sanity check --
improvement_pct = (m0_mae - best_mae) / m0_mae * 100
beat = best_mae < m0_mae
print(f"\nSLIDE 17 SANITY CHECK -- does the winner beat the naive benchmark?")
print(f"  Benchmark MAE : ${m0_mae:,.0f}")
print(f"  Winner MAE    : ${best_mae:,.0f}")
print(f"  Improvement   : {improvement_pct:.1f}%  ->  "
      f"{'PASS' if beat else 'FAIL'}")

# -- Slide 34 complexity check (1 % rule) --
threshold = best_mae * 1.01
selected_idx = best_idx
for i in range(1, best_idx):       # check simpler models (skip M0)
    if mae_vals[i] <= threshold:
        selected_idx = i
        break

selected_name = results.loc[selected_idx, "Model"]
selected_mae  = mae_vals[selected_idx]

print(f"\nSLIDE 34 COMPLEXITY CHECK (prefer simpler if within 1 % of best):")
print(f"  Threshold (best MAE x 1.01): ${threshold:,.0f}")
for i, row in results.iterrows():
    marker = "  <-- SELECTED" if i == selected_idx else (
             "  <-- raw best" if i == best_idx and selected_idx != best_idx else "")
    flag = "within 1%" if mae_vals[i] <= threshold else "outside"
    if i > 0:   # skip M0 in this display
        print(f"  {row['Model']:<28}  MAE=${mae_vals[i]:,.0f}  [{flag}]{marker}")

if selected_idx != best_idx:
    saved = mae_vals[selected_idx] - best_mae
    print(f"\n  Decision: use {selected_name} -- simpler and within 1% of best.")
    print(f"  Complexity of {results.loc[best_idx,'Model']} does NOT earn its keep "
          f"(costs ${saved:,.0f} extra MAE for more parameters).")
else:
    print(f"\n  Decision: {selected_name} is both the best and complexity-adjusted winner.")

# -- Plot 05: Model comparison bar chart (analogue of slide 43) --
# Colour key: red = selected model, light blue = raw MAE winner (if different),
#             orange = M5_alt (the alternative specification), steel blue = all others.
def bar_color(i):
    if i == selected_idx:   return "#d9534f"   # red   -- selected
    if i == best_idx:       return "#5bc0de"   # blue  -- raw MAE winner
    if results["Model"].iloc[i].startswith("M5_alt"): return "#f0ad4e"  # orange
    return "steelblue"

fig, ax = plt.subplots(figsize=(12, 5))
bar_colors = [bar_color(i) for i in range(len(results))]
bars = ax.bar(results["Model"], mae_vals / 1_000, color=bar_colors, edgecolor="white")
ax.axhline(m0_mae / 1_000, color="red", linestyle="--", linewidth=1.8,
           label=f"M0 benchmark (${m0_mae/1_000:.1f}k)")
ax.set_ylabel("5-Fold CV MAE ($ thousands)", fontsize=12)
ax.set_title("Model Comparison -- 5-Fold CV MAE\n"
             "(red = selected | orange = M5_alt: M3 + interactions only)", fontsize=13)
ax.set_xticks(range(len(results["Model"])))
ax.set_xticklabels(results["Model"], rotation=20, ha="right", fontsize=9)
ax.legend(fontsize=10)
for bar, val in zip(bars, mae_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.25,
            f"${val/1_000:.1f}k", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("05_model_comparison.png", dpi=150)
plt.close()
print("\nSaved: 05_model_comparison.png")

# -- Plot 06: Model comparison -- CV Adjusted R2 --
adj_r2_vals = results["CV Adj R2"].values
r2_colors = [bar_color(i) for i in range(len(results))]
fig, ax = plt.subplots(figsize=(12, 5))
bars2 = ax.bar(results["Model"], adj_r2_vals, color=r2_colors, edgecolor="white")
ax.axhline(0, color="red", linestyle="--", linewidth=1.8, label="M0 benchmark (Adj R2 = 0)")
ax.set_ylim(-0.05, 1.0)
ax.set_ylabel("5-Fold CV Adjusted R2", fontsize=12)
ax.set_title("Model Comparison -- 5-Fold CV Adjusted R2\n"
             "(red = selected | orange = M5_alt: M3 + interactions only)", fontsize=13)
ax.set_xticks(range(len(results["Model"])))
ax.set_xticklabels(results["Model"], rotation=20, ha="right", fontsize=9)
ax.legend(fontsize=10)
for bar, val in zip(bars2, adj_r2_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, max(val, 0) + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("06_model_comparison_r2.png", dpi=150)
plt.close()
print("Saved: 06_model_comparison_r2.png")


# =============================================================================
# PART 5 -- FINAL MODEL + PREDICTION
# Slide concepts: refit on full data (slide 34), regression equation,
#                 prediction with ?CV-MAE uncertainty range (slide 45/47).
# =============================================================================

print("\n" + "=" * 70)
print("PART 5 -- FINAL MODEL + PREDICTION")
print("=" * 70)

# Map model index -> pipeline and column lists
model_registry = {
    1: (pipe_m1,     m1_num,     m1_cat),
    2: (pipe_m2,     m2_num,     m2_cat),
    3: (pipe_m3,     m3_num,     m3_cat),
    4: (pipe_m4,     m4_num,     m4_cat),
    5: (pipe_m5,     m5_num,     m5_cat),
    6: (pipe_m5_alt, m5_alt_num, m5_alt_cat),
}
final_pipe, final_num, final_cat = model_registry[selected_idx]

print(f"\nSelected model : {selected_name}")
print(f"  CV MAE  = ${selected_mae:,.0f}")
print(f"  CV RMSE = ${results.loc[selected_idx, 'CV RMSE']:,.0f}")
print(f"  CV Adj R2 = {results.loc[selected_idx, 'CV Adj R2']:.3f}")

# -- Refit on full 1,460-row dataset --
print("\nRefitting on all 1,460 observations ...")
final_pipe.fit(X_df, y)

# -- Recover feature names post-encoding --
ohe_features = []
if final_cat:
    ohe = (final_pipe
           .named_steps["preprocessor"]
           .named_transformers_["cat"]
           .named_steps["onehot"])
    ohe_features = list(ohe.get_feature_names_out(final_cat))

all_feature_names = final_num + ohe_features
coefs     = final_pipe.named_steps["reg"].coef_
intercept = final_pipe.named_steps["reg"].intercept_

coef_df = pd.DataFrame({"Feature": all_feature_names, "Coefficient": coefs})
coef_df_sorted = coef_df.reindex(
    coef_df["Coefficient"].abs().sort_values(ascending=False).index
).reset_index(drop=True)

print(f"\nREGRESSION EQUATION -- top 25 coefficients (sorted by absolute magnitude):")
print(f"  Intercept : {intercept:>15,.2f}")
print(f"  {'Feature':<40} {'Coefficient':>15}")
print(f"  {'-' * 56}")
for _, row in coef_df_sorted.head(25).iterrows():
    print(f"  {row['Feature']:<40} {row['Coefficient']:>15,.2f}")

if len(coef_df_sorted) > 25:
    print(f"  ... ({len(coef_df_sorted) - 25} further coefficients omitted)")

# -- Plot 09: Horizontal coefficient bar chart (top 20 by magnitude) --
# Blue = positive coefficient (raises SalePrice)
# Red  = negative coefficient (lowers SalePrice)
top_n  = 20
plot_df = coef_df_sorted.head(top_n).iloc[::-1]   # reverse so largest is at top

colours = ["#4472C4" if c > 0 else "#C0392B" for c in plot_df["Coefficient"]]

fig, ax = plt.subplots(figsize=(11, 8))
bars = ax.barh(plot_df["Feature"], plot_df["Coefficient"],
               color=colours, edgecolor="white", height=0.65)
ax.axvline(0, color="black", linewidth=1.0)

# Value labels at the end of each bar
for bar, val in zip(bars, plot_df["Coefficient"]):
    pad   = 300 if val >= 0 else -300
    align = "left" if val >= 0 else "right"
    ax.text(val + pad, bar.get_y() + bar.get_height() / 2,
            f"{val:,.0f}", va="center", ha=align, fontsize=8.5)

ax.set_xlabel("Coefficient value (USD)", fontsize=12)
ax.set_title(f"{selected_name} -- Top {top_n} Coefficients by Magnitude\n"
             "Blue = raises SalePrice  |  Red = lowers SalePrice",
             fontsize=13, fontweight="bold")
ax.tick_params(axis="y", labelsize=9)
ax.tick_params(axis="x", labelsize=9)
# Add a light grid on the x-axis only for readability
ax.xaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("09_coefficient_plot.png", dpi=150)
plt.close()
print("\nSaved: 09_coefficient_plot.png")

# -- Plot 10: Variables 1-12 from M5_alt (3 rows x 4 columns) --
# Variables 1-8  (numeric)     : scatter + OLS regression line + R2 + equation
# Variables 9-12 (categorical) : box plot per category level (median line shown)
# This gives an honest visual for each variable type.
num_vars = m3_num                        # 8 numeric  (variables 1-8)
cat_vars = m3_cat                        # 4 categorical (variables 9-12)
all_12   = num_vars + cat_vars

print(f"\nPlot 10 panels -- M5_alt variables 1-12:")
for i, v in enumerate(all_12, 1):
    print(f"  {i:>2}. {v}")

fig, axes = plt.subplots(3, 4, figsize=(22, 15))

for ax, var in zip(axes.flat, all_12):

    ysp_full = df_feat["SalePrice"] / 1_000

    if var in num_vars:
        # ── Numeric panel: scatter + OLS line ──────────────────────────────
        x   = df_feat[var].dropna().values
        idx = df_feat[var].dropna().index
        ysp = ysp_full.loc[idx].values

        b, a  = np.polyfit(x, ysp, 1)
        r_val = np.corrcoef(x, ysp)[0, 1]
        r2    = r_val ** 2

        ax.scatter(x, ysp, alpha=0.20, color="steelblue", s=10, linewidths=0)
        x_line = np.linspace(x.min(), x.max(), 300)
        ax.plot(x_line, b * x_line + a, color="red", linewidth=2.0)

        sign   = "+" if a >= 0 else "-"
        eq_txt = f"y = {b:.1f}x {sign} {abs(a):.0f}\nR2 = {r2:.3f}"
        y_range = ysp.max() - ysp.min()
        x_range = x.max() - x.min()
        ax.text(x.min() + 0.03 * x_range,
                ysp.max() - 0.05 * y_range,
                eq_txt, fontsize=8.5, color="red",
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#cccccc", alpha=0.85))
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda val, _: f"${val:.0f}k")
        )

    else:
        # ── Categorical panel: box plot per category level ──────────────────
        col = df_feat[var].fillna("Missing")
        # Order categories by median SalePrice for readability
        order = (df_feat.assign(sp=ysp_full, cat=col)
                 .groupby("cat")["sp"].median()
                 .sort_values()
                 .index.tolist())
        groups = [ysp_full[col == lvl].values for lvl in order]
        bp = ax.boxplot(groups, patch_artist=True, medianprops=dict(color="red", linewidth=2))
        for patch in bp["boxes"]:
            patch.set_facecolor("#d0e4f7")
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels(order, rotation=30, ha="right", fontsize=7)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda val, _: f"${val:.0f}k")
        )

    ax.set_xlabel(var, fontsize=10)
    ax.set_ylabel("SalePrice ($k)", fontsize=10)
    ax.set_title(f"SalePrice vs {var}", fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)

fig.suptitle(
    "M5_alt Variables 1-12: SalePrice Relationships\n"
    "Numeric (1-8): scatter + OLS line  |  Categorical (9-12): box plot by category",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
plt.savefig("10_simple_regressions.png", dpi=150)
plt.close()
print("Saved: 10_simple_regressions.png")

# -- Sample prediction --
print("""
SAMPLE PREDICTION -- 'Reasonable mid-market profile based on data medians':
  OverallQual   : 6   (slightly above average -- median is 6)
  GrLivArea     : 1,400 sq ft  (median GrLivArea ~1,464)
  TotalBsmtSF   : 800 sq ft
  GarageCars    : 2
  YearBuilt     : 1985
  LotArea       : 9,000 sq ft  (median ~9,478)
  FullBath      : 2
  TotRmsAbvGrd  : 7
  Neighborhood  : NAmes (North Ames -- largest, most typical neighbourhood)
  MSZoning      : RL   (Residential Low Density -- most common, 79 % of data)
  KitchenQual   : TA   (Typical/Average -- modal category)
  ExterQual     : TA
  OverallCond   : 5   (average)
""")

# Build the sample row -- fill all columns that may be referenced by the pipeline
sample = {col: np.nan for col in df_feat.columns}
sample.update({
    # Core numeric features
    "OverallQual"    : 6,
    "GrLivArea"      : 1_400,
    "TotalBsmtSF"    : 800,
    "GarageCars"     : 2,
    "YearBuilt"      : 1_985,
    "LotArea"        : 9_000,
    "FullBath"       : 2,
    "TotRmsAbvGrd"   : 7,
    # Categorical features
    "Neighborhood"   : "NAmes",
    "MSZoning"       : "RL",
    "KitchenQual"    : "TA",
    "ExterQual"      : "TA",
    # For interaction / squared terms (derived columns need values too)
    "OverallCond"    : 5,
    "GrLivArea_sq"   : 1_400 ** 2,
    "YearBuilt_sq"   : 1_985 ** 2,
    "LotArea_sq"     : 9_000 ** 2,
    "NridgHt_flag"   : 0,                       # NAmes != NridgHt
    "Qual_x_GrLiv"   : 6 * 1_400,
    "NridgHt_x_GrLiv": 0,
    "YrBlt_x_Cond"   : 1_985 * 5,
})
sample_df = pd.DataFrame([sample])

predicted_price = final_pipe.predict(sample_df)[0]
low  = predicted_price - selected_mae
high = predicted_price + selected_mae

print(f"  Point estimate      : ${predicted_price:,.0f}")
print(f"  Reported as range   : ${low:,.0f}  to  ${high:,.0f}")
print(f"  (+-${selected_mae:,.0f} CV MAE -- slide 45/47: never report a bare point estimate)")

# -- Residual and predicted-vs-actual plots (in-sample, full refit) --
y_pred_full = final_pipe.predict(X_df)
residuals   = y - y_pred_full

# Plot 03: Residuals vs Predicted
fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(y_pred_full / 1_000, residuals / 1_000,
           alpha=0.25, color="steelblue", s=12)
ax.axhline(0, color="red", linestyle="--", linewidth=1.8)
ax.set_xlabel("Predicted Sale Price ($ thousands)", fontsize=12)
ax.set_ylabel("Residual ($ thousands)", fontsize=12)
ax.set_title(f"Residual Plot -- {selected_name} (in-sample, full refit)\n"
             "Widening spread at high prices indicates heteroskedasticity",
             fontsize=12)
plt.tight_layout()
plt.savefig("03_residual_plot.png", dpi=150)
plt.close()
print("\nSaved: 03_residual_plot.png")

# Plot 04: Predicted vs Actual
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y / 1_000, y_pred_full / 1_000, alpha=0.30, color="steelblue", s=12)
lo = min(y.min(), y_pred_full.min()) / 1_000 - 10
hi = max(y.max(), y_pred_full.max()) / 1_000 + 10
ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.8, label="Perfect prediction (45-degree line)")
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_xlabel("Actual Sale Price ($ thousands)", fontsize=12)
ax.set_ylabel("Predicted Sale Price ($ thousands)", fontsize=12)
ax.set_title(f"Predicted vs Actual -- {selected_name} (in-sample)", fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("04_predicted_vs_actual.png", dpi=150)
plt.close()
print("Saved: 04_predicted_vs_actual.png")

# -- Final summary --
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print("""
Output files:
  01_target_distribution.png   -- SalePrice histogram with mean/median lines
  02_top_predictor_scatter.png -- OverallQual vs SalePrice scatter
  03_residual_plot.png         -- residuals vs predicted (final model)
  04_predicted_vs_actual.png   -- predicted vs actual with 45-degree line
  05_model_comparison.png      -- CV MAE bar chart M0 to M5 + M5_alt
  06_model_comparison_r2.png   -- CV Adjusted R2 bar chart M0 to M5 + M5_alt
  07_summary_statistics.png    -- summary stats table (mean, std, min, max) for key variables
  08_correlation_plots.png     -- 2x2 scatter plots: SalePrice vs GrLivArea, OverallQual, YearBuilt, TotalBsmtSF
  09_coefficient_plot.png      -- horizontal bar chart: top 20 coefficients (blue = raises price, red = lowers price)
  10_simple_regressions.png    -- 3x4 grid: simple OLS for SalePrice vs all 11 M5_alt numeric variables
""")
