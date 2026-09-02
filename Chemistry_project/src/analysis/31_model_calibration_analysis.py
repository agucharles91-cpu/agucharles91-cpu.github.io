"""
SCRIPT 31 — MODEL CALIBRATION & ERROR STRATIFICATION

Purpose
-------
Assess whether the nonlinear Gradient Boosting model is calibrated across
the observed and predicted solubility range.

The analysis examines:

1. Overall observed-vs-predicted calibration
2. Calibration slope and intercept
3. Prediction compression
4. Calibration by observed solubility region
5. Calibration across predicted-solubility bins
6. Directional prediction bias
7. Compound-level calibration across repeated held-out predictions
8. Relationship between prediction uncertainty and calibration error

Important methodological note
-----------------------------
This script uses the repeated scaffold-aware held-out predictions generated
by Script 26.

The repeated predictions are aggregated at compound level where appropriate.

Calibration is descriptive. A calibration slope below 1 indicates that
predictions vary less than observed values, while systematic regional
residuals indicate conditional bias.

This script does not recalibrate the model or alter the underlying model.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# =====================================================================
# PATHS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "molecular_features.csv"
)

PREDICTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nonlinear_residual_analysis.csv"
)

UNCERTAINTY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "prediction_uncertainty_compound_level.csv"
)

CALIBRATION_REGION_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "calibration_by_solubility_region.csv"
)

CALIBRATION_BIN_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "calibration_by_prediction_bin.csv"
)

DIRECTION_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "calibration_directional_error.csv"
)

COMPOUND_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_calibration_compound_level.csv"
)

UNCERTAINTY_CALIBRATION_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "calibration_by_uncertainty.csv"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "model_calibration_analysis.txt"
)


# =====================================================================
# CONFIGURATION
# =====================================================================

TARGET = "Solubility"
PREDICTION = "predicted_solubility"
ID = "ID"

SOLUBILITY_BINS = [-np.inf, -6, -4, -2, 0, np.inf]
SOLUBILITY_LABELS = [
    "< -6",
    "-6 to < -4",
    "-4 to < -2",
    "-2 to < 0",
    ">= 0",
]

N_PREDICTION_BINS = 10


# =====================================================================
# HEADER
# =====================================================================

print("=" * 70)
print("SCRIPT 31 — MODEL CALIBRATION & ERROR STRATIFICATION")
print("=" * 70)

print(f"Project root: {PROJECT_ROOT}")


# =====================================================================
# LOAD MOLECULAR FEATURES
# =====================================================================

print("\n" + "=" * 70)
print("Loading molecular features")
print("=" * 70)

features = pd.read_csv(FEATURE_FILE)

print(f"Feature shape: {features.shape}")

if "standard_analytical_domain" not in features.columns:
    raise ValueError(
        "standard_analytical_domain missing from molecular_features.csv"
    )

population_c = features[
    features["standard_analytical_domain"] == True
].copy()

print(f"Population C rows: {len(population_c):,}")

if len(population_c) != 8643:
    raise ValueError(
        f"Expected Population C = 8643 rows, found {len(population_c)}"
    )

print("Population C verified.")


# =====================================================================
# LOAD SCRIPT 26 PREDICTIONS
# =====================================================================

print("\n" + "=" * 70)
print("Loading Script 26 repeated held-out predictions")
print("=" * 70)

print(f"Prediction source: {PREDICTION_FILE}")

pred = pd.read_csv(PREDICTION_FILE)

print(f"Prediction dataset shape: {pred.shape}")

required_columns = [
    "ID",
    "SMILES",
    "scaffold",
    "Solubility",
    "predicted_solubility",
    "repetition",
    "seed",
]

missing = [
    col for col in required_columns
    if col not in pred.columns
]

if missing:
    raise ValueError(
        f"Missing required prediction columns: {missing}"
    )

print("\nColumns found:")

for col in pred.columns:
    print(f"  {col}")


# =====================================================================
# INPUT VALIDATION
# =====================================================================

print("\n" + "=" * 70)
print("INPUT VALIDATION")
print("=" * 70)

pred = pred.dropna(
    subset=[
        ID,
        TARGET,
        PREDICTION,
    ]
).copy()

print(f"Valid prediction rows: {len(pred):,}")
print(f"Unique compounds represented: {pred[ID].nunique():,}")

if pred[ID].nunique() != 8160:
    raise ValueError(
        f"Expected 8,160 represented compounds, "
        f"found {pred[ID].nunique()}"
    )


# =====================================================================
# RECOMPUTE DIAGNOSTICS
# =====================================================================

print("\n" + "=" * 70)
print("Recomputing prediction diagnostics")
print("=" * 70)

pred["residual_recomputed"] = (
    pred[TARGET] - pred[PREDICTION]
)

pred["absolute_error_recomputed"] = (
    pred["residual_recomputed"].abs()
)

pred["squared_error_recomputed"] = (
    pred["residual_recomputed"] ** 2
)

pred["error_direction_recomputed"] = np.where(
    pred["residual_recomputed"] > 0,
    "Underprediction",
    "Overprediction",
)

print("Prediction diagnostics recomputed.")


# =====================================================================
# OVERALL CALIBRATION
# =====================================================================

print("\n" + "=" * 70)
print("OVERALL CALIBRATION")
print("=" * 70)

y_true = pred[TARGET].to_numpy()
y_pred = pred[PREDICTION].to_numpy()

overall_mae = mean_absolute_error(
    y_true,
    y_pred,
)

overall_rmse = np.sqrt(
    mean_squared_error(
        y_true,
        y_pred,
    )
)

overall_r2 = r2_score(
    y_true,
    y_pred,
)

pearson_r, pearson_p = pearsonr(
    y_true,
    y_pred,
)

spearman_r, spearman_p = spearmanr(
    y_true,
    y_pred,
)

mean_residual = pred["residual_recomputed"].mean()
median_residual = pred["residual_recomputed"].median()

print(f"Observations: {len(pred):,}")
print(f"Mean observed solubility: {y_true.mean():+.4f}")
print(f"Mean predicted solubility: {y_pred.mean():+.4f}")
print(f"Mean residual: {mean_residual:+.4f}")
print(f"Median residual: {median_residual:+.4f}")
print(f"MAE: {overall_mae:.4f}")
print(f"RMSE: {overall_rmse:.4f}")
print(f"R²: {overall_r2:.4f}")
print(
    f"Observed vs predicted Pearson r: "
    f"{pearson_r:+.4f} p={pearson_p:.4e}"
)
print(
    f"Observed vs predicted Spearman r: "
    f"{spearman_r:+.4f} p={spearman_p:.4e}"
)


# =====================================================================
# CALIBRATION REGRESSION
# =====================================================================

print("\n" + "=" * 70)
print("CALIBRATION REGRESSION")
print("=" * 70)

# Predicted = intercept + slope * Observed

X_cal = sm.add_constant(
    pred[TARGET]
)

calibration_model = sm.OLS(
    pred[PREDICTION],
    X_cal,
).fit()

# Convert parameters safely to NumPy values.
calibration_params = np.asarray(
    calibration_model.params
)

calibration_intercept = float(
    calibration_params[0]
)

calibration_slope = float(
    calibration_params[1]
)

calibration_slope_p = float(
    calibration_model.pvalues.iloc[1]
    if hasattr(calibration_model.pvalues, "iloc")
    else np.asarray(calibration_model.pvalues)[1]
)

print(
    "Calibration equation:"
)

print(
    f"Predicted = "
    f"{calibration_intercept:+.4f} + "
    f"{calibration_slope:.4f} × Observed"
)

print(
    f"Calibration slope: "
    f"{calibration_slope:.4f}"
)

print(
    f"Calibration intercept: "
    f"{calibration_intercept:+.4f}"
)

print(
    f"Calibration slope p-value: "
    f"{calibration_slope_p:.4e}"
)

if calibration_slope < 1:
    print(
        "\nInterpretation:"
        "\n  Prediction compression detected: "
        "predictions vary less than observed values."
    )

elif calibration_slope > 1:
    print(
        "\nInterpretation:"
        "\n  Prediction expansion detected: "
        "predictions vary more than observed values."
    )

else:
    print(
        "\nInterpretation:"
        "\n  Calibration slope is approximately 1."
    )


# =====================================================================
# CALIBRATION BY OBSERVED SOLUBILITY REGION
# =====================================================================

print("\n" + "=" * 70)
print("CALIBRATION BY OBSERVED SOLUBILITY REGION")
print("=" * 70)

pred["observed_solubility_region_cal"] = pd.cut(
    pred[TARGET],
    bins=SOLUBILITY_BINS,
    labels=SOLUBILITY_LABELS,
    right=False,
)

region_rows = []

for region, group in pred.groupby(
    "observed_solubility_region_cal",
    observed=True,
):

    observed = group[TARGET].to_numpy()
    predicted = group[PREDICTION].to_numpy()
    residual = group["residual_recomputed"].to_numpy()

    region_rows.append(
        {
            "region": str(region),
            "n": len(group),
            "mean_observed": observed.mean(),
            "mean_predicted": predicted.mean(),
            "mean_residual": residual.mean(),
            "median_residual": np.median(residual),
            "mean_absolute_error": np.mean(np.abs(residual)),
            "median_absolute_error": np.median(np.abs(residual)),
            "rmse": np.sqrt(np.mean(residual ** 2)),
            "r2": (
                r2_score(observed, predicted)
                if len(group) > 1
                else np.nan
            ),
        }
    )

region_summary = pd.DataFrame(region_rows)

print(region_summary.to_string(index=False))

region_summary.to_csv(
    CALIBRATION_REGION_OUTPUT,
    index=False,
)

print(
    f"\nSaved: {CALIBRATION_REGION_OUTPUT}"
)


# =====================================================================
# CALIBRATION BY PREDICTED SOLUBILITY
# =====================================================================

print("\n" + "=" * 70)
print("CALIBRATION BY PREDICTED SOLUBILITY")
print("=" * 70)

pred["prediction_bin"] = pd.qcut(
    pred[PREDICTION],
    q=N_PREDICTION_BINS,
    duplicates="drop",
)

bin_rows = []

for region, group in pred.groupby(
    "prediction_bin",
    observed=True,
):

    observed = group[TARGET].to_numpy()
    predicted = group[PREDICTION].to_numpy()
    residual = group["residual_recomputed"].to_numpy()

    mean_observed = observed.mean()
    mean_predicted = predicted.mean()

    if abs(mean_observed) > 1e-12:
        prediction_bias_ratio = (
            mean_predicted / mean_observed
        )
    else:
        prediction_bias_ratio = np.nan

    bin_rows.append(
        {
            "n": len(group),
            "mean_observed": mean_observed,
            "mean_predicted": mean_predicted,
            "mean_residual": residual.mean(),
            "median_residual": np.median(residual),
            "mean_absolute_error": np.mean(np.abs(residual)),
            "median_absolute_error": np.median(np.abs(residual)),
            "rmse": np.sqrt(np.mean(residual ** 2)),
            "r2": (
                r2_score(observed, predicted)
                if len(group) > 1
                else np.nan
            ),
            "prediction_bias_ratio": prediction_bias_ratio,
            "region": str(region),
        }
    )

bin_summary = pd.DataFrame(bin_rows)

print(bin_summary.to_string(index=False))

bin_summary.to_csv(
    CALIBRATION_BIN_OUTPUT,
    index=False,
)

print(
    f"\nSaved: {CALIBRATION_BIN_OUTPUT}"
)


# =====================================================================
# DIRECTIONAL ERROR
# =====================================================================

print("\n" + "=" * 70)
print("DIRECTIONAL ERROR")
print("=" * 70)

direction_summary = (
    pred
    .groupby("error_direction_recomputed")
    .agg(
        compound_or_row_count=(ID, "size"),
        mean_absolute_error=(
            "absolute_error_recomputed",
            "mean",
        ),
        mean_residual=(
            "residual_recomputed",
            "mean",
        ),
    )
    .reset_index()
)

direction_summary["percentage"] = (
    direction_summary["compound_or_row_count"]
    / len(pred)
    * 100
)

print(direction_summary.to_string(index=False))

direction_summary.to_csv(
    DIRECTION_OUTPUT,
    index=False,
)

print(
    f"\nSaved: {DIRECTION_OUTPUT}"
)


# =====================================================================
# AGGREGATE REPEATED PREDICTIONS
# =====================================================================

print("\n" + "=" * 70)
print("AGGREGATING REPEATED PREDICTIONS")
print("=" * 70)

compound = (
    pred
    .groupby(ID)
    .agg(
        observed_solubility=(TARGET, "first"),
        mean_prediction=(PREDICTION, "mean"),
        median_prediction=(PREDICTION, "median"),
        prediction_sd=(PREDICTION, "std"),
        prediction_min=(PREDICTION, "min"),
        prediction_max=(PREDICTION, "max"),
        prediction_count=(PREDICTION, "count"),
        mean_residual=("residual_recomputed", "mean"),
        mean_absolute_error=(
            "absolute_error_recomputed",
            "mean",
        ),
    )
    .reset_index()
)

compound["prediction_range"] = (
    compound["prediction_max"]
    - compound["prediction_min"]
)

compound["compound_absolute_error"] = (
    compound["observed_solubility"]
    - compound["mean_prediction"]
).abs()

compound["compound_residual"] = (
    compound["observed_solubility"]
    - compound["mean_prediction"]
)

compound["compound_squared_error"] = (
    compound["compound_residual"] ** 2
)

compound["compound_error_direction"] = np.where(
    compound["compound_residual"] > 0,
    "Underprediction",
    "Overprediction",
)

print(
    f"Compound-level observations: "
    f"{len(compound):,}"
)

compound.to_csv(
    COMPOUND_OUTPUT,
    index=False,
)

print(
    f"Saved: {COMPOUND_OUTPUT}"
)


# =====================================================================
# COMPOUND-LEVEL CALIBRATION
# =====================================================================

print("\n" + "=" * 70)
print("COMPOUND-LEVEL CALIBRATION")
print("=" * 70)

compound_X = sm.add_constant(
    compound["observed_solubility"]
)

compound_calibration = sm.OLS(
    compound["mean_prediction"],
    compound_X,
).fit()

# ---------------------------------------------------------------------
# IMPORTANT:
# statsmodels may return params as either a Series or ndarray depending
# on the input structure/version. Access positionally to avoid the
# previous IndexError.
# ---------------------------------------------------------------------

compound_params = np.asarray(
    compound_calibration.params
)

compound_intercept = float(
    compound_params[0]
)

compound_slope = float(
    compound_params[1]
)

compound_pvalues = np.asarray(
    compound_calibration.pvalues
)

compound_slope_p = float(
    compound_pvalues[1]
)

compound_mae = mean_absolute_error(
    compound["observed_solubility"],
    compound["mean_prediction"],
)

compound_rmse = np.sqrt(
    mean_squared_error(
        compound["observed_solubility"],
        compound["mean_prediction"],
    )
)

compound_r2 = r2_score(
    compound["observed_solubility"],
    compound["mean_prediction"],
)

compound_pearson_r, compound_pearson_p = pearsonr(
    compound["observed_solubility"],
    compound["mean_prediction"],
)

compound_spearman_r, compound_spearman_p = spearmanr(
    compound["observed_solubility"],
    compound["mean_prediction"],
)

print(
    "Calibration equation:"
)

print(
    f"Mean prediction = "
    f"{compound_intercept:+.4f} + "
    f"{compound_slope:.4f} × Observed"
)

print(
    f"Calibration slope: {compound_slope:.4f}"
)

print(
    f"Calibration intercept: {compound_intercept:+.4f}"
)

print(
    f"Calibration slope p-value: "
    f"{compound_slope_p:.4e}"
)

print(
    f"MAE: {compound_mae:.4f}"
)

print(
    f"RMSE: {compound_rmse:.4f}"
)

print(
    f"R²: {compound_r2:.4f}"
)

print(
    f"Pearson r: "
    f"{compound_pearson_r:+.4f} "
    f"p={compound_pearson_p:.4e}"
)

print(
    f"Spearman r: "
    f"{compound_spearman_r:+.4f} "
    f"p={compound_spearman_p:.4e}"
)


# =====================================================================
# COMPOUND-LEVEL ERROR STRATIFICATION
# =====================================================================

print("\n" + "=" * 70)
print("COMPOUND-LEVEL ERROR STRATIFICATION")
print("=" * 70)

compound["observed_solubility_region"] = pd.cut(
    compound["observed_solubility"],
    bins=SOLUBILITY_BINS,
    labels=SOLUBILITY_LABELS,
    right=False,
)

compound_region_rows = []

for region, group in compound.groupby(
    "observed_solubility_region",
    observed=True,
):

    residual = group["compound_residual"].to_numpy()

    compound_region_rows.append(
        {
            "region": str(region),
            "compound_count": len(group),
            "mean_observed": group[
                "observed_solubility"
            ].mean(),
            "mean_prediction": group[
                "mean_prediction"
            ].mean(),
            "mean_residual": residual.mean(),
            "median_residual": np.median(residual),
            "mean_absolute_error": np.mean(np.abs(residual)),
            "median_absolute_error": np.median(np.abs(residual)),
            "rmse": np.sqrt(np.mean(residual ** 2)),
            "mean_prediction_sd": group[
                "prediction_sd"
            ].mean(),
        }
    )

compound_region_summary = pd.DataFrame(
    compound_region_rows
)

print(
    compound_region_summary.to_string(index=False)
)


# =====================================================================
# CALIBRATION BY PREDICTION UNCERTAINTY
# =====================================================================

print("\n" + "=" * 70)
print("CALIBRATION BY PREDICTION UNCERTAINTY")
print("=" * 70)

compound["uncertainty_bin"] = pd.qcut(
    compound["prediction_sd"],
    q=4,
    labels=[
        "Lowest uncertainty",
        "Low-moderate uncertainty",
        "Moderate-high uncertainty",
        "Highest uncertainty",
    ],
    duplicates="drop",
)

uncertainty_rows = []

for region, group in compound.groupby(
    "uncertainty_bin",
    observed=True,
):

    residual = group["compound_residual"].to_numpy()

    uncertainty_rows.append(
        {
            "uncertainty_region": str(region),
            "compound_count": len(group),
            "mean_prediction_sd": group[
                "prediction_sd"
            ].mean(),
            "median_prediction_sd": group[
                "prediction_sd"
            ].median(),
            "mean_absolute_error": np.mean(
                np.abs(residual)
            ),
            "median_absolute_error": np.median(
                np.abs(residual)
            ),
            "rmse": np.sqrt(
                np.mean(residual ** 2)
            ),
            "mean_residual": residual.mean(),
        }
    )

uncertainty_summary = pd.DataFrame(
    uncertainty_rows
)

print(
    uncertainty_summary.to_string(index=False)
)

uncertainty_summary.to_csv(
    UNCERTAINTY_CALIBRATION_OUTPUT,
    index=False,
)

print(
    f"\nSaved: {UNCERTAINTY_CALIBRATION_OUTPUT}"
)


# =====================================================================
# REPORT
# =====================================================================

print("\n" + "=" * 70)
print("WRITING REPORT")
print("=" * 70)

report_lines = []

report_lines.append(
    "SCRIPT 31 — MODEL CALIBRATION & ERROR STRATIFICATION"
)

report_lines.append("=" * 70)

report_lines.append("")
report_lines.append("PURPOSE")
report_lines.append("-------")
report_lines.append(
    "Assess calibration of the nonlinear Gradient Boosting "
    "model across observed and predicted solubility ranges, "
    "and determine whether systematic error varies across "
    "solubility and prediction-uncertainty strata."
)

report_lines.append("")
report_lines.append("DATA DESIGN")
report_lines.append("-----------")
report_lines.append(
    f"Population C: {len(population_c):,} compounds"
)
report_lines.append(
    f"Repeated held-out prediction rows: {len(pred):,}"
)
report_lines.append(
    f"Unique compounds represented: {pred[ID].nunique():,}"
)
report_lines.append(
    "Prediction source: Script 26 nonlinear residual analysis"
)

report_lines.append("")
report_lines.append("OVERALL CALIBRATION")
report_lines.append("-------------------")
report_lines.append(
    f"Observations: {len(pred):,}"
)
report_lines.append(
    f"Mean observed solubility: {y_true.mean():+.4f}"
)
report_lines.append(
    f"Mean predicted solubility: {y_pred.mean():+.4f}"
)
report_lines.append(
    f"Mean residual: {mean_residual:+.4f}"
)
report_lines.append(
    f"Median residual: {median_residual:+.4f}"
)
report_lines.append(
    f"MAE: {overall_mae:.4f}"
)
report_lines.append(
    f"RMSE: {overall_rmse:.4f}"
)
report_lines.append(
    f"R²: {overall_r2:.4f}"
)
report_lines.append(
    f"Pearson r: {pearson_r:+.4f}, p={pearson_p:.4e}"
)
report_lines.append(
    f"Spearman r: {spearman_r:+.4f}, "
    f"p={spearman_p:.4e}"
)

report_lines.append("")
report_lines.append("CALIBRATION REGRESSION")
report_lines.append("----------------------")
report_lines.append(
    f"Calibration equation: "
    f"Predicted = {calibration_intercept:+.4f} "
    f"+ {calibration_slope:.4f} × Observed"
)
report_lines.append(
    f"Calibration slope: {calibration_slope:.4f}"
)
report_lines.append(
    f"Calibration intercept: {calibration_intercept:+.4f}"
)
report_lines.append(
    f"Calibration slope p-value: "
    f"{calibration_slope_p:.4e}"
)

if calibration_slope < 1:
    report_lines.append(
        "Interpretation: prediction compression detected; "
        "predictions vary less than observed values."
    )
elif calibration_slope > 1:
    report_lines.append(
        "Interpretation: prediction expansion detected; "
        "predictions vary more than observed values."
    )
else:
    report_lines.append(
        "Interpretation: calibration slope is approximately 1."
    )

report_lines.append("")
report_lines.append("CALIBRATION BY OBSERVED SOLUBILITY")
report_lines.append("----------------------------------")
report_lines.append(
    region_summary.to_string(index=False)
)

report_lines.append("")
report_lines.append("CALIBRATION BY PREDICTED SOLUBILITY")
report_lines.append("-----------------------------------")
report_lines.append(
    bin_summary.to_string(index=False)
)

report_lines.append("")
report_lines.append("DIRECTIONAL ERROR")
report_lines.append("-----------------")
report_lines.append(
    direction_summary.to_string(index=False)
)

report_lines.append("")
report_lines.append("COMPOUND-LEVEL CALIBRATION")
report_lines.append("---------------------------")
report_lines.append(
    f"Compound observations: {len(compound):,}"
)
report_lines.append(
    f"Calibration equation: "
    f"Mean prediction = {compound_intercept:+.4f} "
    f"+ {compound_slope:.4f} × Observed"
)
report_lines.append(
    f"Calibration slope: {compound_slope:.4f}"
)
report_lines.append(
    f"Calibration intercept: {compound_intercept:+.4f}"
)
report_lines.append(
    f"Calibration slope p-value: "
    f"{compound_slope_p:.4e}"
)
report_lines.append(
    f"MAE: {compound_mae:.4f}"
)
report_lines.append(
    f"RMSE: {compound_rmse:.4f}"
)
report_lines.append(
    f"R²: {compound_r2:.4f}"
)
report_lines.append(
    f"Pearson r: {compound_pearson_r:+.4f}, "
    f"p={compound_pearson_p:.4e}"
)
report_lines.append(
    f"Spearman r: {compound_spearman_r:+.4f}, "
    f"p={compound_spearman_p:.4e}"
)

report_lines.append("")
report_lines.append("COMPOUND-LEVEL CALIBRATION BY SOLUBILITY")
report_lines.append("-----------------------------------------")
report_lines.append(
    compound_region_summary.to_string(index=False)
)

report_lines.append("")
report_lines.append("CALIBRATION BY PREDICTION UNCERTAINTY")
report_lines.append("-------------------------------------")
report_lines.append(
    uncertainty_summary.to_string(index=False)
)

report_lines.append("")
report_lines.append("INTERPRETATION FRAMEWORK")
report_lines.append("------------------------")
report_lines.append(
    "A calibration slope below 1 indicates prediction compression: "
    "the model reproduces the central range more strongly than the "
    "extreme observed values."
)
report_lines.append(
    "Regional residual patterns identify conditional prediction bias "
    "that can be hidden by a near-zero overall mean residual."
)
report_lines.append(
    "High error in a particular solubility region does not by itself "
    "establish a causal mechanism."
)
report_lines.append(
    "Compound-level calibration summarizes repeated scaffold-aware "
    "held-out predictions rather than fitted training predictions."
)

REPORT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "\n".join(report_lines)
    )

print(
    f"Report saved: {REPORT_FILE}"
)


# =====================================================================
# FINAL OUTPUT SUMMARY
# =====================================================================

print("\n" + "=" * 70)
print("SCRIPT 31 COMPLETE")
print("=" * 70)

print(
    f"Population C: {len(population_c):,}"
)

print(
    f"Prediction rows: {len(pred):,}"
)

print(
    f"Compounds analysed: {len(compound):,}"
)

print(
    f"Overall calibration slope: "
    f"{calibration_slope:.4f}"
)

print(
    f"Compound-level calibration slope: "
    f"{compound_slope:.4f}"
)

print("\nOutput files:")

print(
    f"  {CALIBRATION_REGION_OUTPUT}"
)

print(
    f"  {CALIBRATION_BIN_OUTPUT}"
)

print(
    f"  {DIRECTION_OUTPUT}"
)

print(
    f"  {COMPOUND_OUTPUT}"
)

print(
    f"  {UNCERTAINTY_CALIBRATION_OUTPUT}"
)

print(
    f"  {REPORT_FILE}"
)

print("=" * 70)