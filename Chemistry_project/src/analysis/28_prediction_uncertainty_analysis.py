"""
SCRIPT 28 — PREDICTION UNCERTAINTY & STABILITY ANALYSIS

Purpose
-------
Quantify how stable the Gradient Boosting predictions are across repeated
scaffold-aware held-out evaluations and determine whether prediction
instability is associated with prediction error, solubility regime,
molecular-property regions, or scaffold representation.

Data source
-----------
Script 26 repeated scaffold-aware held-out predictions.

Important interpretation
------------------------
Prediction SD and prediction range measure instability across repeated
train/test splits. They are not formal probabilistic prediction intervals
and should not be interpreted as calibrated confidence intervals.

All error quantities are calculated explicitly at the compound level from
the repeated held-out predictions.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


# ======================================================================
# PROJECT PATHS
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "molecular_features.csv"
)

RESIDUAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nonlinear_residual_analysis.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================================
# MODEL FEATURES
# ======================================================================

FEATURE_COLUMNS = [
    "rdkit_molwt",
    "rdkit_mollogp",
    "rdkit_ring_count",
    "rdkit_aromatic_rings",
    "rdkit_rotatable_bonds",
    "rdkit_fraction_csp3",
]


# ======================================================================
# HELPER FUNCTIONS
# ======================================================================

def safe_pearson(x, y):
    """Return Pearson correlation and p-value, handling invalid inputs."""

    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < 3:
        return np.nan, np.nan

    if x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return np.nan, np.nan

    result = pearsonr(x[mask], y[mask])

    return result.statistic, result.pvalue


def safe_spearman(x, y):
    """Return Spearman correlation and p-value, handling invalid inputs."""

    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < 3:
        return np.nan, np.nan

    if x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return np.nan, np.nan

    result = spearmanr(x[mask], y[mask])

    return result.statistic, result.pvalue


def region_summary(df, region_column):
    """
    Calculate compound-level uncertainty and error metrics by region.

    Required columns:
        prediction_sd
        prediction_range
        absolute_error
        residual
    """

    rows = []

    for region, group in df.groupby(
        region_column,
        observed=False,
        dropna=False,
    ):

        if len(group) == 0:
            continue

        residual_values = pd.to_numeric(
            group["residual"],
            errors="coerce",
        ).to_numpy()

        absolute_error_values = pd.to_numeric(
            group["absolute_error"],
            errors="coerce",
        ).to_numpy()

        prediction_sd_values = pd.to_numeric(
            group["prediction_sd"],
            errors="coerce",
        ).to_numpy()

        prediction_range_values = pd.to_numeric(
            group["prediction_range"],
            errors="coerce",
        ).to_numpy()

        rows.append(
            {
                "region": region,
                "compound_count": len(group),

                "mean_prediction_sd": np.nanmean(
                    prediction_sd_values
                ),
                "median_prediction_sd": np.nanmedian(
                    prediction_sd_values
                ),

                "mean_prediction_range": np.nanmean(
                    prediction_range_values
                ),
                "median_prediction_range": np.nanmedian(
                    prediction_range_values
                ),

                "mean_absolute_error": np.nanmean(
                    absolute_error_values
                ),
                "median_absolute_error": np.nanmedian(
                    absolute_error_values
                ),

                "rmse": np.sqrt(
                    np.nanmean(residual_values ** 2)
                ),

                "mean_residual": np.nanmean(
                    residual_values
                ),

                "underprediction_pct": (
                    np.mean(residual_values > 0) * 100
                ),

                "overprediction_pct": (
                    np.mean(residual_values < 0) * 100
                ),
            }
        )

    return pd.DataFrame(rows)


def feature_region_summary(
    df,
    feature,
    n_bins=6,
):
    """
    Divide a molecular descriptor into quantile regions and calculate
    prediction uncertainty and error by region.
    """

    working = df[
        [
            feature,
            "prediction_sd",
            "prediction_range",
            "absolute_error",
            "residual",
        ]
    ].copy()

    working[feature] = pd.to_numeric(
        working[feature],
        errors="coerce",
    )

    working = working.dropna(subset=[feature])

    if len(working) < n_bins:
        return pd.DataFrame()

    try:
        working["region"] = pd.qcut(
            working[feature],
            q=n_bins,
            duplicates="drop",
        )
    except ValueError:
        return pd.DataFrame()

    summary = region_summary(
        working,
        "region",
    )

    summary.insert(
        0,
        "feature",
        feature,
    )

    return summary


# ======================================================================
# HEADER
# ======================================================================

print("=" * 70)
print("SCRIPT 28 — PREDICTION UNCERTAINTY & STABILITY ANALYSIS")
print("=" * 70)

print(f"Project root: {PROJECT_ROOT}")


# ======================================================================
# LOAD MOLECULAR FEATURES
# ======================================================================

print("\n" + "=" * 70)
print("Loading molecular features")
print("=" * 70)

features = pd.read_csv(FEATURE_PATH)

print(f"Feature shape: {features.shape}")

required_feature_columns = [
    "ID"
] + FEATURE_COLUMNS

missing_features = [
    column
    for column in required_feature_columns
    if column not in features.columns
]

if missing_features:
    raise ValueError(
        "Missing required columns from molecular_features.csv:\n"
        + "\n".join(missing_features)
    )

features = features[
    required_feature_columns
].copy()

features["ID"] = features["ID"].astype(str)

if features["ID"].duplicated().any():
    raise ValueError(
        "Duplicate compound IDs found in molecular_features.csv."
    )

print(
    f"Unique compounds in molecular features: "
    f"{features['ID'].nunique():,}"
)


# ======================================================================
# LOAD SCRIPT 26 PREDICTIONS
# ======================================================================

print("\n" + "=" * 70)
print("Loading Script 26 repeated held-out predictions")
print("=" * 70)

predictions = pd.read_csv(RESIDUAL_PATH)

print(
    f"Residual dataset shape: {predictions.shape}"
)

print("\nColumns found:")

for column in predictions.columns:
    print(f"  {column}")


# ======================================================================
# RESOLVE SCRIPT 26 COLUMNS
# ======================================================================

required_prediction_columns = [
    "ID",
    "predicted_solubility",
]

missing_predictions = [
    column
    for column in required_prediction_columns
    if column not in predictions.columns
]

if missing_predictions:
    raise ValueError(
        "Missing required columns from nonlinear_residual_analysis.csv:\n"
        + "\n".join(missing_predictions)
    )


predictions["ID"] = predictions["ID"].astype(str)

predictions["predicted_solubility"] = pd.to_numeric(
    predictions["predicted_solubility"],
    errors="coerce",
)

predictions = predictions.dropna(
    subset=[
        "ID",
        "predicted_solubility",
    ]
).copy()

print(
    f"Valid prediction rows: {len(predictions):,}"
)

print(
    f"Unique compounds represented: "
    f"{predictions['ID'].nunique():,}"
)


# ======================================================================
# INPUT VALIDATION
# ======================================================================

print("\n" + "=" * 70)
print("INPUT VALIDATION")
print("=" * 70)

feature_ids = set(features["ID"])

prediction_ids = set(
    predictions["ID"]
)

unmatched_ids = prediction_ids - feature_ids

if unmatched_ids:
    raise ValueError(
        f"{len(unmatched_ids):,} prediction IDs do not match "
        "molecular_features.csv."
    )

print(
    "All prediction IDs successfully matched "
    "to molecular features."
)


# ======================================================================
# RESOLVE OBSERVED SOLUBILITY
# ======================================================================

if "Solubility" in predictions.columns:

    predictions["observed_solubility"] = pd.to_numeric(
        predictions["Solubility"],
        errors="coerce",
    )

elif "observed_solubility" in predictions.columns:

    predictions["observed_solubility"] = pd.to_numeric(
        predictions["observed_solubility"],
        errors="coerce",
    )

else:

    raise ValueError(
        "Script 26 output does not contain an observed solubility column."
    )


predictions = predictions.dropna(
    subset=["observed_solubility"]
).copy()


# ======================================================================
# RECOMPUTE ERROR FIELDS
# ======================================================================

print("\n" + "=" * 70)
print("Recomputing prediction errors")
print("=" * 70)

# IMPORTANT:
# Do not rely on residual / absolute_error columns from Script 26.
# Recalculate them directly from observed and predicted values.

predictions["row_residual"] = (
    predictions["observed_solubility"]
    - predictions["predicted_solubility"]
)

predictions["row_absolute_error"] = (
    predictions["row_residual"].abs()
)

print(
    f"Prediction rows after validation: "
    f"{len(predictions):,}"
)


# ======================================================================
# AGGREGATE REPEATED PREDICTIONS BY COMPOUND
# ======================================================================

print("\n" + "=" * 70)
print("Aggregating repeated predictions")
print("=" * 70)

compound_rows = []

for compound_id, group in predictions.groupby(
    "ID",
    sort=False,
):

    prediction_values = (
        group["predicted_solubility"]
        .astype(float)
        .to_numpy()
    )

    observed_values = (
        group["observed_solubility"]
        .astype(float)
        .to_numpy()
    )

    mean_prediction = np.mean(
        prediction_values
    )

    prediction_sd = (
        np.std(
            prediction_values,
            ddof=1,
        )
        if len(prediction_values) > 1
        else 0.0
    )

    prediction_range = (
        np.max(prediction_values)
        - np.min(prediction_values)
    )

    observed_value = np.mean(
        observed_values
    )

    # Compound-level error is based on the mean prediction
    # across repeated held-out evaluations.
    residual = (
        observed_value
        - mean_prediction
    )

    absolute_error = abs(
        residual
    )

    compound_rows.append(
        {
            "ID": compound_id,
            "observed_solubility": observed_value,
            "mean_prediction": mean_prediction,
            "prediction_sd": prediction_sd,
            "prediction_range": prediction_range,
            "n_predictions": len(prediction_values),
            "residual": residual,
            "absolute_error": absolute_error,
        }
    )


compound_errors = pd.DataFrame(
    compound_rows
)

print(
    f"Compound-level prediction table: "
    f"{len(compound_errors):,}"
)

print(
    "Mean predictions per compound: "
    f"{compound_errors['n_predictions'].mean():.2f}"
)

print(
    "Median predictions per compound: "
    f"{compound_errors['n_predictions'].median():.0f}"
)

print(
    "Maximum predictions for one compound: "
    f"{compound_errors['n_predictions'].max():.0f}"
)


# ======================================================================
# MERGE MOLECULAR FEATURES
# ======================================================================

compound_errors = compound_errors.merge(
    features,
    on="ID",
    how="left",
    validate="one_to_one",
)

if compound_errors[FEATURE_COLUMNS].isna().any().any():

    missing_counts = (
        compound_errors[FEATURE_COLUMNS]
        .isna()
        .sum()
    )

    print("\nMissing feature counts:")

    print(
        missing_counts[
            missing_counts > 0
        ]
    )

    raise ValueError(
        "Missing molecular features after compound-level merge."
    )


# ======================================================================
# OVERALL PREDICTION STABILITY
# ======================================================================

print("\n" + "=" * 70)
print("OVERALL PREDICTION STABILITY")
print("=" * 70)

overall_mean_sd = (
    compound_errors["prediction_sd"].mean()
)

overall_median_sd = (
    compound_errors["prediction_sd"].median()
)

overall_mean_range = (
    compound_errors["prediction_range"].mean()
)

overall_median_range = (
    compound_errors["prediction_range"].median()
)

overall_mae = (
    compound_errors["absolute_error"].mean()
)

overall_rmse = np.sqrt(
    np.mean(
        compound_errors["residual"] ** 2
    )
)

print(
    f"Compounds analysed: "
    f"{len(compound_errors):,}"
)

print(
    f"Mean prediction SD: "
    f"{overall_mean_sd:.4f}"
)

print(
    f"Median prediction SD: "
    f"{overall_median_sd:.4f}"
)

print(
    f"Mean prediction range: "
    f"{overall_mean_range:.4f}"
)

print(
    f"Median prediction range: "
    f"{overall_median_range:.4f}"
)

print(
    f"Mean absolute error of mean prediction: "
    f"{overall_mae:.4f}"
)

print(
    f"RMSE of mean prediction: "
    f"{overall_rmse:.4f}"
)


# ======================================================================
# UNCERTAINTY ASSOCIATION WITH ERROR
# ======================================================================

print("\n" + "=" * 70)
print("UNCERTAINTY ASSOCIATION WITH PREDICTION ERROR")
print("=" * 70)

sd_pearson_r, sd_pearson_p = safe_pearson(
    compound_errors["prediction_sd"],
    compound_errors["absolute_error"],
)

sd_spearman_r, sd_spearman_p = safe_spearman(
    compound_errors["prediction_sd"],
    compound_errors["absolute_error"],
)

range_pearson_r, range_pearson_p = safe_pearson(
    compound_errors["prediction_range"],
    compound_errors["absolute_error"],
)

range_spearman_r, range_spearman_p = safe_spearman(
    compound_errors["prediction_range"],
    compound_errors["absolute_error"],
)

print(
    "Prediction SD vs absolute error: "
    f"Pearson r={sd_pearson_r:+.4f} "
    f"p={sd_pearson_p:.4e}"
)

print(
    "Prediction SD vs absolute error: "
    f"Spearman r={sd_spearman_r:+.4f} "
    f"p={sd_spearman_p:.4e}"
)

print(
    "Prediction range vs absolute error: "
    f"Pearson r={range_pearson_r:+.4f} "
    f"p={range_pearson_p:.4e}"
)

print(
    "Prediction range vs absolute error: "
    f"Spearman r={range_spearman_r:+.4f} "
    f"p={range_spearman_p:.4e}"
)


# ======================================================================
# SOLUBILITY REGIONS
# ======================================================================

solubility_bins = [
    -np.inf,
    -6,
    -4,
    -2,
    0,
    2,
    np.inf,
]

solubility_labels = [
    "< -6",
    "-6 to < -4",
    "-4 to < -2",
    "-2 to < 0",
    "0 to < 2",
    ">= 2",
]

compound_errors["observed_solubility_region"] = pd.cut(
    compound_errors["observed_solubility"],
    bins=solubility_bins,
    labels=solubility_labels,
    right=False,
)


# ======================================================================
# UNCERTAINTY BY SOLUBILITY REGION
# ======================================================================

print("\n" + "=" * 70)
print("UNCERTAINTY BY OBSERVED SOLUBILITY REGION")
print("=" * 70)

solubility_summary = region_summary(
    compound_errors,
    "observed_solubility_region",
)

solubility_output = (
    OUTPUT_DIR
    / "prediction_uncertainty_by_solubility.csv"
)

solubility_summary.to_csv(
    solubility_output,
    index=False,
)

print(
    f"Saved: {solubility_output}"
)

print()

print(
    solubility_summary.to_string(
        index=False
    )
)


# ======================================================================
# MOLECULAR PROPERTY REGIONS
# ======================================================================

print("\n" + "=" * 70)
print("UNCERTAINTY BY MOLECULAR-PROPERTY REGION")
print("=" * 70)

feature_region_tables = []

for feature in FEATURE_COLUMNS:

    print(
        f"Analysing {feature}..."
    )

    table = feature_region_summary(
        compound_errors,
        feature,
        n_bins=6,
    )

    if len(table) > 0:
        feature_region_tables.append(
            table
        )


if feature_region_tables:

    feature_region_summary_table = pd.concat(
        feature_region_tables,
        ignore_index=True,
    )

else:

    feature_region_summary_table = pd.DataFrame()


feature_output = (
    OUTPUT_DIR
    / "prediction_uncertainty_by_feature.csv"
)

feature_region_summary_table.to_csv(
    feature_output,
    index=False,
)

print(
    f"Saved: {feature_output}"
)


# ======================================================================
# ERROR / UNCERTAINTY CORRELATIONS WITH MOLECULAR FEATURES
# ======================================================================

print("\n" + "=" * 70)
print("MOLECULAR PROPERTY ASSOCIATIONS")
print("=" * 70)

correlation_rows = []

for feature in FEATURE_COLUMNS:

    pearson_sd_r, pearson_sd_p = safe_pearson(
        compound_errors[feature],
        compound_errors["prediction_sd"],
    )

    spearman_sd_r, spearman_sd_p = safe_spearman(
        compound_errors[feature],
        compound_errors["prediction_sd"],
    )

    pearson_range_r, pearson_range_p = safe_pearson(
        compound_errors[feature],
        compound_errors["prediction_range"],
    )

    spearman_range_r, spearman_range_p = safe_spearman(
        compound_errors[feature],
        compound_errors["prediction_range"],
    )

    pearson_error_r, pearson_error_p = safe_pearson(
        compound_errors[feature],
        compound_errors["absolute_error"],
    )

    spearman_error_r, spearman_error_p = safe_spearman(
        compound_errors[feature],
        compound_errors["absolute_error"],
    )

    correlation_rows.append(
        {
            "feature": feature,
            "n": len(compound_errors),

            "pearson_prediction_sd": pearson_sd_r,
            "pearson_prediction_sd_p": pearson_sd_p,

            "spearman_prediction_sd": spearman_sd_r,
            "spearman_prediction_sd_p": spearman_sd_p,

            "pearson_prediction_range": pearson_range_r,
            "pearson_prediction_range_p": pearson_range_p,

            "spearman_prediction_range": spearman_range_r,
            "spearman_prediction_range_p": spearman_range_p,

            "pearson_absolute_error": pearson_error_r,
            "pearson_absolute_error_p": pearson_error_p,

            "spearman_absolute_error": spearman_error_r,
            "spearman_absolute_error_p": spearman_error_p,
        }
    )


correlation_summary = pd.DataFrame(
    correlation_rows
)

correlation_output = (
    OUTPUT_DIR
    / "prediction_uncertainty_correlations.csv"
)

correlation_summary.to_csv(
    correlation_output,
    index=False,
)

print(
    correlation_summary.to_string(
        index=False
    )
)

print(
    f"\nSaved: {correlation_output}"
)


# ======================================================================
# HIGH-UNCERTAINTY COMPOUNDS
# ======================================================================

print("\n" + "=" * 70)
print("HIGH-UNCERTAINTY COMPOUNDS")
print("=" * 70)

uncertainty_threshold = (
    compound_errors["prediction_sd"]
    .quantile(0.95)
)

high_uncertainty = (
    compound_errors[
        compound_errors["prediction_sd"]
        >= uncertainty_threshold
    ]
    .sort_values(
        "prediction_sd",
        ascending=False,
    )
    .copy()
)

high_uncertainty_output = (
    OUTPUT_DIR
    / "high_uncertainty_compounds.csv"
)

high_uncertainty.to_csv(
    high_uncertainty_output,
    index=False,
)

print(
    f"95th-percentile prediction SD threshold: "
    f"{uncertainty_threshold:.4f}"
)

print(
    f"High-uncertainty compounds: "
    f"{len(high_uncertainty):,}"
)

print(
    f"Saved: {high_uncertainty_output}"
)


# ======================================================================
# EXTREME UNCERTAINTY / ERROR COMPOUNDS
# ======================================================================

compound_errors["uncertainty_rank"] = (
    compound_errors["prediction_sd"]
    .rank(
        ascending=False,
        method="min",
    )
)

compound_errors["error_rank"] = (
    compound_errors["absolute_error"]
    .rank(
        ascending=False,
        method="min",
    )
)

extreme_output = (
    OUTPUT_DIR
    / "uncertainty_error_compounds.csv"
)

compound_errors.sort_values(
    [
        "prediction_sd",
        "absolute_error",
    ],
    ascending=False,
).head(100).to_csv(
    extreme_output,
    index=False,
)

print(
    f"Saved: {extreme_output}"
)


# ======================================================================
# FINAL COMPOUND-LEVEL OUTPUT
# ======================================================================

compound_output = (
    OUTPUT_DIR
    / "prediction_uncertainty_compound_level.csv"
)

compound_errors.to_csv(
    compound_output,
    index=False,
)

print(
    f"\nSaved: {compound_output}"
)


# ======================================================================
# REPORT
# ======================================================================

report_path = (
    REPORT_DIR
    / "prediction_uncertainty_analysis.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as report:

    report.write(
        "SCRIPT 28 — PREDICTION UNCERTAINTY & "
        "STABILITY ANALYSIS\n"
    )

    report.write("=" * 70 + "\n\n")

    report.write(
        "PURPOSE\n"
    )

    report.write(
        "-------\n"
    )

    report.write(
        "Quantify prediction instability across repeated "
        "scaffold-aware held-out evaluations and determine "
        "whether instability is associated with prediction "
        "error, solubility regime, or molecular-property region.\n\n"
    )

    report.write(
        "VALIDATION / DATA DESIGN\n"
    )

    report.write(
        "------------------------\n"
    )

    report.write(
        f"Molecular feature population: "
        f"{len(features):,} compounds\n"
    )

    report.write(
        f"Script 26 prediction rows: "
        f"{len(predictions):,}\n"
    )

    report.write(
        f"Compounds represented in Script 26: "
        f"{predictions['ID'].nunique():,}\n"
    )

    report.write(
        f"Compound-level uncertainty analysis: "
        f"{len(compound_errors):,} compounds\n"
    )

    report.write(
        f"Mean predictions per compound: "
        f"{compound_errors['n_predictions'].mean():.2f}\n"
    )

    report.write(
        f"Median predictions per compound: "
        f"{compound_errors['n_predictions'].median():.0f}\n"
    )

    report.write(
        f"Maximum predictions per compound: "
        f"{compound_errors['n_predictions'].max():.0f}\n\n"
    )

    report.write(
        "IMPORTANT INTERPRETATION NOTE\n"
    )

    report.write(
        "------------------------------\n"
    )

    report.write(
        "Prediction SD and prediction range quantify "
        "instability across repeated scaffold-aware "
        "train/test splits. They are not calibrated "
        "probabilistic prediction intervals.\n\n"
    )

    report.write(
        "Overall prediction stability\n"
    )

    report.write(
        "-----------------------------\n"
    )

    report.write(
        f"Mean prediction SD: "
        f"{overall_mean_sd:.4f}\n"
    )

    report.write(
        f"Median prediction SD: "
        f"{overall_median_sd:.4f}\n"
    )

    report.write(
        f"Mean prediction range: "
        f"{overall_mean_range:.4f}\n"
    )

    report.write(
        f"Median prediction range: "
        f"{overall_median_range:.4f}\n"
    )

    report.write(
        f"Mean absolute error of mean prediction: "
        f"{overall_mae:.4f}\n"
    )

    report.write(
        f"RMSE of mean prediction: "
        f"{overall_rmse:.4f}\n\n"
    )

    report.write(
        "UNCERTAINTY ASSOCIATION WITH ERROR\n"
    )

    report.write(
        "----------------------------------\n"
    )

    report.write(
        f"Prediction SD vs absolute error "
        f"(Pearson): r={sd_pearson_r:+.4f}, "
        f"p={sd_pearson_p:.4e}\n"
    )

    report.write(
        f"Prediction SD vs absolute error "
        f"(Spearman): r={sd_spearman_r:+.4f}, "
        f"p={sd_spearman_p:.4e}\n"
    )

    report.write(
        f"Prediction range vs absolute error "
        f"(Pearson): r={range_pearson_r:+.4f}, "
        f"p={range_pearson_p:.4e}\n"
    )

    report.write(
        f"Prediction range vs absolute error "
        f"(Spearman): r={range_spearman_r:+.4f}, "
        f"p={range_spearman_p:.4e}\n\n"
    )

    report.write(
        "UNCERTAINTY BY OBSERVED SOLUBILITY REGION\n"
    )

    report.write(
        "-----------------------------------------\n"
    )

    report.write(
        solubility_summary.to_string(
            index=False
        )
    )

    report.write(
        "\n\nMOLECULAR PROPERTY ASSOCIATIONS\n"
    )

    report.write(
        "-------------------------------\n"
    )

    report.write(
        correlation_summary.to_string(
            index=False
        )
    )

    report.write(
        "\n\nHIGH-UNCERTAINTY DEFINITION\n"
    )

    report.write(
        "---------------------------\n"
    )

    report.write(
        "High uncertainty is defined descriptively as "
        "prediction SD at or above the 95th percentile "
        "of compound-level prediction SD.\n"
    )

    report.write(
        f"Threshold: {uncertainty_threshold:.4f}\n"
    )

    report.write(
        f"Compounds above threshold: "
        f"{len(high_uncertainty):,}\n"
    )


# ======================================================================
# FINAL OUTPUT
# ======================================================================

print("\n" + "=" * 70)
print("OUTPUTS")
print("=" * 70)

print(
    f"Compound-level uncertainty table:\n"
    f"{compound_output}"
)

print(
    f"\nUncertainty by solubility region:\n"
    f"{solubility_output}"
)

print(
    f"\nUncertainty by molecular-property region:\n"
    f"{feature_output}"
)

print(
    f"\nUncertainty correlations:\n"
    f"{correlation_output}"
)

print(
    f"\nHigh-uncertainty compounds:\n"
    f"{high_uncertainty_output}"
)

print(
    f"\nUncertainty/error compounds:\n"
    f"{extreme_output}"
)

print(
    f"\nReport:\n"
    f"{report_path}"
)

print("\n" + "=" * 70)
print("SCRIPT 28 COMPLETE")
print("=" * 70)