"""
SCRIPT 29 — MODEL FAILURE MODE ANALYSIS

Purpose
-------
Synthesize the residual, uncertainty, and molecular-property analyses from
Scripts 26–28 to identify interpretable failure modes of the six-descriptor
Gradient Boosting solubility model.

The analysis distinguishes:
1. Stable / accurate predictions
2. High-error but relatively stable predictions
3. High-uncertainty but relatively accurate predictions
4. High-error + high-uncertainty predictions
5. Directionally biased predictions in chemical regions

Important:
---------
This is a diagnostic analysis of the existing repeated scaffold-aware
held-out predictions. It does not retrain the model and does not establish
causality.
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

UNCERTAINTY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "prediction_uncertainty_compound_level.csv"
)

UNCERTAINTY_CORR_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "prediction_uncertainty_correlations.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================================
# CONFIGURATION
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
# HELPERS
# ======================================================================

def safe_corr(x, y, method="pearson"):
    """Return correlation coefficient and p-value safely."""

    mask = (
        pd.notna(x)
        & pd.notna(y)
        & np.isfinite(x)
        & np.isfinite(y)
    )

    if mask.sum() < 3:
        return np.nan, np.nan

    if method == "pearson":
        result = pearsonr(x[mask], y[mask])
    else:
        result = spearmanr(x[mask], y[mask])

    return result.statistic, result.pvalue


def classify_failure_mode(row, error_threshold, uncertainty_threshold):
    """
    Assign each compound to a diagnostic failure mode.

    Error threshold:
        75th percentile absolute error.

    Uncertainty threshold:
        75th percentile prediction SD.

    These thresholds are descriptive, not probabilistic.
    """

    high_error = row["mean_absolute_error"] >= error_threshold
    high_uncertainty = row["prediction_sd"] >= uncertainty_threshold

    if high_error and high_uncertainty:
        return "High-error + high-uncertainty"

    if high_error and not high_uncertainty:
        return "High-error + stable"

    if not high_error and high_uncertainty:
        return "Low-error + high-uncertainty"

    return "Low-error + stable"


def property_region_summary(df, feature):
    """Summarize failure-mode composition across quartile regions."""

    temp = df.copy()

    try:
        temp["region"] = pd.qcut(
            temp[feature],
            q=4,
            duplicates="drop",
        )
    except ValueError:
        return pd.DataFrame()

    rows = []

    for region, group in temp.groupby(
        "region",
        observed=False,
    ):

        if len(group) == 0:
            continue

        rows.append(
            {
                "feature": feature,
                "region": str(region),
                "compound_count": len(group),
                "mean_absolute_error": group[
                    "mean_absolute_error"
                ].mean(),
                "mean_prediction_sd": group[
                    "prediction_sd"
                ].mean(),
                "high_error_pct": (
                    group["failure_mode"]
                    .isin(
                        [
                            "High-error + high-uncertainty",
                            "High-error + stable",
                        ]
                    )
                    .mean()
                    * 100
                ),
                "high_uncertainty_pct": (
                    group["failure_mode"]
                    .isin(
                        [
                            "High-error + high-uncertainty",
                            "Low-error + high-uncertainty",
                        ]
                    )
                    .mean()
                    * 100
                ),
                "high_error_high_uncertainty_pct": (
                    (
                        group["failure_mode"]
                        == "High-error + high-uncertainty"
                    ).mean()
                    * 100
                ),
                "mean_residual": group[
                    "mean_residual"
                ].mean(),
            }
        )

    return pd.DataFrame(rows)


# ======================================================================
# HEADER
# ======================================================================

print("=" * 70)
print("SCRIPT 29 — MODEL FAILURE MODE ANALYSIS")
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

required_features = ["ID"] + FEATURE_COLUMNS

missing = [
    column
    for column in required_features
    if column not in features.columns
]

if missing:
    raise ValueError(
        "Missing required molecular feature columns:\n"
        + "\n".join(missing)
    )

features = features[
    required_features
].copy()

features["ID"] = features["ID"].astype(str)

if features["ID"].duplicated().any():
    raise ValueError(
        "Duplicate compound IDs found in molecular features."
    )

print(
    f"Unique compounds in molecular features: "
    f"{features['ID'].nunique():,}"
)


# ======================================================================
# LOAD SCRIPT 26 RESIDUAL DATA
# ======================================================================

print("\n" + "=" * 70)
print("Loading Script 26 residual analysis")
print("=" * 70)

residuals = pd.read_csv(RESIDUAL_PATH)

print(f"Residual dataset shape: {residuals.shape}")

required_residual = [
    "ID",
    "Solubility",
    "predicted_solubility",
    "residual",
    "absolute_error",
]

missing = [
    column
    for column in required_residual
    if column not in residuals.columns
]

if missing:
    raise ValueError(
        "Missing required Script 26 columns:\n"
        + "\n".join(missing)
    )

residuals["ID"] = residuals["ID"].astype(str)

residuals = residuals[
    required_residual
    + [
        column
        for column in [
            "repetition",
            "seed",
            "scaffold",
        ]
        if column in residuals.columns
    ]
].copy()

residuals = residuals.dropna(
    subset=[
        "ID",
        "Solubility",
        "predicted_solubility",
    ]
)

print(
    f"Valid residual rows: {len(residuals):,}"
)

print(
    f"Unique compounds represented: "
    f"{residuals['ID'].nunique():,}"
)


# ======================================================================
# LOAD SCRIPT 28 UNCERTAINTY DATA
# ======================================================================

print("\n" + "=" * 70)
print("Loading Script 28 prediction uncertainty")
print("=" * 70)

uncertainty = pd.read_csv(UNCERTAINTY_PATH)

print(
    f"Uncertainty dataset shape: "
    f"{uncertainty.shape}"
)

required_uncertainty = [
    "ID",
    "prediction_sd",
    "prediction_range",
]

missing = [
    column
    for column in required_uncertainty
    if column not in uncertainty.columns
]

if missing:
    raise ValueError(
        "Missing required Script 28 columns:\n"
        + "\n".join(missing)
    )

uncertainty["ID"] = uncertainty["ID"].astype(str)

print(
    f"Unique uncertainty compounds: "
    f"{uncertainty['ID'].nunique():,}"
)


# ======================================================================
# AGGREGATE SCRIPT 26 TO COMPOUND LEVEL
# ======================================================================

print("\n" + "=" * 70)
print("Aggregating repeated held-out predictions")
print("=" * 70)

compound_error = (
    residuals
    .groupby("ID")
    .agg(
        observed_solubility=(
            "Solubility",
            "mean",
        ),
        mean_prediction=(
            "predicted_solubility",
            "mean",
        ),
        mean_residual=(
            "residual",
            "mean",
        ),
        mean_absolute_error=(
            "absolute_error",
            "mean",
        ),
        median_absolute_error=(
            "absolute_error",
            "median",
        ),
        prediction_count=(
            "predicted_solubility",
            "count",
        ),
    )
    .reset_index()
)

compound_error["mean_squared_error"] = (
    compound_error["mean_residual"] ** 2
)

print(
    f"Compound-level error table: "
    f"{len(compound_error):,}"
)


# ======================================================================
# MERGE ERROR + UNCERTAINTY + FEATURES
# ======================================================================

print("\n" + "=" * 70)
print("Constructing integrated diagnostic dataset")
print("=" * 70)

analysis = (
    compound_error
    .merge(
        uncertainty[
            [
                "ID",
                "prediction_sd",
                "prediction_range",
            ]
        ],
        on="ID",
        how="inner",
        validate="one_to_one",
    )
    .merge(
        features,
        on="ID",
        how="inner",
        validate="one_to_one",
    )
)

print(
    f"Integrated compounds: {len(analysis):,}"
)

if len(analysis) == 0:
    raise ValueError(
        "No compounds remained after integrating "
        "error, uncertainty, and molecular features."
    )


# ======================================================================
# VERIFY CORE METRICS
# ======================================================================

analysis["recomputed_absolute_error"] = (
    analysis["observed_solubility"]
    - analysis["mean_prediction"]
).abs()

analysis["recomputed_residual"] = (
    analysis["observed_solubility"]
    - analysis["mean_prediction"]
)

if not np.allclose(
    analysis["mean_absolute_error"],
    analysis["recomputed_absolute_error"],
    atol=0.25,
    equal_nan=True,
):
    print(
        "WARNING: Compound-level mean absolute error "
        "differs from absolute error of mean prediction."
    )


# ======================================================================
# FAILURE THRESHOLDS
# ======================================================================

print("\n" + "=" * 70)
print("Defining diagnostic thresholds")
print("=" * 70)

error_threshold = analysis[
    "mean_absolute_error"
].quantile(0.75)

uncertainty_threshold = analysis[
    "prediction_sd"
].quantile(0.75)

print(
    f"75th-percentile absolute-error threshold: "
    f"{error_threshold:.4f}"
)

print(
    f"75th-percentile prediction-SD threshold: "
    f"{uncertainty_threshold:.4f}"
)

analysis["failure_mode"] = analysis.apply(
    classify_failure_mode,
    axis=1,
    error_threshold=error_threshold,
    uncertainty_threshold=uncertainty_threshold,
)


# ======================================================================
# FAILURE MODE SUMMARY
# ======================================================================

print("\n" + "=" * 70)
print("FAILURE MODE SUMMARY")
print("=" * 70)

mode_order = [
    "Low-error + stable",
    "High-error + stable",
    "Low-error + high-uncertainty",
    "High-error + high-uncertainty",
]

mode_rows = []

for mode in mode_order:

    group = analysis[
        analysis["failure_mode"] == mode
    ]

    mode_rows.append(
        {
            "failure_mode": mode,
            "compound_count": len(group),
            "percentage": (
                len(group)
                / len(analysis)
                * 100
            ),
            "mean_absolute_error": group[
                "mean_absolute_error"
            ].mean(),
            "median_absolute_error": group[
                "mean_absolute_error"
            ].median(),
            "mean_prediction_sd": group[
                "prediction_sd"
            ].mean(),
            "median_prediction_sd": group[
                "prediction_sd"
            ].median(),
            "mean_prediction_range": group[
                "prediction_range"
            ].mean(),
            "mean_residual": group[
                "mean_residual"
            ].mean(),
        }
    )

failure_summary = pd.DataFrame(
    mode_rows
)

print(
    failure_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ======================================================================
# DIRECTIONAL ERROR CATEGORIES
# ======================================================================

print("\n" + "=" * 70)
print("DIRECTIONAL ERROR AMONG HIGH-ERROR COMPOUNDS")
print("=" * 70)

analysis["error_direction"] = np.select(
    [
        analysis["mean_residual"] > 0,
        analysis["mean_residual"] < 0,
    ],
    [
        "Underprediction",
        "Overprediction",
    ],
    default="Near-zero",
)

high_error = analysis[
    analysis["mean_absolute_error"]
    >= error_threshold
].copy()

directional_summary = (
    high_error
    .groupby("error_direction")
    .agg(
        compound_count=("ID", "count"),
        mean_absolute_error=(
            "mean_absolute_error",
            "mean",
        ),
        mean_prediction_sd=(
            "prediction_sd",
            "mean",
        ),
        mean_residual=(
            "mean_residual",
            "mean",
        ),
    )
    .reset_index()
)

directional_summary["percentage"] = (
    directional_summary["compound_count"]
    / len(high_error)
    * 100
)

print(
    directional_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ======================================================================
# SOLUBILITY FAILURE REGIONS
# ======================================================================

print("\n" + "=" * 70)
print("FAILURE MODES BY OBSERVED SOLUBILITY")
print("=" * 70)

analysis["solubility_region"] = pd.cut(
    analysis["observed_solubility"],
    bins=[
        -np.inf,
        -6,
        -4,
        -2,
        0,
        np.inf,
    ],
    labels=[
        "< -6",
        "-6 to < -4",
        "-4 to < -2",
        "-2 to < 0",
        ">= 0",
    ],
    right=False,
)

solubility_failure = (
    analysis
    .groupby(
        "solubility_region",
        observed=False,
    )
    .agg(
        compound_count=("ID", "count"),
        mean_absolute_error=(
            "mean_absolute_error",
            "mean",
        ),
        mean_prediction_sd=(
            "prediction_sd",
            "mean",
        ),
        high_error_pct=(
            "failure_mode",
            lambda x: (
                x.isin(
                    [
                        "High-error + high-uncertainty",
                        "High-error + stable",
                    ]
                ).mean()
                * 100
            ),
        ),
        high_uncertainty_pct=(
            "failure_mode",
            lambda x: (
                x.isin(
                    [
                        "High-error + high-uncertainty",
                        "Low-error + high-uncertainty",
                    ]
                ).mean()
                * 100
            ),
        ),
        high_error_high_uncertainty_pct=(
            "failure_mode",
            lambda x: (
                (
                    x
                    == "High-error + high-uncertainty"
                ).mean()
                * 100
            ),
        ),
        mean_residual=(
            "mean_residual",
            "mean",
        ),
    )
    .reset_index()
)

print(
    solubility_failure.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ======================================================================
# MOLECULAR PROPERTY FAILURE REGIONS
# ======================================================================

print("\n" + "=" * 70)
print("FAILURE MODES BY MOLECULAR-PROPERTY REGION")
print("=" * 70)

property_tables = []

for feature in FEATURE_COLUMNS:

    print(f"Analysing {feature}...")

    table = property_region_summary(
        analysis,
        feature,
    )

    if not table.empty:
        property_tables.append(table)

property_failure = pd.concat(
    property_tables,
    ignore_index=True,
)

property_failure_path = (
    OUTPUT_DIR
    / "failure_modes_by_property_region.csv"
)

property_failure.to_csv(
    property_failure_path,
    index=False,
)

print(
    f"Saved: {property_failure_path}"
)


# ======================================================================
# CORRELATION WITH ERROR / UNCERTAINTY
# ======================================================================

print("\n" + "=" * 70)
print("ERROR AND UNCERTAINTY ASSOCIATIONS")
print("=" * 70)

correlation_rows = []

targets = {
    "mean_absolute_error": "error_magnitude",
    "prediction_sd": "prediction_uncertainty",
    "prediction_range": "prediction_range",
}

for feature in FEATURE_COLUMNS:

    for target, target_name in targets.items():

        pearson_r, pearson_p = safe_corr(
            analysis[feature],
            analysis[target],
            method="pearson",
        )

        spearman_r, spearman_p = safe_corr(
            analysis[feature],
            analysis[target],
            method="spearman",
        )

        correlation_rows.append(
            {
                "feature": feature,
                "target": target_name,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
            }
        )

correlations = pd.DataFrame(
    correlation_rows
)

print(
    correlations.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

correlation_path = (
    OUTPUT_DIR
    / "failure_mode_correlations.csv"
)

correlations.to_csv(
    correlation_path,
    index=False,
)

print(
    f"Saved: {correlation_path}"
)


# ======================================================================
# HIGH-PRIORITY FAILURE COMPOUNDS
# ======================================================================

print("\n" + "=" * 70)
print("HIGH-PRIORITY FAILURE COMPOUNDS")
print("=" * 70)

analysis["failure_priority_score"] = (
    analysis["mean_absolute_error"]
    / analysis["mean_absolute_error"].median()
    + analysis["prediction_sd"]
    / analysis["prediction_sd"].median()
)

priority = (
    analysis
    .sort_values(
        [
            "failure_mode",
            "failure_priority_score",
        ],
        ascending=[True, False],
    )
    .copy()
)

priority_columns = [
    "ID",
    "SMILES",
    "observed_solubility",
    "mean_prediction",
    "mean_residual",
    "mean_absolute_error",
    "prediction_sd",
    "prediction_range",
    "failure_mode",
] + [
    column
    for column in FEATURE_COLUMNS
    if column in priority.columns
]

# Include SMILES if it exists in molecular features.
if "SMILES" not in priority.columns:
    priority_columns = [
        column
        for column in priority_columns
        if column != "SMILES"
    ]

priority_output = priority[
    priority_columns
].copy()

priority_path = (
    OUTPUT_DIR
    / "model_failure_priority_compounds.csv"
)

priority_output.to_csv(
    priority_path,
    index=False,
)

print(
    f"Saved: {priority_path}"
)


# ======================================================================
# HIGH-ERROR + HIGH-UNCERTAINTY SUBSET
# ======================================================================

high_high = analysis[
    analysis["failure_mode"]
    == "High-error + high-uncertainty"
].copy()

high_high = high_high.sort_values(
    [
        "mean_absolute_error",
        "prediction_sd",
    ],
    ascending=False,
)

high_high_path = (
    OUTPUT_DIR
    / "high_error_high_uncertainty_compounds.csv"
)

high_high.to_csv(
    high_high_path,
    index=False,
)

print(
    f"High-error + high-uncertainty compounds: "
    f"{len(high_high):,}"
)

print(
    f"Saved: {high_high_path}"
)


# ======================================================================
# OVERALL DIAGNOSTIC CORRELATION
# ======================================================================

error_uncertainty_pearson_r, error_uncertainty_pearson_p = (
    safe_corr(
        analysis["mean_absolute_error"],
        analysis["prediction_sd"],
        method="pearson",
    )
)

error_uncertainty_spearman_r, error_uncertainty_spearman_p = (
    safe_corr(
        analysis["mean_absolute_error"],
        analysis["prediction_sd"],
        method="spearman",
    )
)

print("\n" + "=" * 70)
print("ERROR VS UNCERTAINTY")
print("=" * 70)

print(
    "Absolute error vs prediction SD:"
    f" Pearson r={error_uncertainty_pearson_r:+.4f}"
    f" p={error_uncertainty_pearson_p:.4e}"
)

print(
    "Absolute error vs prediction SD:"
    f" Spearman r={error_uncertainty_spearman_r:+.4f}"
    f" p={error_uncertainty_spearman_p:.4e}"
)


# ======================================================================
# SAVE INTEGRATED COMPOUND TABLE
# ======================================================================

integrated_columns = [
    "ID",
    "observed_solubility",
    "mean_prediction",
    "mean_residual",
    "mean_absolute_error",
    "median_absolute_error",
    "prediction_count",
    "prediction_sd",
    "prediction_range",
    "failure_mode",
    "error_direction",
    "solubility_region",
] + FEATURE_COLUMNS

integrated_columns = [
    column
    for column in integrated_columns
    if column in analysis.columns
]

integrated_path = (
    OUTPUT_DIR
    / "model_failure_mode_compounds.csv"
)

analysis[
    integrated_columns
].to_csv(
    integrated_path,
    index=False,
)

print(
    f"\nSaved: {integrated_path}"
)


# ======================================================================
# REPORT
# ======================================================================

report_path = (
    REPORT_DIR
    / "model_failure_mode_analysis.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as report:

    report.write(
        "SCRIPT 29 — MODEL FAILURE MODE ANALYSIS\n"
    )
    report.write("=" * 70 + "\n\n")

    report.write("PURPOSE\n")
    report.write("-------\n")
    report.write(
        "Synthesize repeated scaffold-aware held-out "
        "prediction error and prediction instability "
        "to identify interpretable model failure modes.\n\n"
    )

    report.write("DATA DESIGN\n")
    report.write("-----------\n")
    report.write(
        f"Molecular feature population: "
        f"{len(features):,}\n"
    )
    report.write(
        f"Integrated compounds analysed: "
        f"{len(analysis):,}\n"
    )
    report.write(
        "Error source: Script 26 repeated "
        "scaffold-aware held-out predictions\n"
    )
    report.write(
        "Uncertainty source: Script 28 repeated "
        "prediction stability analysis\n\n"
    )

    report.write("IMPORTANT INTERPRETATION NOTE\n")
    report.write("------------------------------\n")
    report.write(
        "Failure-mode categories are descriptive "
        "diagnostic groupings. They do not establish "
        "causal relationships between molecular "
        "properties and model failure.\n\n"
    )

    report.write(
        "FAILURE MODE THRESHOLDS\n"
    )
    report.write(
        f"75th-percentile absolute error: "
        f"{error_threshold:.4f}\n"
    )
    report.write(
        f"75th-percentile prediction SD: "
        f"{uncertainty_threshold:.4f}\n\n"
    )

    report.write(
        "FAILURE MODE SUMMARY\n"
    )
    report.write("--------------------\n")
    report.write(
        failure_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    report.write("\n\n")

    report.write(
        "DIRECTIONAL ERROR AMONG HIGH-ERROR COMPOUNDS\n"
    )
    report.write(
        "--------------------------------------------\n"
    )
    report.write(
        directional_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    report.write("\n\n")

    report.write(
        "FAILURE MODES BY OBSERVED SOLUBILITY\n"
    )
    report.write(
        "------------------------------------\n"
    )
    report.write(
        solubility_failure.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    report.write("\n\n")

    report.write(
        "ERROR / UNCERTAINTY CORRELATIONS\n"
    )
    report.write(
        "--------------------------------\n"
    )
    report.write(
        correlations.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )
    report.write("\n\n")

    report.write(
        "OVERALL ERROR VS UNCERTAINTY\n"
    )
    report.write(
        "----------------------------\n"
    )
    report.write(
        f"Pearson r={error_uncertainty_pearson_r:+.4f}, "
        f"p={error_uncertainty_pearson_p:.4e}\n"
    )
    report.write(
        f"Spearman r={error_uncertainty_spearman_r:+.4f}, "
        f"p={error_uncertainty_spearman_p:.4e}\n\n"
    )

    report.write(
        "HIGH-ERROR + HIGH-UNCERTAINTY COMPOUNDS\n"
    )
    report.write(
        "---------------------------------------\n"
    )
    report.write(
        f"Count: {len(high_high):,}\n"
    )

    report.write("\n\n")
    report.write(
        "INTERPRETATION FRAMEWORK\n"
    )
    report.write(
        "------------------------\n"
    )
    report.write(
        "The most important failure category is the "
        "High-error + high-uncertainty group because "
        "these compounds combine poor predictive "
        "accuracy with instability across repeated "
        "scaffold-aware splits.\n\n"
    )
    report.write(
        "High-error + stable compounds indicate "
        "systematic predictive difficulty that is "
        "relatively insensitive to split variation.\n\n"
    )
    report.write(
        "Low-error + high-uncertainty compounds indicate "
        "prediction instability that does not currently "
        "translate into large observed errors.\n\n"
    )
    report.write(
        "Low-error + stable compounds represent the "
        "most reliable region of the current model's "
        "applicability within the analysed population.\n"
    )


# ======================================================================
# FINAL OUTPUT
# ======================================================================

print("\n" + "=" * 70)
print("OUTPUTS")
print("=" * 70)

print(
    f"Integrated compound failure table:\n"
    f"{integrated_path}"
)

print(
    f"\nFailure modes by molecular-property region:\n"
    f"{property_failure_path}"
)

print(
    f"\nFailure-mode correlations:\n"
    f"{correlation_path}"
)

print(
    f"\nHigh-priority failure compounds:\n"
    f"{priority_path}"
)

print(
    f"\nHigh-error + high-uncertainty compounds:\n"
    f"{high_high_path}"
)

print(
    f"\nReport:\n"
    f"{report_path}"
)

print("\n" + "=" * 70)
print("SCRIPT 29 COMPLETE")
print("=" * 70)