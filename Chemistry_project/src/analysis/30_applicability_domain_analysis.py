"""
SCRIPT 30 — CHEMICAL APPLICABILITY DOMAIN ANALYSIS

Purpose
-------
Assess where the six-descriptor Gradient Boosting representation has strong
versus weak chemical-space support.

The analysis uses standardized molecular descriptors and k-nearest-neighbour
chemical-space distance to construct a descriptive applicability-domain
measure.

Important methodological note
-----------------------------
Script 26 contains repeated scaffold-aware held-out predictions, but it does
not retain the exact training compound IDs for every repetition. Therefore,
the chemical-space distance calculated here is a descriptive Population C
applicability-domain measure rather than a fold-specific training-set
distance.

The analysis is used to determine whether compounds that are more isolated
in the molecular descriptor space also tend to exhibit:

    - larger prediction errors
    - greater prediction instability
    - more frequent model failure modes

No causal interpretation is made.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


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

FAILURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_failure_mode_compounds.csv"
)

OUTPUT_AD = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "applicability_domain_compounds.csv"
)

OUTPUT_REGIONS = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "applicability_domain_regions.csv"
)

OUTPUT_CORRELATIONS = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "applicability_domain_correlations.csv"
)

OUTPUT_PRIORITY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "applicability_domain_priority_compounds.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "applicability_domain_analysis.txt"
)


# ======================================================================
# FEATURES
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
# SETTINGS
# ======================================================================

K_NEIGHBOURS = 5

# Descriptive thresholds based on the Population C chemical-space
# distribution.
BORDERLINE_PERCENTILE = 90
OUT_OF_DOMAIN_PERCENTILE = 95


# ======================================================================
# HELPERS
# ======================================================================

def safe_correlation(x, y, method="pearson"):
    """
    Calculate correlation while safely handling missing/non-finite values.
    """

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    if mask.sum() < 3:
        return np.nan, np.nan

    if method == "pearson":
        result = pearsonr(x[mask], y[mask])
    else:
        result = spearmanr(x[mask], y[mask])

    return result.statistic, result.pvalue


def classify_domain(score, borderline_threshold, out_of_domain_threshold):
    """
    Assign descriptive applicability-domain categories.
    """

    if score <= borderline_threshold:
        return "In-domain"

    if score <= out_of_domain_threshold:
        return "Borderline"

    return "Out-of-domain"


def region_summary(df):
    """
    Summarise error, uncertainty and failure metrics by AD region.
    """

    rows = []

    for region, group in df.groupby(
        "applicability_domain",
        observed=False
    ):

        if len(group) == 0:
            continue

        rows.append(
            {
                "applicability_domain": region,
                "compound_count": len(group),
                "percentage": 100 * len(group) / len(df),

                "mean_knn_distance": group[
                    "mean_knn_distance"
                ].mean(),

                "median_knn_distance": group[
                    "mean_knn_distance"
                ].median(),

                "mean_nearest_distance": group[
                    "nearest_neighbor_distance"
                ].mean(),

                "median_nearest_distance": group[
                    "nearest_neighbor_distance"
                ].median(),

                "mean_absolute_error": group[
                    "absolute_error"
                ].mean(),

                "median_absolute_error": group[
                    "absolute_error"
                ].median(),

                "rmse": np.sqrt(
                    np.mean(
                        group["absolute_error"] ** 2
                    )
                ),

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
                    "residual"
                ].mean(),

                "high_error_pct": 100 * group[
                    "high_error"
                ].mean(),

                "high_uncertainty_pct": 100 * group[
                    "high_uncertainty"
                ].mean(),

                "high_error_high_uncertainty_pct": 100 * group[
                    "high_error_high_uncertainty"
                ].mean(),
            }
        )

    return pd.DataFrame(rows)


# ======================================================================
# HEADER
# ======================================================================

print("=" * 70)
print("SCRIPT 30 — CHEMICAL APPLICABILITY DOMAIN ANALYSIS")
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

required_columns = ["ID"] + FEATURE_COLUMNS

missing = [
    column
    for column in required_columns
    if column not in features.columns
]

if missing:
    raise ValueError(
        "Missing required molecular feature columns:\n"
        + "\n".join(missing)
    )

features = features[
    required_columns
].copy()

features = features.drop_duplicates(
    subset="ID"
).reset_index(drop=True)

print(
    f"Unique compounds in molecular features: "
    f"{features['ID'].nunique():,}"
)


# ======================================================================
# POPULATION C VALIDATION
# ======================================================================

print("\n" + "=" * 70)
print("POPULATION C VALIDATION")
print("=" * 70)

if len(features) != 8643:
    raise ValueError(
        f"Expected Population C to contain 8,643 compounds; "
        f"found {len(features):,}."
    )

print("Population C verified: 8,643 compounds.")


# ======================================================================
# LOAD SCRIPT 26 RESIDUAL DATA
# ======================================================================

print("\n" + "=" * 70)
print("Loading Script 26 residual analysis")
print("=" * 70)

residuals = pd.read_csv(RESIDUAL_PATH)

print(
    f"Residual dataset shape: {residuals.shape}"
)

print("\nColumns found:")
for column in residuals.columns:
    print(f"  {column}")


# ======================================================================
# RESOLVE SCRIPT 26 COLUMNS
# ======================================================================

required_residual_candidates = {
    "ID": ["ID"],
    "observed": ["observed_solubility", "Solubility"],
    "predicted": ["predicted_solubility"],
    "residual": ["residual"],
    "absolute_error": ["absolute_error"],
}


def resolve_column(df, candidates, label):

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        f"Could not resolve required Script 26 column: {label}\n"
        f"Accepted names: {candidates}"
    )


id_column = resolve_column(
    residuals,
    required_residual_candidates["ID"],
    "ID"
)

observed_column = resolve_column(
    residuals,
    required_residual_candidates["observed"],
    "observed solubility"
)

predicted_column = resolve_column(
    residuals,
    required_residual_candidates["predicted"],
    "predicted solubility"
)

residual_column = resolve_column(
    residuals,
    required_residual_candidates["residual"],
    "residual"
)

absolute_error_column = resolve_column(
    residuals,
    required_residual_candidates["absolute_error"],
    "absolute error"
)


residuals = residuals.rename(
    columns={
        id_column: "ID",
        observed_column: "observed_solubility",
        predicted_column: "predicted_solubility",
        residual_column: "residual",
        absolute_error_column: "absolute_error",
    }
)


required_residual_columns = [
    "ID",
    "observed_solubility",
    "predicted_solubility",
    "residual",
    "absolute_error",
]


residuals = residuals[
    required_residual_columns
].copy()


residuals = residuals.dropna(
    subset=required_residual_columns
)


print(
    f"\nValid Script 26 prediction rows: "
    f"{len(residuals):,}"
)

print(
    f"Unique compounds represented: "
    f"{residuals['ID'].nunique():,}"
)


# ======================================================================
# AGGREGATE SCRIPT 26 PREDICTIONS
# ======================================================================

print("\n" + "=" * 70)
print("Aggregating repeated held-out predictions")
print("=" * 70)

compound_predictions = (
    residuals
    .groupby("ID")
    .agg(
        observed_solubility=(
            "observed_solubility",
            "first"
        ),
        mean_prediction=(
            "predicted_solubility",
            "mean"
        ),
        prediction_sd=(
            "predicted_solubility",
            "std"
        ),
        prediction_range=(
            "predicted_solubility",
            lambda x: x.max() - x.min()
        ),
        mean_residual=(
            "residual",
            "mean"
        ),
        mean_absolute_error=(
            "absolute_error",
            "mean"
        ),
        prediction_count=(
            "predicted_solubility",
            "count"
        ),
    )
    .reset_index()
)


# Replace undefined SD for single predictions.
compound_predictions[
    "prediction_sd"
] = compound_predictions[
    "prediction_sd"
].fillna(0)


compound_predictions[
    "absolute_error"
] = (
    compound_predictions["observed_solubility"]
    - compound_predictions["mean_prediction"]
).abs()


compound_predictions[
    "residual"
] = (
    compound_predictions["observed_solubility"]
    - compound_predictions["mean_prediction"]
)


print(
    f"Compound-level prediction table: "
    f"{len(compound_predictions):,}"
)

print(
    "Mean predictions per compound: "
    f"{compound_predictions['prediction_count'].mean():.2f}"
)

print(
    "Median predictions per compound: "
    f"{compound_predictions['prediction_count'].median():.0f}"
)


# ======================================================================
# LOAD SCRIPT 28 UNCERTAINTY
# ======================================================================

print("\n" + "=" * 70)
print("Loading Script 28 prediction uncertainty")
print("=" * 70)

uncertainty = pd.read_csv(
    UNCERTAINTY_PATH
)

print(
    f"Uncertainty dataset shape: "
    f"{uncertainty.shape}"
)

if "ID" not in uncertainty.columns:
    raise ValueError(
        "Script 28 uncertainty dataset does not contain ID."
    )


# ======================================================================
# RESOLVE UNCERTAINTY COLUMNS
# ======================================================================

uncertainty_column_map = {}

if "prediction_sd" in uncertainty.columns:
    uncertainty_column_map[
        "prediction_sd"
    ] = "prediction_sd"

if "prediction_range" in uncertainty.columns:
    uncertainty_column_map[
        "prediction_range"
    ] = "prediction_range"


if (
    "prediction_sd" not in uncertainty_column_map
    or "prediction_range" not in uncertainty_column_map
):
    raise ValueError(
        "Script 28 uncertainty dataset must contain "
        "prediction_sd and prediction_range."
    )


uncertainty = uncertainty[
    [
        "ID",
        "prediction_sd",
        "prediction_range",
    ]
].drop_duplicates(
    subset="ID"
)


# ======================================================================
# LOAD SCRIPT 29 FAILURE MODES
# ======================================================================

print("\n" + "=" * 70)
print("Loading Script 29 failure-mode analysis")
print("=" * 70)

failure = pd.read_csv(
    FAILURE_PATH
)

print(
    f"Failure-mode dataset shape: "
    f"{failure.shape}"
)


if "ID" not in failure.columns:
    raise ValueError(
        "Script 29 failure-mode dataset does not contain ID."
    )


# Resolve useful Script 29 columns.

failure_keep = ["ID"]

for column in [
    "failure_mode",
    "high_error",
    "high_uncertainty",
    "high_error_high_uncertainty",
]:

    if column in failure.columns:
        failure_keep.append(column)


failure = failure[
    failure_keep
].drop_duplicates(
    subset="ID"
)


# ======================================================================
# INTEGRATED DIAGNOSTIC DATASET
# ======================================================================

print("\n" + "=" * 70)
print("Constructing integrated diagnostic dataset")
print("=" * 70)

diagnostic = (
    features
    .merge(
        compound_predictions,
        on="ID",
        how="inner"
    )
    .merge(
        uncertainty,
        on="ID",
        how="left",
        suffixes=("", "_script28")
    )
    .merge(
        failure,
        on="ID",
        how="left"
    )
)


# Prefer Script 28 uncertainty values when available.
diagnostic[
    "prediction_sd"
] = diagnostic[
    "prediction_sd_script28"
].fillna(
    diagnostic["prediction_sd"]
)

diagnostic[
    "prediction_range"
] = diagnostic[
    "prediction_range_script28"
].fillna(
    diagnostic["prediction_range"]
)


diagnostic = diagnostic.drop(
    columns=[
        "prediction_sd_script28",
        "prediction_range_script28",
    ],
    errors="ignore"
)


print(
    f"Integrated compounds: "
    f"{len(diagnostic):,}"
)


# ======================================================================
# INPUT VALIDATION
# ======================================================================

print("\n" + "=" * 70)
print("INPUT VALIDATION")
print("=" * 70)

missing_ids = set(
    diagnostic["ID"]
) - set(
    features["ID"]
)

if missing_ids:
    raise ValueError(
        "Some integrated IDs are absent from molecular features."
    )

print(
    "All integrated compounds matched to molecular features."
)

print(
    f"Compounds available for applicability-domain analysis: "
    f"{len(diagnostic):,}"
)


# ======================================================================
# PREPARE CHEMICAL DESCRIPTOR MATRIX
# ======================================================================

print("\n" + "=" * 70)
print("Preparing standardized molecular descriptor space")
print("=" * 70)

descriptor_data = diagnostic[
    FEATURE_COLUMNS
].copy()


if descriptor_data.isna().any().any():
    missing_counts = descriptor_data.isna().sum()

    raise ValueError(
        "Missing molecular descriptor values detected:\n"
        + missing_counts[
            missing_counts > 0
        ].to_string()
    )


if not np.isfinite(
    descriptor_data.to_numpy()
).all():

    raise ValueError(
        "Non-finite molecular descriptor values detected."
    )


scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    descriptor_data
)


print(
    f"Descriptors standardised: "
    f"{len(FEATURE_COLUMNS)}"
)

for feature in FEATURE_COLUMNS:
    print(f"  - {feature}")


# ======================================================================
# K-NEAREST-NEIGHBOUR CHEMICAL SPACE DISTANCE
# ======================================================================

print("\n" + "=" * 70)
print("Calculating chemical-space proximity")
print("=" * 70)

print(
    f"Nearest neighbours used: {K_NEIGHBOURS}"
)

print(
    "Distance metric: Euclidean distance in standardized "
    "six-descriptor space"
)


# +1 because the first neighbour of every compound is itself.
knn = NearestNeighbors(
    n_neighbors=K_NEIGHBOURS + 1,
    metric="euclidean"
)

knn.fit(X_scaled)

distances, indices = knn.kneighbors(
    X_scaled
)


# First neighbour is the compound itself.
neighbour_distances = distances[
    :, 1:
]


diagnostic[
    "nearest_neighbor_distance"
] = neighbour_distances[
    :, 0
]


diagnostic[
    "mean_knn_distance"
] = neighbour_distances.mean(
    axis=1
)


diagnostic[
    "median_knn_distance"
] = np.median(
    neighbour_distances,
    axis=1
)


diagnostic[
    "maximum_knn_distance"
] = neighbour_distances.max(
    axis=1
)


print(
    "Chemical-space distances calculated successfully."
)

print(
    f"Mean nearest-neighbour distance: "
    f"{diagnostic['nearest_neighbor_distance'].mean():.4f}"
)

print(
    f"Median nearest-neighbour distance: "
    f"{diagnostic['nearest_neighbor_distance'].median():.4f}"
)

print(
    f"Mean {K_NEIGHBOURS}-NN distance: "
    f"{diagnostic['mean_knn_distance'].mean():.4f}"
)

print(
    f"Median {K_NEIGHBOURS}-NN distance: "
    f"{diagnostic['mean_knn_distance'].median():.4f}"
)


# ======================================================================
# APPLICABILITY-DOMAIN THRESHOLDS
# ======================================================================

print("\n" + "=" * 70)
print("Defining descriptive applicability-domain thresholds")
print("=" * 70)

borderline_threshold = np.percentile(
    diagnostic["mean_knn_distance"],
    BORDERLINE_PERCENTILE
)

out_of_domain_threshold = np.percentile(
    diagnostic["mean_knn_distance"],
    OUT_OF_DOMAIN_PERCENTILE
)


print(
    f"{BORDERLINE_PERCENTILE}th-percentile threshold: "
    f"{borderline_threshold:.4f}"
)

print(
    f"{OUT_OF_DOMAIN_PERCENTILE}th-percentile threshold: "
    f"{out_of_domain_threshold:.4f}"
)


diagnostic[
    "applicability_domain"
] = diagnostic[
    "mean_knn_distance"
].apply(
    classify_domain,
    args=(
        borderline_threshold,
        out_of_domain_threshold,
    )
)


# ======================================================================
# ERROR / UNCERTAINTY THRESHOLDS
# ======================================================================

print("\n" + "=" * 70)
print("Defining diagnostic error thresholds")
print("=" * 70)

error_threshold = np.percentile(
    diagnostic["absolute_error"],
    75
)

uncertainty_threshold = np.percentile(
    diagnostic["prediction_sd"],
    75
)


print(
    f"75th-percentile absolute error: "
    f"{error_threshold:.4f}"
)

print(
    f"75th-percentile prediction SD: "
    f"{uncertainty_threshold:.4f}"
)


diagnostic[
    "high_error"
] = (
    diagnostic["absolute_error"]
    >= error_threshold
)


diagnostic[
    "high_uncertainty"
] = (
    diagnostic["prediction_sd"]
    >= uncertainty_threshold
)


diagnostic[
    "high_error_high_uncertainty"
] = (
    diagnostic["high_error"]
    & diagnostic["high_uncertainty"]
)


# ======================================================================
# SOLUBILITY REGIONS
# ======================================================================

print("\n" + "=" * 70)
print("Assigning observed solubility regions")
print("=" * 70)


def solubility_region(value):

    if value < -6:
        return "< -6"

    if value < -4:
        return "-6 to < -4"

    if value < -2:
        return "-4 to < -2"

    if value < 0:
        return "-2 to < 0"

    return ">= 0"


diagnostic[
    "observed_solubility_region"
] = diagnostic[
    "observed_solubility"
].apply(
    solubility_region
)


# ======================================================================
# APPLICABILITY DOMAIN SUMMARY
# ======================================================================

print("\n" + "=" * 70)
print("APPLICABILITY-DOMAIN SUMMARY")
print("=" * 70)

domain_summary = region_summary(
    diagnostic
)

print(
    domain_summary.to_string(
        index=False
    )
)


# ======================================================================
# SAVE REGION SUMMARY
# ======================================================================

domain_summary.to_csv(
    OUTPUT_REGIONS,
    index=False
)

print(
    f"\nSaved: {OUTPUT_REGIONS}"
)


# ======================================================================
# CHEMICAL DISTANCE VS ERROR CORRELATIONS
# ======================================================================

print("\n" + "=" * 70)
print("CHEMICAL-SPACE DISTANCE ASSOCIATIONS")
print("=" * 70)

correlation_rows = []

targets = {
    "absolute_error": diagnostic[
        "absolute_error"
    ].to_numpy(),

    "prediction_sd": diagnostic[
        "prediction_sd"
    ].to_numpy(),

    "prediction_range": diagnostic[
        "prediction_range"
    ].to_numpy(),

    "residual": diagnostic[
        "residual"
    ].to_numpy(),
}


distance_metrics = {
    "nearest_neighbor_distance": diagnostic[
        "nearest_neighbor_distance"
    ].to_numpy(),

    "mean_knn_distance": diagnostic[
        "mean_knn_distance"
    ].to_numpy(),

    "maximum_knn_distance": diagnostic[
        "maximum_knn_distance"
    ].to_numpy(),
}


for distance_name, x in distance_metrics.items():

    for target_name, y in targets.items():

        pearson_r, pearson_p = safe_correlation(
            x,
            y,
            method="pearson"
        )

        spearman_r, spearman_p = safe_correlation(
            x,
            y,
            method="spearman"
        )

        correlation_rows.append(
            {
                "distance_metric": distance_name,
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
        index=False
    )
)


correlations.to_csv(
    OUTPUT_CORRELATIONS,
    index=False
)

print(
    f"\nSaved: {OUTPUT_CORRELATIONS}"
)


# ======================================================================
# HIGH-PRIORITY OUT-OF-DOMAIN COMPOUNDS
# ======================================================================

print("\n" + "=" * 70)
print("HIGH-PRIORITY APPLICABILITY-DOMAIN COMPOUNDS")
print("=" * 70)


priority = diagnostic[
    (
        diagnostic[
            "applicability_domain"
        ] == "Out-of-domain"
    )
    &
    (
        diagnostic[
            "high_error_high_uncertainty"
        ]
    )
].copy()


priority = priority.sort_values(
    [
        "mean_knn_distance",
        "absolute_error",
        "prediction_sd",
    ],
    ascending=False
)


print(
    "Out-of-domain + high-error + "
    f"high-uncertainty compounds: {len(priority):,}"
)


priority_columns = [
    "ID",
    "observed_solubility",
    "mean_prediction",
    "residual",
    "absolute_error",
    "prediction_sd",
    "prediction_range",
    "nearest_neighbor_distance",
    "mean_knn_distance",
    "maximum_knn_distance",
    "applicability_domain",
    "failure_mode",
] + FEATURE_COLUMNS


priority_columns = [
    column
    for column in priority_columns
    if column in priority.columns
]


priority[
    priority_columns
].to_csv(
    OUTPUT_PRIORITY,
    index=False
)


print(
    f"Saved: {OUTPUT_PRIORITY}"
)


# ======================================================================
# SAVE COMPLETE COMPOUND-LEVEL DATASET
# ======================================================================

output_columns = [
    "ID",
    "observed_solubility",
    "mean_prediction",
    "residual",
    "absolute_error",
    "prediction_sd",
    "prediction_range",
    "prediction_count",
    "nearest_neighbor_distance",
    "mean_knn_distance",
    "median_knn_distance",
    "maximum_knn_distance",
    "applicability_domain",
    "observed_solubility_region",
    "high_error",
    "high_uncertainty",
    "high_error_high_uncertainty",
    "failure_mode",
] + FEATURE_COLUMNS


output_columns = [
    column
    for column in output_columns
    if column in diagnostic.columns
]


diagnostic[
    output_columns
].to_csv(
    OUTPUT_AD,
    index=False
)


print(
    f"Saved: {OUTPUT_AD}"
)


# ======================================================================
# REPORT
# ======================================================================

print("\n" + "=" * 70)
print("Writing report")
print("=" * 70)


domain_counts = (
    diagnostic[
        "applicability_domain"
    ]
    .value_counts()
)


report_lines = []

report_lines.append(
    "SCRIPT 30 — CHEMICAL APPLICABILITY DOMAIN ANALYSIS"
)

report_lines.append(
    "=" * 70
)

report_lines.append("")

report_lines.append(
    "PURPOSE"
)

report_lines.append(
    "-------"
)

report_lines.append(
    "Assess the chemical-space support of the six-descriptor "
    "Gradient Boosting representation and determine whether "
    "chemical-space isolation is associated with prediction "
    "error, prediction instability, or model failure."
)

report_lines.append("")

report_lines.append(
    "DATA DESIGN"
)

report_lines.append(
    "-----------"
)

report_lines.append(
    f"Population C: {len(features):,} compounds"
)

report_lines.append(
    f"Compounds represented in Script 26: "
    f"{residuals['ID'].nunique():,}"
)

report_lines.append(
    f"Compounds integrated for AD analysis: "
    f"{len(diagnostic):,}"
)

report_lines.append(
    f"Descriptor dimensions: {len(FEATURE_COLUMNS)}"
)

report_lines.append(
    f"k-nearest neighbours: {K_NEIGHBOURS}"
)

report_lines.append(
    "Distance metric: Euclidean distance after "
    "standardisation of molecular descriptors"
)

report_lines.append("")

report_lines.append(
    "IMPORTANT METHODOLOGICAL NOTE"
)

report_lines.append(
    "------------------------------"
)

report_lines.append(
    "Script 26 does not retain the exact training compound IDs "
    "for each repeated scaffold-aware split. Consequently, "
    "the applicability-domain distance is a descriptive "
    "Population C chemical-space measure rather than a "
    "fold-specific training-set applicability-domain estimate."
)

report_lines.append(
    "No causal interpretation is assigned to associations "
    "between chemical-space distance and prediction error."
)

report_lines.append("")

report_lines.append(
    "CHEMICAL-SPACE DISTANCE"
)

report_lines.append(
    "-----------------------"
)

report_lines.append(
    f"Mean nearest-neighbour distance: "
    f"{diagnostic['nearest_neighbor_distance'].mean():.4f}"
)

report_lines.append(
    f"Median nearest-neighbour distance: "
    f"{diagnostic['nearest_neighbor_distance'].median():.4f}"
)

report_lines.append(
    f"Mean {K_NEIGHBOURS}-NN distance: "
    f"{diagnostic['mean_knn_distance'].mean():.4f}"
)

report_lines.append(
    f"Median {K_NEIGHBOURS}-NN distance: "
    f"{diagnostic['mean_knn_distance'].median():.4f}"
)

report_lines.append("")

report_lines.append(
    "APPLICABILITY-DOMAIN THRESHOLDS"
)

report_lines.append(
    "-------------------------------"
)

report_lines.append(
    f"Borderline threshold "
    f"({BORDERLINE_PERCENTILE}th percentile): "
    f"{borderline_threshold:.4f}"
)

report_lines.append(
    f"Out-of-domain threshold "
    f"({OUT_OF_DOMAIN_PERCENTILE}th percentile): "
    f"{out_of_domain_threshold:.4f}"
)

report_lines.append("")

report_lines.append(
    "APPLICABILITY-DOMAIN DISTRIBUTION"
)

report_lines.append(
    "---------------------------------"
)

for region in [
    "In-domain",
    "Borderline",
    "Out-of-domain",
]:

    count = int(
        domain_counts.get(
            region,
            0
        )
    )

    percentage = (
        100 * count / len(diagnostic)
    )

    report_lines.append(
        f"{region}: {count:,} "
        f"({percentage:.2f}%)"
    )

report_lines.append("")

report_lines.append(
    "ERROR / UNCERTAINTY ASSOCIATIONS"
)

report_lines.append(
    "--------------------------------"
)

for _, row in correlations.iterrows():

    report_lines.append(
        f"{row['distance_metric']} vs "
        f"{row['target']}: "
        f"Pearson r={row['pearson_r']:+.4f}, "
        f"p={row['pearson_p']:.4e}; "
        f"Spearman r={row['spearman_r']:+.4f}, "
        f"p={row['spearman_p']:.4e}"
    )

report_lines.append("")

report_lines.append(
    "APPLICABILITY-DOMAIN PERFORMANCE"
)

report_lines.append(
    "--------------------------------"
)

report_lines.append(
    domain_summary.to_string(
        index=False
    )
)

report_lines.append("")

report_lines.append(
    "HIGH-PRIORITY OUT-OF-DOMAIN COMPOUNDS"
)

report_lines.append(
    "------------------------------------"
)

report_lines.append(
    f"Out-of-domain + high-error + "
    f"high-uncertainty compounds: "
    f"{len(priority):,}"
)

report_lines.append("")

report_lines.append(
    "INTERPRETATION FRAMEWORK"
)

report_lines.append(
    "------------------------"
)

report_lines.append(
    "In-domain compounds occupy relatively well-supported "
    "regions of the standardized six-descriptor chemical space."
)

report_lines.append(
    "Borderline compounds occupy less densely represented "
    "regions and warrant additional prediction caution."
)

report_lines.append(
    "Out-of-domain compounds are relatively isolated from "
    "the Population C descriptor distribution under the "
    "k-nearest-neighbour distance definition used here."
)

report_lines.append(
    "Higher chemical-space distance combined with high "
    "prediction error and high prediction instability provides "
    "a stronger diagnostic signal than distance alone."
)

report_lines.append(
    "These categories describe model applicability and "
    "diagnostic behaviour; they do not demonstrate that "
    "chemical-space isolation causes prediction error."
)

REPORT_PATH.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)


print(
    f"Report saved: {REPORT_PATH}"
)


# ======================================================================
# FINAL OUTPUT
# ======================================================================

print("\n" + "=" * 70)
print("SCRIPT 30 COMPLETE")
print("=" * 70)

print(
    f"Population C: {len(features):,}"
)

print(
    f"Compounds analysed: {len(diagnostic):,}"
)

print(
    "Applicability-domain distribution:"
)

for region in [
    "In-domain",
    "Borderline",
    "Out-of-domain",
]:

    count = int(
        domain_counts.get(
            region,
            0
        )
    )

    print(
        f"  {region}: {count:,}"
    )

print(
    "\nOutput files:"
)

print(
    f"  {OUTPUT_AD}"
)

print(
    f"  {OUTPUT_REGIONS}"
)

print(
    f"  {OUTPUT_CORRELATIONS}"
)

print(
    f"  {OUTPUT_PRIORITY}"
)

print(
    f"  {REPORT_PATH}"
)

print("=" * 70)