"""
SCRIPT 27 — CHEMICAL ERROR HETEROGENEITY ANALYSIS

Purpose
-------
Determine whether Gradient Boosting prediction error varies systematically
across chemical-property regions, solubility regimes, molecular complexity,
and scaffold representation.

This script builds directly on Script 26.

Important methodological principles
-----------------------------------
1. Population C remains locked at 8,643 compounds.
2. No new model is trained.
3. No random molecule-level validation is introduced.
4. Script 26 held-out predictions are used as the source of model errors.
5. Repeated predictions are aggregated to the compound level before
   subgroup analysis so compounds are not disproportionately weighted
   simply because they appeared in more test repetitions.
6. Extreme-error compounds are retained.
7. No causal conclusions are drawn from subgroup associations.

Outputs
-------
data/processed/error_heterogeneity_by_feature.csv
data/processed/error_heterogeneity_by_solubility.csv
data/processed/error_heterogeneity_by_scaffold.csv
data/processed/error_heterogeneity_summary.csv

reports/error_heterogeneity_analysis.txt
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "molecular_features.csv"
)

RESIDUAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nonlinear_residual_analysis.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_POPULATION = 8643

FEATURES = [
    "rdkit_molwt",
    "rdkit_mollogp",
    "rdkit_ring_count",
    "rdkit_aromatic_rings",
    "rdkit_rotatable_bonds",
    "rdkit_fraction_csp3",
]

ID_COL = "ID"
SMILES_COL = "SMILES"
TARGET_COL = "Solubility"


# ============================================================
# HELPERS
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def safe_numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def calculate_metrics(group):
    """
    Calculate error statistics for a group of compound-level observations.
    """

    residual = group["mean_residual"]
    abs_error = group["mean_absolute_error"]

    return pd.Series({
        "compound_count": len(group),
        "mean_residual": residual.mean(),
        "median_residual": residual.median(),
        "mean_absolute_error": abs_error.mean(),
        "median_absolute_error": abs_error.median(),
        "rmse": np.sqrt(np.mean(residual ** 2)),
        "underprediction_pct": (residual > 0).mean() * 100,
        "overprediction_pct": (residual < 0).mean() * 100,
    })


def make_quantile_bins(series, q=5):
    """
    Create approximately equal-sized quantile bins.

    Duplicated quantile boundaries can occur when a descriptor contains
    many repeated values, so duplicates='drop' is used.
    """

    try:
        result = pd.qcut(
            series,
            q=q,
            duplicates="drop"
        )

        return result

    except ValueError:
        return pd.Series(
            ["All observations"] * len(series),
            index=series.index,
            dtype="object"
        )


def scaffold_from_smiles(smiles):
    """
    Generate a Bemis-Murcko scaffold from SMILES.
    """

    if pd.isna(smiles):
        return None

    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:
        return None

    try:
        return MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol,
            includeChirality=False
        )
    except Exception:
        return None


# ============================================================
# START
# ============================================================

print_header("SCRIPT 27 — CHEMICAL ERROR HETEROGENEITY ANALYSIS")

print(f"Project root: {PROJECT_ROOT}")

# ============================================================
# LOAD MOLECULAR FEATURES
# ============================================================

print_header("Loading molecular features")

if not FEATURE_FILE.exists():
    print(f"ERROR: Molecular feature file not found:")
    print(FEATURE_FILE)
    sys.exit(1)

features = pd.read_csv(FEATURE_FILE)

print(f"Feature shape: {features.shape}")

required_feature_columns = [
    ID_COL,
    SMILES_COL,
    TARGET_COL,
    *FEATURES,
]

missing_features = [
    col for col in required_feature_columns
    if col not in features.columns
]

if missing_features:
    print("ERROR: Missing required molecular feature columns:")
    for col in missing_features:
        print(f"  - {col}")
    sys.exit(1)

features = features[required_feature_columns].copy()

features = safe_numeric(
    features,
    [TARGET_COL] + FEATURES
)

# ------------------------------------------------------------
# Population C check
# ------------------------------------------------------------

if len(features) != EXPECTED_POPULATION:
    print(
        f"ERROR: Expected Population C to contain "
        f"{EXPECTED_POPULATION:,} compounds, "
        f"but found {len(features):,}."
    )
    sys.exit(1)

print(
    f"Population C verified: "
    f"{len(features):,} compounds"
)

if features[ID_COL].nunique() != EXPECTED_POPULATION:
    print("ERROR: ID is not unique within Population C.")
    sys.exit(1)

print("Unique compound IDs verified.")


# ============================================================
# LOAD SCRIPT 26 PREDICTIONS / RESIDUALS
# ============================================================

print_header("Loading Script 26 residual analysis")

if not RESIDUAL_FILE.exists():
    print("ERROR: Script 26 residual file not found:")
    print(RESIDUAL_FILE)
    print()
    print(
        "Run Script 26 first and ensure that "
        "nonlinear_residual_analysis.csv exists."
    )
    sys.exit(1)

residuals = pd.read_csv(RESIDUAL_FILE)

print(f"Residual dataset shape: {residuals.shape}")
print()
print("Columns found:")
for col in residuals.columns:
    print(f"  {col}")


# ============================================================
# IDENTIFY SCRIPT 26 COLUMN NAMES
# ============================================================

"""
Script 26 may contain slightly different naming conventions depending
on the exact implementation. We therefore identify the required
columns without silently guessing values.
"""

column_aliases = {
    "ID": [
        "ID",
        "id",
        "compound_id",
    ],

    "observed": [
        "Solubility",
        "observed",
        "Observed",
        "observed_solubility",
        "Observed_Solubility",
        "y_true",
        "actual",
    ],

    "predicted": [
        "predicted",
        "Predicted",
        "predicted_solubility",
        "Predicted_Solubility",
        "y_pred",
        "prediction",
    ],

    "residual": [
        "residual",
        "Residual",
        "error",
    ],

    "absolute_error": [
        "absolute_error",
        "Absolute_Error",
        "abs_error",
        "absolute_residual",
    ],
}


def find_column(df, aliases):
    for alias in aliases:
        if alias in df.columns:
            return alias
    return None


resolved = {}

for key, aliases in column_aliases.items():
    resolved[key] = find_column(residuals, aliases)


print()
print("Resolved Script 26 columns:")

for key, value in resolved.items():
    print(f"  {key}: {value}")


required_residual_keys = [
    "ID",
    "observed",
    "predicted",
]

missing_residual_keys = [
    key
    for key in required_residual_keys
    if resolved[key] is None
]

if missing_residual_keys:
    print()
    print("ERROR: Could not identify required Script 26 columns:")
    for key in missing_residual_keys:
        print(f"  - {key}")

    print()
    print(
        "Do not modify this script to guess column meanings. "
        "Inspect Script 26 output and map the exact columns."
    )

    sys.exit(1)


# ============================================================
# STANDARDIZE RESIDUAL DATA
# ============================================================

rename_map = {
    resolved["ID"]: "ID",
    resolved["observed"]: "observed",
    resolved["predicted"]: "predicted",
}

if resolved["residual"] is not None:
    rename_map[resolved["residual"]] = "residual"

if resolved["absolute_error"] is not None:
    rename_map[resolved["absolute_error"]] = "absolute_error"

residuals = residuals.rename(columns=rename_map)

residuals = safe_numeric(
    residuals,
    [
        "observed",
        "predicted",
        "residual",
        "absolute_error",
    ],
)


# ============================================================
# RECONSTRUCT ERROR VARIABLES IF NECESSARY
# ============================================================

if "residual" not in residuals.columns:
    residuals["residual"] = (
        residuals["observed"]
        - residuals["predicted"]
    )

if "absolute_error" not in residuals.columns:
    residuals["absolute_error"] = (
        residuals["residual"].abs()
    )


# ============================================================
# BASIC VALIDATION
# ============================================================

residuals = residuals.dropna(
    subset=[
        "ID",
        "observed",
        "predicted",
        "residual",
        "absolute_error",
    ]
).copy()

print()
print(
    f"Valid Script 26 prediction rows: "
    f"{len(residuals):,}"
)

print(
    f"Unique compounds represented: "
    f"{residuals['ID'].nunique():,}"
)


if not set(residuals["ID"]).issubset(set(features["ID"])):
    unmatched = set(residuals["ID"]) - set(features["ID"])

    print()
    print(
        f"ERROR: {len(unmatched):,} Script 26 IDs "
        "are not present in Population C."
    )

    sys.exit(1)


# ============================================================
# REPEATED PREDICTIONS → COMPOUND-LEVEL ERROR
# ============================================================

print_header("Aggregating repeated held-out predictions")

"""
Each compound may appear in multiple scaffold-aware test repetitions.

For subgroup analysis we first aggregate to one row per compound.

This prevents a compound from being given extra influence simply because
its scaffold happened to be selected as test more frequently across the
10 repetitions.
"""

compound_error = (
    residuals
    .groupby("ID", as_index=False)
    .agg(
        mean_observed=("observed", "mean"),
        mean_predicted=("predicted", "mean"),
        mean_residual=("residual", "mean"),
        mean_absolute_error=("absolute_error", "mean"),
        sd_residual=("residual", "std"),
        prediction_count=("residual", "count"),
    )
)

compound_error["sd_residual"] = (
    compound_error["sd_residual"].fillna(0)
)

print(
    f"Compound-level error table: "
    f"{len(compound_error):,} compounds"
)

print(
    f"Mean predictions per compound: "
    f"{compound_error['prediction_count'].mean():.2f}"
)


# ============================================================
# MERGE MOLECULAR FEATURES
# ============================================================

analysis = compound_error.merge(
    features,
    on="ID",
    how="left",
    validate="one_to_one",
)

if len(analysis) != len(compound_error):
    print("ERROR: Merge changed compound count.")
    sys.exit(1)

missing_feature_rows = analysis[FEATURES].isna().any(axis=1).sum()

if missing_feature_rows > 0:
    print(
        f"WARNING: {missing_feature_rows:,} compounds "
        "have missing molecular descriptors."
    )


# ============================================================
# 1. ERROR BY MOLECULAR PROPERTY QUINTILES
# ============================================================

print_header("Error heterogeneity across molecular-property regions")

feature_tables = []

for feature in FEATURES:

    subset = analysis[
        [ID_COL, feature, "mean_residual", "mean_absolute_error"]
    ].dropna(subset=[feature]).copy()

    subset["feature"] = feature

    subset["region"] = make_quantile_bins(
        subset[feature],
        q=5
    )

    summary = (
        subset
        .groupby(
            ["feature", "region"],
            observed=True
        )
        .apply(
            calculate_metrics,
            include_groups=False
        )
        .reset_index()
    )

    summary["feature_min"] = (
        subset.groupby(
            ["feature", "region"],
            observed=True
        )[feature]
        .min()
        .values
    )

    summary["feature_max"] = (
        subset.groupby(
            ["feature", "region"],
            observed=True
        )[feature]
        .max()
        .values
    )

    feature_tables.append(summary)


error_by_feature = pd.concat(
    feature_tables,
    ignore_index=True
)

feature_output = (
    OUTPUT_DIR
    / "error_heterogeneity_by_feature.csv"
)

error_by_feature.to_csv(
    feature_output,
    index=False
)

print(
    f"Saved: {feature_output}"
)


# ============================================================
# 2. ERROR BY SOLUBILITY REGIME
# ============================================================

print_header("Error heterogeneity across solubility regimes")

"""
These ranges follow the established AqSolDB LogS interpretation and
the same broad regions examined in Script 26.

AqSolDB describes LogS >= 0 as highly soluble, 0 to -2 as soluble,
-2 to -4 as slightly soluble, and values below -4 as insoluble.
"""

def solubility_region(value):

    if value < -6:
        return "< -6"

    if value < -4:
        return "-6 to < -4"

    if value < -2:
        return "-4 to < -2"

    if value < 0:
        return "-2 to < 0"

    if value < 2:
        return "0 to < 2"

    return ">= 2"


analysis["solubility_region"] = (
    analysis["mean_observed"]
    .apply(solubility_region)
)

solubility_order = [
    "< -6",
    "-6 to < -4",
    "-4 to < -2",
    "-2 to < 0",
    "0 to < 2",
    ">= 2",
]

error_by_solubility = (
    analysis
    .groupby(
        "solubility_region",
        observed=False
    )
    .apply(
        calculate_metrics,
        include_groups=False
    )
    .reset_index()
)

error_by_solubility["solubility_region"] = pd.Categorical(
    error_by_solubility["solubility_region"],
    categories=solubility_order,
    ordered=True
)

error_by_solubility = (
    error_by_solubility
    .sort_values("solubility_region")
)

solubility_output = (
    OUTPUT_DIR
    / "error_heterogeneity_by_solubility.csv"
)

error_by_solubility.to_csv(
    solubility_output,
    index=False
)

print(
    f"Saved: {solubility_output}"
)


# ============================================================
# 3. SCAFFOLD REPRESENTATION / RARITY
# ============================================================

print_header("Generating Bemis-Murcko scaffold representation")

scaffold_cache = {}

for smiles in analysis[SMILES_COL].dropna().unique():

    scaffold_cache[smiles] = scaffold_from_smiles(smiles)

analysis["murcko_scaffold"] = (
    analysis[SMILES_COL]
    .map(scaffold_cache)
)

valid_scaffolds = analysis["murcko_scaffold"].notna().sum()

print(
    f"Compounds with valid Murcko scaffolds: "
    f"{valid_scaffolds:,}"
)

print(
    f"Unique Murcko scaffolds: "
    f"{analysis['murcko_scaffold'].nunique():,}"
)


# ------------------------------------------------------------
# Scaffold frequency
# ------------------------------------------------------------

scaffold_counts = (
    analysis
    .groupby("murcko_scaffold")["ID"]
    .transform("count")
)

analysis["scaffold_frequency"] = scaffold_counts


def scaffold_frequency_region(n):

    if pd.isna(n):
        return "Unknown"

    if n == 1:
        return "Singleton"

    if n <= 5:
        return "2–5 compounds"

    if n <= 20:
        return "6–20 compounds"

    return ">20 compounds"


analysis["scaffold_frequency_region"] = (
    analysis["scaffold_frequency"]
    .apply(scaffold_frequency_region)
)

scaffold_order = [
    "Singleton",
    "2–5 compounds",
    "6–20 compounds",
    ">20 compounds",
    "Unknown",
]


error_by_scaffold = (
    analysis
    .groupby(
        "scaffold_frequency_region",
        observed=False
    )
    .apply(
        calculate_metrics,
        include_groups=False
    )
    .reset_index()
)

error_by_scaffold["scaffold_frequency_region"] = pd.Categorical(
    error_by_scaffold["scaffold_frequency_region"],
    categories=scaffold_order,
    ordered=True
)

error_by_scaffold = (
    error_by_scaffold
    .sort_values("scaffold_frequency_region")
)

scaffold_output = (
    OUTPUT_DIR
    / "error_heterogeneity_by_scaffold.csv"
)

error_by_scaffold.to_csv(
    scaffold_output,
    index=False
)

print(
    f"Saved: {scaffold_output}"
)


# ============================================================
# 4. CORRELATION OF MOLECULAR FEATURES WITH ERROR MAGNITUDE
# ============================================================

print_header("Correlation between molecular properties and error magnitude")

correlation_rows = []

for feature in FEATURES:

    subset = analysis[
        [feature, "mean_residual", "mean_absolute_error"]
    ].dropna()

    if len(subset) < 3:
        continue

    pearson_abs, pearson_p = pearsonr(
        subset[feature],
        subset["mean_absolute_error"]
    )

    spearman_abs, spearman_p = spearmanr(
        subset[feature],
        subset["mean_absolute_error"]
    )

    pearson_resid, pearson_resid_p = pearsonr(
        subset[feature],
        subset["mean_residual"]
    )

    spearman_resid, spearman_resid_p = spearmanr(
        subset[feature],
        subset["mean_residual"]
    )

    correlation_rows.append({
        "feature": feature,
        "n": len(subset),

        "pearson_error": pearson_abs,
        "pearson_error_p": pearson_p,

        "spearman_error": spearman_abs,
        "spearman_error_p": spearman_p,

        "pearson_residual": pearson_resid,
        "pearson_residual_p": pearson_resid_p,

        "spearman_residual": spearman_resid,
        "spearman_residual_p": spearman_resid_p,
    })

correlations = pd.DataFrame(correlation_rows)


# ============================================================
# 5. IDENTIFY HIGH-ERROR CHEMICAL REGIONS
# ============================================================

print_header("Identifying high-error molecular-property regions")

high_error_rows = []

for feature in FEATURES:

    subset = analysis[
        [ID_COL, feature, "mean_absolute_error"]
    ].dropna().copy()

    if len(subset) == 0:
        continue

    subset["feature"] = feature

    subset["region"] = make_quantile_bins(
        subset[feature],
        q=5
    )

    region_stats = (
        subset
        .groupby(
            ["feature", "region"],
            observed=True
        )
        .agg(
            compound_count=("ID", "count"),
            mean_absolute_error=(
                "mean_absolute_error",
                "mean"
            ),
            median_absolute_error=(
                "mean_absolute_error",
                "median"
            ),
        )
        .reset_index()
    )

    overall_mae = subset["mean_absolute_error"].mean()

    region_stats["overall_mae"] = overall_mae

    region_stats["mae_ratio_to_overall"] = (
        region_stats["mean_absolute_error"]
        / overall_mae
    )

    high_error_rows.append(region_stats)


high_error_regions = pd.concat(
    high_error_rows,
    ignore_index=True
)

high_error_regions = high_error_regions.sort_values(
    "mae_ratio_to_overall",
    ascending=False
)


# ============================================================
# 6. SUMMARY TABLE
# ============================================================

summary_rows = []

# Overall compound-level performance
summary_rows.append({
    "analysis": "Overall compound-level error",
    "group": "All Population C compounds",
    "compound_count": len(analysis),
    "mean_residual": analysis["mean_residual"].mean(),
    "mean_absolute_error": analysis["mean_absolute_error"].mean(),
    "rmse": np.sqrt(
        np.mean(
            analysis["mean_residual"] ** 2
        )
    ),
})

# Highest-error region for each descriptor
for feature in FEATURES:

    subset = high_error_regions[
        high_error_regions["feature"] == feature
    ]

    if len(subset) == 0:
        continue

    top = subset.iloc[0]

    summary_rows.append({
        "analysis": "Highest-error descriptor region",
        "group": (
            f"{feature}: {top['region']}"
        ),
        "compound_count": int(
            top["compound_count"]
        ),
        "mean_residual": np.nan,
        "mean_absolute_error": (
            top["mean_absolute_error"]
        ),
        "rmse": np.nan,
    })


# Highest-error solubility region
if len(error_by_solubility) > 0:

    top_sol = error_by_solubility.loc[
        error_by_solubility["mean_absolute_error"].idxmax()
    ]

    summary_rows.append({
        "analysis": "Highest-error solubility region",
        "group": str(top_sol["solubility_region"]),
        "compound_count": int(
            top_sol["compound_count"]
        ),
        "mean_residual": top_sol["mean_residual"],
        "mean_absolute_error": (
            top_sol["mean_absolute_error"]
        ),
        "rmse": top_sol["rmse"],
    })


# Scaffold rarity
if len(error_by_scaffold) > 0:

    scaffold_known = error_by_scaffold[
        error_by_scaffold[
            "scaffold_frequency_region"
        ].astype(str) != "Unknown"
    ]

    if len(scaffold_known) > 0:

        top_scaffold = scaffold_known.loc[
            scaffold_known[
                "mean_absolute_error"
            ].idxmax()
        ]

        summary_rows.append({
            "analysis": "Highest-error scaffold-frequency region",
            "group": str(
                top_scaffold[
                    "scaffold_frequency_region"
                ]
            ),
            "compound_count": int(
                top_scaffold["compound_count"]
            ),
            "mean_residual": (
                top_scaffold["mean_residual"]
            ),
            "mean_absolute_error": (
                top_scaffold[
                    "mean_absolute_error"
                ]
            ),
            "rmse": top_scaffold["rmse"],
        })


summary = pd.DataFrame(summary_rows)

summary_output = (
    OUTPUT_DIR
    / "error_heterogeneity_summary.csv"
)

summary.to_csv(
    summary_output,
    index=False
)

print(
    f"Saved: {summary_output}"
)


# ============================================================
# 7. REPORT
# ============================================================

print_header("Writing report")

report_file = (
    REPORT_DIR
    / "error_heterogeneity_analysis.txt"
)

with open(report_file, "w", encoding="utf-8") as f:

    f.write(
        "SCRIPT 27 — CHEMICAL ERROR HETEROGENEITY ANALYSIS\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        "PURPOSE\n"
        "-------\n"
    )

    f.write(
        "Determine whether Gradient Boosting prediction error varies "
        "systematically across molecular-property regions, solubility "
        "regimes, and scaffold representation.\n\n"
    )

    f.write(
        "VALIDATION / DATA DESIGN\n"
        "------------------------\n"
    )

    f.write(
        f"Population C: {EXPECTED_POPULATION:,} compounds\n"
    )

    f.write(
        "Model source: Script 26 repeated scaffold-aware "
        "held-out predictions\n"
    )

    f.write(
        "Compound-level aggregation: mean error across "
        "available held-out repetitions\n\n"
    )

    f.write(
        "IMPORTANT INTERPRETATION NOTE\n"
        "------------------------------\n"
    )

    f.write(
        "Subgroup differences describe where the model performs "
        "better or worse. They do not establish causal relationships "
        "between molecular descriptors and prediction error.\n\n"
    )

    # Overall
    f.write(
        "OVERALL COMPOUND-LEVEL ERROR\n"
        "----------------------------\n"
    )

    f.write(
        f"Compounds analysed: {len(analysis):,}\n"
    )

    f.write(
        f"Mean residual: "
        f"{analysis['mean_residual'].mean():.4f}\n"
    )

    f.write(
        f"Mean absolute error: "
        f"{analysis['mean_absolute_error'].mean():.4f}\n"
    )

    f.write(
        f"RMSE: "
        f"{np.sqrt(np.mean(analysis['mean_residual'] ** 2)):.4f}\n"
    )

    f.write("\n")

    # Solubility
    f.write(
        "ERROR BY OBSERVED SOLUBILITY REGION\n"
        "------------------------------------\n"
    )

    f.write(
        error_by_solubility.to_string(
            index=False
        )
    )

    f.write("\n\n")

    # Scaffold
    f.write(
        "ERROR BY SCAFFOLD FREQUENCY\n"
        "---------------------------\n"
    )

    f.write(
        error_by_scaffold.to_string(
            index=False
        )
    )

    f.write("\n\n")

    # Correlations
    f.write(
        "ERROR CORRELATIONS\n"
        "------------------\n"
    )

    f.write(
        correlations.to_string(
            index=False
        )
    )

    f.write("\n\n")

    # Highest error regions
    f.write(
        "HIGHEST-ERROR MOLECULAR PROPERTY REGIONS\n"
        "-----------------------------------------\n"
    )

    for feature in FEATURES:

        subset = high_error_regions[
            high_error_regions["feature"] == feature
        ]

        if len(subset) == 0:
            continue

        top = subset.iloc[0]

        f.write(
            f"\n{feature}\n"
        )

        f.write(
            f"  Highest-error region: {top['region']}\n"
        )

        f.write(
            f"  Compounds: "
            f"{int(top['compound_count']):,}\n"
        )

        f.write(
            f"  Mean MAE: "
            f"{top['mean_absolute_error']:.4f}\n"
        )

        f.write(
            f"  Overall MAE ratio: "
            f"{top['mae_ratio_to_overall']:.3f}\n"
        )

    f.write("\n\n")

    # Top regions overall
    f.write(
        "TOP 15 HIGHEST-ERROR DESCRIPTOR REGIONS\n"
        "---------------------------------------\n"
    )

    f.write(
        high_error_regions.head(15).to_string(
            index=False
        )
    )

    f.write("\n\n")

    # Conclusion framework
    f.write(
        "INTERPRETATION FRAMEWORK\n"
        "------------------------\n"
    )

    f.write(
        "The analysis should be interpreted as evidence of "
        "error heterogeneity rather than evidence of causation. "
        "Descriptor regions with higher MAE indicate areas where "
        "the current six-descriptor Gradient Boosting representation "
        "has greater predictive difficulty. Differences in mean "
        "residual indicate conditional directional bias within "
        "those regions.\n\n"
    )

    f.write(
        "Particular attention should be given to regions where:\n"
        "1. compound count is sufficiently large;\n"
        "2. MAE is substantially above the overall MAE;\n"
        "3. residual direction is consistently positive or negative;\n"
        "4. the result is supported by more than one descriptor;\n"
        "5. scaffold frequency suggests limited structural representation.\n"
    )


print(
    f"Report saved: {report_file}"
)


# ============================================================
# FINAL CONSOLE SUMMARY
# ============================================================

print_header("SCRIPT 27 COMPLETE")

print(
    f"Population C: {len(features):,}"
)

print(
    f"Compound-level predictions analysed: "
    f"{len(analysis):,}"
)

print(
    f"Unique Murcko scaffolds: "
    f"{analysis['murcko_scaffold'].nunique():,}"
)

print()

print(
    "Overall compound-level error:"
)

print(
    f"  Mean residual: "
    f"{analysis['mean_residual'].mean():.4f}"
)

print(
    f"  Mean absolute error: "
    f"{analysis['mean_absolute_error'].mean():.4f}"
)

print(
    f"  RMSE: "
    f"{np.sqrt(np.mean(analysis['mean_residual'] ** 2)):.4f}"
)

print()

print("Output files:")

print(
    f"  {OUTPUT_DIR / 'error_heterogeneity_by_feature.csv'}"
)

print(
    f"  {OUTPUT_DIR / 'error_heterogeneity_by_solubility.csv'}"
)

print(
    f"  {OUTPUT_DIR / 'error_heterogeneity_by_scaffold.csv'}"
)

print(
    f"  {OUTPUT_DIR / 'error_heterogeneity_summary.csv'}"
)

print(
    f"  {REPORT_DIR / 'error_heterogeneity_analysis.txt'}"
)

print()
print("=" * 70)