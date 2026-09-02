"""
SCRIPT 26 — NONLINEAR MODEL RESIDUAL & ERROR ANALYSIS

Purpose
-------
Investigate where the Gradient Boosting model succeeds and where it
systematically struggles under repeated scaffold-aware evaluation.

The analysis focuses on held-out predictions from the same ten
scaffold-aware repetitions used in Scripts 23–25.

Model
-----
Gradient Boosting using the six structural descriptors:

    rdkit_molwt
    rdkit_mollogp
    rdkit_ring_count
    rdkit_aromatic_rings
    rdkit_rotatable_bonds
    rdkit_fraction_csp3

Method
------
For each of the ten scaffold-aware repetitions:

1. Recreate the same deterministic scaffold split.
2. Fit Gradient Boosting only on the training compounds.
3. Generate predictions only for the held-out test compounds.
4. Calculate residuals and absolute errors.
5. Examine error patterns against observed/predicted solubility
   and molecular descriptors.

Residual definition
-------------------
residual = observed_solubility - predicted_solubility

Therefore:

    positive residual -> model underpredicts solubility
    negative residual -> model overpredicts solubility

This script does NOT alter Population C.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "curated-solubility-dataset.csv"
)

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "molecular_features.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nonlinear_residual_analysis.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nonlinear_residual_summary.csv"
)

EXTREME_ERROR_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nonlinear_extreme_errors.csv"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "nonlinear_residual_analysis.txt"
)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

N_REPEATS = 10
TEST_FRACTION = 0.20

RANDOM_SEEDS = list(range(100, 100 + N_REPEATS))

TARGET = "Solubility"

NONLINEAR_FEATURES = [
    "rdkit_molwt",
    "rdkit_mollogp",
    "rdkit_ring_count",
    "rdkit_aromatic_rings",
    "rdkit_rotatable_bonds",
    "rdkit_fraction_csp3",
]

# Exact Gradient Boosting configuration from Script 25.
GB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "loss": "squared_error",
    "random_state": 42,
}


# ---------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------

print("=" * 70)
print("SCRIPT 26 — NONLINEAR MODEL RESIDUAL & ERROR ANALYSIS")
print("=" * 70)


# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------

print("\nLoading raw dataset:")
print(RAW_FILE)

raw = pd.read_csv(RAW_FILE)

print(f"Raw dataset shape: {raw.shape}")


print("\nLoading molecular features:")
print(FEATURE_FILE)

features = pd.read_csv(FEATURE_FILE)

print(f"Molecular feature shape: {features.shape}")


# ---------------------------------------------------------------------
# VERIFY POPULATION C
# ---------------------------------------------------------------------

print("\n" + "-" * 70)
print("POPULATION C")
print("-" * 70)

if "standard_analytical_domain" not in features.columns:
    raise ValueError(
        "standard_analytical_domain is missing from "
        "molecular_features.csv"
    )

population = features[
    features["standard_analytical_domain"] == True
].copy()

print(
    "Rows flagged as standard analytical domain:",
    len(population)
)

if len(population) != 8643:
    raise ValueError(
        f"Expected Population C = 8643 rows, found {len(population)}"
    )

print("Population C verified: 8,643 rows")


# ---------------------------------------------------------------------
# ENSURE TARGET EXISTS
# ---------------------------------------------------------------------

if TARGET not in population.columns:

    if TARGET not in raw.columns:
        raise ValueError(
            f"{TARGET} not found in either molecular_features.csv "
            "or raw dataset."
        )

    target_lookup = raw[["ID", TARGET]].copy()

    population = population.drop(
        columns=[TARGET],
        errors="ignore"
    )

    population = population.merge(
        target_lookup,
        on="ID",
        how="left",
        validate="one_to_one",
    )


if population[TARGET].isna().any():
    raise ValueError(
        "Missing target values detected."
    )

print(
    f"Rows after solubility merge: {len(population):,}"
)


# ---------------------------------------------------------------------
# VERIFY REQUIRED FEATURES
# ---------------------------------------------------------------------

required_features = sorted(
    set(
        NONLINEAR_FEATURES
        + [TARGET, "SMILES", "ID"]
    )
)

missing_features = [
    col
    for col in required_features
    if col not in population.columns
]

if missing_features:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing_features)
    )

print(
    f"Rows after required-feature filtering: "
    f"{len(population):,}"
)


# ---------------------------------------------------------------------
# GENERATE BEMIS-MURCKO SCAFFOLDS
# ---------------------------------------------------------------------

print("\n" + "-" * 70)
print("SCAFFOLD GENERATION")
print("-" * 70)

print("Generating Bemis-Murcko scaffolds...")


def get_scaffold(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol,
        includeChirality=False
    )

    return scaffold


population["scaffold"] = population["SMILES"].apply(
    get_scaffold
)


if population["scaffold"].isna().any():

    invalid_count = population["scaffold"].isna().sum()

    raise ValueError(
        f"{invalid_count} compounds have invalid scaffold generation."
    )


n_scaffolds = population["scaffold"].nunique()

print(
    f"Unique scaffold groups: {n_scaffolds:,}"
)


# ---------------------------------------------------------------------
# SCAFFOLD SPLITTER
# ---------------------------------------------------------------------

def make_scaffold_split(
    df,
    test_fraction,
    seed,
):
    """
    Reproduce the scaffold-level splitting procedure used in
    Scripts 22–25.
    """

    rng = np.random.default_rng(seed)

    scaffold_sizes = (
        df.groupby("scaffold")
        .size()
        .sort_values(ascending=False)
    )

    scaffolds = scaffold_sizes.index.to_numpy()

    shuffled = scaffolds.copy()

    rng.shuffle(shuffled)

    target_test_n = int(
        round(len(df) * test_fraction)
    )

    test_scaffolds = []

    test_n = 0

    for scaffold in shuffled:

        if test_n >= target_test_n:
            break

        test_scaffolds.append(scaffold)

        test_n += scaffold_sizes.loc[scaffold]

    test_scaffolds = set(test_scaffolds)

    test_mask = df["scaffold"].isin(
        test_scaffolds
    )

    train_idx = df.index[~test_mask]

    test_idx = df.index[test_mask]

    return train_idx, test_idx


# ---------------------------------------------------------------------
# REPEATED HELD-OUT RESIDUAL ANALYSIS
# ---------------------------------------------------------------------

print("\n" + "-" * 70)
print("REPEATED SCAFFOLD-AWARE RESIDUAL ANALYSIS")
print("-" * 70)

print(
    f"Repeats: {N_REPEATS}"
)

print(
    f"Target test fraction: {TEST_FRACTION:.0%}"
)

print(
    "Gradient Boosting configuration:"
)

for key, value in GB_PARAMS.items():
    print(
        f"  {key}: {value}"
    )

print("\nResidual definition:")
print(
    "  residual = observed solubility - predicted solubility"
)

print(
    "  positive residual = underprediction"
)

print(
    "  negative residual = overprediction"
)


# ---------------------------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------------------------

all_predictions = []

repetition_summary = []


for repetition, seed in enumerate(
    RANDOM_SEEDS,
    start=1
):

    train_idx, test_idx = make_scaffold_split(
        population,
        TEST_FRACTION,
        seed,
    )

    train_df = population.loc[
        train_idx
    ].copy()

    test_df = population.loc[
        test_idx
    ].copy()

    train_scaffolds = set(
        train_df["scaffold"]
    )

    test_scaffolds = set(
        test_df["scaffold"]
    )

    overlap = train_scaffolds.intersection(
        test_scaffolds
    )

    if overlap:
        raise ValueError(
            f"Scaffold leakage detected in "
            f"repetition {repetition}."
        )

    # -------------------------------------------------------------
    # FIT GRADIENT BOOSTING
    # -------------------------------------------------------------

    X_train = train_df[
        NONLINEAR_FEATURES
    ]

    y_train = train_df[
        TARGET
    ]

    X_test = test_df[
        NONLINEAR_FEATURES
    ]

    y_test = test_df[
        TARGET
    ]

    model = GradientBoostingRegressor(
        **GB_PARAMS
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    residuals = (
        y_test.to_numpy()
        - predictions
    )

    absolute_errors = np.abs(
        residuals
    )

    squared_errors = (
        residuals ** 2
    )

    # -------------------------------------------------------------
    # PERFORMANCE
    # -------------------------------------------------------------

    r2 = r2_score(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mean_error = np.mean(
        residuals
    )

    median_error = np.median(
        residuals
    )

    error_sd = np.std(
        residuals,
        ddof=1
    )

    # -------------------------------------------------------------
    # STORE ROW-LEVEL PREDICTIONS
    # -------------------------------------------------------------

    test_output = test_df[
        [
            "ID",
            "SMILES",
            "scaffold",
            TARGET,
        ]
        + NONLINEAR_FEATURES
    ].copy()

    test_output["repetition"] = repetition

    test_output["seed"] = seed

    test_output["predicted_solubility"] = predictions

    test_output["residual"] = residuals

    test_output["absolute_error"] = absolute_errors

    test_output["squared_error"] = squared_errors

    test_output["prediction_error_direction"] = np.where(
        residuals > 0,
        "underprediction",
        np.where(
            residuals < 0,
            "overprediction",
            "exact"
        )
    )

    test_output["train_size"] = len(
        train_df
    )

    test_output["test_size"] = len(
        test_df
    )

    test_output["test_scaffold_count"] = len(
        test_scaffolds
    )

    all_predictions.append(
        test_output
    )

    repetition_summary.append(
        {
            "repetition": repetition,
            "seed": seed,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "n_test_scaffolds": len(test_scaffolds),
            "r_squared": r2,
            "rmse": rmse,
            "mae": mae,
            "mean_error": mean_error,
            "median_error": median_error,
            "residual_sd": error_sd,
            "mean_absolute_error": np.mean(
                absolute_errors
            ),
            "median_absolute_error": np.median(
                absolute_errors
            ),
            "underprediction_fraction": np.mean(
                residuals > 0
            ),
            "overprediction_fraction": np.mean(
                residuals < 0
            ),
        }
    )

    print(
        f"\nRepetition {repetition:02d} "
        f"(seed={seed})"
    )

    print(
        f"  Train: {len(train_df):,}"
        f" | Test: {len(test_df):,}"
        f" | Test scaffolds: "
        f"{len(test_scaffolds):,}"
    )

    print(
        f"  R²={r2:.4f}"
        f" | RMSE={rmse:.4f}"
        f" | MAE={mae:.4f}"
    )

    print(
        f"  Mean residual={mean_error:+.4f}"
        f" | Median residual={median_error:+.4f}"
    )

    print(
        f"  Underprediction="
        f"{np.mean(residuals > 0):.1%}"
        f" | Overprediction="
        f"{np.mean(residuals < 0):.1%}"
    )


# ---------------------------------------------------------------------
# COMBINE PREDICTIONS
# ---------------------------------------------------------------------

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)

summary_df = pd.DataFrame(
    repetition_summary
)


# ---------------------------------------------------------------------
# ERROR DISTRIBUTION
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("OVERALL HELD-OUT ERROR DISTRIBUTION")
print("=" * 70)

residual = predictions_df[
    "residual"
]

absolute_error = predictions_df[
    "absolute_error"
]

print(
    f"Total held-out predictions: "
    f"{len(predictions_df):,}"
)

print(
    f"Mean residual: "
    f"{residual.mean():+.4f}"
)

print(
    f"Median residual: "
    f"{residual.median():+.4f}"
)

print(
    f"Residual SD: "
    f"{residual.std(ddof=1):.4f}"
)

print(
    f"Mean absolute error: "
    f"{absolute_error.mean():.4f}"
)

print(
    f"Median absolute error: "
    f"{absolute_error.median():.4f}"
)

print(
    f"90th percentile absolute error: "
    f"{absolute_error.quantile(0.90):.4f}"
)

print(
    f"95th percentile absolute error: "
    f"{absolute_error.quantile(0.95):.4f}"
)

print(
    f"99th percentile absolute error: "
    f"{absolute_error.quantile(0.99):.4f}"
)

print(
    f"Underprediction fraction: "
    f"{np.mean(residual > 0):.1%}"
)

print(
    f"Overprediction fraction: "
    f"{np.mean(residual < 0):.1%}"
)


# ---------------------------------------------------------------------
# EXTREME ERRORS
# ---------------------------------------------------------------------

print("\n" + "-" * 70)
print("EXTREME PREDICTION ERRORS")
print("-" * 70)

extreme_rows = []

# Largest underpredictions:
largest_under = predictions_df.nlargest(
    20,
    "residual"
).copy()

largest_under["error_type"] = (
    "largest_underprediction"
)

# Largest overpredictions:
largest_over = predictions_df.nsmallest(
    20,
    "residual"
).copy()

largest_over["error_type"] = (
    "largest_overprediction"
)

# Largest absolute errors:
largest_absolute = predictions_df.nlargest(
    20,
    "absolute_error"
).copy()

largest_absolute["error_type"] = (
    "largest_absolute_error"
)

extreme_errors = pd.concat(
    [
        largest_under,
        largest_over,
        largest_absolute,
    ],
    ignore_index=True
)

extreme_errors = extreme_errors.drop_duplicates(
    subset=[
        "ID",
        "repetition",
        "error_type",
    ]
)

print(
    "\nLargest absolute held-out errors:"
)

for _, row in largest_absolute.head(10).iterrows():

    print(
        f"  ID={row['ID']} "
        f"| observed={row[TARGET]:.4f} "
        f"| predicted={row['predicted_solubility']:.4f} "
        f"| residual={row['residual']:+.4f} "
        f"| |error|={row['absolute_error']:.4f}"
    )


# ---------------------------------------------------------------------
# ERROR BY OBSERVED SOLUBILITY REGION
# ---------------------------------------------------------------------

print("\n" + "-" * 70)
print("ERROR BY OBSERVED SOLUBILITY REGION")
print("-" * 70)

# Use fixed bins based on observed solubility.
# These bins are descriptive rather than inferential.

observed_bins = [
    -np.inf,
    -6,
    -4,
    -2,
    0,
    2,
    np.inf,
]

observed_labels = [
    "< -6",
    "-6 to < -4",
    "-4 to < -2",
    "-2 to < 0",
    "0 to < 2",
    ">= 2",
]

predictions_df["observed_solubility_region"] = pd.cut(
    predictions_df[TARGET],
    bins=observed_bins,
    labels=observed_labels,
    right=False
)

observed_region_summary = (
    predictions_df
    .groupby(
        "observed_solubility_region",
        observed=False
    )
    .agg(
        n=("residual", "size"),
        mean_observed=(TARGET, "mean"),
        mean_predicted=(
            "predicted_solubility",
            "mean"
        ),
        mean_residual=(
            "residual",
            "mean"
        ),
        mean_absolute_error=(
            "absolute_error",
            "mean"
        ),
        median_absolute_error=(
            "absolute_error",
            "median"
        ),
        rmse=(
            "squared_error",
            lambda x: np.sqrt(
                np.mean(x)
            )
        ),
    )
    .reset_index()
)

for _, row in observed_region_summary.iterrows():

    print(
        f"{str(row['observed_solubility_region']):12s}"
        f" n={int(row['n']):5d}"
        f" | mean residual="
        f"{row['mean_residual']:+.4f}"
        f" | MAE="
        f"{row['mean_absolute_error']:.4f}"
        f" | RMSE="
        f"{row['rmse']:.4f}"
    )


# ---------------------------------------------------------------------
# ERROR BY PREDICTED SOLUBILITY REGION
# ---------------------------------------------------------------------

print("\n" + "-" * 70)
print("ERROR BY PREDICTED SOLUBILITY REGION")
print("-" * 70)

predicted_bins = [
    -np.inf,
    -6,
    -4,
    -2,
    0,
    2,
    np.inf,
]

predictions_df["predicted_solubility_region"] = pd.cut(
    predictions_df["predicted_solubility"],
    bins=predicted_bins,
    labels=observed_labels,
    right=False
)

predicted_region_summary = (
    predictions_df
    .groupby(
        "predicted_solubility_region",
        observed=False
    )
    .agg(
        n=("residual", "size"),
        mean_observed=(TARGET, "mean"),
        mean_predicted=(
            "predicted_solubility",
            "mean"
        ),
        mean_residual=(
            "residual",
            "mean"
        ),
        mean_absolute_error=(
            "absolute_error",
            "mean"
        ),
        rmse=(
            "squared_error",
            lambda x: np.sqrt(
                np.mean(x)
            )
        ),
    )
    .reset_index()
)

for _, row in predicted_region_summary.iterrows():

    print(
        f"{str(row['predicted_solubility_region']):12s}"
        f" n={int(row['n']):5d}"
        f" | mean residual="
        f"{row['mean_residual']:+.4f}"
        f" | MAE="
        f"{row['mean_absolute_error']:.4f}"
        f" | RMSE="
        f"{row['rmse']:.4f}"
    )


# ---------------------------------------------------------------------
# ERROR BY MOLECULAR FEATURE
# ---------------------------------------------------------------------

print("\n" + "-" * 70)
print("ERROR ASSOCIATION WITH MOLECULAR FEATURES")
print("-" * 70)

feature_error_rows = []

for feature in NONLINEAR_FEATURES:

    x = predictions_df[feature]

    residual_corr = x.corr(
        predictions_df["residual"]
    )

    absolute_error_corr = x.corr(
        predictions_df["absolute_error"]
    )

    feature_error_rows.append(
        {
            "feature": feature,
            "residual_correlation": residual_corr,
            "absolute_error_correlation": absolute_error_corr,
        }
    )

    print(
        f"{feature:30s}"
        f" residual r={residual_corr:+.4f}"
        f" | absolute-error r="
        f"{absolute_error_corr:+.4f}"
    )

feature_error_df = pd.DataFrame(
    feature_error_rows
)


# ---------------------------------------------------------------------
# OBSERVED VS PREDICTED RELATIONSHIP
# ---------------------------------------------------------------------

print("\n" + "-" * 70)
print("OBSERVED VS PREDICTED SOLUBILITY")
print("-" * 70)

observed_predicted_r = predictions_df[
    TARGET
].corr(
    predictions_df[
        "predicted_solubility"
    ]
)

print(
    f"Correlation between observed and predicted "
    f"solubility: r={observed_predicted_r:.4f}"
)

print(
    "This correlation is descriptive; the repeated "
    "held-out R² values above remain the primary "
    "performance measure."
)


# ---------------------------------------------------------------------
# SYSTEMATIC ERROR CHECK
# ---------------------------------------------------------------------

print("\n" + "-" * 70)
print("SYSTEMATIC ERROR CHECK")
print("-" * 70)

mean_residual = residual.mean()

underprediction_fraction = np.mean(
    residual > 0
)

overprediction_fraction = np.mean(
    residual < 0
)

if abs(mean_residual) < 0.05:
    bias_interpretation = (
        "Overall mean residual is close to zero, "
        "suggesting little aggregate directional bias."
    )
else:
    if mean_residual > 0:
        bias_interpretation = (
            "The model shows aggregate underprediction "
            "of observed solubility."
        )
    else:
        bias_interpretation = (
            "The model shows aggregate overprediction "
            "of observed solubility."
        )

print(
    f"Mean residual: {mean_residual:+.4f}"
)

print(
    f"Underprediction: "
    f"{underprediction_fraction:.1%}"
)

print(
    f"Overprediction: "
    f"{overprediction_fraction:.1%}"
)

print(
    bias_interpretation
)


# ---------------------------------------------------------------------
# REPEATED PERFORMANCE SUMMARY
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("REPEATED ERROR SUMMARY")
print("=" * 70)

for metric in [
    "r_squared",
    "rmse",
    "mae",
    "mean_error",
    "median_error",
    "residual_sd",
    "mean_absolute_error",
    "median_absolute_error",
    "underprediction_fraction",
    "overprediction_fraction",
]:

    values = summary_df[
        metric
    ].to_numpy(
        dtype=float
    )

    mean_value = np.mean(values)

    sd_value = np.std(
        values,
        ddof=1
    )

    print(
        f"{metric:30s}"
        f" mean={mean_value:+.4f}"
        f" | SD={sd_value:.4f}"
        f" | min={np.min(values):+.4f}"
        f" | max={np.max(values):+.4f}"
    )


# ---------------------------------------------------------------------
# SAVE ROW-LEVEL RESULTS
# ---------------------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

predictions_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("OUTPUTS")
print("=" * 70)

print(
    "\nRow-level held-out predictions and residuals:"
)

print(
    OUTPUT_FILE
)


# ---------------------------------------------------------------------
# SAVE SUMMARY
# ---------------------------------------------------------------------

summary_output = summary_df.copy()

summary_output.to_csv(
    SUMMARY_FILE,
    index=False
)

print(
    "\nRepeated-repetition error summary:"
)

print(
    SUMMARY_FILE
)


# ---------------------------------------------------------------------
# SAVE EXTREME ERRORS
# ---------------------------------------------------------------------

extreme_errors.to_csv(
    EXTREME_ERROR_FILE,
    index=False
)

print(
    "\nExtreme-error compounds:"
)

print(
    EXTREME_ERROR_FILE
)


# ---------------------------------------------------------------------
# WRITE REPORT
# ---------------------------------------------------------------------

report_lines = []

report_lines.append(
    "SCRIPT 26 — NONLINEAR MODEL RESIDUAL & ERROR ANALYSIS"
)

report_lines.append(
    "=" * 70
)

report_lines.append(
    f"Population C: {len(population):,} compounds"
)

report_lines.append(
    f"Unique scaffolds: {n_scaffolds:,}"
)

report_lines.append(
    f"Repeats: {N_REPEATS}"
)

report_lines.append(
    f"Test fraction: {TEST_FRACTION:.0%}"
)

report_lines.append("")

report_lines.append(
    "MODEL"
)

report_lines.append(
    "-" * 70
)

report_lines.append(
    "Gradient Boosting Regressor"
)

for key, value in GB_PARAMS.items():

    report_lines.append(
        f"{key}: {value}"
    )

report_lines.append("")

report_lines.append(
    "FEATURES"
)

for feature in NONLINEAR_FEATURES:

    report_lines.append(
        f"- {feature}"
    )

report_lines.append("")

report_lines.append(
    "RESIDUAL DEFINITION"
)

report_lines.append(
    "residual = observed solubility - predicted solubility"
)

report_lines.append(
    "Positive residual indicates underprediction."
)

report_lines.append(
    "Negative residual indicates overprediction."
)

report_lines.append("")

report_lines.append(
    "REPEATED PERFORMANCE"
)

report_lines.append(
    "-" * 70
)

for _, row in summary_df.iterrows():

    report_lines.append(
        f"Repetition {int(row['repetition']):02d} "
        f"| seed={int(row['seed'])} "
        f"| R2={row['r_squared']:.6f} "
        f"| RMSE={row['rmse']:.6f} "
        f"| MAE={row['mae']:.6f} "
        f"| mean_error={row['mean_error']:+.6f}"
    )

report_lines.append("")

report_lines.append(
    "OVERALL HELD-OUT ERROR DISTRIBUTION"
)

report_lines.append(
    "-" * 70
)

report_lines.append(
    f"Total held-out predictions: "
    f"{len(predictions_df):,}"
)

report_lines.append(
    f"Mean residual: "
    f"{residual.mean():+.6f}"
)

report_lines.append(
    f"Median residual: "
    f"{residual.median():+.6f}"
)

report_lines.append(
    f"Residual SD: "
    f"{residual.std(ddof=1):.6f}"
)

report_lines.append(
    f"Mean absolute error: "
    f"{absolute_error.mean():.6f}"
)

report_lines.append(
    f"Median absolute error: "
    f"{absolute_error.median():.6f}"
)

report_lines.append(
    f"90th percentile absolute error: "
    f"{absolute_error.quantile(0.90):.6f}"
)

report_lines.append(
    f"95th percentile absolute error: "
    f"{absolute_error.quantile(0.95):.6f}"
)

report_lines.append(
    f"99th percentile absolute error: "
    f"{absolute_error.quantile(0.99):.6f}"
)

report_lines.append(
    f"Underprediction fraction: "
    f"{underprediction_fraction:.6f}"
)

report_lines.append(
    f"Overprediction fraction: "
    f"{overprediction_fraction:.6f}"
)

report_lines.append("")

report_lines.append(
    "ERROR ASSOCIATION WITH MOLECULAR FEATURES"
)

report_lines.append(
    "-" * 70
)

for _, row in feature_error_df.iterrows():

    report_lines.append(
        f"{row['feature']} | "
        f"residual_correlation="
        f"{row['residual_correlation']:+.6f} | "
        f"absolute_error_correlation="
        f"{row['absolute_error_correlation']:+.6f}"
    )

report_lines.append("")

report_lines.append(
    "ERROR BY OBSERVED SOLUBILITY REGION"
)

report_lines.append(
    "-" * 70
)

for _, row in observed_region_summary.iterrows():

    report_lines.append(
        f"{row['observed_solubility_region']} | "
        f"n={int(row['n'])} | "
        f"mean_observed="
        f"{row['mean_observed']:.6f} | "
        f"mean_predicted="
        f"{row['mean_predicted']:.6f} | "
        f"mean_residual="
        f"{row['mean_residual']:+.6f} | "
        f"MAE="
        f"{row['mean_absolute_error']:.6f} | "
        f"RMSE="
        f"{row['rmse']:.6f}"
    )

report_lines.append("")

report_lines.append(
    "OBSERVED VS PREDICTED"
)

report_lines.append(
    "-" * 70
)

report_lines.append(
    f"Observed/predicted correlation: "
    f"{observed_predicted_r:.6f}"
)

report_lines.append("")

report_lines.append(
    "EXTREME ERRORS"
)

report_lines.append(
    "-" * 70
)

for _, row in largest_absolute.head(20).iterrows():

    report_lines.append(
        f"ID={row['ID']} | "
        f"repetition={int(row['repetition'])} | "
        f"observed={row[TARGET]:.6f} | "
        f"predicted={row['predicted_solubility']:.6f} | "
        f"residual={row['residual']:+.6f} | "
        f"absolute_error={row['absolute_error']:.6f}"
    )

report_lines.append("")

report_lines.append(
    "METHODOLOGICAL INTERPRETATION"
)

report_lines.append(
    "-" * 70
)

report_lines.append(
    "Residuals were calculated exclusively from held-out "
    "scaffold-aware test observations."
)

report_lines.append(
    "The same ten scaffold-level seeds used in Scripts "
    "22–25 were used to maintain methodological continuity."
)

report_lines.append(
    "Positive residuals represent cases where the model "
    "underpredicted measured solubility."
)

report_lines.append(
    "Negative residuals represent cases where the model "
    "overpredicted measured solubility."
)

report_lines.append(
    "Feature-error correlations are descriptive and do not "
    "establish causal relationships."
)

report_lines.append(
    "Large residuals identify compounds or structural regions "
    "where the six-descriptor Gradient Boosting model has "
    "limited predictive accuracy."
)

report_lines.append(
    "Error patterns should be interpreted alongside the "
    "scaffold-aware performance results and SHAP analysis "
    "from Scripts 23–25."
)

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(report_lines)
    )


print(
    "\nReport:"
)

print(
    REPORT_FILE
)

print("\n" + "=" * 70)
print("SCRIPT 26 COMPLETE")
print("=" * 70)