"""
SCRIPT 18 — MODEL EXPLAINABILITY

Purpose
-------
Determine which molecular descriptors contribute most to the predictive
performance of the selected M4 model using permutation importance on the
held-out scaffold test set.

M4:
    Solubility ~ MolWt + MolLogP + RingCount + AromaticRings
                 + RotatableBonds + FractionCSP3

Primary method:
    Manual permutation importance (R² decrease) on the scaffold-aware
    test set, evaluated directly against the fitted statsmodels model.

Secondary method:
    Standardized coefficients from the model fitted on the scaffold
    training population.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ======================================================================
# PATHS
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "curated-solubility-dataset.csv"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "molecular_features.csv"
SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "model_evaluation_splits.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures" / "model_explainability"
REPORT_PATH = PROJECT_ROOT / "reports" / "model_explainability.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================================
# CONFIGURATION
# ======================================================================

TARGET = "Solubility"

FEATURES = [
    "rdkit_molwt",
    "rdkit_mollogp",
    "rdkit_ring_count",
    "rdkit_aromatic_rings",
    "rdkit_rotatable_bonds",
    "rdkit_fraction_csp3",
]

RANDOM_STATE = 42
N_REPEATS = 50


# ======================================================================
# HEADER
# ======================================================================

print("=" * 70)
print("SCRIPT 18 — MODEL EXPLAINABILITY")
print("=" * 70)


# ======================================================================
# LOAD DATA
# ======================================================================

print("\nLoading raw dataset:")
print(RAW_PATH)

raw = pd.read_csv(RAW_PATH)

print(f"Raw dataset shape: {raw.shape}")


print("\nLoading molecular features:")
print(FEATURES_PATH)

features = pd.read_csv(FEATURES_PATH)

print(f"Molecular feature shape: {features.shape}")


print("\nLoading scaffold split assignments:")
print(SPLIT_PATH)

splits = pd.read_csv(SPLIT_PATH)

print(f"Split assignment shape: {splits.shape}")


# ======================================================================
# IDENTIFY SPLIT COLUMN
# ======================================================================

if "ID" not in splits.columns:
    raise ValueError(
        "model_evaluation_splits.csv does not contain an ID column."
    )

split_column = None

for column in splits.columns:
    if column == "ID":
        continue

    values = set(
        splits[column]
        .dropna()
        .astype(str)
        .str.lower()
        .unique()
    )

    if {"train", "validation", "test"}.issubset(values):
        split_column = column
        break

if split_column is None:
    raise ValueError(
        "Could not identify the train/validation/test split column.\n"
        f"Available columns: {list(splits.columns)}"
    )

print(f"Using split column: {split_column}")


# ======================================================================
# VERIFY REQUIRED COLUMNS
# ======================================================================

required_columns = [
    "ID",
    TARGET,
    "standard_analytical_domain",
] + FEATURES

missing_columns = [
    col for col in required_columns
    if col not in features.columns
]

if missing_columns:
    raise ValueError(
        "Required columns missing from molecular_features.csv:\n"
        f"{missing_columns}"
    )


# ======================================================================
# RECONSTRUCT POPULATION C
# ======================================================================

print("\n" + "-" * 70)
print("POPULATION C")
print("-" * 70)

population = features[
    features["standard_analytical_domain"] == True
].copy()

print(
    f"Rows flagged as standard analytical domain: "
    f"{len(population):,}"
)

if len(population) != 8643:
    raise ValueError(
        f"Population C mismatch. Expected 8643 rows, "
        f"found {len(population):,}."
    )

print("Population C verified: 8,643 rows")


# ======================================================================
# MERGE SPLIT ASSIGNMENTS
# ======================================================================

split_subset = splits[["ID", split_column]].copy()

split_subset = split_subset.rename(
    columns={split_column: "data_split"}
)

if split_subset["ID"].duplicated().any():
    raise ValueError(
        "Duplicate IDs found in model evaluation split assignments."
    )

data = population.merge(
    split_subset,
    on="ID",
    how="inner",
    validate="one_to_one",
)

print(f"Rows after split merge: {len(data):,}")

if len(data) != 8643:
    raise ValueError(
        "Split merge did not preserve all Population C rows."
    )

data["data_split"] = (
    data["data_split"]
    .astype(str)
    .str.lower()
)


# ======================================================================
# SPLIT COUNTS
# ======================================================================

print("\nScaffold-aware split:")
print(
    data["data_split"]
    .value_counts()
    .sort_index()
)


expected_splits = {"train", "validation", "test"}

actual_splits = set(data["data_split"].unique())

if not expected_splits.issubset(actual_splits):
    raise ValueError(
        f"Expected train/validation/test splits. "
        f"Found: {actual_splits}"
    )


# ======================================================================
# CREATE TRAIN / VALIDATION / TEST SETS
# ======================================================================

train = data[
    data["data_split"] == "train"
].copy()

validation = data[
    data["data_split"] == "validation"
].copy()

test = data[
    data["data_split"] == "test"
].copy()


X_train = train[FEATURES].copy()
y_train = train[TARGET].copy()

X_test = test[FEATURES].copy()
y_test = test[TARGET].copy()


print("\nTraining rows:   {:,}".format(len(train)))
print("Validation rows: {:,}".format(len(validation)))
print("Test rows:       {:,}".format(len(test)))


# ======================================================================
# FIT M4 MODEL
# ======================================================================

print("\n" + "=" * 70)
print("M4 MODEL")
print("=" * 70)

print(
    """
Solubility ~ rdkit_molwt
           + rdkit_mollogp
           + rdkit_ring_count
           + rdkit_aromatic_rings
           + rdkit_rotatable_bonds
           + rdkit_fraction_csp3
"""
)

X_train_sm = sm.add_constant(
    X_train,
    has_constant="add"
)

X_test_sm = sm.add_constant(
    X_test,
    has_constant="add"
)

model = sm.OLS(
    y_train,
    X_train_sm
).fit(
    cov_type="HC3"
)


# ======================================================================
# TEST PERFORMANCE
# ======================================================================

test_predictions = model.predict(X_test_sm)

test_r2 = r2_score(
    y_test,
    test_predictions
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_predictions
    )
)

test_mae = mean_absolute_error(
    y_test,
    test_predictions
)

test_mean_error = np.mean(
    y_test.to_numpy()
    - test_predictions.to_numpy()
)


print("\nScaffold-test performance:")
print(f"R²         = {test_r2:.6f}")
print(f"RMSE       = {test_rmse:.6f}")
print(f"MAE        = {test_mae:.6f}")
print(f"Mean error = {test_mean_error:.6f}")


# ======================================================================
# PERMUTATION IMPORTANCE (MANUAL — statsmodels-compatible)
# ======================================================================

print("\n" + "=" * 70)
print("PERMUTATION IMPORTANCE")
print("=" * 70)

print(
    "\nCalculating permutation importance on the held-out "
    "scaffold test set..."
)


def statsmodels_predict(fitted_model, X):
    """Predict from a fitted statsmodels OLS model given a raw feature frame."""
    X_sm = sm.add_constant(X, has_constant="add")
    return np.asarray(fitted_model.predict(X_sm))


def manual_permutation_importance(
    fitted_model, X_test_df, y_test_series, feature_cols,
    n_repeats=30, random_state=42,
):
    """
    Manual permutation importance, scored as decrease in R².

    For each feature: shuffle its values in the test set, re-predict
    using the ALREADY-FITTED model (no refitting), and measure how
    much R² drops relative to the unpermuted baseline. Repeated
    n_repeats times per feature to get a mean and std.
    """
    rng = np.random.RandomState(random_state)

    y_true = y_test_series.to_numpy()

    baseline_pred = statsmodels_predict(fitted_model, X_test_df)
    baseline_r2 = r2_score(y_true, baseline_pred)

    means = []
    stds = []

    for feature in feature_cols:
        drops = []

        for _ in range(n_repeats):
            X_permuted = X_test_df.copy()
            X_permuted[feature] = rng.permutation(X_permuted[feature].to_numpy())

            permuted_pred = statsmodels_predict(fitted_model, X_permuted)
            permuted_r2 = r2_score(y_true, permuted_pred)

            # Positive value = permuting this feature HURT performance
            drops.append(baseline_r2 - permuted_r2)

        means.append(np.mean(drops))
        stds.append(np.std(drops))

    return baseline_r2, np.array(means), np.array(stds)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    baseline_r2_check, importances_mean, importances_std = manual_permutation_importance(
        model,
        X_test,
        y_test,
        FEATURES,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

print(f"\nBaseline R² (unpermuted, sanity check): {baseline_r2_check:.6f}")
print(f"(Should match scaffold-test R² above: {test_r2:.6f})")

importance = pd.DataFrame(
    {
        "feature": FEATURES,
        "mean_importance": importances_mean,
        "std_importance": importances_std,
    }
)

importance["ci_approx_low"] = (
    importance["mean_importance"]
    - 1.96 * importance["std_importance"]
)

importance["ci_approx_high"] = (
    importance["mean_importance"]
    + 1.96 * importance["std_importance"]
)

importance["importance_rank"] = (
    importance["mean_importance"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)

importance = (
    importance
    .sort_values(
        "mean_importance",
        ascending=False
    )
    .reset_index(drop=True)
)


print("\nPermutation importance:")
print(
    importance[
        [
            "feature",
            "mean_importance",
            "std_importance",
            "ci_approx_low",
            "ci_approx_high",
            "importance_rank",
        ]
    ].to_string(index=False)
)


# ======================================================================
# STANDARDIZED COEFFICIENTS
# ======================================================================

print("\n" + "=" * 70)
print("STANDARDIZED COEFFICIENTS")
print("=" * 70)


train_sd = X_train.std(
    ddof=1
)

confidence_intervals = model.conf_int()


coefficient_records = []

for feature in FEATURES:

    coefficient = model.params[feature]

    standardized_coefficient = (
        coefficient
        * train_sd[feature]
    )

    coefficient_records.append(
        {
            "feature": feature,
            "coefficient": coefficient,
            "robust_se": model.bse[feature],
            "train_sd": train_sd[feature],
            "standardized_coefficient":
                standardized_coefficient,
            "absolute_standardized_coefficient":
                abs(standardized_coefficient),
            "p_value": model.pvalues[feature],
            "ci_low":
                confidence_intervals.loc[
                    feature,
                    0
                ],
            "ci_high":
                confidence_intervals.loc[
                    feature,
                    1
                ],
        }
    )


coefficients_df = pd.DataFrame(
    coefficient_records
)

coefficients_df = (
    coefficients_df
    .sort_values(
        "absolute_standardized_coefficient",
        ascending=False
    )
    .reset_index(drop=True)
)


print(
    coefficients_df[
        [
            "feature",
            "coefficient",
            "standardized_coefficient",
            "p_value",
            "ci_low",
            "ci_high",
        ]
    ].to_string(index=False)
)


# ======================================================================
# SAVE CSV OUTPUTS
# ======================================================================

importance_path = (
    OUTPUT_DIR
    / "model_permutation_importance.csv"
)

coefficients_path = (
    OUTPUT_DIR
    / "model_standardized_coefficients.csv"
)

importance.to_csv(
    importance_path,
    index=False
)

coefficients_df.to_csv(
    coefficients_path,
    index=False
)


# ======================================================================
# PERMUTATION IMPORTANCE FIGURE
# ======================================================================

plot_data = importance.sort_values(
    "mean_importance",
    ascending=True
)


plt.figure(
    figsize=(9, 6)
)

plt.barh(
    plot_data["feature"],
    plot_data["mean_importance"],
    xerr=plot_data["std_importance"]
)

plt.axvline(
    0,
    linewidth=1
)

plt.xlabel(
    "Mean decrease in scaffold-test R²"
)

plt.ylabel(
    "Molecular descriptor"
)

plt.title(
    "M4 Scaffold-Test Permutation Importance"
)

plt.tight_layout()


figure_path = (
    FIGURE_DIR
    / "permutation_importance.png"
)

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ======================================================================
# WRITE REPORT
# ======================================================================

report_lines = []

report_lines.append(
    "SCRIPT 18 — MODEL EXPLAINABILITY"
)

report_lines.append(
    "=" * 70
)

report_lines.append(
    "\nPurpose:\n"
    "Determine which molecular descriptors contribute most to the "
    "predictive performance of M4 using manual permutation importance "
    "(R² decrease) on the held-out scaffold test set."
)

report_lines.append(
    f"\nPopulation C:\n{len(data):,} compounds"
)

report_lines.append(
    "\nScaffold-aware split:\n"
    f"Training:   {len(train):,}\n"
    f"Validation: {len(validation):,}\n"
    f"Test:       {len(test):,}"
)

report_lines.append(
    "\nM4 scaffold-test performance:\n"
    f"R² = {test_r2:.6f}\n"
    f"RMSE = {test_rmse:.6f}\n"
    f"MAE = {test_mae:.6f}\n"
    f"Mean error = {test_mean_error:.6f}"
)

report_lines.append(
    "\nPermutation importance:\n"
)

report_lines.append(
    importance[
        [
            "feature",
            "mean_importance",
            "std_importance",
            "ci_approx_low",
            "ci_approx_high",
        ]
    ].to_string(index=False)
)

report_lines.append(
    "\nStandardized coefficients:\n"
)

report_lines.append(
    coefficients_df[
        [
            "feature",
            "coefficient",
            "standardized_coefficient",
            "p_value",
            "ci_low",
            "ci_high",
        ]
    ].to_string(index=False)
)

report_lines.append(
    "\nInterpretation guidance:\n"
    "- Permutation importance is calculated manually on the held-out "
    "scaffold test set by shuffling one feature at a time and "
    "re-predicting with the fixed, already-fitted model.\n"
    "- Larger positive importance means that permuting the feature "
    "causes a larger decrease in test R².\n"
    "- Importance is predictive, not causal.\n"
    "- Correlated descriptors can share predictive information.\n"
    "- Standardized coefficients describe the fitted linear model "
    "and are not causal effects.\n"
    "- Ring-related importance should be interpreted as evidence "
    "that structural information adds predictive value beyond "
    "MolWt and MolLogP, not as direct proof of lattice-energy effects."
)

REPORT_PATH.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)


# ======================================================================
# FINAL OUTPUT
# ======================================================================

print("\n" + "=" * 70)
print("OUTPUTS")
print("=" * 70)

print("\nPermutation importance:")
print(importance_path)

print("\nStandardized coefficients:")
print(coefficients_path)

print("\nReport:")
print(REPORT_PATH)

print("\nFigure:")
print(figure_path)

print("\n" + "=" * 70)
print("SCRIPT 18 COMPLETE")
print("=" * 70)