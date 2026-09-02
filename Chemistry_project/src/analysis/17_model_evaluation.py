"""
SCRIPT 17 — MODEL EVALUATION
Random and scaffold-aware train/test evaluation of the solubility models.

Purpose
-------
Evaluate whether the incremental improvements observed in Scripts 15–16
generalize to unseen compounds.

Models
------
M0: MolWt + MolLogP
M3: MolWt + MolLogP + RingCount + AromaticRings
M4: M3 + RotatableBonds + FractionCSP3

Evaluation
----------
1. Random train/test split
2. Scaffold-aware train/validation/test split

Important
---------
All model coefficients are estimated on the training set only.
Performance metrics are calculated on held-out data.

Scaffold split uses Bemis-Murcko scaffolds where available.
Acyclic compounds are treated as individual groups rather than one
large shared scaffold group.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "curated-solubility-dataset.csv"
FEATURE_PATH = PROJECT_ROOT / "data" / "processed" / "molecular_features.csv"

OUTPUT_MODEL = (
    PROJECT_ROOT / "data" / "processed" / "model_evaluation_results.csv"
)

OUTPUT_SPLIT = (
    PROJECT_ROOT / "data" / "processed" / "model_evaluation_splits.csv"
)

OUTPUT_REPORT = (
    PROJECT_ROOT / "reports" / "model_evaluation.txt"
)


# ---------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------

RANDOM_STATE = 42

# Random split:
RANDOM_TEST_SIZE = 0.20

# Scaffold split:
# 70% train / 15% validation / 15% test
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15


# ---------------------------------------------------------------------
# MODEL DEFINITIONS
# ---------------------------------------------------------------------

MODELS = {
    "M0_baseline": [
        "rdkit_molwt",
        "rdkit_mollogp",
    ],
    "M3_ring_model": [
        "rdkit_molwt",
        "rdkit_mollogp",
        "rdkit_ring_count",
        "rdkit_aromatic_rings",
    ],
    "M4_selective_model": [
        "rdkit_molwt",
        "rdkit_mollogp",
        "rdkit_ring_count",
        "rdkit_aromatic_rings",
        "rdkit_rotatable_bonds",
        "rdkit_fraction_csp3",
    ],
}


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def calculate_metrics(y_true, y_pred):
    """Calculate held-out regression metrics."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    residuals = y_true - y_pred

    mae = np.mean(np.abs(residuals))
    mean_error = np.mean(residuals)

    return {
        "r_squared": r2,
        "rmse": rmse,
        "mae": mae,
        "mean_error": mean_error,
        "n": len(y_true),
    }


def fit_and_evaluate(train_df, test_df, predictors, model_name, split_name):
    """
    Fit OLS on training data and evaluate on held-out test data.
    """

    X_train = sm.add_constant(
        train_df[predictors],
        has_constant="add"
    )

    y_train = train_df["Solubility"]

    X_test = sm.add_constant(
        test_df[predictors],
        has_constant="add"
    )

    y_test = test_df["Solubility"]

    model = sm.OLS(y_train, X_train).fit()

    predictions = model.predict(X_test)

    metrics = calculate_metrics(y_test, predictions)

    baseline_prediction = np.mean(y_train)
    baseline_predictions = np.repeat(
        baseline_prediction,
        len(y_test)
    )

    baseline_metrics = calculate_metrics(
        y_test,
        baseline_predictions
    )

    return {
        "split": split_name,
        "model": model_name,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "r_squared": metrics["r_squared"],
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "mean_error": metrics["mean_error"],
        "train_target_mean": y_train.mean(),
        "test_target_mean": y_test.mean(),
        "delta_r_squared_vs_mean_baseline": (
            metrics["r_squared"] - baseline_metrics["r_squared"]
        ),
    }, model


def get_scaffold_key(smiles, row_id):
    """
    Generate a Bemis-Murcko scaffold key.

    For molecules without a non-empty Murcko scaffold, the molecule
    itself is assigned a unique group. This prevents all acyclic
    molecules from becoming one enormous scaffold group.
    """

    if pd.isna(smiles):
        return f"NO_SCAFFOLD_{row_id}"

    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:
        return f"INVALID_{row_id}"

    scaffold = MurckoScaffold.GetScaffoldForMol(mol)

    if scaffold is None:
        return f"ACYCLIC_{row_id}"

    scaffold_smiles = Chem.MolToSmiles(
        scaffold,
        canonical=True
    )

    if not scaffold_smiles:
        return f"ACYCLIC_{row_id}"

    return scaffold_smiles


def create_scaffold_split(df):
    """
    Create a deterministic scaffold-aware 70/15/15 split.

    Scaffolds are shuffled and assigned to train, validation, and test
    without allowing the same scaffold to appear in multiple sets.
    """

    scaffold_sizes = (
        df.groupby("scaffold_key")
        .size()
        .sort_values(ascending=False)
    )

    scaffold_ids = scaffold_sizes.index.tolist()

    rng = np.random.default_rng(RANDOM_STATE)

    rng.shuffle(scaffold_ids)

    total_rows = len(df)

    target_train = total_rows * TRAIN_FRACTION
    target_validation = total_rows * VALIDATION_FRACTION

    train_scaffolds = []
    validation_scaffolds = []
    test_scaffolds = []

    train_rows = 0
    validation_rows = 0

    for scaffold in scaffold_ids:

        size = scaffold_sizes.loc[scaffold]

        if train_rows < target_train:
            train_scaffolds.append(scaffold)
            train_rows += size

        elif validation_rows < target_validation:
            validation_scaffolds.append(scaffold)
            validation_rows += size

        else:
            test_scaffolds.append(scaffold)

    train_mask = df["scaffold_key"].isin(train_scaffolds)
    validation_mask = df["scaffold_key"].isin(validation_scaffolds)
    test_mask = df["scaffold_key"].isin(test_scaffolds)

    train_df = df.loc[train_mask].copy()
    validation_df = df.loc[validation_mask].copy()
    test_df = df.loc[test_mask].copy()

    return train_df, validation_df, test_df


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

print("=" * 70)
print("SCRIPT 17 — MODEL EVALUATION")
print("=" * 70)


# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------

print("\nLoading raw dataset:")
print(RAW_PATH)

raw = pd.read_csv(RAW_PATH)

print(f"Raw dataset shape: {raw.shape}")


print("\nLoading molecular features:")
print(FEATURE_PATH)

features = pd.read_csv(FEATURE_PATH)

print(f"Molecular feature shape: {features.shape}")


# ---------------------------------------------------------------------
# VERIFY POPULATION C
# ---------------------------------------------------------------------

if "standard_analytical_domain" not in features.columns:
    raise ValueError(
        "standard_analytical_domain flag not found in "
        "molecular_features.csv"
    )

pop_c = features[
    features["standard_analytical_domain"] == True
].copy()

print("\n" + "-" * 70)
print("POPULATION C")
print("-" * 70)

print(
    f"Rows flagged as standard analytical domain: {len(pop_c):,}"
)

if len(pop_c) != 8643:
    raise ValueError(
        f"Population C verification failed. "
        f"Expected 8643, found {len(pop_c)}."
    )

print("Population C verified: 8,643 rows")


# ---------------------------------------------------------------------
# SOLUBILITY MERGE
# ---------------------------------------------------------------------

if "Solubility" not in pop_c.columns:

    if "Solubility" not in raw.columns:
        raise ValueError(
            "Solubility not available in either feature or raw dataset."
        )

    solubility = raw[["ID", "Solubility"]].copy()

    pop_c = pop_c.drop(columns=["Solubility"], errors="ignore")

    pop_c = pop_c.merge(
        solubility,
        on="ID",
        how="left",
        validate="one_to_one"
    )


if pop_c["Solubility"].isna().any():
    raise ValueError(
        "Missing Solubility values found after merge."
    )

print(f"Rows after solubility merge: {len(pop_c):,}")


# ---------------------------------------------------------------------
# REQUIRED COLUMNS
# ---------------------------------------------------------------------

required_columns = [
    "ID",
    "SMILES",
    "Solubility",
    "rdkit_molwt",
    "rdkit_mollogp",
    "rdkit_ring_count",
    "rdkit_aromatic_rings",
    "rdkit_rotatable_bonds",
    "rdkit_fraction_csp3",
]

missing_columns = [
    col for col in required_columns
    if col not in pop_c.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns missing: {missing_columns}"
    )


# ---------------------------------------------------------------------
# COMPLETE CASE CHECK
# ---------------------------------------------------------------------

all_predictors = sorted(
    set(
        predictor
        for predictors in MODELS.values()
        for predictor in predictors
    )
)

analysis_columns = ["ID", "SMILES", "Solubility"] + all_predictors

before = len(pop_c)

pop_c = pop_c.dropna(
    subset=analysis_columns
).copy()

after = len(pop_c)

if before != after:
    print(
        f"\nRows removed due to missing analytical variables: "
        f"{before - after}"
    )

if len(pop_c) != 8643:
    raise ValueError(
        f"Unexpected complete-case count: {len(pop_c)}"
    )


# =====================================================================
# RANDOM SPLIT
# =====================================================================

print("\n" + "=" * 70)
print("RANDOM 80/20 HOLDOUT")
print("=" * 70)

train_random, test_random = train_test_split(
    pop_c,
    test_size=RANDOM_TEST_SIZE,
    random_state=RANDOM_STATE,
)

print(f"Training rows: {len(train_random):,}")
print(f"Test rows:     {len(test_random):,}")


random_results = []

for model_name, predictors in MODELS.items():

    result, fitted_model = fit_and_evaluate(
        train_random,
        test_random,
        predictors,
        model_name,
        "random_80_20"
    )

    random_results.append(result)


random_results_df = pd.DataFrame(random_results)


print("\nRandom-split performance:")
print(
    random_results_df[
        [
            "model",
            "n_train",
            "n_test",
            "r_squared",
            "rmse",
            "mae",
            "mean_error",
        ]
    ].to_string(index=False)
)


# ---------------------------------------------------------------------
# RANDOM SPLIT INCREMENTAL COMPARISON
# ---------------------------------------------------------------------

baseline_random = random_results_df.loc[
    random_results_df["model"] == "M0_baseline"
].iloc[0]

random_results_df["delta_r_squared_vs_M0"] = (
    random_results_df["r_squared"]
    - baseline_random["r_squared"]
)

random_results_df["rmse_change_vs_M0"] = (
    random_results_df["rmse"]
    - baseline_random["rmse"]
)


print("\nRandom-split improvement versus M0:")

print(
    random_results_df[
        [
            "model",
            "delta_r_squared_vs_M0",
            "rmse_change_vs_M0",
        ]
    ].to_string(index=False)
)


# =====================================================================
# SCAFFOLD ASSIGNMENT
# =====================================================================

print("\n" + "=" * 70)
print("SCAFFOLD-AWARE SPLIT")
print("=" * 70)

print("\nGenerating Bemis-Murcko scaffold groups...")

pop_c["scaffold_key"] = [
    get_scaffold_key(smiles, row_id)
    for smiles, row_id
    in zip(pop_c["SMILES"], pop_c["ID"])
]

n_scaffolds = pop_c["scaffold_key"].nunique()

print(f"Unique scaffold groups: {n_scaffolds:,}")


train_scaffold, validation_scaffold, test_scaffold = (
    create_scaffold_split(pop_c)
)


print("\nScaffold split sizes:")
print(f"Training:   {len(train_scaffold):,}")
print(f"Validation: {len(validation_scaffold):,}")
print(f"Test:       {len(test_scaffold):,}")


# ---------------------------------------------------------------------
# VERIFY SCAFFOLD SEPARATION
# ---------------------------------------------------------------------

train_scaffolds = set(train_scaffold["scaffold_key"])
validation_scaffolds = set(validation_scaffold["scaffold_key"])
test_scaffolds = set(test_scaffold["scaffold_key"])

train_validation_overlap = (
    train_scaffolds & validation_scaffolds
)

train_test_overlap = (
    train_scaffolds & test_scaffolds
)

validation_test_overlap = (
    validation_scaffolds & test_scaffolds
)

print("\nScaffold overlap checks:")
print(
    f"Train ∩ Validation: {len(train_validation_overlap)}"
)
print(
    f"Train ∩ Test:       {len(train_test_overlap)}"
)
print(
    f"Validation ∩ Test:  {len(validation_test_overlap)}"
)

if (
    train_validation_overlap
    or train_test_overlap
    or validation_test_overlap
):
    raise ValueError(
        "Scaffold leakage detected between splits."
    )


# ---------------------------------------------------------------------
# SAVE SPLIT ASSIGNMENTS
# ---------------------------------------------------------------------

split_lookup = pd.concat(
    [
        train_scaffold.assign(split="train"),
        validation_scaffold.assign(split="validation"),
        test_scaffold.assign(split="test"),
    ],
    ignore_index=True
)[["ID", "scaffold_key", "split"]]

split_lookup.to_csv(
    OUTPUT_SPLIT,
    index=False
)


# =====================================================================
# SCAFFOLD MODEL EVALUATION
# =====================================================================

print("\n" + "=" * 70)
print("SCAFFOLD-AWARE MODEL EVALUATION")
print("=" * 70)

scaffold_results = []


for model_name, predictors in MODELS.items():

    # -------------------------------------------------------------
    # Development model:
    # Train only
    # -------------------------------------------------------------

    X_train = sm.add_constant(
        train_scaffold[predictors],
        has_constant="add"
    )

    y_train = train_scaffold["Solubility"]

    development_model = sm.OLS(
        y_train,
        X_train
    ).fit()

    # -------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------

    X_validation = sm.add_constant(
        validation_scaffold[predictors],
        has_constant="add"
    )

    y_validation = validation_scaffold["Solubility"]

    validation_prediction = development_model.predict(
        X_validation
    )

    validation_metrics = calculate_metrics(
        y_validation,
        validation_prediction
    )

    # -------------------------------------------------------------
    # Final model:
    # Train + validation
    # -------------------------------------------------------------

    train_validation = pd.concat(
        [
            train_scaffold,
            validation_scaffold,
        ],
        ignore_index=True
    )

    X_train_final = sm.add_constant(
        train_validation[predictors],
        has_constant="add"
    )

    y_train_final = train_validation["Solubility"]

    final_model = sm.OLS(
        y_train_final,
        X_train_final
    ).fit()

    # -------------------------------------------------------------
    # Test
    # -------------------------------------------------------------

    X_test = sm.add_constant(
        test_scaffold[predictors],
        has_constant="add"
    )

    y_test = test_scaffold["Solubility"]

    test_prediction = final_model.predict(
        X_test
    )

    test_metrics = calculate_metrics(
        y_test,
        test_prediction
    )

    scaffold_results.append(
        {
            "split": "scaffold_70_15_15",
            "model": model_name,
            "n_train": len(train_scaffold),
            "n_validation": len(validation_scaffold),
            "n_train_final": len(train_validation),
            "n_test": len(test_scaffold),

            "validation_r_squared":
                validation_metrics["r_squared"],

            "validation_rmse":
                validation_metrics["rmse"],

            "validation_mae":
                validation_metrics["mae"],

            "test_r_squared":
                test_metrics["r_squared"],

            "test_rmse":
                test_metrics["rmse"],

            "test_mae":
                test_metrics["mae"],

            "test_mean_error":
                test_metrics["mean_error"],
        }
    )


scaffold_results_df = pd.DataFrame(
    scaffold_results
)


print("\nScaffold validation performance:")

print(
    scaffold_results_df[
        [
            "model",
            "validation_r_squared",
            "validation_rmse",
            "validation_mae",
        ]
    ].to_string(index=False)
)


print("\nScaffold test performance:")

print(
    scaffold_results_df[
        [
            "model",
            "test_r_squared",
            "test_rmse",
            "test_mae",
            "test_mean_error",
        ]
    ].to_string(index=False)
)


# ---------------------------------------------------------------------
# SCAFFOLD TEST INCREMENTAL COMPARISON
# ---------------------------------------------------------------------

baseline_scaffold = scaffold_results_df.loc[
    scaffold_results_df["model"] == "M0_baseline"
].iloc[0]

scaffold_results_df["delta_test_r_squared_vs_M0"] = (
    scaffold_results_df["test_r_squared"]
    - baseline_scaffold["test_r_squared"]
)

scaffold_results_df["test_rmse_change_vs_M0"] = (
    scaffold_results_df["test_rmse"]
    - baseline_scaffold["test_rmse"]
)


print("\nScaffold-test improvement versus M0:")

print(
    scaffold_results_df[
        [
            "model",
            "delta_test_r_squared_vs_M0",
            "test_rmse_change_vs_M0",
        ]
    ].to_string(index=False)
)


# =====================================================================
# COMBINE RESULTS
# =====================================================================

random_output = random_results_df.copy()

random_output["validation_r_squared"] = np.nan
random_output["validation_rmse"] = np.nan
random_output["validation_mae"] = np.nan
random_output["test_r_squared"] = random_output["r_squared"]
random_output["test_rmse"] = random_output["rmse"]
random_output["test_mae"] = random_output["mae"]
random_output["test_mean_error"] = random_output["mean_error"]

random_output["n_validation"] = np.nan
random_output["n_train_final"] = random_output["n_train"]

random_output["delta_test_r_squared_vs_M0"] = (
    random_output["r_squared"]
    - random_results_df.loc[
        random_results_df["model"] == "M0_baseline",
        "r_squared"
    ].iloc[0]
)

random_output["test_rmse_change_vs_M0"] = (
    random_output["rmse"]
    - random_results_df.loc[
        random_results_df["model"] == "M0_baseline",
        "rmse"
    ].iloc[0]
)


combined_results = pd.concat(
    [
        random_output[
            [
                "split",
                "model",
                "n_train",
                "n_validation",
                "n_train_final",
                "n_test",
                "validation_r_squared",
                "validation_rmse",
                "validation_mae",
                "test_r_squared",
                "test_rmse",
                "test_mae",
                "test_mean_error",
                "delta_test_r_squared_vs_M0",
                "test_rmse_change_vs_M0",
            ]
        ],
        scaffold_results_df[
            [
                "split",
                "model",
                "n_train",
                "n_validation",
                "n_train_final",
                "n_test",
                "validation_r_squared",
                "validation_rmse",
                "validation_mae",
                "test_r_squared",
                "test_rmse",
                "test_mae",
                "test_mean_error",
                "delta_test_r_squared_vs_M0",
                "test_rmse_change_vs_M0",
            ]
        ],
    ],
    ignore_index=True
)


# =====================================================================
# SAVE MODEL RESULTS
# =====================================================================

combined_results.to_csv(
    OUTPUT_MODEL,
    index=False
)


# =====================================================================
# REPORT
# =====================================================================

report_lines = []

report_lines.append(
    "SCRIPT 17 — MODEL EVALUATION"
)
report_lines.append(
    "Random and scaffold-aware held-out evaluation"
)
report_lines.append("=" * 70)
report_lines.append("")

report_lines.append(
    f"Population C: {len(pop_c):,} compounds"
)
report_lines.append(
    f"Unique scaffold groups: {n_scaffolds:,}"
)
report_lines.append("")

report_lines.append(
    "MODEL DEFINITIONS"
)
report_lines.append("-" * 70)

for model_name, predictors in MODELS.items():
    report_lines.append(
        f"{model_name}: {', '.join(predictors)}"
    )

report_lines.append("")
report_lines.append(
    "RANDOM 80/20 HOLDOUT"
)
report_lines.append("-" * 70)

report_lines.append(
    random_results_df[
        [
            "model",
            "r_squared",
            "rmse",
            "mae",
            "mean_error",
        ]
    ].to_string(index=False)
)

report_lines.append("")
report_lines.append(
    "SCAFFOLD-AWARE VALIDATION"
)
report_lines.append("-" * 70)

report_lines.append(
    scaffold_results_df[
        [
            "model",
            "validation_r_squared",
            "validation_rmse",
            "validation_mae",
        ]
    ].to_string(index=False)
)

report_lines.append("")
report_lines.append(
    "SCAFFOLD-AWARE TEST"
)
report_lines.append("-" * 70)

report_lines.append(
    scaffold_results_df[
        [
            "model",
            "test_r_squared",
            "test_rmse",
            "test_mae",
            "test_mean_error",
        ]
    ].to_string(index=False)
)

report_lines.append("")
report_lines.append(
    "SCAFFOLD SEPARATION"
)
report_lines.append("-" * 70)

report_lines.append(
    f"Train scaffolds:      {len(train_scaffolds):,}"
)
report_lines.append(
    f"Validation scaffolds: {len(validation_scaffolds):,}"
)
report_lines.append(
    f"Test scaffolds:       {len(test_scaffolds):,}"
)
report_lines.append(
    f"Train/validation overlap: {len(train_validation_overlap)}"
)
report_lines.append(
    f"Train/test overlap:       {len(train_test_overlap)}"
)
report_lines.append(
    f"Validation/test overlap:  {len(validation_test_overlap)}"
)

report_lines.append("")
report_lines.append(
    "OUTPUTS"
)
report_lines.append("-" * 70)
report_lines.append(str(OUTPUT_MODEL))
report_lines.append(str(OUTPUT_SPLIT))
report_lines.append(str(OUTPUT_REPORT))

OUTPUT_REPORT.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)


# =====================================================================
# FINAL OUTPUT
# =====================================================================

print("\n" + "=" * 70)
print("OUTPUTS")
print("=" * 70)

print(f"\nModel evaluation:")
print(OUTPUT_MODEL)

print(f"\nSplit assignments:")
print(OUTPUT_SPLIT)

print(f"\nReport:")
print(OUTPUT_REPORT)

print("\n" + "=" * 70)
print("SCRIPT 17 COMPLETE")
print("=" * 70)