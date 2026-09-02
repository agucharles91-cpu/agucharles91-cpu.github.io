"""
SCRIPT 24 — NONLINEAR MODEL FEATURE IMPORTANCE

Purpose
-------
Interpret the nonlinear Gradient Boosting model established in Script 23.

The analysis asks:

    Which molecular descriptors contribute most consistently to
    nonlinear aqueous-solubility prediction?

Method
------
The same repeated scaffold-aware evaluation framework used in Scripts 22
and 23 is reproduced:

    - Population C = 8,643 compounds
    - Bemis-Murcko scaffold grouping
    - 10 scaffold-level repetitions
    - seeds 100–109
    - approximately 20% scaffold-held-out test data
    - no scaffold leakage between train and test

For each repetition:

    1. Fit Gradient Boosting using the six M4 descriptors.
    2. Extract native tree-based feature_importances_.
    3. Calculate permutation importance on the held-out test set.
    4. Record the importance of each descriptor.

Two importance measures are reported:

    Native GB importance
        Mean impurity-based importance from the fitted Gradient Boosting
        trees.

    Permutation importance
        Increase in held-out RMSE when a feature is randomly permuted.
        Larger positive values indicate greater predictive dependence.

Permutation importance is treated as the primary interpretive measure
because it is evaluated on unseen scaffold-held-out compounds.

This script does NOT alter Population C or the previous evaluation outputs.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error


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
    / "nonlinear_feature_importance.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nonlinear_feature_importance_summary.csv"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "nonlinear_feature_importance.txt"
)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

N_REPEATS = 10
TEST_FRACTION = 0.20

RANDOM_SEEDS = list(range(100, 110))

TARGET = "Solubility"

NONLINEAR_FEATURES = [
    "rdkit_molwt",
    "rdkit_mollogp",
    "rdkit_ring_count",
    "rdkit_aromatic_rings",
    "rdkit_rotatable_bonds",
    "rdkit_fraction_csp3",
]


# ---------------------------------------------------------------------
# GRADIENT BOOSTING CONFIGURATION
# ---------------------------------------------------------------------
#
# Script 23 established Gradient Boosting as the best nonlinear model.
#
# The estimator below deliberately uses a fixed configuration across
# all repetitions. The random_state is tied to the repetition seed so
# that each repetition remains independently reproducible.
#
# If Script 23 used explicit hyperparameters beyond these defaults,
# those should be copied here exactly before treating the numerical
# importance values as a strict continuation of Script 23.
# ---------------------------------------------------------------------

GB_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 3,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "subsample": 1.0,
    "loss": "squared_error",
}


# ---------------------------------------------------------------------
# PERMUTATION CONFIGURATION
# ---------------------------------------------------------------------

PERMUTATION_REPEATS = 10

PERMUTATION_SCORING = "neg_root_mean_squared_error"


# ---------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------

print("=" * 70)
print("SCRIPT 24 — NONLINEAR MODEL FEATURE IMPORTANCE")
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
        "standard_analytical_domain is missing from molecular_features.csv"
    )

population = features[
    features["standard_analytical_domain"] == True
].copy()

print(
    "Rows flagged as standard analytical domain:",
    f"{len(population):,}"
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


# ---------------------------------------------------------------------
# VERIFY NUMERIC FEATURES
# ---------------------------------------------------------------------

for feature in NONLINEAR_FEATURES:

    if not pd.api.types.is_numeric_dtype(
        population[feature]
    ):
        raise TypeError(
            f"Feature '{feature}' is not numeric."
        )

    if population[feature].isna().any():
        raise ValueError(
            f"Feature '{feature}' contains missing values."
        )


# ---------------------------------------------------------------------
# SCAFFOLD GENERATION
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
    Reproduce the scaffold-level split logic used in Script 22.

    Entire scaffold groups are assigned together.
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
# SUMMARY FUNCTION
# ---------------------------------------------------------------------

def summarize(values):

    values = np.asarray(
        values,
        dtype=float
    )

    mean = np.mean(values)

    sd = np.std(
        values,
        ddof=1
    )

    se = sd / np.sqrt(
        len(values)
    )

    ci_low = mean - 1.96 * se

    ci_high = mean + 1.96 * se

    return {
        "mean": mean,
        "sd": sd,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "min": np.min(values),
        "max": np.max(values),
    }


# ---------------------------------------------------------------------
# REPEATED FEATURE-IMPORTANCE EVALUATION
# ---------------------------------------------------------------------

print("\n" + "-" * 70)
print("REPEATED SCAFFOLD-AWARE FEATURE IMPORTANCE")
print("-" * 70)

print(
    f"Repeats: {N_REPEATS}"
)

print(
    f"Target test fraction: {TEST_FRACTION:.0%}"
)

print(
    "Seeds:",
    ", ".join(str(seed) for seed in RANDOM_SEEDS)
)

print(
    f"Permutation repetitions per feature: "
    f"{PERMUTATION_REPEATS}"
)

print(
    "\nImportance interpretation:"
)

print(
    "  Native importance = tree impurity-based importance."
)

print(
    "  Permutation importance = held-out RMSE increase "
    "after feature permutation."
)


importance_rows = []

repetition_summary_rows = []


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
            f"Scaffold leakage detected in repetition "
            f"{repetition}."
        )


    X_train = train_df[
        NONLINEAR_FEATURES
    ].copy()

    y_train = train_df[
        TARGET
    ].copy()

    X_test = test_df[
        NONLINEAR_FEATURES
    ].copy()

    y_test = test_df[
        TARGET
    ].copy()


    # ---------------------------------------------------------------
    # FIT GRADIENT BOOSTING
    # ---------------------------------------------------------------

    model = GradientBoostingRegressor(
        random_state=seed,
        **GB_PARAMS,
    )

    model.fit(
        X_train,
        y_train
    )


    # ---------------------------------------------------------------
    # BASELINE TEST PERFORMANCE
    # ---------------------------------------------------------------

    baseline_predictions = model.predict(
        X_test
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            baseline_predictions
        )
    )


    # ---------------------------------------------------------------
    # NATIVE FEATURE IMPORTANCE
    # ---------------------------------------------------------------

    native_importance = (
        model.feature_importances_
    )


    # ---------------------------------------------------------------
    # HELD-OUT PERMUTATION IMPORTANCE
    # ---------------------------------------------------------------

    permutation = permutation_importance(
        model,
        X_test,
        y_test,
        scoring=PERMUTATION_SCORING,
        n_repeats=PERMUTATION_REPEATS,
        random_state=seed,
        n_jobs=-1,
    )


    permutation_mean = (
        permutation.importances_mean
    )

    permutation_sd = (
        permutation.importances_std
    )


    # ---------------------------------------------------------------
    # FEATURE RANKS
    # ---------------------------------------------------------------

    native_order = np.argsort(
        -native_importance
    )

    permutation_order = np.argsort(
        -permutation_mean
    )

    native_rank = np.empty(
        len(NONLINEAR_FEATURES),
        dtype=int
    )

    permutation_rank = np.empty(
        len(NONLINEAR_FEATURES),
        dtype=int
    )

    for rank, idx in enumerate(
        native_order,
        start=1
    ):
        native_rank[idx] = rank

    for rank, idx in enumerate(
        permutation_order,
        start=1
    ):
        permutation_rank[idx] = rank


    # ---------------------------------------------------------------
    # STORE FEATURE RESULTS
    # ---------------------------------------------------------------

    for i, feature in enumerate(
        NONLINEAR_FEATURES
    ):

        importance_rows.append(
            {
                "repetition": repetition,
                "seed": seed,
                "n_train": len(train_df),
                "n_test": len(test_df),
                "n_train_scaffolds": len(
                    train_scaffolds
                ),
                "n_test_scaffolds": len(
                    test_scaffolds
                ),
                "baseline_test_rmse": baseline_rmse,
                "feature": feature,
                "native_importance": native_importance[i],
                "native_rank": native_rank[i],
                "permutation_importance": permutation_mean[i],
                "permutation_importance_sd": permutation_sd[i],
                "permutation_rank": permutation_rank[i],
            }
        )


    # ---------------------------------------------------------------
    # REPETITION OUTPUT
    # ---------------------------------------------------------------

    best_native_idx = np.argmax(
        native_importance
    )

    best_permutation_idx = np.argmax(
        permutation_mean
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
        f"  Held-out RMSE: "
        f"{baseline_rmse:.4f}"
    )

    print(
        "  Top native importance: "
        f"{NONLINEAR_FEATURES[best_native_idx]}"
        f" ({native_importance[best_native_idx]:.4f})"
    )

    print(
        "  Top permutation importance: "
        f"{NONLINEAR_FEATURES[best_permutation_idx]}"
        f" ({permutation_mean[best_permutation_idx]:+.4f})"
    )


# ---------------------------------------------------------------------
# CREATE DETAILED DATAFRAME
# ---------------------------------------------------------------------

importance_df = pd.DataFrame(
    importance_rows
)


# ---------------------------------------------------------------------
# AGGREGATED FEATURE SUMMARY
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("AGGREGATED FEATURE IMPORTANCE")
print("=" * 70)


summary_rows = []


for feature in NONLINEAR_FEATURES:

    subset = importance_df[
        importance_df["feature"] == feature
    ].copy()


    native_stats = summarize(
        subset["native_importance"]
    )

    permutation_stats = summarize(
        subset["permutation_importance"]
    )


    mean_native_rank = np.mean(
        subset["native_rank"]
    )

    mean_permutation_rank = np.mean(
        subset["permutation_rank"]
    )


    top_native_fraction = np.mean(
        subset["native_rank"] == 1
    )

    top_permutation_fraction = np.mean(
        subset["permutation_rank"] == 1
    )


    positive_permutation_fraction = np.mean(
        subset["permutation_importance"] > 0
    )


    summary_rows.append(
        {
            "feature": feature,

            "native_mean":
                native_stats["mean"],

            "native_sd":
                native_stats["sd"],

            "native_ci_low":
                native_stats["ci_low"],

            "native_ci_high":
                native_stats["ci_high"],

            "native_min":
                native_stats["min"],

            "native_max":
                native_stats["max"],

            "native_mean_rank":
                mean_native_rank,

            "native_top_rank_fraction":
                top_native_fraction,

            "permutation_mean":
                permutation_stats["mean"],

            "permutation_sd":
                permutation_stats["sd"],

            "permutation_ci_low":
                permutation_stats["ci_low"],

            "permutation_ci_high":
                permutation_stats["ci_high"],

            "permutation_min":
                permutation_stats["min"],

            "permutation_max":
                permutation_stats["max"],

            "permutation_mean_rank":
                mean_permutation_rank,

            "permutation_top_rank_fraction":
                top_permutation_fraction,

            "permutation_positive_fraction":
                positive_permutation_fraction,
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


# ---------------------------------------------------------------------
# SORT BY PERMUTATION IMPORTANCE
# ---------------------------------------------------------------------

summary_df = summary_df.sort_values(
    "permutation_mean",
    ascending=False
).reset_index(
    drop=True
)


# ---------------------------------------------------------------------
# PRINT SUMMARY
# ---------------------------------------------------------------------

print(
    "\nFeatures ranked by mean held-out permutation importance:"
)

for rank, row in summary_df.iterrows():

    print(
        f"{rank + 1:2d}. "
        f"{row['feature']:28s} "
        f"Permutation="
        f"{row['permutation_mean']:+.4f} "
        f"95% CI=["
        f"{row['permutation_ci_low']:+.4f}, "
        f"{row['permutation_ci_high']:+.4f}] "
        f"Mean rank="
        f"{row['permutation_mean_rank']:.2f}"
    )


print(
    "\nNative Gradient Boosting importance:"
)

native_ranked = summary_df.sort_values(
    "native_mean",
    ascending=False
)

for rank, row in native_ranked.iterrows():

    print(
        f"{rank + 1:2d}. "
        f"{row['feature']:28s} "
        f"Importance="
        f"{row['native_mean']:.4f} "
        f"95% CI=["
        f"{row['native_ci_low']:.4f}, "
        f"{row['native_ci_high']:.4f}] "
        f"Mean rank="
        f"{row['native_mean_rank']:.2f}"
    )


# ---------------------------------------------------------------------
# RANK CONSISTENCY
# ---------------------------------------------------------------------

print("\n" + "-" * 70)
print("RANK CONSISTENCY")
print("-" * 70)


for _, row in summary_df.iterrows():

    print(
        f"{row['feature']:28s} "
        f"Permutation top-1="
        f"{row['permutation_top_rank_fraction']:.1%} "
        f"| positive="
        f"{row['permutation_positive_fraction']:.1%}"
    )


# ---------------------------------------------------------------------
# OUTPUT DIRECTORIES
# ---------------------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------------------
# SAVE DETAILED RESULTS
# ---------------------------------------------------------------------

importance_df.to_csv(
    OUTPUT_FILE,
    index=False
)


summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


# ---------------------------------------------------------------------
# WRITE REPORT
# ---------------------------------------------------------------------

report_lines = []


report_lines.append(
    "SCRIPT 24 — NONLINEAR MODEL FEATURE IMPORTANCE"
)

report_lines.append(
    "=" * 70
)

report_lines.append(
    f"Population C: {len(population):,} compounds"
)

report_lines.append(
    f"Unique Bemis-Murcko scaffolds: {n_scaffolds:,}"
)

report_lines.append(
    f"Repeated scaffold splits: {N_REPEATS}"
)

report_lines.append(
    f"Test fraction: {TEST_FRACTION:.0%}"
)

report_lines.append(
    "Seeds: 100–109"
)

report_lines.append("")


report_lines.append(
    "MODEL"
)

report_lines.append(
    "-" * 70
)

report_lines.append(
    "Gradient Boosting regression"
)

report_lines.append(
    "Features: MolWt, MolLogP, RingCount, "
    "AromaticRings, RotatableBonds, FractionCSP3"
)

report_lines.append(
    f"n_estimators={GB_PARAMS['n_estimators']}"
)

report_lines.append(
    f"learning_rate={GB_PARAMS['learning_rate']}"
)

report_lines.append(
    f"max_depth={GB_PARAMS['max_depth']}"
)

report_lines.append("")


report_lines.append(
    "FEATURE IMPORTANCE SUMMARY"
)

report_lines.append(
    "-" * 70
)


for rank, row in summary_df.iterrows():

    report_lines.append(
        f"{rank + 1}. {row['feature']} | "
        f"permutation_mean="
        f"{row['permutation_mean']:.6f} | "
        f"permutation_SD="
        f"{row['permutation_sd']:.6f} | "
        f"95% CI="
        f"[{row['permutation_ci_low']:.6f}, "
        f"{row['permutation_ci_high']:.6f}] | "
        f"mean_rank="
        f"{row['permutation_mean_rank']:.3f} | "
        f"top_rank_fraction="
        f"{row['permutation_top_rank_fraction']:.3f} | "
        f"positive_fraction="
        f"{row['permutation_positive_fraction']:.3f}"
    )


report_lines.append("")


report_lines.append(
    "NATIVE TREE IMPORTANCE"
)

report_lines.append(
    "-" * 70
)


for rank, row in enumerate(
    native_ranked.itertuples(),
    start=1
):

    report_lines.append(
        f"{rank}. {row.feature} | "
        f"mean={row.native_mean:.6f} | "
        f"SD={row.native_sd:.6f} | "
        f"95% CI="
        f"[{row.native_ci_low:.6f}, "
        f"{row.native_ci_high:.6f}] | "
        f"mean_rank="
        f"{row.native_mean_rank:.3f}"
    )


report_lines.append("")


report_lines.append(
    "METHODOLOGICAL INTERPRETATION"
)

report_lines.append(
    "-" * 70
)

report_lines.append(
    "The analysis evaluates feature importance for the Gradient "
    "Boosting model under the same repeated scaffold-aware "
    "evaluation framework used in Script 23."
)

report_lines.append(
    "Permutation importance is calculated on the held-out test "
    "set for each repetition. It therefore measures the extent "
    "to which disrupting a descriptor degrades predictive "
    "performance on previously unseen scaffold groups."
)

report_lines.append(
    "Native Gradient Boosting importance is impurity-based and "
    "should be interpreted as a complementary measure rather "
    "than as a causal effect."
)

report_lines.append(
    "Features with consistently positive permutation importance "
    "and low mean rank across repetitions provide stronger evidence "
    "of stable predictive contribution."
)

report_lines.append(
    "Feature importance does not establish causality. Correlated "
    "molecular descriptors can share predictive information, causing "
    "individual importance values to depend on the presence of the "
    "other descriptors in the model."
)

report_lines.append(
    "These results describe predictive importance within the "
    "specified descriptor set and should not be interpreted as "
    "independent physicochemical effects."
)


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(report_lines)
    )


# ---------------------------------------------------------------------
# FINAL OUTPUT
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("OUTPUTS")
print("=" * 70)

print(
    "\nDetailed feature importance:"
)

print(
    OUTPUT_FILE
)

print(
    "\nAggregated feature summary:"
)

print(
    SUMMARY_FILE
)

print(
    "\nReport:"
)

print(
    REPORT_FILE
)

print("\n" + "=" * 70)
print("SCRIPT 24 COMPLETE")
print("=" * 70)