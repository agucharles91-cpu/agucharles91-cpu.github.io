"""
SCRIPT 22 — REPEATED SCAFFOLD-AWARE MODEL EVALUATION

Purpose
-------
Test whether the improvement from the structural models over the simple
MolWt + MolLogP baseline is robust across multiple independent
scaffold-aware train/test splits.

Models
------
M0: Solubility ~ MolWt + MolLogP

M3: Solubility ~ MolWt + MolLogP + RingCount + AromaticRings

M4: Solubility ~ MolWt + MolLogP + RingCount + AromaticRings
                + RotatableBonds + FractionCSP3

Method
------
Repeated scaffold-group holdout evaluation.

No scaffold may appear in both train and test within a repetition.

The model is fitted independently within each training set and evaluated
only on its corresponding held-out test set.

This script does NOT alter the locked Population C.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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
    / "repeated_scaffold_evaluation.csv"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "repeated_scaffold_evaluation.txt"
)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

N_REPEATS = 10
TEST_FRACTION = 0.20

RANDOM_SEEDS = list(range(100, 100 + N_REPEATS))

TARGET = "Solubility"

M0_FEATURES = [
    "rdkit_molwt",
    "rdkit_mollogp",
]

M3_FEATURES = [
    "rdkit_molwt",
    "rdkit_mollogp",
    "rdkit_ring_count",
    "rdkit_aromatic_rings",
]

M4_FEATURES = [
    "rdkit_molwt",
    "rdkit_mollogp",
    "rdkit_ring_count",
    "rdkit_aromatic_rings",
    "rdkit_rotatable_bonds",
    "rdkit_fraction_csp3",
]


# ---------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------

print("=" * 70)
print("SCRIPT 22 — REPEATED SCAFFOLD-AWARE MODEL EVALUATION")
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

    population = population.drop(columns=[TARGET], errors="ignore")

    population = population.merge(
        target_lookup,
        on="ID",
        how="left",
        validate="one_to_one",
    )


if population[TARGET].isna().any():
    raise ValueError("Missing target values detected.")

print(f"Rows after solubility merge: {len(population):,}")


# ---------------------------------------------------------------------
# VERIFY REQUIRED FEATURES
# ---------------------------------------------------------------------

required_features = sorted(
    set(M4_FEATURES + [TARGET, "SMILES", "ID"])
)

missing_features = [
    col for col in required_features
    if col not in population.columns
]

if missing_features:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing_features)
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


population["scaffold"] = population["SMILES"].apply(get_scaffold)


if population["scaffold"].isna().any():

    invalid_count = population["scaffold"].isna().sum()

    raise ValueError(
        f"{invalid_count} compounds have invalid scaffold generation."
    )


n_scaffolds = population["scaffold"].nunique()

print(f"Unique scaffold groups: {n_scaffolds:,}")


# ---------------------------------------------------------------------
# SCAFFOLD SPLITTER
# ---------------------------------------------------------------------

def make_scaffold_split(
    df,
    test_fraction,
    seed,
):
    """
    Create a scaffold-aware train/test split.

    Entire scaffold groups are assigned together.

    The groups are shuffled deterministically using the supplied seed,
    then assigned to test until approximately the requested fraction
    of compounds is reached.

    This intentionally operates at scaffold-group level rather than
    molecule level.
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

    target_test_n = int(round(len(df) * test_fraction))

    test_scaffolds = []
    test_n = 0

    for scaffold in shuffled:

        if test_n >= target_test_n:
            break

        test_scaffolds.append(scaffold)
        test_n += scaffold_sizes.loc[scaffold]

    test_scaffolds = set(test_scaffolds)

    test_mask = df["scaffold"].isin(test_scaffolds)

    train_idx = df.index[~test_mask]
    test_idx = df.index[test_mask]

    return train_idx, test_idx


# ---------------------------------------------------------------------
# MODEL FITTING
# ---------------------------------------------------------------------

def fit_predict(train_df, test_df, predictors):

    X_train = train_df[predictors].copy()
    y_train = train_df[TARGET].copy()

    X_test = test_df[predictors].copy()
    y_test = test_df[TARGET].copy()

    X_train = sm.add_constant(
        X_train,
        has_constant="add"
    )

    X_test = sm.add_constant(
        X_test,
        has_constant="add"
    )

    model = sm.OLS(
        y_train,
        X_train
    ).fit()

    predictions = model.predict(X_test)

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
        y_test - predictions
    )

    return {
        "r_squared": r2,
        "rmse": rmse,
        "mae": mae,
        "mean_error": mean_error,
    }


# ---------------------------------------------------------------------
# REPEATED EVALUATION
# ---------------------------------------------------------------------

print("\n" + "-" * 70)
print("REPEATED SCAFFOLD-AWARE EVALUATION")
print("-" * 70)

print(
    f"Repeats: {N_REPEATS}"
)

print(
    f"Target test fraction: {TEST_FRACTION:.0%}"
)

print(
    "Each repetition uses a different scaffold-level random seed."
)

results = []

for repetition, seed in enumerate(
    RANDOM_SEEDS,
    start=1
):

    train_idx, test_idx = make_scaffold_split(
        population,
        TEST_FRACTION,
        seed,
    )

    train_df = population.loc[train_idx].copy()
    test_df = population.loc[test_idx].copy()

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
            f"Scaffold leakage detected in repetition {repetition}."
        )

    print(
        f"\nRepetition {repetition:02d} "
        f"(seed={seed})"
    )

    print(
        f"  Train: {len(train_df):,}"
        f" | Test: {len(test_df):,}"
        f" | Test scaffolds: {len(test_scaffolds):,}"
    )

    m0 = fit_predict(
        train_df,
        test_df,
        M0_FEATURES
    )

    m3 = fit_predict(
        train_df,
        test_df,
        M3_FEATURES
    )

    m4 = fit_predict(
        train_df,
        test_df,
        M4_FEATURES
    )

    models = {
        "M0_baseline": m0,
        "M3_ring_model": m3,
        "M4_selective_model": m4,
    }

    for model_name, metrics in models.items():

        results.append(
            {
                "repetition": repetition,
                "seed": seed,
                "model": model_name,
                "n_train": len(train_df),
                "n_test": len(test_df),
                "n_train_scaffolds": len(train_scaffolds),
                "n_test_scaffolds": len(test_scaffolds),
                **metrics,
            }
        )

    print(
        f"  M0: R²={m0['r_squared']:.4f}, "
        f"RMSE={m0['rmse']:.4f}, "
        f"MAE={m0['mae']:.4f}"
    )

    print(
        f"  M3: R²={m3['r_squared']:.4f}, "
        f"RMSE={m3['rmse']:.4f}, "
        f"MAE={m3['mae']:.4f}"
    )

    print(
        f"  M4: R²={m4['r_squared']:.4f}, "
        f"RMSE={m4['rmse']:.4f}, "
        f"MAE={m4['mae']:.4f}"
    )

    print(
        f"  M4 ΔR² vs M0: "
        f"{m4['r_squared'] - m0['r_squared']:+.4f}"
    )

    print(
        f"  M4 ΔRMSE vs M0: "
        f"{m4['rmse'] - m0['rmse']:+.4f}"
    )


results_df = pd.DataFrame(results)


# ---------------------------------------------------------------------
# CALCULATE WITHIN-REPETITION IMPROVEMENTS
# ---------------------------------------------------------------------

print("\n" + "-" * 70)
print("WITHIN-REPETITION MODEL IMPROVEMENT")
print("-" * 70)

pivot = results_df.pivot(
    index="repetition",
    columns="model",
    values=[
        "r_squared",
        "rmse",
        "mae",
    ]
)

improvement_rows = []

for repetition in sorted(
    results_df["repetition"].unique()
):

    m0_row = results_df[
        (results_df["repetition"] == repetition)
        & (results_df["model"] == "M0_baseline")
    ].iloc[0]

    m3_row = results_df[
        (results_df["repetition"] == repetition)
        & (results_df["model"] == "M3_ring_model")
    ].iloc[0]

    m4_row = results_df[
        (results_df["repetition"] == repetition)
        & (results_df["model"] == "M4_selective_model")
    ].iloc[0]

    improvement_rows.append(
        {
            "repetition": repetition,

            "m0_r_squared": m0_row["r_squared"],
            "m3_r_squared": m3_row["r_squared"],
            "m4_r_squared": m4_row["r_squared"],

            "m0_rmse": m0_row["rmse"],
            "m3_rmse": m3_row["rmse"],
            "m4_rmse": m4_row["rmse"],

            "m0_mae": m0_row["mae"],
            "m3_mae": m3_row["mae"],
            "m4_mae": m4_row["mae"],

            "m3_delta_r2_vs_m0":
                m3_row["r_squared"]
                - m0_row["r_squared"],

            "m4_delta_r2_vs_m0":
                m4_row["r_squared"]
                - m0_row["r_squared"],

            "m3_delta_rmse_vs_m0":
                m3_row["rmse"]
                - m0_row["rmse"],

            "m4_delta_rmse_vs_m0":
                m4_row["rmse"]
                - m0_row["rmse"],

            "m3_delta_mae_vs_m0":
                m3_row["mae"]
                - m0_row["mae"],

            "m4_delta_mae_vs_m0":
                m4_row["mae"]
                - m0_row["mae"],
        }
    )


improvements = pd.DataFrame(
    improvement_rows
)


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

    se = sd / np.sqrt(len(values))

    ci_low = mean - 1.96 * se
    ci_high = mean + 1.96 * se

    return {
        "mean": mean,
        "sd": sd,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "min": np.min(values),
        "max": np.max(values),
        "positive_fraction": np.mean(values > 0),
    }


# ---------------------------------------------------------------------
# PERFORMANCE SUMMARY
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("REPEATED-SPLIT SUMMARY")
print("=" * 70)

summary_rows = []

for model in [
    "M0_baseline",
    "M3_ring_model",
    "M4_selective_model",
]:

    subset = results_df[
        results_df["model"] == model
    ]

    for metric in [
        "r_squared",
        "rmse",
        "mae",
    ]:

        stats = summarize(
            subset[metric]
        )

        summary_rows.append(
            {
                "model": model,
                "metric": metric,
                **stats,
            }
        )

        print(
            f"{model:22s} "
            f"{metric:10s} "
            f"mean={stats['mean']:.4f} "
            f"SD={stats['sd']:.4f} "
            f"95% CI=["
            f"{stats['ci_low']:.4f}, "
            f"{stats['ci_high']:.4f}] "
            f"range=["
            f"{stats['min']:.4f}, "
            f"{stats['max']:.4f}]"
        )


summary_df = pd.DataFrame(
    summary_rows
)


# ---------------------------------------------------------------------
# IMPROVEMENT SUMMARY
# ---------------------------------------------------------------------

print("\n" + "-" * 70)
print("M4 IMPROVEMENT OVER M0")
print("-" * 70)

for metric in [
    "m4_delta_r2_vs_m0",
    "m4_delta_rmse_vs_m0",
    "m4_delta_mae_vs_m0",
]:

    stats = summarize(
        improvements[metric]
    )

    print(
        f"{metric:25s} "
        f"mean={stats['mean']:+.4f} "
        f"SD={stats['sd']:.4f} "
        f"95% CI=["
        f"{stats['ci_low']:+.4f}, "
        f"{stats['ci_high']:+.4f}] "
        f"range=["
        f"{stats['min']:+.4f}, "
        f"{stats['max']:+.4f}] "
        f"positive={stats['positive_fraction']:.1%}"
    )


# ---------------------------------------------------------------------
# M3 IMPROVEMENT SUMMARY
# ---------------------------------------------------------------------

print("\n" + "-" * 70)
print("M3 IMPROVEMENT OVER M0")
print("-" * 70)

for metric in [
    "m3_delta_r2_vs_m0",
    "m3_delta_rmse_vs_m0",
    "m3_delta_mae_vs_m0",
]:

    stats = summarize(
        improvements[metric]
    )

    print(
        f"{metric:25s} "
        f"mean={stats['mean']:+.4f} "
        f"SD={stats['sd']:.4f} "
        f"95% CI=["
        f"{stats['ci_low']:+.4f}, "
        f"{stats['ci_high']:+.4f}] "
        f"range=["
        f"{stats['min']:+.4f}, "
        f"{stats['max']:+.4f}] "
        f"positive={stats['positive_fraction']:.1%}"
    )


# ---------------------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("OUTPUTS")
print("=" * 70)

print(
    "\nDetailed repetition results:"
)

print(
    OUTPUT_FILE
)


# ---------------------------------------------------------------------
# WRITE REPORT
# ---------------------------------------------------------------------

report_lines = []

report_lines.append(
    "SCRIPT 22 — REPEATED SCAFFOLD-AWARE MODEL EVALUATION"
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
    "Models:"
)

report_lines.append(
    "M0: MolWt + MolLogP"
)

report_lines.append(
    "M3: MolWt + MolLogP + RingCount + AromaticRings"
)

report_lines.append(
    "M4: M3 + RotatableBonds + FractionCSP3"
)

report_lines.append("")

report_lines.append(
    "REPEATED PERFORMANCE SUMMARY"
)

report_lines.append(
    "-" * 70
)

for _, row in summary_df.iterrows():

    report_lines.append(
        f"{row['model']} | "
        f"{row['metric']} | "
        f"mean={row['mean']:.6f} | "
        f"SD={row['sd']:.6f} | "
        f"95% CI="
        f"[{row['ci_low']:.6f}, "
        f"{row['ci_high']:.6f}] | "
        f"range="
        f"[{row['min']:.6f}, "
        f"{row['max']:.6f}]"
    )


report_lines.append("")

report_lines.append(
    "M4 IMPROVEMENT OVER M0"
)

report_lines.append(
    "-" * 70
)

for metric in [
    "m4_delta_r2_vs_m0",
    "m4_delta_rmse_vs_m0",
    "m4_delta_mae_vs_m0",
]:

    stats = summarize(
        improvements[metric]
    )

    report_lines.append(
        f"{metric}: "
        f"mean={stats['mean']:.6f}, "
        f"SD={stats['sd']:.6f}, "
        f"95% CI="
        f"[{stats['ci_low']:.6f}, "
        f"{stats['ci_high']:.6f}], "
        f"range="
        f"[{stats['min']:.6f}, "
        f"{stats['max']:.6f}], "
        f"positive_fraction="
        f"{stats['positive_fraction']:.3f}"
    )


report_lines.append("")

report_lines.append(
    "METHODOLOGICAL INTERPRETATION"
)

report_lines.append(
    "-" * 70
)

report_lines.append(
    "The purpose of this analysis is to determine whether the "
    "structural-model improvement observed in the original "
    "scaffold-aware split is stable across multiple independent "
    "scaffold-level holdouts."
)

report_lines.append(
    "Each repetition keeps entire Bemis-Murcko scaffold groups "
    "together, preventing the same scaffold from appearing in both "
    "training and test data."
)

report_lines.append(
    "Positive M4 delta R-squared indicates that M4 explains more "
    "held-out variance than the MolWt + MolLogP baseline."
)

report_lines.append(
    "Negative M4 delta RMSE or MAE indicates improved prediction "
    "error relative to the baseline."
)

report_lines.append(
    "These results should be interpreted as robustness evidence, "
    "not as a final estimate of universal model performance."
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
print("SCRIPT 22 COMPLETE")
print("=" * 70)