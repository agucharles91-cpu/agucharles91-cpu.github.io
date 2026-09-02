"""
SCRIPT 23 — NONLINEAR MODEL EVALUATION

Purpose
-------
Test whether nonlinear machine-learning models capture additional
predictive structure in aqueous solubility beyond the locked linear
models M0 and M4.

This script uses the EXACT scaffold-generation and repeated scaffold
splitting procedure from Script 22.

Models
------
M0:
    Solubility ~ MolWt + MolLogP

M4:
    Solubility ~ MolWt + MolLogP + RingCount + AromaticRings
                 + RotatableBonds + FractionCSP3

RF:
    Random Forest using the six M4 descriptors

GB:
    Gradient Boosting using the six M4 descriptors

Method
------
Repeated scaffold-aware holdout evaluation.

10 independent repetitions are generated using the same scaffold
splitting function and seeds (100–109) used in Script 22.

Entire Bemis-Murcko scaffold groups remain together within each
train/test split.

The purpose is NOT to optimize the nonlinear models aggressively.
The purpose is to determine whether nonlinear functional forms provide
a meaningful and reproducible improvement over the already-established
linear structural model.

This script does NOT alter the locked Population C.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

import statsmodels.api as sm

from sklearn.ensemble import RandomForestRegressor
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
    / "nonlinear_model_evaluation.csv"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "nonlinear_model_evaluation.txt"
)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

N_REPEATS = 10
TEST_FRACTION = 0.20

RANDOM_SEEDS = list(range(100, 100 + N_REPEATS))

TARGET = "Solubility"


# ---------------------------------------------------------------------
# MODEL FEATURES
# ---------------------------------------------------------------------

M0_FEATURES = [
    "rdkit_molwt",
    "rdkit_mollogp",
]


M4_FEATURES = [
    "rdkit_molwt",
    "rdkit_mollogp",
    "rdkit_ring_count",
    "rdkit_aromatic_rings",
    "rdkit_rotatable_bonds",
    "rdkit_fraction_csp3",
]


NONLINEAR_FEATURES = M4_FEATURES.copy()


# ---------------------------------------------------------------------
# NONLINEAR MODEL CONFIGURATION
# ---------------------------------------------------------------------
#
# These are deliberately fixed rather than tuned separately inside
# every scaffold split. The goal is a fair structural-form comparison,
# not hyperparameter optimization.
#
# Both models use the same six descriptors as M4.
# ---------------------------------------------------------------------

RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_leaf": 5,
    "max_features": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}


GB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "max_depth": 3,
    "min_samples_leaf": 5,
    "loss": "squared_error",
    "random_state": 42,
}


# ---------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------

print("=" * 70)
print("SCRIPT 23 — NONLINEAR MODEL EVALUATION")
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
    f"{len(population):,}"
)


if len(population) != 8643:
    raise ValueError(
        f"Expected Population C = 8643 rows, "
        f"found {len(population)}"
    )


print("Population C verified: 8,643 rows")


# ---------------------------------------------------------------------
# ENSURE TARGET EXISTS
# ---------------------------------------------------------------------

if TARGET not in population.columns:

    if TARGET not in raw.columns:
        raise ValueError(
            f"{TARGET} not found in either "
            "molecular_features.csv or raw dataset."
        )

    target_lookup = raw[
        ["ID", TARGET]
    ].copy()

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


# ---------------------------------------------------------------------
# VERIFY REQUIRED FEATURES
# ---------------------------------------------------------------------

required_features = sorted(
    set(
        M4_FEATURES
        + [
            TARGET,
            "SMILES",
            "ID",
        ]
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
#
# IMPORTANT:
# This is copied from Script 22 intentionally.
# Do not substitute another scaffold implementation here.
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


population["scaffold"] = population[
    "SMILES"
].apply(get_scaffold)


if population["scaffold"].isna().any():

    invalid_count = population[
        "scaffold"
    ].isna().sum()

    raise ValueError(
        f"{invalid_count} compounds have invalid "
        "scaffold generation."
    )


n_scaffolds = population[
    "scaffold"
].nunique()


print(
    f"Unique scaffold groups: {n_scaffolds:,}"
)


# ---------------------------------------------------------------------
# SCAFFOLD SPLITTER
# ---------------------------------------------------------------------
#
# EXACTLY the same procedure as Script 22.
# ---------------------------------------------------------------------

def make_scaffold_split(
    df,
    test_fraction,
    seed,
):
    """
    Create a scaffold-aware train/test split.

    Entire scaffold groups are assigned together.

    Scaffold groups are shuffled deterministically using the supplied
    seed, then assigned to test until approximately the requested
    fraction of compounds is reached.
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
        round(
            len(df) * test_fraction
        )
    )

    test_scaffolds = []

    test_n = 0

    for scaffold in shuffled:

        if test_n >= target_test_n:
            break

        test_scaffolds.append(
            scaffold
        )

        test_n += scaffold_sizes.loc[
            scaffold
        ]

    test_scaffolds = set(
        test_scaffolds
    )

    test_mask = df[
        "scaffold"
    ].isin(test_scaffolds)

    train_idx = df.index[
        ~test_mask
    ]

    test_idx = df.index[
        test_mask
    ]

    return train_idx, test_idx


# ---------------------------------------------------------------------
# LINEAR MODEL
# ---------------------------------------------------------------------

def fit_predict_linear(
    train_df,
    test_df,
    predictors,
):

    X_train = train_df[
        predictors
    ].copy()

    y_train = train_df[
        TARGET
    ].copy()

    X_test = test_df[
        predictors
    ].copy()

    y_test = test_df[
        TARGET
    ].copy()

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

    predictions = model.predict(
        X_test
    )

    return calculate_metrics(
        y_test,
        predictions
    )


# ---------------------------------------------------------------------
# NONLINEAR MODEL
# ---------------------------------------------------------------------

def fit_predict_nonlinear(
    train_df,
    test_df,
    model,
    predictors,
):

    # Convert explicitly to NumPy arrays.
    #
    # This avoids the duplicate-column / pandas metadata problem that
    # caused the previous Random Forest failure.
    X_train = train_df[
        predictors
    ].to_numpy(
        dtype=float
    )

    y_train = train_df[
        TARGET
    ].to_numpy(
        dtype=float
    )

    X_test = test_df[
        predictors
    ].to_numpy(
        dtype=float
    )

    y_test = test_df[
        TARGET
    ].to_numpy(
        dtype=float
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    return calculate_metrics(
        y_test,
        predictions
    )


# ---------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------

def calculate_metrics(
    y_true,
    predictions,
):

    r2 = r2_score(
        y_true,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions
        )
    )

    mae = mean_absolute_error(
        y_true,
        predictions
    )

    mean_error = np.mean(
        y_true - predictions
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
print("REPEATED SCAFFOLD-AWARE NONLINEAR EVALUATION")
print("-" * 70)

print(
    f"Repeats: {N_REPEATS}"
)

print(
    f"Target test fraction: {TEST_FRACTION:.0%}"
)

print(
    "Same scaffold-level seeds as Script 22: "
    "100–109"
)


results = []


for repetition, seed in enumerate(
    RANDOM_SEEDS,
    start=1
):

    # ---------------------------------------------------------------
    # REPRODUCE SCRIPT 22 SPLIT
    # ---------------------------------------------------------------

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
        train_df[
            "scaffold"
        ]
    )

    test_scaffolds = set(
        test_df[
            "scaffold"
        ]
    )


    overlap = (
        train_scaffolds
        .intersection(
            test_scaffolds
        )
    )


    if overlap:

        raise ValueError(
            f"Scaffold leakage detected in "
            f"repetition {repetition}."
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


    # ---------------------------------------------------------------
    # M0
    # ---------------------------------------------------------------

    m0 = fit_predict_linear(
        train_df,
        test_df,
        M0_FEATURES
    )


    # ---------------------------------------------------------------
    # M4
    # ---------------------------------------------------------------

    m4 = fit_predict_linear(
        train_df,
        test_df,
        M4_FEATURES
    )


    # ---------------------------------------------------------------
    # RANDOM FOREST
    # ---------------------------------------------------------------

    rf = RandomForestRegressor(
        **RF_PARAMS
    )


    rf_metrics = fit_predict_nonlinear(
        train_df,
        test_df,
        rf,
        NONLINEAR_FEATURES
    )


    # ---------------------------------------------------------------
    # GRADIENT BOOSTING
    # ---------------------------------------------------------------

    gb = GradientBoostingRegressor(
        **GB_PARAMS
    )


    gb_metrics = fit_predict_nonlinear(
        train_df,
        test_df,
        gb,
        NONLINEAR_FEATURES
    )


    models = {
        "M0_baseline": m0,
        "M4_linear_structural": m4,
        "RF_random_forest": rf_metrics,
        "GB_gradient_boosting": gb_metrics,
    }


    # ---------------------------------------------------------------
    # SAVE RESULTS
    # ---------------------------------------------------------------

    for model_name, metrics in models.items():

        results.append(
            {
                "repetition": repetition,
                "seed": seed,
                "model": model_name,
                "n_train": len(train_df),
                "n_test": len(test_df),
                "n_train_scaffolds": len(
                    train_scaffolds
                ),
                "n_test_scaffolds": len(
                    test_scaffolds
                ),
                **metrics,
            }
        )


    # ---------------------------------------------------------------
    # DISPLAY
    # ---------------------------------------------------------------

    print(
        f"  M0: "
        f"R²={m0['r_squared']:.4f}, "
        f"RMSE={m0['rmse']:.4f}, "
        f"MAE={m0['mae']:.4f}"
    )

    print(
        f"  M4: "
        f"R²={m4['r_squared']:.4f}, "
        f"RMSE={m4['rmse']:.4f}, "
        f"MAE={m4['mae']:.4f}"
    )

    print(
        f"  RF: "
        f"R²={rf_metrics['r_squared']:.4f}, "
        f"RMSE={rf_metrics['rmse']:.4f}, "
        f"MAE={rf_metrics['mae']:.4f}"
    )

    print(
        f"  GB: "
        f"R²={gb_metrics['r_squared']:.4f}, "
        f"RMSE={gb_metrics['rmse']:.4f}, "
        f"MAE={gb_metrics['mae']:.4f}"
    )


# ---------------------------------------------------------------------
# RESULTS DATAFRAME
# ---------------------------------------------------------------------

results_df = pd.DataFrame(
    results
)


# ---------------------------------------------------------------------
# WITHIN-REPETITION IMPROVEMENT
# ---------------------------------------------------------------------

print("\n" + "-" * 70)
print("NONLINEAR MODEL IMPROVEMENT OVER M4")
print("-" * 70)


improvement_rows = []


for repetition in sorted(
    results_df[
        "repetition"
    ].unique()
):

    subset = results_df[
        results_df[
            "repetition"
        ] == repetition
    ]


    m4_row = subset[
        subset["model"]
        == "M4_linear_structural"
    ].iloc[0]


    rf_row = subset[
        subset["model"]
        == "RF_random_forest"
    ].iloc[0]


    gb_row = subset[
        subset["model"]
        == "GB_gradient_boosting"
    ].iloc[0]


    improvement_rows.append(
        {
            "repetition": repetition,

            "rf_delta_r2_vs_m4":
                rf_row["r_squared"]
                - m4_row["r_squared"],

            "rf_delta_rmse_vs_m4":
                rf_row["rmse"]
                - m4_row["rmse"],

            "rf_delta_mae_vs_m4":
                rf_row["mae"]
                - m4_row["mae"],

            "gb_delta_r2_vs_m4":
                gb_row["r_squared"]
                - m4_row["r_squared"],

            "gb_delta_rmse_vs_m4":
                gb_row["rmse"]
                - m4_row["rmse"],

            "gb_delta_mae_vs_m4":
                gb_row["mae"]
                - m4_row["mae"],
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

    mean = np.mean(
        values
    )

    sd = np.std(
        values,
        ddof=1
    )

    se = sd / np.sqrt(
        len(values)
    )

    ci_low = (
        mean
        - 1.96 * se
    )

    ci_high = (
        mean
        + 1.96 * se
    )

    return {
        "mean": mean,
        "sd": sd,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "min": np.min(values),
        "max": np.max(values),
        "positive_fraction":
            np.mean(values > 0),
    }


# ---------------------------------------------------------------------
# PERFORMANCE SUMMARY
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("REPEATED-SPLIT PERFORMANCE SUMMARY")
print("=" * 70)


summary_rows = []


for model in [
    "M0_baseline",
    "M4_linear_structural",
    "RF_random_forest",
    "GB_gradient_boosting",
]:

    subset = results_df[
        results_df[
            "model"
        ] == model
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
            f"{model:25s} "
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
# RF IMPROVEMENT SUMMARY
# ---------------------------------------------------------------------

print("\n" + "-" * 70)
print("RANDOM FOREST IMPROVEMENT OVER M4")
print("-" * 70)


for metric in [
    "rf_delta_r2_vs_m4",
    "rf_delta_rmse_vs_m4",
    "rf_delta_mae_vs_m4",
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
# GB IMPROVEMENT SUMMARY
# ---------------------------------------------------------------------

print("\n" + "-" * 70)
print("GRADIENT BOOSTING IMPROVEMENT OVER M4")
print("-" * 70)


for metric in [
    "gb_delta_r2_vs_m4",
    "gb_delta_rmse_vs_m4",
    "gb_delta_mae_vs_m4",
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
# DETERMINE BEST MODEL BY MEAN PERFORMANCE
# ---------------------------------------------------------------------

mean_performance = (
    results_df
    .groupby("model")
    [
        [
            "r_squared",
            "rmse",
            "mae",
        ]
    ]
    .mean()
)


best_r2_model = (
    mean_performance[
        "r_squared"
    ]
    .idxmax()
)


best_rmse_model = (
    mean_performance[
        "rmse"
    ]
    .idxmin()
)


best_mae_model = (
    mean_performance[
        "mae"
    ]
    .idxmin()
)


print("\n" + "-" * 70)
print("MEAN PERFORMANCE COMPARISON")
print("-" * 70)


print(
    f"Best mean R²:   {best_r2_model}"
)

print(
    f"Best mean RMSE: {best_rmse_model}"
)

print(
    f"Best mean MAE:  {best_mae_model}"
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


# ---------------------------------------------------------------------
# WRITE REPORT
# ---------------------------------------------------------------------

report_lines = []


report_lines.append(
    "SCRIPT 23 — NONLINEAR MODEL EVALUATION"
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
    "MODELS"
)

report_lines.append(
    "-" * 70
)

report_lines.append(
    "M0: MolWt + MolLogP"
)

report_lines.append(
    "M4: MolWt + MolLogP + RingCount + "
    "AromaticRings + RotatableBonds + FractionCSP3"
)

report_lines.append(
    "RF: Random Forest using the six M4 descriptors"
)

report_lines.append(
    "GB: Gradient Boosting using the six M4 descriptors"
)

report_lines.append("")


report_lines.append(
    "NONLINEAR MODEL PARAMETERS"
)

report_lines.append(
    "-" * 70
)

report_lines.append(
    f"Random Forest: {RF_PARAMS}"
)

report_lines.append(
    f"Gradient Boosting: {GB_PARAMS}"
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
    "RANDOM FOREST IMPROVEMENT OVER M4"
)

report_lines.append(
    "-" * 70
)


for metric in [
    "rf_delta_r2_vs_m4",
    "rf_delta_rmse_vs_m4",
    "rf_delta_mae_vs_m4",
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
    "GRADIENT BOOSTING IMPROVEMENT OVER M4"
)

report_lines.append(
    "-" * 70
)


for metric in [
    "gb_delta_r2_vs_m4",
    "gb_delta_rmse_vs_m4",
    "gb_delta_mae_vs_m4",
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
    "This analysis tests whether nonlinear machine-learning "
    "models capture additional predictive structure beyond the "
    "linear M4 structural model."
)

report_lines.append(
    "The same Population C and the same deterministic scaffold "
    "splitting procedure used by Script 22 are reproduced here."
)

report_lines.append(
    "Entire Bemis-Murcko scaffold groups remain together within "
    "each train/test repetition."
)

report_lines.append(
    "Random Forest and Gradient Boosting receive exactly the same "
    "six molecular descriptors used by M4."
)

report_lines.append(
    "Therefore, any improvement over M4 is attributable to the "
    "nonlinear model form rather than the introduction of "
    "additional molecular descriptors."
)

report_lines.append(
    "The nonlinear models are evaluated using fixed hyperparameters "
    "rather than separately tuned configurations within each split, "
    "avoiding an additional layer of optimization during the "
    "comparative experiment."
)

report_lines.append(
    "A nonlinear model should only be considered substantively "
    "better if its improvement is consistent across scaffold-aware "
    "repetitions rather than driven by a small number of favorable "
    "splits."
)

report_lines.append("")


report_lines.append(
    "BEST MEAN-PERFORMANCE MODELS"
)

report_lines.append(
    "-" * 70
)

report_lines.append(
    f"Best mean R²: {best_r2_model}"
)

report_lines.append(
    f"Best mean RMSE: {best_rmse_model}"
)

report_lines.append(
    f"Best mean MAE: {best_mae_model}"
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
    "\nDetailed nonlinear evaluation results:"
)

print(
    OUTPUT_FILE
)


print(
    "\nReport:"
)

print(
    REPORT_FILE
)


print("\n" + "=" * 70)
print("SCRIPT 23 COMPLETE")
print("=" * 70)