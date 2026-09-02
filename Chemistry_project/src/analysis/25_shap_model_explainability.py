"""
SCRIPT 25 — GRADIENT BOOSTING SHAP MODEL EXPLAINABILITY

Purpose
-------
Explain how the best-performing nonlinear model from Script 23 uses the
six molecular descriptors to predict aqueous solubility.

The analysis uses the same:
    - locked Population C
    - six M4 molecular descriptors
    - Gradient Boosting model specification
    - scaffold-aware repeated evaluation framework
    - random seeds 100–109

This script focuses on model interpretation rather than model selection.

Outputs
-------
1. Detailed SHAP values across all repeated scaffold test sets.
2. Aggregated global SHAP feature importance.
3. Mean signed SHAP effect by feature.
4. A reproducible textual report.

Interpretation
--------------
SHAP magnitude indicates how strongly a feature contributes to an
individual prediction.

Mean absolute SHAP value measures global feature influence.

Mean signed SHAP value describes the average direction of contribution
across the evaluated test observations. Because positive and negative
effects can cancel, signed SHAP should not be interpreted as a complete
description of nonlinear relationships.

The analysis is observational/model-based and does not establish
causality.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.ensemble import GradientBoostingRegressor

import shap


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

OUTPUT_VALUES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "shap_values.csv"
)

OUTPUT_SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "shap_feature_summary.csv"
)

OUTPUT_EFFECT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "shap_feature_effects.csv"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "shap_model_explainability.txt"
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

# Same Gradient Boosting specification used in Script 23.
GB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "loss": "squared_error",
    "random_state": 42,
}

# Number of test observations retained per repetition for detailed
# SHAP output. None means retain every test observation.
MAX_SHAP_ROWS_PER_REPETITION = None


# ---------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------

print("=" * 70)
print("SCRIPT 25 — GRADIENT BOOSTING SHAP MODEL EXPLAINABILITY")
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
# VERIFY FEATURES
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

    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol,
        includeChirality=False
    )


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
    Create the same deterministic scaffold-level split strategy used
    by Script 22 and Script 23.
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

    test_scaffolds = set(
        test_scaffolds
    )

    test_mask = df["scaffold"].isin(
        test_scaffolds
    )

    train_idx = df.index[~test_mask]
    test_idx = df.index[test_mask]

    return train_idx, test_idx


# ---------------------------------------------------------------------
# SHAP COMPATIBILITY HELPER
# ---------------------------------------------------------------------

def calculate_shap_values(model, X):

    """
    Calculate SHAP values for a scikit-learn GradientBoostingRegressor.

    TreeExplainer is used because GradientBoostingRegressor is a tree
    ensemble and therefore supports exact tree-based SHAP calculations.
    """

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    shap_values = np.asarray(
        shap_values
    )

    if shap_values.ndim == 3:
        shap_values = shap_values[0]

    return shap_values


# ---------------------------------------------------------------------
# REPEATED SHAP ANALYSIS
# ---------------------------------------------------------------------

print("\n" + "-" * 70)
print("REPEATED SCAFFOLD-AWARE SHAP ANALYSIS")
print("-" * 70)

print(
    f"Repeats: {N_REPEATS}"
)

print(
    f"Target test fraction: {TEST_FRACTION:.0%}"
)

print(
    "Gradient Boosting model:"
)

for key, value in GB_PARAMS.items():
    print(
        f"  {key}: {value}"
    )

print(
    "\nFeatures:"
)

for feature in NONLINEAR_FEATURES:
    print(
        f"  - {feature}"
    )

print(
    "\nSHAP interpretation:"
)

print(
    "  Mean absolute SHAP = average magnitude of model contribution."
)

print(
    "  Signed SHAP = average direction of model contribution."
)


# ---------------------------------------------------------------------
# STORAGE
# ---------------------------------------------------------------------

shap_rows = []
summary_rows = []


# ---------------------------------------------------------------------
# REPEATED EVALUATION
# ---------------------------------------------------------------------

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

    # --------------------------------------------------------------
    # FIT MODEL
    # --------------------------------------------------------------

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

    # --------------------------------------------------------------
    # SHAP
    # --------------------------------------------------------------

    shap_values = calculate_shap_values(
        model,
        X_test
    )

    if shap_values.shape != X_test.shape:
        raise ValueError(
            f"Unexpected SHAP shape in repetition "
            f"{repetition}: "
            f"{shap_values.shape} vs "
            f"{X_test.shape}"
        )

    # --------------------------------------------------------------
    # OPTIONAL ROW LIMIT
    # --------------------------------------------------------------

    if MAX_SHAP_ROWS_PER_REPETITION is not None:

        rng = np.random.default_rng(
            seed
        )

        n_keep = min(
            MAX_SHAP_ROWS_PER_REPETITION,
            len(X_test)
        )

        selected_positions = rng.choice(
            len(X_test),
            size=n_keep,
            replace=False
        )

    else:

        selected_positions = np.arange(
            len(X_test)
        )

    # --------------------------------------------------------------
    # STORE DETAILED SHAP VALUES
    # --------------------------------------------------------------

    selected_X = X_test.iloc[
        selected_positions
    ]

    selected_shap = shap_values[
        selected_positions
    ]

    selected_y = y_test.iloc[
        selected_positions
    ]

    selected_predictions = predictions[
        selected_positions
    ]

    selected_ids = test_df[
        "ID"
    ].iloc[
        selected_positions
    ]

    selected_scaffolds = test_df[
        "scaffold"
    ].iloc[
        selected_positions
    ]

    for row_position in range(
        len(selected_positions)
    ):

        output_row = {
            "repetition": repetition,
            "seed": seed,
            "ID": selected_ids.iloc[
                row_position
            ],
            "scaffold": selected_scaffolds.iloc[
                row_position
            ],
            "observed_solubility": selected_y.iloc[
                row_position
            ],
            "predicted_solubility": selected_predictions[
                row_position
            ],
        }

        for feature_index, feature in enumerate(
            NONLINEAR_FEATURES
        ):

            output_row[
                f"{feature}_value"
            ] = selected_X.iloc[
                row_position,
                feature_index
            ]

            output_row[
                f"{feature}_shap"
            ] = selected_shap[
                row_position,
                feature_index
            ]

        shap_rows.append(
            output_row
        )

    # --------------------------------------------------------------
    # REPETITION SUMMARY
    # --------------------------------------------------------------

    print(
        f"\nRepetition {repetition:02d} "
        f"(seed={seed})"
    )

    print(
        f"  Train: {len(train_df):,}"
        f" | Test: {len(test_df):,}"
        f" | Test scaffolds: {len(test_scaffolds):,}"
    )

    repetition_importance = np.mean(
        np.abs(shap_values),
        axis=0
    )

    repetition_signed = np.mean(
        shap_values,
        axis=0
    )

    ranked_features = sorted(
        zip(
            NONLINEAR_FEATURES,
            repetition_importance,
        ),
        key=lambda x: x[1],
        reverse=True
    )

    print(
        f"  Held-out mean absolute SHAP: "
        f"{np.mean(np.abs(shap_values)):.4f}"
    )

    print(
        f"  Top feature: "
        f"{ranked_features[0][0]} "
        f"({ranked_features[0][1]:.4f})"
    )

    for feature_index, feature in enumerate(
        NONLINEAR_FEATURES
    ):

        summary_rows.append(
            {
                "repetition": repetition,
                "seed": seed,
                "feature": feature,
                "mean_absolute_shap":
                    repetition_importance[
                        feature_index
                    ],
                "mean_signed_shap":
                    repetition_signed[
                        feature_index
                    ],
                "mean_shap_rank":
                    None,
                "n_test":
                    len(test_df),
                "n_test_scaffolds":
                    len(test_scaffolds),
            }
        )

    # Add ranks within repetition.

    rank_lookup = {
        feature: rank
        for rank, (
            feature,
            _
        ) in enumerate(
            ranked_features,
            start=1
        )
    }

    start_index = len(
        summary_rows
    ) - len(
        NONLINEAR_FEATURES
    )

    for i, feature in enumerate(
        NONLINEAR_FEATURES
    ):

        summary_rows[
            start_index + i
        ][
            "mean_shap_rank"
        ] = rank_lookup[
            feature
        ]


# ---------------------------------------------------------------------
# DATAFRAMES
# ---------------------------------------------------------------------

shap_values_df = pd.DataFrame(
    shap_rows
)

summary_df = pd.DataFrame(
    summary_rows
)


# ---------------------------------------------------------------------
# AGGREGATED FEATURE IMPORTANCE
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("AGGREGATED SHAP FEATURE IMPORTANCE")
print("=" * 70)


aggregated_rows = []

for feature in NONLINEAR_FEATURES:

    subset = summary_df[
        summary_df["feature"] == feature
    ]

    values = subset[
        "mean_absolute_shap"
    ].to_numpy()

    signed_values = subset[
        "mean_signed_shap"
    ].to_numpy()

    ranks = subset[
        "mean_shap_rank"
    ].to_numpy()

    mean_abs = np.mean(
        values
    )

    sd_abs = np.std(
        values,
        ddof=1
    )

    se_abs = (
        sd_abs
        / np.sqrt(len(values))
    )

    ci_low_abs = (
        mean_abs
        - 1.96 * se_abs
    )

    ci_high_abs = (
        mean_abs
        + 1.96 * se_abs
    )

    mean_signed = np.mean(
        signed_values
    )

    sd_signed = np.std(
        signed_values,
        ddof=1
    )

    se_signed = (
        sd_signed
        / np.sqrt(len(signed_values))
    )

    ci_low_signed = (
        mean_signed
        - 1.96 * se_signed
    )

    ci_high_signed = (
        mean_signed
        + 1.96 * se_signed
    )

    aggregated_rows.append(
        {
            "feature": feature,

            "mean_absolute_shap":
                mean_abs,

            "sd_absolute_shap":
                sd_abs,

            "ci_low_absolute_shap":
                ci_low_abs,

            "ci_high_absolute_shap":
                ci_high_abs,

            "mean_signed_shap":
                mean_signed,

            "sd_signed_shap":
                sd_signed,

            "ci_low_signed_shap":
                ci_low_signed,

            "ci_high_signed_shap":
                ci_high_signed,

            "mean_rank":
                np.mean(ranks),

            "top1_fraction":
                np.mean(ranks == 1),

            "positive_signed_fraction":
                np.mean(signed_values > 0),

            "negative_signed_fraction":
                np.mean(signed_values < 0),
        }
    )


aggregated_df = pd.DataFrame(
    aggregated_rows
)

aggregated_df = aggregated_df.sort_values(
    "mean_absolute_shap",
    ascending=False
).reset_index(
    drop=True
)

aggregated_df[
    "importance_rank"
] = np.arange(
    1,
    len(aggregated_df) + 1
)


# ---------------------------------------------------------------------
# PRINT AGGREGATED IMPORTANCE
# ---------------------------------------------------------------------

print(
    "\nFeatures ranked by mean absolute SHAP:"
)

for _, row in aggregated_df.iterrows():

    print(
        f"{int(row['importance_rank']):2d}. "
        f"{row['feature']:30s} "
        f"Mean |SHAP|="
        f"{row['mean_absolute_shap']:.4f} "
        f"95% CI=["
        f"{row['ci_low_absolute_shap']:.4f}, "
        f"{row['ci_high_absolute_shap']:.4f}] "
        f"Mean rank="
        f"{row['mean_rank']:.2f} "
        f"Top-1="
        f"{row['top1_fraction']:.1%}"
    )


# ---------------------------------------------------------------------
# SIGNED SHAP SUMMARY
# ---------------------------------------------------------------------

print("\n" + "-" * 70)
print("SIGNED SHAP EFFECT SUMMARY")
print("-" * 70)

print(
    "\nMean signed SHAP values:"
)

for _, row in aggregated_df.iterrows():

    print(
        f"{row['feature']:30s} "
        f"Mean={row['mean_signed_shap']:+.4f} "
        f"95% CI=["
        f"{row['ci_low_signed_shap']:+.4f}, "
        f"{row['ci_high_signed_shap']:+.4f}] "
        f"Positive repetitions="
        f"{row['positive_signed_fraction']:.1%}"
    )


# ---------------------------------------------------------------------
# MODEL CONSISTENCY
# ---------------------------------------------------------------------

print("\n" + "-" * 70)
print("SHAP RANK CONSISTENCY")
print("-" * 70)

for _, row in aggregated_df.iterrows():

    print(
        f"{row['feature']:30s} "
        f"Top-1="
        f"{row['top1_fraction']:.1%} "
        f"Mean rank="
        f"{row['mean_rank']:.2f}"
    )


# ---------------------------------------------------------------------
# SAVE OUTPUTS
# ---------------------------------------------------------------------

OUTPUT_VALUES_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

shap_values_df.to_csv(
    OUTPUT_VALUES_FILE,
    index=False
)

aggregated_df.to_csv(
    OUTPUT_SUMMARY_FILE,
    index=False
)

summary_df.to_csv(
    OUTPUT_EFFECT_FILE,
    index=False
)


# ---------------------------------------------------------------------
# WRITE REPORT
# ---------------------------------------------------------------------

report_lines = []

report_lines.append(
    "SCRIPT 25 — GRADIENT BOOSTING SHAP MODEL EXPLAINABILITY"
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

report_lines.append(
    "-" * 70
)

for feature in NONLINEAR_FEATURES:

    report_lines.append(
        feature
    )

report_lines.append("")

report_lines.append(
    "GLOBAL SHAP IMPORTANCE"
)

report_lines.append(
    "-" * 70
)

for _, row in aggregated_df.iterrows():

    report_lines.append(
        f"{row['feature']} | "
        f"mean_absolute_shap="
        f"{row['mean_absolute_shap']:.6f} | "
        f"SD="
        f"{row['sd_absolute_shap']:.6f} | "
        f"95% CI="
        f"["
        f"{row['ci_low_absolute_shap']:.6f}, "
        f"{row['ci_high_absolute_shap']:.6f}"
        f"] | "
        f"mean_rank="
        f"{row['mean_rank']:.3f} | "
        f"top1_fraction="
        f"{row['top1_fraction']:.3f}"
    )

report_lines.append("")

report_lines.append(
    "SIGNED SHAP EFFECTS"
)

report_lines.append(
    "-" * 70
)

for _, row in aggregated_df.iterrows():

    report_lines.append(
        f"{row['feature']} | "
        f"mean_signed_shap="
        f"{row['mean_signed_shap']:.6f} | "
        f"SD="
        f"{row['sd_signed_shap']:.6f} | "
        f"95% CI="
        f"["
        f"{row['ci_low_signed_shap']:.6f}, "
        f"{row['ci_high_signed_shap']:.6f}"
        f"] | "
        f"positive_fraction="
        f"{row['positive_signed_fraction']:.3f}"
    )

report_lines.append("")

report_lines.append(
    "INTERPRETATION"
)

report_lines.append(
    "-" * 70
)

report_lines.append(
    "SHAP values quantify the contribution of each molecular "
    "descriptor to individual Gradient Boosting predictions."
)

report_lines.append(
    "Mean absolute SHAP magnitude is used to rank global feature "
    "importance across the repeated scaffold-aware test sets."
)

report_lines.append(
    "Mean signed SHAP indicates the average direction of a feature's "
    "contribution, but cancellation between positive and negative "
    "effects means it should not be interpreted as a complete "
    "description of nonlinear feature behavior."
)

report_lines.append(
    "The repeated scaffold-aware design evaluates explainability "
    "across structurally separated held-out compounds rather than "
    "random molecule-level splits."
)

report_lines.append(
    "SHAP-based importance describes the fitted model's behavior and "
    "does not establish causal relationships between molecular "
    "properties and aqueous solubility."
)

report_lines.append("")

report_lines.append(
    "OUTPUTS"
)

report_lines.append(
    "-" * 70
)

report_lines.append(
    str(OUTPUT_VALUES_FILE)
)

report_lines.append(
    str(OUTPUT_SUMMARY_FILE)
)

report_lines.append(
    str(OUTPUT_EFFECT_FILE)
)

report_lines.append(
    str(REPORT_FILE)
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
# COMPLETION
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("OUTPUTS")
print("=" * 70)

print(
    "\nDetailed SHAP values:"
)

print(
    OUTPUT_VALUES_FILE
)

print(
    "\nAggregated SHAP feature summary:"
)

print(
    OUTPUT_SUMMARY_FILE
)

print(
    "\nRepeated feature-level SHAP summary:"
)

print(
    OUTPUT_EFFECT_FILE
)

print(
    "\nReport:"
)

print(
    REPORT_FILE
)

print("\n" + "=" * 70)
print("SCRIPT 25 COMPLETE")
print("=" * 70)