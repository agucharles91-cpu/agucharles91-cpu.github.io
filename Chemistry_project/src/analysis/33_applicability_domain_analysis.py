"""
SCRIPT 33 — APPLICABILITY DOMAIN & CHEMICAL-SPACE RELIABILITY

Purpose
-------
Determine whether Gradient Boosting prediction reliability varies
with chemical-space representation under the repeated scaffold-aware
validation design.

The analysis does NOT fit, tune, or modify a predictive model.

It evaluates:
1. Descriptor-space distance from each held-out compound to its
   corresponding training population.
2. Chemical-space novelty strata.
3. Relationship between novelty and prediction error.
4. Scaffold representation characteristics.
5. Prediction uncertainty alongside chemical-space novelty.

Population C and the repeated held-out prediction dataset from
Script 26 are preserved.

Important methodological rule
-----------------------------
Applicability-domain diagnostics are calculated separately within
each scaffold-aware repetition using only the training compounds
from that repetition as the reference chemical space.

This prevents the held-out test compounds from defining their own
reference distribution.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


# =====================================================================
# CONFIGURATION
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

FEATURE_FILE = DATA_DIR / "molecular_features.csv"
PREDICTION_FILE = DATA_DIR / "nonlinear_residual_analysis.csv"

OUTPUT_NOVELTY_FILE = (
    DATA_DIR / "applicability_domain_compound_level.csv"
)

OUTPUT_STRATA_FILE = (
    DATA_DIR / "applicability_domain_by_novelty.csv"
)

OUTPUT_SCAFFOLD_FILE = (
    DATA_DIR / "applicability_domain_scaffold_representation.csv"
)

OUTPUT_SUMMARY_FILE = (
    DATA_DIR / "applicability_domain_summary.csv"
)

REPORT_FILE = (
    REPORT_DIR / "applicability_domain_analysis.txt"
)


# The six descriptors used by the established M4 / GB models.
FEATURES = [
    "rdkit_molwt",
    "rdkit_mollogp",
    "rdkit_ring_count",
    "rdkit_aromatic_rings",
    "rdkit_rotatable_bonds",
    "rdkit_fraction_csp3",
]


# =====================================================================
# HEADER
# =====================================================================

print("=" * 70)
print("SCRIPT 33 — APPLICABILITY DOMAIN & CHEMICAL-SPACE RELIABILITY")
print("=" * 70)

print("\nProject root:")
print(PROJECT_ROOT)


# =====================================================================
# INPUT VALIDATION
# =====================================================================

print("\n" + "=" * 70)
print("INPUT VALIDATION")
print("=" * 70)

required_files = {
    "molecular_features": FEATURE_FILE,
    "held_out_predictions": PREDICTION_FILE,
}

for name, path in required_files.items():

    if path.exists():
        print(f"[OK] {name}: {path}")

    else:
        raise FileNotFoundError(
            f"Required input missing: {name}: {path}"
        )


# =====================================================================
# LOAD FEATURES
# =====================================================================

print("\n" + "=" * 70)
print("LOADING MOLECULAR FEATURES")
print("=" * 70)

features = pd.read_csv(FEATURE_FILE)

print(f"Feature shape: {features.shape}")

missing_features = [
    col for col in FEATURES
    if col not in features.columns
]

if missing_features:

    raise ValueError(
        "Required molecular descriptors are missing: "
        + ", ".join(missing_features)
    )

if "ID" not in features.columns:

    raise ValueError(
        "molecular_features.csv must contain ID."
    )


# =====================================================================
# POPULATION C
# =====================================================================

population_c = features[
    ["ID"] + FEATURES
].copy()

population_c = population_c.dropna(
    subset=FEATURES
).drop_duplicates(
    subset="ID"
)

print(
    f"Population C usable descriptor rows: "
    f"{len(population_c):,}"
)


# =====================================================================
# LOAD REPEATED HELD-OUT PREDICTIONS
# =====================================================================

print("\n" + "=" * 70)
print("LOADING REPEATED HELD-OUT PREDICTIONS")
print("=" * 70)

predictions = pd.read_csv(PREDICTION_FILE)

print(
    f"Prediction rows: {len(predictions):,}"
)

required_prediction_columns = [
    "ID",
    "repetition",
    "seed",
    "Solubility",
    "predicted_solubility",
    "absolute_error",
]

missing_prediction_columns = [
    col
    for col in required_prediction_columns
    if col not in predictions.columns
]

if missing_prediction_columns:

    raise ValueError(
        "Required prediction columns are missing: "
        + ", ".join(missing_prediction_columns)
    )


# =====================================================================
# JOIN DESCRIPTORS
# =====================================================================

print("\n" + "=" * 70)
print("JOINING POPULATION C DESCRIPTORS TO HELD-OUT PREDICTIONS")
print("=" * 70)

# The held-out prediction file already contains descriptor columns.
# To avoid duplicate/suffixed descriptor fields and to ensure that
# applicability-domain calculations use the authoritative Population C
# descriptor table, retain only ID + the six required descriptors here.

population_c_descriptor_lookup = population_c[
    ["ID"] + FEATURES
].copy()

# Confirm that Population C contains one descriptor record per compound.
if population_c_descriptor_lookup["ID"].duplicated().any():

    duplicate_ids = (
        population_c_descriptor_lookup.loc[
            population_c_descriptor_lookup["ID"].duplicated(),
            "ID"
        ]
        .nunique()
    )

    raise ValueError(
        f"Population C descriptor table contains "
        f"{duplicate_ids:,} duplicated compound IDs."
    )

# Remove the descriptor columns already present in the prediction file
# before joining the authoritative Population C values.
prediction_base_columns = [
    col
    for col in predictions.columns
    if col not in FEATURES
]

predictions = predictions[
    prediction_base_columns
].merge(
    population_c_descriptor_lookup,
    on="ID",
    how="left",
    validate="many_to_one",
)

missing_join = predictions[FEATURES].isna().any(axis=1).sum()

if missing_join:

    raise ValueError(
        f"{missing_join:,} prediction rows could not be matched "
        "to Population C molecular descriptors."
    )

print(
    f"Validated prediction rows: "
    f"{len(predictions):,}"
)

print(
    f"Unique compounds: "
    f"{predictions['ID'].nunique():,}"
)

print(
    "Population C descriptors successfully attached "
    "without duplicate descriptor columns."
)

# =====================================================================
# TRAINING-REFERENCE RECONSTRUCTION
# =====================================================================

print("\n" + "=" * 70)
print("CALCULATING WITHIN-REPETITION CHEMICAL-SPACE DISTANCE")
print("=" * 70)

"""
For each repetition:

1. Identify compounds appearing in that repetition's held-out
   prediction rows.
2. Treat all other Population C compounds as the training reference.
3. Standardize descriptors using ONLY the training reference.
4. Calculate the nearest-neighbour Euclidean distance from each
   held-out compound to the training chemical space.

This produces a chemically interpretable novelty measure:

distance = distance to the closest training compound in standardized
descriptor space.

Lower distance = better represented chemical space.
Higher distance = greater descriptor-space novelty.
"""

novelty_rows = []

repetitions = (
    predictions[
        ["repetition", "seed"]
    ]
    .drop_duplicates()
    .sort_values(
        ["repetition", "seed"]
    )
)

print(
    f"Repeated validation sets detected: "
    f"{len(repetitions):,}"
)


for repetition, seed in repetitions.itertuples(index=False):

    print(
        f"Processing repetition={repetition}, seed={seed}"
    )

    rep_mask = (
        (predictions["repetition"] == repetition)
        & (predictions["seed"] == seed)
    )

    test_ids = set(
        predictions.loc[
            rep_mask,
            "ID"
        ]
    )

    training_reference = population_c[
        ~population_c["ID"].isin(test_ids)
    ].copy()

    held_out = population_c[
        population_c["ID"].isin(test_ids)
    ].copy()

    if len(held_out) == 0:

        raise ValueError(
            f"No held-out compounds found for repetition {repetition}."
        )

    if len(training_reference) == 0:

        raise ValueError(
            f"No training reference compounds found for "
            f"repetition {repetition}."
        )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        training_reference[FEATURES]
    )

    X_test = scaler.transform(
        held_out[FEATURES]
    )

    distances = pairwise_distances(
        X_test,
        X_train,
        metric="euclidean",
    )

    nearest_distance = distances.min(
        axis=1
    )

    nearest_position = distances.argmin(
        axis=1
    )

    nearest_ids = training_reference.iloc[
        nearest_position
    ]["ID"].to_numpy()

    repetition_output = held_out[
        ["ID"]
    ].copy()

    repetition_output[
        "repetition"
    ] = repetition

    repetition_output[
        "seed"
    ] = seed

    repetition_output[
        "training_reference_count"
    ] = len(training_reference)

    repetition_output[
        "nearest_training_distance"
    ] = nearest_distance

    repetition_output[
        "nearest_training_ID"
    ] = nearest_ids

    novelty_rows.append(
        repetition_output
    )


novelty_df = pd.concat(
    novelty_rows,
    ignore_index=True,
)


# =====================================================================
# MERGE ERROR INFORMATION
# =====================================================================

print("\n" + "=" * 70)
print("LINKING NOVELTY TO PREDICTION ERROR")
print("=" * 70)

prediction_error = (
    predictions[
        [
            "ID",
            "repetition",
            "seed",
            "Solubility",
            "predicted_solubility",
            "absolute_error",
        ]
    ]
    .drop_duplicates(
        subset=[
            "ID",
            "repetition",
            "seed",
        ]
    )
)

novelty_df = novelty_df.merge(
    prediction_error,
    on=[
        "ID",
        "repetition",
        "seed",
    ],
    how="left",
    validate="one_to_one",
)

if novelty_df["absolute_error"].isna().any():

    raise ValueError(
        "Some novelty observations could not be matched "
        "to prediction error."
    )


# =====================================================================
# NOVELTY STRATA
# =====================================================================

print("\n" + "=" * 70)
print("CREATING CHEMICAL-SPACE NOVELTY STRATA")
print("=" * 70)

"""
Strata are defined globally across the resulting compound-level
observations using quartiles.

The strata are descriptive rather than externally defined
applicability-domain thresholds.
"""

novelty_df[
    "novelty_region"
] = pd.qcut(
    novelty_df["nearest_training_distance"],
    q=4,
    labels=[
        "Lowest novelty",
        "Low-moderate novelty",
        "Moderate-high novelty",
        "Highest novelty",
    ],
    duplicates="drop",
)


# =====================================================================
# NOVELTY SUMMARY
# =====================================================================

strata_summary = (
    novelty_df
    .groupby(
        "novelty_region",
        observed=False
    )
    .agg(
        compound_observations=(
            "ID",
            "count",
        ),
        unique_compounds=(
            "ID",
            "nunique",
        ),
        mean_nearest_training_distance=(
            "nearest_training_distance",
            "mean",
        ),
        median_nearest_training_distance=(
            "nearest_training_distance",
            "median",
        ),
        mean_absolute_error=(
            "absolute_error",
            "mean",
        ),
        median_absolute_error=(
            "absolute_error",
            "median",
        ),
        rmse=(
            "absolute_error",
            lambda x: np.sqrt(
                np.mean(
                    np.square(x)
                )
            ),
        ),
    )
    .reset_index()
)


print(strata_summary.to_string(index=False))


# =====================================================================
# SCAFFOLD REPRESENTATION
# =====================================================================

print("\n" + "=" * 70)
print("SCAFFOLD REPRESENTATION")
print("=" * 70)

if "scaffold" in predictions.columns:

    scaffold_data = (
        predictions[
            [
                "ID",
                "repetition",
                "seed",
                "scaffold",
                "absolute_error",
            ]
        ]
        .drop_duplicates(
            subset=[
                "ID",
                "repetition",
                "seed",
            ]
        )
    )

    scaffold_summary = (
        scaffold_data
        .groupby("scaffold")
        .agg(
            compound_observations=(
                "ID",
                "count",
            ),
            unique_compounds=(
                "ID",
                "nunique",
            ),
            mean_absolute_error=(
                "absolute_error",
                "mean",
            ),
            median_absolute_error=(
                "absolute_error",
                "median",
            ),
        )
        .reset_index()
    )

    scaffold_summary[
        "scaffold_frequency_region"
    ] = pd.cut(
        scaffold_summary[
            "unique_compounds"
        ],
        bins=[
            -np.inf,
            1,
            2,
            5,
            np.inf,
        ],
        labels=[
            "Singleton",
            "2 compounds",
            "3–5 compounds",
            "6+ compounds",
        ],
    )

    scaffold_region_summary = (
        scaffold_summary
        .groupby(
            "scaffold_frequency_region",
            observed=False,
        )
        .agg(
            scaffold_count=(
                "scaffold",
                "count",
            ),
            compound_observations=(
                "unique_compounds",
                "sum",
            ),
            mean_scaffold_mae=(
                "mean_absolute_error",
                "mean",
            ),
            median_scaffold_mae=(
                "median_absolute_error",
                "median",
            ),
        )
        .reset_index()
    )

    print(
        scaffold_region_summary.to_string(
            index=False
        )
    )

else:

    print(
        "No scaffold column found. "
        "Scaffold representation analysis skipped."
    )

    scaffold_summary = pd.DataFrame()

    scaffold_region_summary = pd.DataFrame()


# =====================================================================
# OVERALL NOVELTY / ERROR RELATIONSHIP
# =====================================================================

print("\n" + "=" * 70)
print("NOVELTY / ERROR RELATIONSHIP")
print("=" * 70)

pearson_r = novelty_df[
    [
        "nearest_training_distance",
        "absolute_error",
    ]
].corr(
    method="pearson"
).iloc[0, 1]

spearman_r = novelty_df[
    [
        "nearest_training_distance",
        "absolute_error",
    ]
].corr(
    method="spearman"
).iloc[0, 1]

print(
    f"Pearson correlation:  {pearson_r:+.4f}"
)

print(
    f"Spearman correlation: {spearman_r:+.4f}"
)


# =====================================================================
# COMPOUND-LEVEL AGGREGATION
# =====================================================================

print("\n" + "=" * 70)
print("COMPOUND-LEVEL NOVELTY SUMMARY")
print("=" * 70)

compound_level = (
    novelty_df
    .groupby("ID")
    .agg(
        mean_nearest_training_distance=(
            "nearest_training_distance",
            "mean",
        ),
        median_nearest_training_distance=(
            "nearest_training_distance",
            "median",
        ),
        max_nearest_training_distance=(
            "nearest_training_distance",
            "max",
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
            "absolute_error",
            "count",
        ),
    )
    .reset_index()
)


compound_level[
    "novelty_region"
] = pd.qcut(
    compound_level[
        "mean_nearest_training_distance"
    ],
    q=4,
    labels=[
        "Lowest novelty",
        "Low-moderate novelty",
        "Moderate-high novelty",
        "Highest novelty",
    ],
    duplicates="drop",
)


compound_strata_summary = (
    compound_level
    .groupby(
        "novelty_region",
        observed=False,
    )
    .agg(
        compound_count=(
            "ID",
            "count",
        ),
        mean_novelty_distance=(
            "mean_nearest_training_distance",
            "mean",
        ),
        median_novelty_distance=(
            "mean_nearest_training_distance",
            "median",
        ),
        mean_absolute_error=(
            "mean_absolute_error",
            "mean",
        ),
        median_absolute_error=(
            "median_absolute_error",
            "median",
        ),
        max_novelty_distance=(
            "max_nearest_training_distance",
            "mean",
        ),
    )
    .reset_index()
)


print(
    compound_strata_summary.to_string(
        index=False
    )
)


# =====================================================================
# SUMMARY TABLE
# =====================================================================

summary_rows = [
    {
        "analysis": "Prediction observations",
        "value": len(novelty_df),
    },
    {
        "analysis": "Unique compounds",
        "value": novelty_df["ID"].nunique(),
    },
    {
        "analysis": "Repeated validation sets",
        "value": len(repetitions),
    },
    {
        "analysis": "Mean nearest-training distance",
        "value": novelty_df[
            "nearest_training_distance"
        ].mean(),
    },
    {
        "analysis": "Median nearest-training distance",
        "value": novelty_df[
            "nearest_training_distance"
        ].median(),
    },
    {
        "analysis": "Mean absolute error",
        "value": novelty_df[
            "absolute_error"
        ].mean(),
    },
    {
        "analysis": "Pearson novelty-error correlation",
        "value": pearson_r,
    },
    {
        "analysis": "Spearman novelty-error correlation",
        "value": spearman_r,
    },
]


summary_df = pd.DataFrame(
    summary_rows
)


# =====================================================================
# SAVE OUTPUTS
# =====================================================================

print("\n" + "=" * 70)
print("SAVING OUTPUTS")
print("=" * 70)

OUTPUT_NOVELTY_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

novelty_df.to_csv(
    OUTPUT_NOVELTY_FILE,
    index=False,
)

strata_summary.to_csv(
    OUTPUT_STRATA_FILE,
    index=False,
)

if not scaffold_region_summary.empty:

    scaffold_region_summary.to_csv(
        OUTPUT_SCAFFOLD_FILE,
        index=False,
    )

summary_df.to_csv(
    OUTPUT_SUMMARY_FILE,
    index=False,
)


# =====================================================================
# REPORT
# =====================================================================

report_lines = []

report_lines.append(
    "SCRIPT 33 — APPLICABILITY DOMAIN & "
    "CHEMICAL-SPACE RELIABILITY"
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
    "Determine whether Gradient Boosting prediction reliability "
    "varies with chemical-space representation under the repeated "
    "scaffold-aware validation framework."
)

report_lines.append("")

report_lines.append(
    "DATA DESIGN"
)

report_lines.append(
    "-----------"
)

report_lines.append(
    f"Population C descriptor rows: "
    f"{len(population_c):,}"
)

report_lines.append(
    f"Repeated held-out prediction rows: "
    f"{len(predictions):,}"
)

report_lines.append(
    f"Unique compounds represented: "
    f"{predictions['ID'].nunique():,}"
)

report_lines.append(
    f"Repeated validation sets: "
    f"{len(repetitions):,}"
)

report_lines.append("")

report_lines.append(
    "APPLICABILITY-DOMAIN METHOD"
)

report_lines.append(
    "---------------------------"
)

report_lines.append(
    "For each scaffold-aware validation repetition, the held-out "
    "compounds were compared with the corresponding training "
    "population using the six established molecular descriptors."
)

report_lines.append(
    "Descriptors were standardized using only the training "
    "reference population within each repetition."
)

report_lines.append(
    "Chemical-space novelty was defined as the Euclidean distance "
    "from a held-out compound to its nearest training compound "
    "in standardized descriptor space."
)

report_lines.append(
    "Lower nearest-training distance indicates greater descriptor-"
    "space representation; higher distance indicates greater "
    "chemical-space novelty."
)

report_lines.append(
    "Novelty quartiles are descriptive strata and are not claimed "
    "as universal applicability-domain thresholds."
)

report_lines.append("")

report_lines.append(
    "NOVELTY / ERROR RELATIONSHIP"
)

report_lines.append(
    "----------------------------"
)

report_lines.append(
    f"Pearson correlation: "
    f"{pearson_r:+.6f}"
)

report_lines.append(
    f"Spearman correlation: "
    f"{spearman_r:+.6f}"
)

report_lines.append("")

report_lines.append(
    "NOVELTY STRATA"
)

report_lines.append(
    "--------------"
)

for _, row in strata_summary.iterrows():

    report_lines.append(
        f"{row['novelty_region']} | "
        f"observations="
        f"{int(row['compound_observations'])} | "
        f"unique_compounds="
        f"{int(row['unique_compounds'])} | "
        f"mean_distance="
        f"{row['mean_nearest_training_distance']:.6f} | "
        f"median_distance="
        f"{row['median_nearest_training_distance']:.6f} | "
        f"MAE="
        f"{row['mean_absolute_error']:.6f} | "
        f"RMSE="
        f"{row['rmse']:.6f}"
    )

report_lines.append("")

report_lines.append(
    "INTERPRETATION FRAMEWORK"
)

report_lines.append(
    "------------------------"
)

report_lines.append(
    "Nearest-training distance is a descriptor-space representation "
    "diagnostic, not a direct measure of molecular similarity."
)

report_lines.append(
    "A relationship between chemical-space novelty and prediction "
    "error would indicate that model reliability varies across the "
    "evaluated chemical space."
)

report_lines.append(
    "Absence of a strong novelty-error relationship would not prove "
    "uniform reliability because the six descriptors provide only "
    "a low-dimensional representation of molecular structure."
)

report_lines.append(
    "Scaffold-aware validation remains the primary protection "
    "against overly optimistic performance caused by structural "
    "overlap between training and test compounds."
)

report_lines.append(
    "Applicability-domain diagnostics do not establish causal "
    "relationships between chemical novelty and prediction error."
)

report_lines.append("")

report_lines.append(
    "OUTPUTS"
)

report_lines.append(
    "-------"
)

report_lines.append(
    str(OUTPUT_NOVELTY_FILE)
)

report_lines.append(
    str(OUTPUT_STRATA_FILE)
)

if not scaffold_region_summary.empty:

    report_lines.append(
        str(OUTPUT_SCAFFOLD_FILE)
    )

report_lines.append(
    str(OUTPUT_SUMMARY_FILE)
)

report_lines.append(
    str(REPORT_FILE)
)


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "\n".join(report_lines)
    )


# =====================================================================
# COMPLETION
# =====================================================================

print("\n" + "=" * 70)
print("SCRIPT 33 COMPLETE")
print("=" * 70)

print(
    f"Population C: {len(population_c):,}"
)

print(
    f"Prediction observations: {len(novelty_df):,}"
)

print(
    f"Unique compounds: "
    f"{novelty_df['ID'].nunique():,}"
)

print(
    f"Pearson novelty-error correlation: "
    f"{pearson_r:+.4f}"
)

print(
    f"Spearman novelty-error correlation: "
    f"{spearman_r:+.4f}"
)

print("\nOutputs:")

print(
    OUTPUT_NOVELTY_FILE
)

print(
    OUTPUT_STRATA_FILE
)

if not scaffold_region_summary.empty:

    print(
        OUTPUT_SCAFFOLD_FILE
    )

print(
    OUTPUT_SUMMARY_FILE
)

print(
    REPORT_FILE
)