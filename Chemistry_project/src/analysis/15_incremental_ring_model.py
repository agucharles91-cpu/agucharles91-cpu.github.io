"""
15_incremental_ring_model.py

Test whether ring-related structural descriptors add explanatory
power beyond the canonical MolWt + MolLogP baseline.

Canonical analytical population:
    standard_analytical_domain == True

Models:

M0: Solubility ~ rdkit_molwt + rdkit_mollogp
M1: Solubility ~ rdkit_molwt + rdkit_mollogp + rdkit_ring_count
M2: Solubility ~ rdkit_molwt + rdkit_mollogp + rdkit_aromatic_rings
M3: Solubility ~ rdkit_molwt + rdkit_mollogp
                     + rdkit_ring_count
                     + rdkit_aromatic_rings

This is an explanatory/incremental analysis, not a held-out
generalization evaluation.

Ring count is not equivalent to fused-ring count.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


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

OUTPUT_DATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "incremental_ring_models.csv"
)

VIF_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "incremental_ring_vif.csv"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "incremental_ring_model.txt"
)


# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------

print("=" * 70)
print("SCRIPT 15 — INCREMENTAL RING MODEL")
print("=" * 70)

print("\nLoading raw dataset:")
print(RAW_FILE)

raw = pd.read_csv(RAW_FILE)

print(f"Raw dataset shape: {raw.shape}")

print("\nLoading molecular features:")
print(FEATURE_FILE)

features = pd.read_csv(FEATURE_FILE)

print(f"Molecular feature shape: {features.shape}")


# ---------------------------------------------------------------------
# VALIDATE REQUIRED COLUMNS
# ---------------------------------------------------------------------

required_raw = [
    "ID",
    "Solubility"
]

required_features = [
    "ID",
    "standard_analytical_domain",
    "rdkit_molwt",
    "rdkit_mollogp",
    "rdkit_ring_count",
    "rdkit_aromatic_rings"
]

missing_raw = [
    col for col in required_raw
    if col not in raw.columns
]

missing_features = [
    col for col in required_features
    if col not in features.columns
]

if missing_raw:
    raise ValueError(
        "Missing required raw columns: "
        + ", ".join(missing_raw)
    )

if missing_features:
    raise ValueError(
        "Missing required feature columns: "
        + ", ".join(missing_features)
    )


# ---------------------------------------------------------------------
# VALIDATE IDS
# ---------------------------------------------------------------------

if raw["ID"].duplicated().any():
    raise ValueError(
        "Duplicate IDs detected in raw dataset."
    )

if features["ID"].duplicated().any():
    raise ValueError(
        "Duplicate IDs detected in molecular_features.csv."
    )


# ---------------------------------------------------------------------
# USE THE LOCKED ANALYTICAL-DOMAIN FLAG
# ---------------------------------------------------------------------

print("\nUsing established standard_analytical_domain flag.")

population_features = features[
    features["standard_analytical_domain"] == True
].copy()

print(
    f"Rows flagged as standard analytical domain: "
    f"{len(population_features):,}"
)

if len(population_features) != 8643:
    raise ValueError(
        "Locked Population C verification failed. "
        f"Expected 8,643 rows, found "
        f"{len(population_features):,}."
    )

print("Population C verified: 8,643 rows")


# ---------------------------------------------------------------------
# MERGE SOLUBILITY FROM RAW DATA
# ---------------------------------------------------------------------

population = population_features[
    [
        "ID",
        "rdkit_molwt",
        "rdkit_mollogp",
        "rdkit_ring_count",
        "rdkit_aromatic_rings"
    ]
].merge(
    raw[
        [
            "ID",
            "Solubility"
        ]
    ],
    on="ID",
    how="inner",
    validate="one_to_one"
)

print(
    f"Rows after solubility merge: {len(population):,}"
)

if len(population) != 8643:
    raise ValueError(
        "Population C merge failed. "
        f"Expected 8,643 rows, found {len(population):,}."
    )


# ---------------------------------------------------------------------
# PREPARE VARIABLES
# ---------------------------------------------------------------------

numeric_columns = [
    "Solubility",
    "rdkit_molwt",
    "rdkit_mollogp",
    "rdkit_ring_count",
    "rdkit_aromatic_rings"
]

for column in numeric_columns:

    population[column] = pd.to_numeric(
        population[column],
        errors="coerce"
    )

if population[numeric_columns].isna().any().any():

    missing_counts = (
        population[numeric_columns]
        .isna()
        .sum()
    )

    raise ValueError(
        "Missing values detected:\n"
        + missing_counts.to_string()
    )


# ---------------------------------------------------------------------
# CORRELATION BETWEEN RING VARIABLES
# ---------------------------------------------------------------------

print("\n" + "-" * 70)
print("CORRELATION MATRIX")
print("-" * 70)

correlation_matrix = population[
    [
        "rdkit_molwt",
        "rdkit_mollogp",
        "rdkit_ring_count",
        "rdkit_aromatic_rings"
    ]
].corr()

print(
    correlation_matrix.to_string()
)


# ---------------------------------------------------------------------
# MODEL DEFINITIONS
# ---------------------------------------------------------------------

models = {

    "M0_baseline": [
        "rdkit_molwt",
        "rdkit_mollogp"
    ],

    "M1_baseline_plus_ring_count": [
        "rdkit_molwt",
        "rdkit_mollogp",
        "rdkit_ring_count"
    ],

    "M2_baseline_plus_aromatic_rings": [
        "rdkit_molwt",
        "rdkit_mollogp",
        "rdkit_aromatic_rings"
    ],

    "M3_baseline_plus_both_ring_features": [
        "rdkit_molwt",
        "rdkit_mollogp",
        "rdkit_ring_count",
        "rdkit_aromatic_rings"
    ]
}


# ---------------------------------------------------------------------
# FIT MODELS
# ---------------------------------------------------------------------

y = population["Solubility"]

fitted_models = {}
model_rows = []

for model_name, predictors in models.items():

    X = sm.add_constant(
        population[predictors]
    )

    model = sm.OLS(
        y,
        X
    ).fit(
        cov_type="HC3"
    )

    fitted_models[model_name] = model

    predictions = model.predict(X)

    residuals = y - predictions

    rmse = np.sqrt(
        np.mean(residuals ** 2)
    )

    model_rows.append({
        "model": model_name,
        "n": len(population),
        "predictors": ", ".join(predictors),
        "r_squared": model.rsquared,
        "adjusted_r_squared": model.rsquared_adj,
        "rmse": rmse,
        "aic": model.aic,
        "bic": model.bic
    })


results_df = pd.DataFrame(
    model_rows
)


# ---------------------------------------------------------------------
# INCREMENTAL R-SQUARED
# ---------------------------------------------------------------------

baseline_r2 = results_df.loc[
    results_df["model"] == "M0_baseline",
    "r_squared"
].iloc[0]

results_df["delta_r_squared_vs_baseline"] = (
    results_df["r_squared"] - baseline_r2
)


# ---------------------------------------------------------------------
# PRINT MODEL PERFORMANCE
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(
    results_df[
        [
            "model",
            "r_squared",
            "adjusted_r_squared",
            "rmse",
            "aic",
            "bic",
            "delta_r_squared_vs_baseline"
        ]
    ].to_string(index=False)
)


# ---------------------------------------------------------------------
# COEFFICIENT RESULTS
# ---------------------------------------------------------------------

coefficient_rows = []

for model_name, model in fitted_models.items():

    for predictor in model.params.index:

        if predictor == "const":
            continue

        ci = model.conf_int().loc[predictor]

        coefficient_rows.append({
            "model": model_name,
            "predictor": predictor,
            "coefficient": model.params[predictor],
            "robust_se": model.bse[predictor],
            "p_value": model.pvalues[predictor],
            "ci_low": ci.iloc[0],
            "ci_high": ci.iloc[1]
        })


coefficients_df = pd.DataFrame(
    coefficient_rows
)


# ---------------------------------------------------------------------
# PRINT RING COEFFICIENTS
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("RING-RELATED COEFFICIENTS")
print("=" * 70)

ring_coefficients = coefficients_df[
    coefficients_df["predictor"].isin(
        [
            "rdkit_ring_count",
            "rdkit_aromatic_rings"
        ]
    )
]

print(
    ring_coefficients.to_string(
        index=False
    )
)


# ---------------------------------------------------------------------
# NESTED MODEL COMPARISONS
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("NESTED MODEL COMPARISONS")
print("=" * 70)

comparisons = [

    (
        "M0_baseline",
        "M1_baseline_plus_ring_count"
    ),

    (
        "M0_baseline",
        "M2_baseline_plus_aromatic_rings"
    ),

    (
        "M0_baseline",
        "M3_baseline_plus_both_ring_features"
    )
]

nested_rows = []

for reduced_name, full_name in comparisons:

    reduced_predictors = models[reduced_name]
    full_predictors = models[full_name]

    X_reduced = sm.add_constant(
        population[reduced_predictors]
    )

    X_full = sm.add_constant(
        population[full_predictors]
    )

    # Separate conventional OLS fits are used here because
    # compare_f_test() is not valid with HC3 robust covariance.
    reduced_classical = sm.OLS(
        y,
        X_reduced
    ).fit()

    full_classical = sm.OLS(
        y,
        X_full
    ).fit()

    f_stat, p_value, df_diff = (
        full_classical.compare_f_test(
            reduced_classical
        )
    )

    nested_rows.append({
        "reduced_model": reduced_name,
        "full_model": full_name,
        "f_statistic": f_stat,
        "p_value": p_value,
        "df_difference": df_diff
    })

    print(
        f"{reduced_name} -> {full_name}: "
        f"F = {f_stat:.4f}, "
        f"p = {p_value:.6g}, "
        f"df = {df_diff}"
    )


nested_df = pd.DataFrame(
    nested_rows
)


# ---------------------------------------------------------------------
# VIF
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("VIF — FULL RING MODEL")
print("=" * 70)

vif_predictors = [
    "rdkit_molwt",
    "rdkit_mollogp",
    "rdkit_ring_count",
    "rdkit_aromatic_rings"
]

X_vif = population[
    vif_predictors
].copy()

vif_rows = []

for i, column in enumerate(
    X_vif.columns
):

    vif_rows.append({
        "predictor": column,
        "VIF": variance_inflation_factor(
            X_vif.values,
            i
        )
    })

vif_df = pd.DataFrame(
    vif_rows
)

print(
    vif_df.to_string(index=False)
)


# ---------------------------------------------------------------------
# SAVE OUTPUTS
# ---------------------------------------------------------------------

results_df.to_csv(
    OUTPUT_DATA,
    index=False
)

vif_df.to_csv(
    VIF_OUTPUT,
    index=False
)


# ---------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------

report_lines = []

report_lines.append(
    "SCRIPT 15 — INCREMENTAL RING MODEL"
)

report_lines.append("=" * 70)
report_lines.append("")

report_lines.append(
    "OBJECTIVE"
)

report_lines.append(
    "Test whether ring-related descriptors add explanatory "
    "power beyond the canonical MolWt + MolLogP baseline."
)

report_lines.append("")

report_lines.append(
    "POPULATION"
)

report_lines.append(
    "Population C = 8,643 compounds."
)

report_lines.append(
    "Population C was taken directly from the established "
    "standard_analytical_domain flag in molecular_features.csv."
)

report_lines.append("")

report_lines.append(
    "MODELS"
)

report_lines.append(
    "M0: Solubility ~ rdkit_molwt + rdkit_mollogp"
)

report_lines.append(
    "M1: M0 + rdkit_ring_count"
)

report_lines.append(
    "M2: M0 + rdkit_aromatic_rings"
)

report_lines.append(
    "M3: M0 + rdkit_ring_count + rdkit_aromatic_rings"
)

report_lines.append("")

report_lines.append(
    "MODEL PERFORMANCE"
)

report_lines.append(
    results_df.to_string(index=False)
)

report_lines.append("")

report_lines.append(
    "RING-RELATED COEFFICIENTS"
)

report_lines.append(
    ring_coefficients.to_string(index=False)
)

report_lines.append("")

report_lines.append(
    "NESTED MODEL TESTS"
)

report_lines.append(
    "Nested F-tests use conventional OLS covariance because "
    "statsmodels compare_f_test() is not valid with HC3 robust covariance. "
    "HC3 robust inference is retained for individual model coefficients."
)

report_lines.append(
    nested_df.to_string(index=False)
)

report_lines.append("")

report_lines.append(
    "VIF"
)

report_lines.append(
    vif_df.to_string(index=False)
)

report_lines.append("")

report_lines.append(
    "INTERPRETATION CAUTION"
)

report_lines.append(
    "This is an explanatory analysis rather than a held-out "
    "prediction evaluation."
)

report_lines.append(
    "Ring count is not equivalent to fused-ring count."
)

report_lines.append(
    "Ring count is not a direct measurement of crystal packing, "
    "lattice energy or melting point."
)

report_lines.append(
    "A statistically significant ring coefficient does not by "
    "itself establish a causal ring effect."
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

print("\nModel results:")
print(OUTPUT_DATA)

print("\nVIF results:")
print(VIF_OUTPUT)

print("\nReport:")
print(REPORT_FILE)

print("\n" + "=" * 70)
print("SCRIPT 15 COMPLETE")
print("=" * 70)