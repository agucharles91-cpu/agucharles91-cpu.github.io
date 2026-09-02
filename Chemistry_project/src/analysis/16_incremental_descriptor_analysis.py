"""
16_incremental_descriptor_analysis.py

Incremental chemical descriptor analysis for the AqSolDB
aqueous-solubility project.

Objective
---------
Determine which chemically interpretable molecular descriptors add
explanatory power beyond the established:

    MolWt + MolLogP + RingCount + AromaticRings

baseline.

Candidate descriptors are tested individually first to quantify their
incremental contribution. A selective multivariable model is then
constructed from descriptors that provide useful additional information.

Important methodological choices
---------------------------------
1. Population C is taken directly from the established
   standard_analytical_domain flag.

2. The baseline is the canonical Script 15 M3 model:

       Solubility ~ MolWt + MolLogP
                  + RingCount + AromaticRings

3. Highly redundant size descriptors such as HeavyAtomCount, MolMR and
   LabuteASA are deliberately excluded from the first incremental screen
   because Script 07 established their strong correlation with MolWt.

4. Individual model coefficients use HC3 robust standard errors.

5. Nested-model F-tests use conventional OLS covariance because
   statsmodels compare_f_test() is not valid with HC3 covariance.

6. This remains an explanatory analysis. No held-out prediction
   performance is claimed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor


# =====================================================================
# PATHS
# =====================================================================

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
    / "incremental_descriptor_models.csv"
)

COEFFICIENT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "incremental_descriptor_coefficients.csv"
)

VIF_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "incremental_descriptor_vif.csv"
)

CORRELATION_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "incremental_descriptor_correlations.csv"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "incremental_descriptor_analysis.txt"
)


# =====================================================================
# HEADER
# =====================================================================

print("=" * 70)
print("SCRIPT 16 — INCREMENTAL CHEMICAL DESCRIPTOR ANALYSIS")
print("=" * 70)


# =====================================================================
# LOAD DATA
# =====================================================================

print("\nLoading raw dataset:")
print(RAW_FILE)

raw = pd.read_csv(RAW_FILE)

print(f"Raw dataset shape: {raw.shape}")

print("\nLoading molecular features:")
print(FEATURE_FILE)

features = pd.read_csv(FEATURE_FILE)

print(f"Molecular feature shape: {features.shape}")


# =====================================================================
# REQUIRED COLUMNS
# =====================================================================

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
    "rdkit_aromatic_rings",
    "rdkit_tpsa",
    "rdkit_hbd",
    "rdkit_hba",
    "rdkit_rotatable_bonds",
    "rdkit_fraction_csp3",
    "rdkit_bertz_ct"
]

missing_raw = [
    col
    for col in required_raw
    if col not in raw.columns
]

missing_features = [
    col
    for col in required_features
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


# =====================================================================
# VALIDATE IDS
# =====================================================================

if raw["ID"].duplicated().any():
    raise ValueError(
        "Duplicate IDs detected in raw dataset."
    )

if features["ID"].duplicated().any():
    raise ValueError(
        "Duplicate IDs detected in molecular_features.csv."
    )


# =====================================================================
# ESTABLISH POPULATION C
# =====================================================================

print("\n" + "-" * 70)
print("POPULATION C")
print("-" * 70)

population_features = features[
    features["standard_analytical_domain"] == True
].copy()

print(
    "Rows flagged as standard analytical domain: "
    f"{len(population_features):,}"
)

if len(population_features) != 8643:
    raise ValueError(
        "Population C verification failed. "
        f"Expected 8,643 rows, found "
        f"{len(population_features):,}."
    )

population = population_features[
    [
        "ID",
        "rdkit_molwt",
        "rdkit_mollogp",
        "rdkit_ring_count",
        "rdkit_aromatic_rings",
        "rdkit_tpsa",
        "rdkit_hbd",
        "rdkit_hba",
        "rdkit_rotatable_bonds",
        "rdkit_fraction_csp3",
        "rdkit_bertz_ct"
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
        "Population C merge failed."
    )


# =====================================================================
# NUMERIC VALIDATION
# =====================================================================

predictor_columns = [
    "rdkit_molwt",
    "rdkit_mollogp",
    "rdkit_ring_count",
    "rdkit_aromatic_rings",
    "rdkit_tpsa",
    "rdkit_hbd",
    "rdkit_hba",
    "rdkit_rotatable_bonds",
    "rdkit_fraction_csp3",
    "rdkit_bertz_ct"
]

for column in ["Solubility"] + predictor_columns:

    population[column] = pd.to_numeric(
        population[column],
        errors="coerce"
    )

missing_values = (
    population[
        ["Solubility"] + predictor_columns
    ]
    .isna()
    .sum()
)

if missing_values.any():

    print("\nMissing values detected:")

    print(
        missing_values[
            missing_values > 0
        ].to_string()
    )

    raise ValueError(
        "Missing values present in analytical dataset."
    )


# =====================================================================
# DESCRIPTOR DEFINITIONS
# =====================================================================

baseline_predictors = [
    "rdkit_molwt",
    "rdkit_mollogp",
    "rdkit_ring_count",
    "rdkit_aromatic_rings"
]

candidate_descriptors = [
    "rdkit_tpsa",
    "rdkit_hbd",
    "rdkit_hba",
    "rdkit_rotatable_bonds",
    "rdkit_fraction_csp3",
    "rdkit_bertz_ct"
]


# =====================================================================
# DESCRIPTOR CORRELATIONS
# =====================================================================

print("\n" + "=" * 70)
print("CANDIDATE DESCRIPTOR CORRELATIONS")
print("=" * 70)

correlation_rows = []

for descriptor in candidate_descriptors:

    for baseline_descriptor in baseline_predictors:

        rho, p_value = spearmanr(
            population[descriptor],
            population[baseline_descriptor]
        )

        correlation_rows.append({
            "candidate_descriptor": descriptor,
            "baseline_descriptor": baseline_descriptor,
            "spearman_rho": rho,
            "p_value": p_value
        })

correlation_df = pd.DataFrame(
    correlation_rows
)

print(
    correlation_df.to_string(index=False)
)

correlation_df.to_csv(
    CORRELATION_OUTPUT,
    index=False
)


# =====================================================================
# MODEL FITTING FUNCTION
# =====================================================================

y = population["Solubility"]


def fit_model(
    predictors,
    robust=True
):
    """
    Fit OLS model.

    HC3 robust covariance is used for coefficient inference when
    robust=True.
    """

    X = sm.add_constant(
        population[predictors]
    )

    if robust:

        model = sm.OLS(
            y,
            X
        ).fit(
            cov_type="HC3"
        )

    else:

        model = sm.OLS(
            y,
            X
        ).fit()

    predictions = model.predict(X)

    residuals = y - predictions

    rmse = np.sqrt(
        np.mean(residuals ** 2)
    )

    return model, rmse


# =====================================================================
# BASELINE MODEL
# =====================================================================

print("\n" + "=" * 70)
print("BASELINE MODEL")
print("=" * 70)

baseline_model, baseline_rmse = fit_model(
    baseline_predictors,
    robust=True
)

baseline_r2 = baseline_model.rsquared

print(
    "Model: Solubility ~ "
    + " + ".join(baseline_predictors)
)

print(
    f"R²   = {baseline_r2:.6f}"
)

print(
    f"RMSE = {baseline_rmse:.6f}"
)


# =====================================================================
# INDIVIDUAL INCREMENTAL MODELS
# =====================================================================

print("\n" + "=" * 70)
print("INDIVIDUAL INCREMENTAL DESCRIPTOR MODELS")
print("=" * 70)

model_rows = []

fitted_models = {
    "M3_ring_baseline": baseline_model
}

for descriptor in candidate_descriptors:

    predictors = (
        baseline_predictors
        + [descriptor]
    )

    model_name = (
        "M3_plus_"
        + descriptor.replace(
            "rdkit_",
            ""
        )
    )

    model, rmse = fit_model(
        predictors,
        robust=True
    )

    fitted_models[model_name] = model

    delta_r2 = (
        model.rsquared
        - baseline_r2
    )

    rmse_change = (
        baseline_rmse
        - rmse
    )

    model_rows.append({
        "model": model_name,
        "added_descriptor": descriptor,
        "n": len(population),
        "r_squared": model.rsquared,
        "adjusted_r_squared": model.rsquared_adj,
        "rmse": rmse,
        "delta_r_squared_vs_M3": delta_r2,
        "rmse_reduction_vs_M3": rmse_change,
        "aic": model.aic,
        "bic": model.bic
    })


incremental_df = pd.DataFrame(
    model_rows
)

incremental_df = incremental_df.sort_values(
    "delta_r_squared_vs_M3",
    ascending=False
)

print(
    incremental_df.to_string(
        index=False
    )
)


# =====================================================================
# ROBUST COEFFICIENTS
# =====================================================================

print("\n" + "=" * 70)
print("CANDIDATE DESCRIPTOR COEFFICIENTS")
print("=" * 70)

coefficient_rows = []

for model_name, model in fitted_models.items():

    for predictor in model.params.index:

        if predictor == "const":
            continue

        ci = model.conf_int().loc[
            predictor
        ]

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

candidate_coefficients = coefficients_df[
    coefficients_df["predictor"].isin(
        candidate_descriptors
    )
]

print(
    candidate_coefficients.to_string(
        index=False
    )
)

candidate_coefficients.to_csv(
    COEFFICIENT_OUTPUT,
    index=False
)


# =====================================================================
# NESTED F-TESTS
# =====================================================================

print("\n" + "=" * 70)
print("NESTED MODEL TESTS")
print("=" * 70)

nested_rows = []

for descriptor in candidate_descriptors:

    full_predictors = (
        baseline_predictors
        + [descriptor]
    )

    X_reduced = sm.add_constant(
        population[baseline_predictors]
    )

    X_full = sm.add_constant(
        population[full_predictors]
    )

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
        "added_descriptor": descriptor,
        "f_statistic": f_stat,
        "p_value": p_value,
        "df_difference": df_diff
    })

    print(
        f"{descriptor}: "
        f"F = {f_stat:.4f}, "
        f"p = {p_value:.6g}, "
        f"df = {df_diff}"
    )

nested_df = pd.DataFrame(
    nested_rows
)


# =====================================================================
# SELECTIVE MULTIVARIABLE MODEL
# =====================================================================

print("\n" + "=" * 70)
print("SELECTIVE MULTIVARIABLE MODEL")
print("=" * 70)

"""
Selection rule:

A descriptor enters the selective model if:

1. It produces a meaningful incremental R² improvement relative
   to the established ring baseline.

2. Its robust coefficient is statistically distinguishable from zero.

3. It is chemically interpretable.

No arbitrary p-value-only forward selection is used.

The threshold below is deliberately modest because the objective is
screening rather than automated model selection.

A descriptor must add at least 0.002 in R².
"""

selected_descriptors = []

for descriptor in candidate_descriptors:

    row = incremental_df[
        incremental_df["added_descriptor"]
        == descriptor
    ].iloc[0]

    coefficient_row = candidate_coefficients[
        (
            candidate_coefficients["model"]
            == "M3_plus_"
            + descriptor.replace(
                "rdkit_",
                ""
            )
        )
        &
        (
            candidate_coefficients["predictor"]
            == descriptor
        )
    ]

    if coefficient_row.empty:
        continue

    p_value = coefficient_row[
        "p_value"
    ].iloc[0]

    delta_r2 = row[
        "delta_r_squared_vs_M3"
    ]

    if (
        delta_r2 >= 0.002
        and p_value < 0.001
    ):

        selected_descriptors.append(
            descriptor
        )


print(
    "Selected candidate descriptors:"
)

if selected_descriptors:

    for descriptor in selected_descriptors:

        print(
            f"  - {descriptor}"
        )

else:

    print(
        "  None met the predefined screening criteria."
    )


selective_predictors = (
    baseline_predictors
    + selected_descriptors
)

selective_model, selective_rmse = fit_model(
    selective_predictors,
    robust=True
)

selective_r2 = selective_model.rsquared

print("\nSelective model predictors:")

for predictor in selective_predictors:

    print(
        f"  - {predictor}"
    )

print(
    f"\nSelective model R²   = {selective_r2:.6f}"
)

print(
    f"Selective model RMSE = {selective_rmse:.6f}"
)

print(
    f"ΔR² vs ring baseline = "
    f"{selective_r2 - baseline_r2:.6f}"
)

print(
    f"RMSE reduction       = "
    f"{baseline_rmse - selective_rmse:.6f}"
)


# =====================================================================
# SELECTIVE MODEL COEFFICIENTS
# =====================================================================

selective_coefficient_rows = []

for predictor in selective_model.params.index:

    if predictor == "const":
        continue

    ci = selective_model.conf_int().loc[
        predictor
    ]

    selective_coefficient_rows.append({
        "predictor": predictor,
        "coefficient": selective_model.params[predictor],
        "robust_se": selective_model.bse[predictor],
        "p_value": selective_model.pvalues[predictor],
        "ci_low": ci.iloc[0],
        "ci_high": ci.iloc[1]
    })

selective_coefficients_df = pd.DataFrame(
    selective_coefficient_rows
)


print("\nSelective model coefficients:")

print(
    selective_coefficients_df.to_string(
        index=False
    )
)


# =====================================================================
# VIF
# =====================================================================

print("\n" + "=" * 70)
print("VIF — SELECTIVE MODEL")
print("=" * 70)

X_vif = population[
    selective_predictors
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
    vif_df.to_string(
        index=False
    )
)

vif_df.to_csv(
    VIF_OUTPUT,
    index=False
)


# =====================================================================
# SAVE MODEL RESULTS
# =====================================================================

incremental_df.to_csv(
    OUTPUT_DATA,
    index=False
)


# =====================================================================
# REPORT
# =====================================================================

report_lines = []

report_lines.append(
    "SCRIPT 16 — INCREMENTAL CHEMICAL DESCRIPTOR ANALYSIS"
)

report_lines.append(
    "=" * 70
)

report_lines.append("")

report_lines.append(
    "OBJECTIVE"
)

report_lines.append(
    "Determine which chemically interpretable descriptors add "
    "explanatory power beyond the established MolWt + MolLogP + "
    "RingCount + AromaticRings model."
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
    "standard_analytical_domain flag."
)

report_lines.append("")

report_lines.append(
    "BASELINE"
)

report_lines.append(
    "Solubility ~ rdkit_molwt + rdkit_mollogp "
    "+ rdkit_ring_count + rdkit_aromatic_rings"
)

report_lines.append("")

report_lines.append(
    "INDIVIDUAL INCREMENTAL MODELS"
)

report_lines.append(
    incremental_df.to_string(
        index=False
    )
)

report_lines.append("")

report_lines.append(
    "ROBUST CANDIDATE COEFFICIENTS"
)

report_lines.append(
    candidate_coefficients.to_string(
        index=False
    )
)

report_lines.append("")

report_lines.append(
    "NESTED F-TESTS"
)

report_lines.append(
    nested_df.to_string(
        index=False
    )
)

report_lines.append("")

report_lines.append(
    "SELECTIVE MODEL"
)

report_lines.append(
    "Selected descriptors: "
    + (
        ", ".join(selected_descriptors)
        if selected_descriptors
        else "None"
    )
)

report_lines.append(
    f"Selective model R² = {selective_r2:.6f}"
)

report_lines.append(
    f"Selective model RMSE = {selective_rmse:.6f}"
)

report_lines.append(
    f"Delta R² vs ring baseline = "
    f"{selective_r2 - baseline_r2:.6f}"
)

report_lines.append("")

report_lines.append(
    "SELECTIVE MODEL COEFFICIENTS"
)

report_lines.append(
    selective_coefficients_df.to_string(
        index=False
    )
)

report_lines.append("")

report_lines.append(
    "VIF"
)

report_lines.append(
    vif_df.to_string(
        index=False
    )
)

report_lines.append("")

report_lines.append(
    "METHODOLOGICAL NOTES"
)

report_lines.append(
    "HC3 robust covariance is used for individual coefficient "
    "inference."
)

report_lines.append(
    "Nested F-tests use conventional OLS covariance."
)

report_lines.append(
    "HeavyAtomCount, MolMR and LabuteASA were not included in "
    "the initial candidate screen because Script 07 established "
    "their strong correlation with MolWt."
)

report_lines.append(
    "Ring count is not equivalent to fused-ring count."
)

report_lines.append(
    "This analysis is explanatory and does not provide held-out "
    "generalization performance."
)

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(report_lines)
    )


# =====================================================================
# FINAL OUTPUT
# =====================================================================

print("\n" + "=" * 70)
print("OUTPUTS")
print("=" * 70)

print("\nModel results:")
print(OUTPUT_DATA)

print("\nCoefficient results:")
print(COEFFICIENT_OUTPUT)

print("\nVIF results:")
print(VIF_OUTPUT)

print("\nCorrelation results:")
print(CORRELATION_OUTPUT)

print("\nReport:")
print(REPORT_FILE)

print("\n" + "=" * 70)
print("SCRIPT 16 COMPLETE")
print("=" * 70)