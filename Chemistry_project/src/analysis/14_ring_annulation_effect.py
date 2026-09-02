"""
14_ring_annulation_effect.py

Quantify the relationship between molecular ring complexity and
baseline solubility residuals across the full Population C.

Canonical baseline:
    Solubility ~ rdkit_molwt + rdkit_mollogp

Residual:
    observed Solubility - baseline predicted Solubility

Primary questions:
    1. Does residual become systematically more negative with increasing
       total ring count?
    2. Does residual become systematically more negative with increasing
       aromatic ring count?
    3. Does residual magnitude change with ring complexity?
    4. Are the observed relationships statistically supported across
       the full analytical population?

Important:
    Ring count is NOT equivalent to fused-ring count.
    This analysis therefore describes ring-count/aromatic-ring associations
    rather than claiming a causal fused-ring effect.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import spearmanr


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESIDUAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "baseline_residuals_popc.csv"
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
    / "ring_annulation_effect.csv"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "ring_annulation_effect.txt"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "ring_annulation"
)

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------

RANDOM_SEED = 42
N_BOOTSTRAP = 2000

np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------

def bootstrap_ci(values, statistic=np.mean, n_bootstrap=N_BOOTSTRAP,
                 random_seed=RANDOM_SEED):
    """
    Calculate a percentile bootstrap 95% confidence interval.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    if len(values) == 1:
        value = statistic(values)
        return value, value

    rng = np.random.default_rng(random_seed)

    bootstrap_statistics = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        sample = rng.choice(
            values,
            size=len(values),
            replace=True
        )
        bootstrap_statistics[i] = statistic(sample)

    lower = np.percentile(bootstrap_statistics, 2.5)
    upper = np.percentile(bootstrap_statistics, 97.5)

    return lower, upper


def grouped_statistics(df, feature):
    """
    Calculate descriptive statistics and bootstrap confidence intervals
    for residual and absolute residual by a ring-related feature.
    """

    rows = []

    for value, group in df.groupby(feature, sort=True):

        residuals = group["Residual"].dropna().to_numpy()
        abs_residuals = group["abs_residual"].dropna().to_numpy()

        residual_mean = np.mean(residuals)
        residual_median = np.median(residuals)

        abs_mean = np.mean(abs_residuals)
        abs_median = np.median(abs_residuals)

        residual_mean_low, residual_mean_high = bootstrap_ci(
            residuals,
            statistic=np.mean,
            random_seed=RANDOM_SEED + int(value)
        )

        residual_median_low, residual_median_high = bootstrap_ci(
            residuals,
            statistic=np.median,
            random_seed=RANDOM_SEED + 1000 + int(value)
        )

        abs_mean_low, abs_mean_high = bootstrap_ci(
            abs_residuals,
            statistic=np.mean,
            random_seed=RANDOM_SEED + 2000 + int(value)
        )

        abs_median_low, abs_median_high = bootstrap_ci(
            abs_residuals,
            statistic=np.median,
            random_seed=RANDOM_SEED + 3000 + int(value)
        )

        rows.append({
            feature: value,
            "n": len(group),

            "mean_residual": residual_mean,
            "median_residual": residual_median,

            "mean_residual_ci_low": residual_mean_low,
            "mean_residual_ci_high": residual_mean_high,

            "median_residual_ci_low": residual_median_low,
            "median_residual_ci_high": residual_median_high,

            "mean_abs_residual": abs_mean,
            "median_abs_residual": abs_median,

            "mean_abs_residual_ci_low": abs_mean_low,
            "mean_abs_residual_ci_high": abs_mean_high,

            "median_abs_residual_ci_low": abs_median_low,
            "median_abs_residual_ci_high": abs_median_high,
        })

    return pd.DataFrame(rows)


def spearman_trend(df, feature, target):
    """
    Spearman rank correlation between ring-related feature and target.
    """
    subset = df[[feature, target]].dropna()

    rho, p_value = spearmanr(
        subset[feature],
        subset[target]
    )

    return {
        "feature": feature,
        "target": target,
        "n": len(subset),
        "spearman_rho": rho,
        "p_value": p_value
    }


def robust_ols(df, feature, target):
    """
    Fit target ~ feature using OLS with HC3 robust standard errors.
    """
    subset = df[[feature, target]].dropna()

    X = sm.add_constant(subset[feature])
    y = subset[target]

    model = sm.OLS(y, X).fit(cov_type="HC3")

    coefficient = model.params[feature]
    p_value = model.pvalues[feature]
    ci_low, ci_high = model.conf_int().loc[feature]

    return {
        "feature": feature,
        "target": target,
        "n": len(subset),
        "slope": coefficient,
        "slope_ci_low": ci_low,
        "slope_ci_high": ci_high,
        "p_value": p_value,
        "r_squared": model.rsquared
    }


def format_p_value(p):
    if p < 0.001:
        return "<0.001"
    return f"{p:.4g}"


# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------

print("=" * 70)
print("SCRIPT 14 — RING ANNULATION / RING-COUNT EFFECT")
print("=" * 70)

print("\nLoading canonical residual dataset:")
print(RESIDUAL_FILE)

residuals = pd.read_csv(RESIDUAL_FILE)

print(f"Residual dataset shape: {residuals.shape}")

print("\nLoading molecular features:")
print(FEATURE_FILE)

features = pd.read_csv(FEATURE_FILE)

print(f"Molecular features shape: {features.shape}")


# ---------------------------------------------------------------------
# VALIDATE REQUIRED COLUMNS
# ---------------------------------------------------------------------

if "ID" not in residuals.columns:
    raise ValueError("Residual dataset does not contain required column: ID")

if "ID" not in features.columns:
    raise ValueError("Molecular features dataset does not contain required column: ID")


# Identify residual column robustly
possible_residual_columns = [
    "Residual",
    "residual",
    "residual_popc"
]

residual_column = None

for column in possible_residual_columns:
    if column in residuals.columns:
        residual_column = column
        break

if residual_column is None:
    raise ValueError(
        "Could not identify residual column. "
        f"Available columns include: {list(residuals.columns)}"
    )

if residual_column != "Residual":
    residuals = residuals.rename(
        columns={residual_column: "Residual"}
    )


required_features = [
    "rdkit_ring_count",
    "rdkit_aromatic_rings"
]

missing_features = [
    column
    for column in required_features
    if column not in features.columns
]

if missing_features:
    raise ValueError(
        "Missing required molecular feature columns: "
        + ", ".join(missing_features)
    )


# ---------------------------------------------------------------------
# CHECK ID UNIQUENESS
# ---------------------------------------------------------------------

if residuals["ID"].duplicated().any():
    raise ValueError(
        "Duplicate IDs detected in residual dataset."
    )

if features["ID"].duplicated().any():
    raise ValueError(
        "Duplicate IDs detected in molecular feature dataset."
    )


# ---------------------------------------------------------------------
# MERGE
# ---------------------------------------------------------------------

analysis = residuals[
    ["ID", "Residual"]
].merge(
    features[
        ["ID", "rdkit_ring_count", "rdkit_aromatic_rings"]
    ],
    on="ID",
    how="inner",
    validate="one_to_one"
)

print("\nMerged analysis dataset:")
print(f"Rows: {len(analysis):,}")

if len(analysis) != 8643:
    raise ValueError(
        f"Expected 8,643 Population C rows after merge, "
        f"but found {len(analysis):,}."
    )

print("Population C verified: 8,643 rows")


# ---------------------------------------------------------------------
# CLEAN / DERIVE VARIABLES
# ---------------------------------------------------------------------

analysis["Residual"] = pd.to_numeric(
    analysis["Residual"],
    errors="coerce"
)

analysis["rdkit_ring_count"] = pd.to_numeric(
    analysis["rdkit_ring_count"],
    errors="coerce"
)

analysis["rdkit_aromatic_rings"] = pd.to_numeric(
    analysis["rdkit_aromatic_rings"],
    errors="coerce"
)

analysis["abs_residual"] = analysis["Residual"].abs()


if analysis["Residual"].isna().any():
    raise ValueError("Missing residual values detected.")

if analysis["rdkit_ring_count"].isna().any():
    raise ValueError("Missing ring-count values detected.")

if analysis["rdkit_aromatic_rings"].isna().any():
    raise ValueError("Missing aromatic-ring values detected.")


# Ring counts should be integer-valued
if not np.allclose(
    analysis["rdkit_ring_count"],
    np.round(analysis["rdkit_ring_count"])
):
    raise ValueError(
        "rdkit_ring_count contains non-integer values."
    )

if not np.allclose(
    analysis["rdkit_aromatic_rings"],
    np.round(analysis["rdkit_aromatic_rings"])
):
    raise ValueError(
        "rdkit_aromatic_rings contains non-integer values."
    )

analysis["rdkit_ring_count"] = (
    analysis["rdkit_ring_count"].astype(int)
)

analysis["rdkit_aromatic_rings"] = (
    analysis["rdkit_aromatic_rings"].astype(int)
)


# ---------------------------------------------------------------------
# BASIC DISTRIBUTION
# ---------------------------------------------------------------------

print("\n" + "-" * 70)
print("RING COUNT DISTRIBUTION")
print("-" * 70)

print(
    analysis["rdkit_ring_count"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\n" + "-" * 70)
print("AROMATIC RING DISTRIBUTION")
print("-" * 70)

print(
    analysis["rdkit_aromatic_rings"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ---------------------------------------------------------------------
# GROUPED STATISTICS
# ---------------------------------------------------------------------

ring_stats = grouped_statistics(
    analysis,
    "rdkit_ring_count"
)

aromatic_stats = grouped_statistics(
    analysis,
    "rdkit_aromatic_rings"
)

ring_stats["feature_type"] = "total_ring_count"
aromatic_stats["feature_type"] = "aromatic_ring_count"

combined_stats = pd.concat(
    [ring_stats, aromatic_stats],
    ignore_index=True
)

combined_stats.to_csv(
    OUTPUT_DATA,
    index=False
)

print("\nGrouped statistics written to:")
print(OUTPUT_DATA)


# ---------------------------------------------------------------------
# TREND TESTS
# ---------------------------------------------------------------------

trend_results = []

for feature in [
    "rdkit_ring_count",
    "rdkit_aromatic_rings"
]:

    trend_results.append(
        spearman_trend(
            analysis,
            feature,
            "Residual"
        )
    )

    trend_results.append(
        spearman_trend(
            analysis,
            feature,
            "abs_residual"
        )
    )

trend_results_df = pd.DataFrame(trend_results)


# ---------------------------------------------------------------------
# ROBUST LINEAR TREND MODELS
# ---------------------------------------------------------------------

ols_results = []

for feature in [
    "rdkit_ring_count",
    "rdkit_aromatic_rings"
]:

    ols_results.append(
        robust_ols(
            analysis,
            feature,
            "Residual"
        )
    )

    ols_results.append(
        robust_ols(
            analysis,
            feature,
            "abs_residual"
        )
    )

ols_results_df = pd.DataFrame(ols_results)


# ---------------------------------------------------------------------
# PRINT TREND RESULTS
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("SPEARMAN TREND TESTS")
print("=" * 70)

for _, row in trend_results_df.iterrows():

    print(
        f"{row['feature']} vs {row['target']}: "
        f"rho = {row['spearman_rho']:.4f}, "
        f"p = {format_p_value(row['p_value'])}, "
        f"n = {int(row['n']):,}"
    )


print("\n" + "=" * 70)
print("ROBUST OLS TREND MODELS")
print("=" * 70)

for _, row in ols_results_df.iterrows():

    print(
        f"{row['target']} ~ {row['feature']}: "
        f"slope = {row['slope']:.4f}, "
        f"95% CI = "
        f"[{row['slope_ci_low']:.4f}, "
        f"{row['slope_ci_high']:.4f}], "
        f"p = {format_p_value(row['p_value'])}, "
        f"R² = {row['r_squared']:.4f}"
    )


# ---------------------------------------------------------------------
# REPORT TABLES
# ---------------------------------------------------------------------

def format_group_table(stats_df):
    """
    Format grouped statistics for inclusion in the report.
    """
    columns = [
        stats_df.columns[0],
        "n",
        "mean_residual",
        "median_residual",
        "mean_abs_residual",
        "median_abs_residual"
    ]

    output = stats_df[columns].copy()

    for column in [
        "mean_residual",
        "median_residual",
        "mean_abs_residual",
        "median_abs_residual"
    ]:
        output[column] = output[column].map(
            lambda x: f"{x:.4f}"
        )

    return output.to_string(index=False)


# ---------------------------------------------------------------------
# VISUALIZATION 1 — RESIDUAL BY TOTAL RING COUNT
# ---------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

groups = [
    analysis.loc[
        analysis["rdkit_ring_count"] == ring,
        "Residual"
    ].values
    for ring in sorted(
        analysis["rdkit_ring_count"].unique()
    )
]

positions = sorted(
    analysis["rdkit_ring_count"].unique()
)

ax.boxplot(
    groups,
    positions=positions,
    showfliers=False
)

ax.axhline(
    0,
    linestyle="--",
    linewidth=1
)

ax.set_xlabel("Total Ring Count")
ax.set_ylabel("Baseline Residual (Observed − Predicted LogS)")
ax.set_title(
    "Baseline Solubility Residuals by Total Ring Count"
)

fig.tight_layout()

ring_boxplot = (
    FIGURE_DIR
    / "residual_by_total_ring_count.png"
)

fig.savefig(
    ring_boxplot,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ---------------------------------------------------------------------
# VISUALIZATION 2 — MEDIAN RESIDUAL WITH BOOTSTRAP CI
# ---------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

plot_stats = ring_stats.sort_values(
    "rdkit_ring_count"
)

x = plot_stats["rdkit_ring_count"].to_numpy()
y = plot_stats["median_residual"].to_numpy()

lower = (
    y
    - plot_stats["median_residual_ci_low"].to_numpy()
)

upper = (
    plot_stats["median_residual_ci_high"].to_numpy()
    - y
)

ax.errorbar(
    x,
    y,
    yerr=[lower, upper],
    fmt="o-",
    capsize=4
)

ax.axhline(
    0,
    linestyle="--",
    linewidth=1
)

ax.set_xlabel("Total Ring Count")
ax.set_ylabel("Median Baseline Residual")
ax.set_title(
    "Median Baseline Residual Across Ring Counts"
)

fig.tight_layout()

ring_trend = (
    FIGURE_DIR
    / "median_residual_by_ring_count.png"
)

fig.savefig(
    ring_trend,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ---------------------------------------------------------------------
# VISUALIZATION 3 — ABSOLUTE RESIDUAL BY TOTAL RING COUNT
# ---------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

groups_abs = [
    analysis.loc[
        analysis["rdkit_ring_count"] == ring,
        "abs_residual"
    ].values
    for ring in positions
]

ax.boxplot(
    groups_abs,
    positions=positions,
    showfliers=False
)

ax.set_xlabel("Total Ring Count")
ax.set_ylabel("|Baseline Residual|")
ax.set_title(
    "Absolute Baseline Residuals by Total Ring Count"
)

fig.tight_layout()

abs_ring_boxplot = (
    FIGURE_DIR
    / "absolute_residual_by_total_ring_count.png"
)

fig.savefig(
    abs_ring_boxplot,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ---------------------------------------------------------------------
# VISUALIZATION 4 — AROMATIC RINGS
# ---------------------------------------------------------------------

aromatic_positions = sorted(
    analysis["rdkit_aromatic_rings"].unique()
)

groups_aromatic = [
    analysis.loc[
        analysis["rdkit_aromatic_rings"] == ring,
        "Residual"
    ].values
    for ring in aromatic_positions
]

fig, ax = plt.subplots(figsize=(10, 6))

ax.boxplot(
    groups_aromatic,
    positions=aromatic_positions,
    showfliers=False
)

ax.axhline(
    0,
    linestyle="--",
    linewidth=1
)

ax.set_xlabel("Aromatic Ring Count")
ax.set_ylabel("Baseline Residual (Observed − Predicted LogS)")
ax.set_title(
    "Baseline Solubility Residuals by Aromatic Ring Count"
)

fig.tight_layout()

aromatic_boxplot = (
    FIGURE_DIR
    / "residual_by_aromatic_ring_count.png"
)

fig.savefig(
    aromatic_boxplot,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ---------------------------------------------------------------------
# WRITE TEXT REPORT
# ---------------------------------------------------------------------

report_lines = []

report_lines.append(
    "SCRIPT 14 — RING-COUNT / STRUCTURAL COMPLEXITY EFFECT"
)

report_lines.append("=" * 70)
report_lines.append("")

report_lines.append(
    "OBJECTIVE"
)

report_lines.append(
    "Quantify whether molecular ring complexity is systematically "
    "associated with baseline solubility residuals across the full "
    "8,643-compound Population C."
)

report_lines.append("")

report_lines.append(
    "CANONICAL BASELINE"
)

report_lines.append(
    "Solubility ~ rdkit_molwt + rdkit_mollogp"
)

report_lines.append(
    "Residual = observed Solubility - baseline predicted Solubility"
)

report_lines.append("")

report_lines.append(
    "IMPORTANT INTERPRETATION"
)

report_lines.append(
    "Positive residual = observed solubility is higher than the "
    "baseline prediction."
)

report_lines.append(
    "Negative residual = observed solubility is lower than the "
    "baseline prediction."
)

report_lines.append(
    "Ring count is not equivalent to fused-ring count and is not "
    "a direct measurement of crystal packing or lattice stability."
)

report_lines.append("")

report_lines.append(
    "DATA VALIDATION"
)

report_lines.append(
    f"Population C rows after merge: {len(analysis):,}"
)

report_lines.append(
    "Expected Population C rows: 8,643"
)

report_lines.append("")

report_lines.append(
    "TOTAL RING COUNT — GROUPED STATISTICS"
)

report_lines.append(
    format_group_table(ring_stats)
)

report_lines.append("")

report_lines.append(
    "AROMATIC RING COUNT — GROUPED STATISTICS"
)

report_lines.append(
    format_group_table(aromatic_stats)
)

report_lines.append("")

report_lines.append(
    "SPEARMAN TREND TESTS"
)

for _, row in trend_results_df.iterrows():

    report_lines.append(
        f"{row['feature']} vs {row['target']}: "
        f"rho = {row['spearman_rho']:.6f}; "
        f"p = {row['p_value']:.6g}; "
        f"n = {int(row['n']):,}"
    )

report_lines.append("")

report_lines.append(
    "ROBUST OLS TREND MODELS"
)

for _, row in ols_results_df.iterrows():

    report_lines.append(
        f"{row['target']} ~ {row['feature']}: "
        f"slope = {row['slope']:.6f}; "
        f"95% CI = "
        f"[{row['slope_ci_low']:.6f}, "
        f"{row['slope_ci_high']:.6f}]; "
        f"p = {row['p_value']:.6g}; "
        f"R² = {row['r_squared']:.6f}"
    )

report_lines.append("")

report_lines.append(
    "FIGURES"
)

report_lines.append(
    str(ring_boxplot)
)

report_lines.append(
    str(ring_trend)
)

report_lines.append(
    str(abs_ring_boxplot)
)

report_lines.append(
    str(aromatic_boxplot)
)

report_lines.append("")

report_lines.append(
    "INTERPRETATION GUIDANCE"
)

report_lines.append(
    "A negative association between ring count and residual indicates "
    "that compounds with greater ring counts tend to have observed "
    "solubilities below the MolWt + MolLogP baseline prediction."
)

report_lines.append(
    "A positive association would indicate the opposite and would "
    "contradict the current working hypothesis."
)

report_lines.append(
    "Statistical significance alone should not be treated as evidence "
    "of a causal ring effect. Ring count is correlated with molecular "
    "size, aromaticity and other molecular properties."
)

report_lines.append(
    "The subsequent multivariable analysis should therefore test "
    "whether ring-related descriptors provide explanatory power beyond "
    "MolWt and MolLogP."
)

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:
    f.write("\n".join(report_lines))


# ---------------------------------------------------------------------
# FINAL OUTPUT
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("OUTPUTS")
print("=" * 70)

print(f"\nGrouped statistics:")
print(OUTPUT_DATA)

print("\nReport:")
print(REPORT_FILE)

print("\nFigures:")

for figure in [
    ring_boxplot,
    ring_trend,
    abs_ring_boxplot,
    aromatic_boxplot
]:
    print(figure)

print("\n" + "=" * 70)
print("SCRIPT 14 COMPLETE")
print("=" * 70)