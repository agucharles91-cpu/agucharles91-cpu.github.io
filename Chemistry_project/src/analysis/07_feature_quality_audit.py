from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "molecular_features.csv"
)

OUTPUT_DATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_quality_audit.csv"
)

OUTPUT_CORRELATION = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_correlation_matrix.csv"
)

OUTPUT_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "feature_quality_audit.txt"
)


# ============================================================
# CONFIGURATION: FEATURE DEFINITIONS
# ============================================================

DESCRIPTOR_COLUMNS = [
    "rdkit_molwt",
    "rdkit_mollogp",
    "rdkit_tpsa",
    "rdkit_hbd",
    "rdkit_hba",
    "rdkit_rotatable_bonds",
    "rdkit_heavy_atom_count",
    "rdkit_heteroatom_count",
    "rdkit_aromatic_rings",
    "rdkit_saturated_rings",
    "rdkit_aliphatic_rings",
    "rdkit_ring_count",
    "rdkit_fraction_csp3",
    "rdkit_molmr",
    "rdkit_labute_asa",
    "rdkit_balaban_j",
    "rdkit_bertz_ct",
]

TARGET_COLUMN = "Solubility"

# Features with <= this percentage of unique values are not
# automatically removed; they are flagged for inspection.
LOW_VARIANCE_UNIQUE_PCT = 1.0

# Correlations at or above this magnitude are flagged as
# potentially redundant. This is an audit threshold, not an
# automatic deletion rule.
HIGH_CORRELATION_THRESHOLD = 0.90


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FEATURE QUALITY AUDIT")
print("=" * 70)

print("\nLoading molecular feature dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df):,}")
print(f"Columns loaded: {len(df.columns):,}")


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    TARGET_COLUMN,
    *DESCRIPTOR_COLUMNS,
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise KeyError(
        "Required columns missing from molecular_features.csv: "
        + ", ".join(missing_columns)
    )


# ============================================================
# BASIC DATASET VALIDATION
# ============================================================

print("\nValidating analytical population...")

if len(df) != 8643:
    raise ValueError(
        "Unexpected analytical population size.\n"
        "Expected: 8,643\n"
        f"Observed: {len(df):,}"
    )

print("Analytical population verified: 8,643 rows")


# ============================================================
# TARGET VALIDATION
# ============================================================

print("\nValidating target variable...")

target_missing = int(df[TARGET_COLUMN].isna().sum())
target_unique = int(df[TARGET_COLUMN].nunique())
target_min = df[TARGET_COLUMN].min()
target_max = df[TARGET_COLUMN].max()
target_mean = df[TARGET_COLUMN].mean()
target_median = df[TARGET_COLUMN].median()
target_std = df[TARGET_COLUMN].std()
target_skew = skew(
    df[TARGET_COLUMN].dropna()
)

print(f"Target missing:  {target_missing:,}")
print(f"Target unique:   {target_unique:,}")
print(f"Target minimum:  {target_min:.6f}")
print(f"Target maximum:  {target_max:.6f}")
print(f"Target mean:     {target_mean:.6f}")
print(f"Target median:   {target_median:.6f}")
print(f"Target std:      {target_std:.6f}")
print(f"Target skewness: {target_skew:.6f}")


# ============================================================
# FEATURE QUALITY SUMMARY
# ============================================================

print("\nCalculating descriptor quality metrics...")

quality_rows = []

for column in DESCRIPTOR_COLUMNS:

    series = df[column]

    non_missing = series.dropna()

    n = len(series)
    missing = int(series.isna().sum())

    unique_count = int(series.nunique(dropna=True))

    unique_pct = (
        unique_count / n * 100
        if n > 0
        else np.nan
    )

    variance = (
        non_missing.var()
        if len(non_missing) > 1
        else np.nan
    )

    standard_deviation = (
        non_missing.std()
        if len(non_missing) > 1
        else np.nan
    )

    minimum = (
        non_missing.min()
        if len(non_missing) > 0
        else np.nan
    )

    maximum = (
        non_missing.max()
        if len(non_missing) > 0
        else np.nan
    )

    mean = (
        non_missing.mean()
        if len(non_missing) > 0
        else np.nan
    )

    median = (
        non_missing.median()
        if len(non_missing) > 0
        else np.nan
    )

    feature_skew = (
        skew(non_missing)
        if len(non_missing) > 2
        else np.nan
    )

    zero_count = int(
        (non_missing == 0).sum()
    )

    zero_pct = (
        zero_count / len(non_missing) * 100
        if len(non_missing) > 0
        else np.nan
    )

    quality_rows.append(
        {
            "feature": column,
            "n": n,
            "missing": missing,
            "missing_pct": missing / n * 100,
            "unique_count": unique_count,
            "unique_pct": unique_pct,
            "variance": variance,
            "std": standard_deviation,
            "mean": mean,
            "median": median,
            "min": minimum,
            "max": maximum,
            "skewness": feature_skew,
            "zero_count": zero_count,
            "zero_pct": zero_pct,
            "low_variance_flag": (
                unique_pct <= LOW_VARIANCE_UNIQUE_PCT
            ),
        }
    )


quality_df = pd.DataFrame(quality_rows)


# ============================================================
# MISSINGNESS AUDIT
# ============================================================

print("\nDescriptor missingness:")

print(
    quality_df[
        [
            "feature",
            "missing",
            "missing_pct",
        ]
    ].to_string(index=False)
)


# ============================================================
# LOW-VARIANCE AUDIT
# ============================================================

low_variance_features = quality_df.loc[
    quality_df["low_variance_flag"],
    "feature"
].tolist()

print("\nLow-variance / low-uniqueness features:")

if low_variance_features:

    for feature in low_variance_features:
        print(f"  {feature}")

else:

    print("  None")


# ============================================================
# CONSTANT FEATURE AUDIT
# ============================================================

constant_features = []

for column in DESCRIPTOR_COLUMNS:

    if df[column].nunique(dropna=True) <= 1:
        constant_features.append(column)

print("\nConstant features:")

if constant_features:

    for feature in constant_features:
        print(f"  {feature}")

else:

    print("  None")


# ============================================================
# CORRELATION MATRIX
# ============================================================

print("\nCalculating Spearman correlation matrix...")

correlation_matrix = (
    df[DESCRIPTOR_COLUMNS]
    .corr(method="spearman")
)

OUTPUT_CORRELATION.parent.mkdir(
    parents=True,
    exist_ok=True
)

correlation_matrix.to_csv(
    OUTPUT_CORRELATION
)


# ============================================================
# HIGH-CORRELATION PAIRS
# ============================================================

high_correlation_pairs = []

for i, feature_a in enumerate(DESCRIPTOR_COLUMNS):

    for feature_b in DESCRIPTOR_COLUMNS[i + 1:]:

        rho = correlation_matrix.loc[
            feature_a,
            feature_b
        ]

        if pd.isna(rho):
            continue

        if abs(rho) >= HIGH_CORRELATION_THRESHOLD:

            high_correlation_pairs.append(
                {
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "spearman_rho": rho,
                    "absolute_rho": abs(rho),
                }
            )


high_correlation_df = pd.DataFrame(
    high_correlation_pairs
)

if not high_correlation_df.empty:

    high_correlation_df = (
        high_correlation_df
        .sort_values(
            "absolute_rho",
            ascending=False
        )
        .reset_index(drop=True)
    )


print(
    "\nHighly correlated descriptor pairs "
    f"(|rho| >= {HIGH_CORRELATION_THRESHOLD:.2f}):"
)

if high_correlation_df.empty:

    print("  None")

else:

    print(
        high_correlation_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )


# ============================================================
# FEATURE / TARGET CORRELATION
# ============================================================

print("\nCalculating descriptor-target relationships...")

target_correlations = []

for feature in DESCRIPTOR_COLUMNS:

    valid = df[
        [feature, TARGET_COLUMN]
    ].dropna()

    if len(valid) < 3:
        rho = np.nan
    else:
        rho = valid[feature].corr(
            valid[TARGET_COLUMN],
            method="spearman"
        )

    target_correlations.append(
        {
            "feature": feature,
            "spearman_rho_with_solubility": rho,
            "absolute_rho": (
                abs(rho)
                if not pd.isna(rho)
                else np.nan
            ),
        }
    )


target_correlation_df = pd.DataFrame(
    target_correlations
).sort_values(
    "absolute_rho",
    ascending=False
)


print(
    target_correlation_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# OUTLIER AUDIT USING IQR
# ============================================================

print("\nCalculating IQR-based outlier counts...")

outlier_rows = []

for feature in DESCRIPTOR_COLUMNS:

    series = df[feature].dropna()

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (
        (series < lower_bound)
        | (series > upper_bound)
    )

    outlier_count = int(
        outlier_mask.sum()
    )

    outlier_pct = (
        outlier_count
        / len(series)
        * 100
        if len(series) > 0
        else np.nan
    )

    outlier_rows.append(
        {
            "feature": feature,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_count": outlier_count,
            "outlier_pct": outlier_pct,
        }
    )


outlier_df = (
    pd.DataFrame(outlier_rows)
    .sort_values(
        "outlier_pct",
        ascending=False
    )
)


# ============================================================
# COMBINED AUDIT DATASET
# ============================================================

audit_df = quality_df.merge(
    target_correlation_df[
        [
            "feature",
            "spearman_rho_with_solubility",
        ]
    ],
    on="feature",
    how="left"
)

audit_df = audit_df.merge(
    outlier_df[
        [
            "feature",
            "outlier_count",
            "outlier_pct",
        ]
    ],
    on="feature",
    how="left"
)


# ============================================================
# SAVE AUDIT DATA
# ============================================================

OUTPUT_DATA.parent.mkdir(
    parents=True,
    exist_ok=True
)

audit_df.to_csv(
    OUTPUT_DATA,
    index=False
)


# ============================================================
# WRITE REPORT
# ============================================================

OUTPUT_REPORT.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_REPORT,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "=" * 70 + "\n"
    )

    report.write(
        "FEATURE QUALITY AUDIT REPORT\n"
    )

    report.write(
        "=" * 70 + "\n\n"
    )

    report.write(
        "DATASET\n"
    )

    report.write(
        "-" * 70 + "\n"
    )

    report.write(
        f"Rows: {len(df):,}\n"
    )

    report.write(
        f"Descriptor count: "
        f"{len(DESCRIPTOR_COLUMNS):,}\n"
    )

    report.write(
        f"Target: {TARGET_COLUMN}\n\n"
    )

    report.write(
        "TARGET SUMMARY\n"
    )

    report.write(
        "-" * 70 + "\n"
    )

    report.write(
        f"Missing: {target_missing:,}\n"
    )

    report.write(
        f"Unique values: {target_unique:,}\n"
    )

    report.write(
        f"Minimum: {target_min:.6f}\n"
    )

    report.write(
        f"Maximum: {target_max:.6f}\n"
    )

    report.write(
        f"Mean: {target_mean:.6f}\n"
    )

    report.write(
        f"Median: {target_median:.6f}\n"
    )

    report.write(
        f"Standard deviation: {target_std:.6f}\n"
    )

    report.write(
        f"Skewness: {target_skew:.6f}\n\n"
    )

    report.write(
        "DESCRIPTOR QUALITY\n"
    )

    report.write(
        "-" * 70 + "\n"
    )

    report.write(
        audit_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    report.write("\n\n")

    report.write(
        "CONSTANT FEATURES\n"
    )

    report.write(
        "-" * 70 + "\n"
    )

    if constant_features:

        report.write(
            "\n".join(constant_features)
        )

    else:

        report.write(
            "None"
        )

    report.write("\n\n")

    report.write(
        "HIGHLY CORRELATED FEATURE PAIRS\n"
    )

    report.write(
        "-" * 70 + "\n"
    )

    if high_correlation_df.empty:

        report.write(
            "No pairs exceeded the "
            f"|rho| >= {HIGH_CORRELATION_THRESHOLD:.2f} "
            "audit threshold."
        )

    else:

        report.write(
            high_correlation_df.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}"
            )
        )

    report.write("\n\n")

    report.write(
        "DESCRIPTOR-TARGET SPEARMAN CORRELATIONS\n"
    )

    report.write(
        "-" * 70 + "\n"
    )

    report.write(
        target_correlation_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    report.write("\n\n")

    report.write(
        "IQR-BASED OUTLIER AUDIT\n"
    )

    report.write(
        "-" * 70 + "\n"
    )

    report.write(
        outlier_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    report.write("\n\n")

    report.write(
        "INTERPRETATION\n"
    )

    report.write(
        "-" * 70 + "\n"
    )

    report.write(
        "This audit is descriptive and does not automatically "
        "remove molecular descriptors. Low variance, strong "
        "inter-feature correlation, skewness, and IQR-defined "
        "outliers are flagged for scientific and modelling "
        "consideration. Feature selection should occur only "
        "after considering descriptor meaning, redundancy, "
        "model requirements, and validation performance.\n"
    )


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE QUALITY SUMMARY")
print("=" * 70)

print(
    f"\nDataset rows: {len(df):,}"
)

print(
    f"Descriptors audited: "
    f"{len(DESCRIPTOR_COLUMNS):,}"
)

print(
    f"Target missing values: "
    f"{target_missing:,}"
)

print(
    f"Constant features: "
    f"{len(constant_features):,}"
)

print(
    f"Highly correlated pairs: "
    f"{len(high_correlation_df):,}"
)

print("\nTop descriptor-target correlations:")

print(
    target_correlation_df.head(10).to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\n" + "=" * 70)
print("FEATURE QUALITY AUDIT COMPLETE")
print("=" * 70)

print("\nAudit dataset saved to:")
print(OUTPUT_DATA)

print("\nCorrelation matrix saved to:")
print(OUTPUT_CORRELATION)

print("\nReport saved to:")
print(OUTPUT_REPORT)