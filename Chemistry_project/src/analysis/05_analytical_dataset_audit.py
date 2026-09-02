from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chemical_domain_audit.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "analytical_dataset_audit.csv"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "analytical_dataset_audit.txt"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("ANALYTICAL DATASET / INCLUSION AUDIT")
print("=" * 70)

print("\nLoading chemical domain audit...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df):,}")
print(f"Columns loaded: {len(df.columns):,}")


# ============================================================
# BASIC VALIDATION
# ============================================================

required_columns = [
    "ID",
    "SMILES",
    "Solubility",
    "PredictedSolubility",
    "Residual",
    "abs_residual",
    "MolWt",
    "MolLogP",
    "has_metal",
    "is_charged",
    "multicomponent_record",
    "component_count",
    "Group",
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(
        f"Required columns are missing: {missing}"
    )


# ============================================================
# DOMAIN FLAGS
# ============================================================

audit = df.copy()

audit["single_component"] = (
    audit["component_count"] == 1
)

audit["multicomponent"] = (
    audit["component_count"] > 1
)

audit["metal_free"] = (
    ~audit["has_metal"].fillna(False)
)

audit["neutral"] = (
    ~audit["is_charged"].fillna(False)
)

# Recalculate percentile thresholds from the actual data.
mw_95 = audit["MolWt"].quantile(0.95)
mw_99 = audit["MolWt"].quantile(0.99)

logp_95 = audit["MolLogP"].quantile(0.95)
logp_99 = audit["MolLogP"].quantile(0.99)

audit["high_mw"] = audit["MolWt"] >= mw_95
audit["very_high_mw"] = audit["MolWt"] >= mw_99

audit["high_logp"] = audit["MolLogP"] >= logp_95
audit["very_high_logp"] = audit["MolLogP"] >= logp_99


# ============================================================
# ANALYTICAL CANDIDATE POPULATIONS
# ============================================================

# Population A:
# Single-component structures only.
audit["candidate_single_component"] = (
    audit["single_component"]
)

# Population B:
# Single-component + metal-free.
audit["candidate_single_neutral"] = (
    audit["single_component"]
    & audit["metal_free"]
    & audit["neutral"]
)

# Population C:
# Single-component + metal-free + neutral
# + exclude extreme MW and LogP observations.
audit["candidate_standard_domain"] = (
    audit["single_component"]
    & audit["metal_free"]
    & audit["neutral"]
    & (~audit["very_high_mw"])
    & (~audit["very_high_logp"])
)

# Population D:
# More conservative chemical domain.
audit["candidate_conservative_domain"] = (
    audit["single_component"]
    & audit["metal_free"]
    & audit["neutral"]
    & (~audit["high_mw"])
    & (~audit["high_logp"])
)


# ============================================================
# POPULATION SUMMARY
# ============================================================

total_n = len(audit)

populations = {
    "Full dataset": np.ones(total_n, dtype=bool),
    "Single-component": audit["candidate_single_component"],
    "Single-component + neutral + metal-free":
        audit["candidate_single_neutral"],
    "Standard analytical domain":
        audit["candidate_standard_domain"],
    "Conservative analytical domain":
        audit["candidate_conservative_domain"],
}


summary_rows = []

for name, mask in populations.items():

    subset = audit.loc[mask]

    summary_rows.append(
        {
            "population": name,
            "n": len(subset),
            "pct_of_full_dataset":
                len(subset) / total_n * 100,
            "mean_solubility":
                subset["Solubility"].mean(),
            "median_solubility":
                subset["Solubility"].median(),
            "mean_abs_residual":
                subset["abs_residual"].mean(),
            "median_abs_residual":
                subset["abs_residual"].median(),
        }
    )

population_summary = pd.DataFrame(summary_rows)


# ============================================================
# EXCLUSION FLAGS
# ============================================================

audit["excluded_multicomponent"] = (
    audit["multicomponent"]
)

audit["excluded_metal"] = (
    audit["has_metal"].fillna(False)
)

audit["excluded_charged"] = (
    audit["is_charged"].fillna(False)
)

audit["excluded_high_mw"] = (
    audit["high_mw"]
)

audit["excluded_high_logp"] = (
    audit["high_logp"]
)


# ============================================================
# OVERLAP SUMMARY
# ============================================================

exclusion_columns = [
    "excluded_multicomponent",
    "excluded_metal",
    "excluded_charged",
    "excluded_high_mw",
    "excluded_high_logp",
]

overlap_rows = []

for column in exclusion_columns:

    mask = audit[column]

    overlap_rows.append(
        {
            "criterion": column,
            "n": int(mask.sum()),
            "pct": mask.mean() * 100,
            "mean_abs_residual":
                audit.loc[mask, "abs_residual"].mean(),
            "median_abs_residual":
                audit.loc[mask, "abs_residual"].median(),
        }
    )

overlap_summary = pd.DataFrame(overlap_rows)


# ============================================================
# TOP EXTREME RECORDS
# ============================================================

top_extremes = (
    audit
    .sort_values("abs_residual", ascending=False)
    [
        [
            "ID",
            "Name",
            "Solubility",
            "PredictedSolubility",
            "Residual",
            "abs_residual",
            "MolWt",
            "MolLogP",
            "has_metal",
            "is_charged",
            "multicomponent_record",
            "component_count",
            "Group",
        ]
    ]
    .head(50)
)


# ============================================================
# SAVE PROCESSED AUDIT
# ============================================================

audit.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# WRITE REPORT
# ============================================================

with open(REPORT_FILE, "w", encoding="utf-8") as f:

    f.write("=" * 70 + "\n")
    f.write("ANALYTICAL DATASET / INCLUSION AUDIT\n")
    f.write("=" * 70 + "\n\n")

    f.write("PURPOSE\n")
    f.write("-" * 70 + "\n")
    f.write(
        "This audit evaluates the chemical domain of the dataset before "
        "construction of the modelling population. It does not delete "
        "or modify the raw observations.\n\n"
    )

    f.write("DATASET SIZE\n")
    f.write("-" * 70 + "\n")
    f.write(f"Full dataset: {total_n:,} rows\n\n")

    f.write("THRESHOLDS\n")
    f.write("-" * 70 + "\n")
    f.write(f"MW 95th percentile: {mw_95:.4f}\n")
    f.write(f"MW 99th percentile: {mw_99:.4f}\n")
    f.write(f"MolLogP 95th percentile: {logp_95:.4f}\n")
    f.write(f"MolLogP 99th percentile: {logp_99:.4f}\n\n")

    f.write("ANALYTICAL POPULATIONS\n")
    f.write("-" * 70 + "\n")
    f.write(
        population_summary.to_string(index=False)
    )
    f.write("\n\n")

    f.write("EXCLUSION CRITERIA\n")
    f.write("-" * 70 + "\n")
    f.write(
        overlap_summary.to_string(index=False)
    )
    f.write("\n\n")

    f.write("TOP 50 ABSOLUTE RESIDUALS\n")
    f.write("-" * 70 + "\n")
    f.write(
        top_extremes.to_string(index=False)
    )
    f.write("\n\n")

    f.write("INTERPRETATION\n")
    f.write("-" * 70 + "\n")
    f.write(
        "The purpose of this analysis is to distinguish chemically "
        "meaningful observations from records whose structure or "
        "composition may place them outside the intended predictive "
        "domain. Population selection should be based on the observed "
        "data and scientific justification rather than residual magnitude "
        "alone.\n"
    )


# ============================================================
# CONSOLE OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("ANALYTICAL POPULATION SUMMARY")
print("=" * 70)

print(
    population_summary.to_string(index=False)
)

print("\n" + "=" * 70)
print("EXCLUSION CRITERIA")
print("=" * 70)

print(
    overlap_summary.to_string(index=False)
)

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)

print("\nReport saved to:")
print(REPORT_FILE)

print("\nProcessed data saved to:")
print(OUTPUT_FILE)