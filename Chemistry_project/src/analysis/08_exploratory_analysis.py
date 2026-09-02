from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

OUTPUT_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "exploratory_analysis.txt"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "eda"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("EXPLORATORY CHEMICAL / SOLUBILITY ANALYSIS")
print("=" * 70)

print("\nLoading molecular feature dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df):,}")
print(f"Columns loaded: {len(df.columns):,}")


# ============================================================
# VALIDATION
# ============================================================

required_columns = [
    "Solubility",
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

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise KeyError(
        "Required columns missing from molecular_features.csv: "
        + ", ".join(missing)
    )


if df["Solubility"].isna().any():
    raise ValueError(
        "Target variable contains missing values."
    )


# ============================================================
# DESCRIPTOR LIST
# ============================================================

descriptor_columns = [
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


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\nAnalyzing target distribution...")

target = df["Solubility"]

target_summary = pd.Series(
    {
        "n": len(target),
        "mean": target.mean(),
        "median": target.median(),
        "std": target.std(),
        "min": target.min(),
        "q01": target.quantile(0.01),
        "q05": target.quantile(0.05),
        "q25": target.quantile(0.25),
        "q75": target.quantile(0.75),
        "q95": target.quantile(0.95),
        "q99": target.quantile(0.99),
        "max": target.max(),
        "skewness": target.skew(),
        "unique_values": target.nunique(),
    }
)


# ============================================================
# TARGET DUPLICATION / REPEATED VALUES
# ============================================================

target_frequency = (
    df["Solubility"]
    .value_counts()
    .rename_axis("Solubility")
    .reset_index(name="count")
)

target_frequency["pct"] = (
    target_frequency["count"]
    / len(df)
    * 100
)

top_target_values = (
    target_frequency
    .head(20)
)


# ============================================================
# DESCRIPTOR SUMMARY
# ============================================================

descriptor_summary = (
    df[descriptor_columns]
    .describe()
    .T
)

descriptor_summary["skewness"] = (
    df[descriptor_columns]
    .skew()
)

descriptor_summary["n_unique"] = (
    df[descriptor_columns]
    .nunique()
)


# ============================================================
# SPEARMAN CORRELATIONS
# ============================================================

print("\nCalculating descriptor-target correlations...")

target_correlations = (
    df[descriptor_columns + ["Solubility"]]
    .corr(method="spearman")["Solubility"]
    .drop("Solubility")
    .sort_values()
)

target_correlation_table = pd.DataFrame(
    {
        "feature": target_correlations.index,
        "spearman_rho": target_correlations.values,
    }
)

target_correlation_table["absolute_rho"] = (
    target_correlation_table["spearman_rho"].abs()
)

target_correlation_table = (
    target_correlation_table
    .sort_values(
        "absolute_rho",
        ascending=False
    )
)


# ============================================================
# CHEMICAL SIZE / LIPOPHILICITY BINS
# ============================================================

print("\nAnalyzing solubility across molecular property bins...")


def quantile_bin_analysis(column):

    temp = df[[column, "Solubility"]].copy()

    temp["bin"] = pd.qcut(
        temp[column],
        q=5,
        duplicates="drop"
    )

    result = (
        temp.groupby(
            "bin",
            observed=True
        )
        .agg(
            n=("Solubility", "size"),
            mean_property=(column, "mean"),
            median_solubility=("Solubility", "median"),
            mean_solubility=("Solubility", "mean"),
            std_solubility=("Solubility", "std"),
        )
        .reset_index()
    )

    return result


mw_bins = quantile_bin_analysis(
    "rdkit_molwt"
)

logp_bins = quantile_bin_analysis(
    "rdkit_mollogp"
)

tpsa_bins = quantile_bin_analysis(
    "rdkit_tpsa"
)


# ============================================================
# STRUCTURAL FEATURE GROUP ANALYSIS
# ============================================================

print("\nAnalyzing structural feature groups...")


def grouped_summary(column):

    result = (
        df.groupby(column)["Solubility"]
        .agg(
            n="size",
            mean="mean",
            median="median",
            std="std",
        )
        .reset_index()
    )

    return result


hbd_summary = grouped_summary(
    "rdkit_hbd"
)

hba_summary = grouped_summary(
    "rdkit_hba"
)

aromatic_ring_summary = grouped_summary(
    "rdkit_aromatic_rings"
)

ring_summary = grouped_summary(
    "rdkit_ring_count"
)


# ============================================================
# EXTREME OBSERVATIONS
# ============================================================

print("\nInspecting extreme solubility observations...")

lowest_solubility = (
    df[
        [
            "Solubility",
            "rdkit_molwt",
            "rdkit_mollogp",
            "rdkit_tpsa",
            "rdkit_hbd",
            "rdkit_hba",
            "rdkit_ring_count",
            "rdkit_fraction_csp3",
        ]
    ]
    .sort_values("Solubility")
    .head(20)
)

highest_solubility = (
    df[
        [
            "Solubility",
            "rdkit_molwt",
            "rdkit_mollogp",
            "rdkit_tpsa",
            "rdkit_hbd",
            "rdkit_hba",
            "rdkit_ring_count",
            "rdkit_fraction_csp3",
        ]
    ]
    .sort_values(
        "Solubility",
        ascending=False
    )
    .head(20)
)


# ============================================================
# FIGURE 1 — TARGET DISTRIBUTION
# ============================================================

print("\nGenerating target distribution figure...")

plt.figure(figsize=(10, 6))

plt.hist(
    target,
    bins=50
)

plt.xlabel("Aqueous solubility (log S)")
plt.ylabel("Number of observations")
plt.title("Distribution of Measured Aqueous Solubility")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "01_target_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# FIGURE 2 — LOGP VS SOLUBILITY
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["rdkit_mollogp"],
    df["Solubility"],
    alpha=0.35,
    s=12
)

plt.xlabel("RDKit MolLogP")
plt.ylabel("Aqueous solubility (log S)")
plt.title("Lipophilicity vs Aqueous Solubility")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "02_logp_vs_solubility.png",
    dpi=300
)

plt.close()


# ============================================================
# FIGURE 3 — MOLECULAR WEIGHT VS SOLUBILITY
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["rdkit_molwt"],
    df["Solubility"],
    alpha=0.35,
    s=12
)

plt.xlabel("Molecular weight (Da)")
plt.ylabel("Aqueous solubility (log S)")
plt.title("Molecular Weight vs Aqueous Solubility")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "03_mw_vs_solubility.png",
    dpi=300
)

plt.close()


# ============================================================
# FIGURE 4 — TPSA VS SOLUBILITY
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["rdkit_tpsa"],
    df["Solubility"],
    alpha=0.35,
    s=12
)

plt.xlabel("Topological polar surface area (Å²)")
plt.ylabel("Aqueous solubility (log S)")
plt.title("TPSA vs Aqueous Solubility")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "04_tpsa_vs_solubility.png",
    dpi=300
)

plt.close()


# ============================================================
# FIGURE 5 — CORRELATION RANKING
# ============================================================

plt.figure(figsize=(10, 7))

plot_data = (
    target_correlation_table
    .sort_values("spearman_rho")
)

plt.barh(
    plot_data["feature"],
    plot_data["spearman_rho"]
)

plt.xlabel("Spearman correlation with log S")
plt.ylabel("Molecular descriptor")
plt.title("Descriptor–Solubility Relationships")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "05_descriptor_target_correlations.png",
    dpi=300
)

plt.close()


# ============================================================
# FIGURE 6 — SOLUBILITY BY HBD COUNT
# ============================================================

plt.figure(figsize=(10, 6))

groups = (
    df.groupby("rdkit_hbd")["Solubility"]
    .median()
)

plt.plot(
    groups.index,
    groups.values,
    marker="o"
)

plt.xlabel("Hydrogen-bond donor count")
plt.ylabel("Median aqueous solubility (log S)")
plt.title("Median Solubility by Hydrogen-Bond Donor Count")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "06_hbd_vs_solubility.png",
    dpi=300
)

plt.close()


# ============================================================
# FIGURE 7 — SOLUBILITY BY AROMATIC RING COUNT
# ============================================================

plt.figure(figsize=(10, 6))

groups = (
    df.groupby("rdkit_aromatic_rings")["Solubility"]
    .median()
)

plt.plot(
    groups.index,
    groups.values,
    marker="o"
)

plt.xlabel("Aromatic ring count")
plt.ylabel("Median aqueous solubility (log S)")
plt.title("Median Solubility by Aromatic Ring Count")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "07_aromatic_rings_vs_solubility.png",
    dpi=300
)

plt.close()


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

    report.write("=" * 70 + "\n")
    report.write(
        "EXPLORATORY CHEMICAL / SOLUBILITY ANALYSIS\n"
    )
    report.write("=" * 70 + "\n\n")

    report.write("DATASET\n")
    report.write("-" * 70 + "\n")
    report.write(
        f"Rows: {len(df):,}\n"
    )
    report.write(
        f"Descriptors: {len(descriptor_columns):,}\n"
    )
    report.write(
        f"Target: Solubility (log S)\n\n"
    )

    report.write("TARGET DISTRIBUTION\n")
    report.write("-" * 70 + "\n")
    report.write(
        target_summary.to_string()
    )
    report.write("\n\n")

    report.write(
        "MOST FREQUENT TARGET VALUES\n"
    )
    report.write("-" * 70 + "\n")
    report.write(
        top_target_values.to_string(
            index=False
        )
    )
    report.write("\n\n")

    report.write(
        "DESCRIPTOR SUMMARY\n"
    )
    report.write("-" * 70 + "\n")
    report.write(
        descriptor_summary.to_string(
            float_format=lambda x: f"{x:.6f}"
        )
    )
    report.write("\n\n")

    report.write(
        "DESCRIPTOR-TARGET SPEARMAN CORRELATIONS\n"
    )
    report.write("-" * 70 + "\n")
    report.write(
        target_correlation_table.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )
    report.write("\n\n")

    report.write(
        "SOLUBILITY ACROSS MOLECULAR WEIGHT QUINTILES\n"
    )
    report.write("-" * 70 + "\n")
    report.write(
        mw_bins.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )
    report.write("\n\n")

    report.write(
        "SOLUBILITY ACROSS MOLLOGP QUINTILES\n"
    )
    report.write("-" * 70 + "\n")
    report.write(
        logp_bins.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )
    report.write("\n\n")

    report.write(
        "SOLUBILITY ACROSS TPSA QUINTILES\n"
    )
    report.write("-" * 70 + "\n")
    report.write(
        tpsa_bins.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )
    report.write("\n\n")

    report.write(
        "HYDROGEN-BOND DONOR ANALYSIS\n"
    )
    report.write("-" * 70 + "\n")
    report.write(
        hbd_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )
    report.write("\n\n")

    report.write(
        "HYDROGEN-BOND ACCEPTOR ANALYSIS\n"
    )
    report.write("-" * 70 + "\n")
    report.write(
        hba_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )
    report.write("\n\n")

    report.write(
        "AROMATIC RING ANALYSIS\n"
    )
    report.write("-" * 70 + "\n")
    report.write(
        aromatic_ring_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )
    report.write("\n\n")

    report.write(
        "RING COUNT ANALYSIS\n"
    )
    report.write("-" * 70 + "\n")
    report.write(
        ring_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )
    report.write("\n\n")

    report.write(
        "LOWEST-SOLUBILITY OBSERVATIONS\n"
    )
    report.write("-" * 70 + "\n")
    report.write(
        lowest_solubility.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )
    report.write("\n\n")

    report.write(
        "HIGHEST-SOLUBILITY OBSERVATIONS\n"
    )
    report.write("-" * 70 + "\n")
    report.write(
        highest_solubility.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )
    report.write("\n\n")

    report.write("=" * 70 + "\n")
    report.write("INTERPRETATION NOTES\n")
    report.write("=" * 70 + "\n\n")

    report.write(
        "The exploratory analysis characterizes the relationship "
        "between molecular descriptors and measured aqueous "
        "solubility before predictive modelling. Correlations are "
        "descriptive rather than causal. Strong descriptor "
        "intercorrelations indicate potential redundancy among "
        "molecular size and surface-area descriptors. These "
        "relationships will be considered during subsequent "
        "feature selection and model interpretation.\n"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("EXPLORATORY ANALYSIS SUMMARY")
print("=" * 70)

print(
    f"\nDataset rows: {len(df):,}"
)

print(
    "\nStrongest descriptor-target relationships:"
)

print(
    target_correlation_table
    .head(10)
    .to_string(index=False)
)

print("\nFigures saved to:")
print(FIGURE_DIR)

print("\nReport saved to:")
print(OUTPUT_REPORT)

print("\n" + "=" * 70)
print("EXPLORATORY ANALYSIS COMPLETE")
print("=" * 70)