from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import (
    Descriptors,
    Crippen,
    Lipinski,
    rdMolDescriptors,
    GraphDescriptors,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chemical_domain_audit.csv"
)

OUTPUT_DATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "molecular_features.csv"
)

OUTPUT_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "molecular_feature_engineering.txt"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("MOLECULAR FEATURE ENGINEERING")
print("=" * 70)

print("\nLoading chemical domain audit...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df):,}")
print(f"Columns loaded: {len(df.columns):,}")


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "SMILES",
    "Solubility",
    "has_metal",
    "is_charged",
    "component_count",
    "MolWt",
    "MolLogP",
]

missing = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing:
    raise KeyError(
        "Required columns missing from "
        "chemical_domain_audit.csv: "
        + ", ".join(missing)
    )


# ============================================================
# RECONSTRUCT STANDARD ANALYTICAL DOMAIN
# ============================================================

print("\nReconstructing standard analytical domain...")

# IMPORTANT:
# This reproduces the exact Population C definition from
# 05_analytical_dataset_audit.py.
#
# Standard analytical domain:
#   1. Single-component structure
#   2. Metal-free
#   3. Neutral
#   4. Exclude very-high MW observations
#   5. Exclude very-high MolLogP observations
#
# The thresholds are calculated from the same full dataset,
# exactly as in script 05.

mw_99 = df["MolWt"].quantile(0.99)
logp_99 = df["MolLogP"].quantile(0.99)

print(
    f"MolWt 99th percentile:   {mw_99:.6f}"
)

print(
    f"MolLogP 99th percentile: {logp_99:.6f}"
)


# ------------------------------------------------------------
# Reproduce domain flags from script 05
# ------------------------------------------------------------

df["single_component"] = (
    df["component_count"] == 1
)

df["metal_free"] = (
    ~df["has_metal"].fillna(False)
)

df["neutral"] = (
    ~df["is_charged"].fillna(False)
)

df["very_high_mw"] = (
    df["MolWt"] >= mw_99
)

df["very_high_logp"] = (
    df["MolLogP"] >= logp_99
)


# ------------------------------------------------------------
# Exact standard analytical domain
# ------------------------------------------------------------

df["standard_analytical_domain"] = (
    df["single_component"]
    & df["metal_free"]
    & df["neutral"]
    & (~df["very_high_mw"])
    & (~df["very_high_logp"])
)


standard_n = int(
    df["standard_analytical_domain"].sum()
)

print(
    f"Standard analytical-domain rows: "
    f"{standard_n:,}"
)


# ============================================================
# VERIFY ANALYTICAL POPULATION
# ============================================================

EXPECTED_STANDARD_N = 8643

if standard_n != EXPECTED_STANDARD_N:

    raise ValueError(
        "\nStandard analytical-domain population does not "
        "match the audited population.\n\n"
        f"Expected: {EXPECTED_STANDARD_N:,}\n"
        f"Observed: {standard_n:,}\n\n"
        "Feature engineering has been stopped because the "
        "analytical population must match the prior audit."
    )

print(
    "\nAnalytical population verified: "
    f"{standard_n:,} rows"
)


# ============================================================
# FILTER ANALYTICAL DATASET
# ============================================================

analysis_df = df[
    df["standard_analytical_domain"]
].copy()

print(
    f"\nRows entering feature engineering: "
    f"{len(analysis_df):,}"
)


# ============================================================
# RDKit STRUCTURE PARSING
# ============================================================

print("\nParsing structures with RDKit...")


def parse_molecule(smiles):

    if pd.isna(smiles):
        return None

    return Chem.MolFromSmiles(str(smiles))


analysis_df["rdkit_mol"] = (
    analysis_df["SMILES"]
    .apply(parse_molecule)
)

analysis_df["rdkit_valid_structure"] = (
    analysis_df["rdkit_mol"].notna()
)

valid_count = int(
    analysis_df["rdkit_valid_structure"].sum()
)

invalid_count = (
    len(analysis_df) - valid_count
)

print(
    f"Valid structures:   {valid_count:,}"
)

print(
    f"Invalid structures: {invalid_count:,}"
)


# ============================================================
# DESCRIPTOR CALCULATION
# ============================================================

print("\nCalculating molecular descriptors...")


def calculate_descriptors(mol):

    if mol is None:
        return {}

    return {
        "rdkit_molwt":
            Descriptors.MolWt(mol),

        "rdkit_mollogp":
            Crippen.MolLogP(mol),

        "rdkit_tpsa":
            rdMolDescriptors.CalcTPSA(mol),

        "rdkit_hbd":
            Lipinski.NumHDonors(mol),

        "rdkit_hba":
            Lipinski.NumHAcceptors(mol),

        "rdkit_rotatable_bonds":
            Lipinski.NumRotatableBonds(mol),

        "rdkit_heavy_atom_count":
            Lipinski.HeavyAtomCount(mol),

        "rdkit_heteroatom_count":
            Lipinski.NumHeteroatoms(mol),

        "rdkit_aromatic_rings":
            rdMolDescriptors.CalcNumAromaticRings(mol),

        "rdkit_saturated_rings":
            rdMolDescriptors.CalcNumSaturatedRings(mol),

        "rdkit_aliphatic_rings":
            rdMolDescriptors.CalcNumAliphaticRings(mol),

        "rdkit_ring_count":
            rdMolDescriptors.CalcNumRings(mol),

        "rdkit_fraction_csp3":
            rdMolDescriptors.CalcFractionCSP3(mol),

        "rdkit_molmr":
            Crippen.MolMR(mol),

        "rdkit_labute_asa":
            rdMolDescriptors.CalcLabuteASA(mol),

        "rdkit_balaban_j":
            GraphDescriptors.BalabanJ(mol),

        "rdkit_bertz_ct":
            GraphDescriptors.BertzCT(mol),
    }


descriptor_records = [
    calculate_descriptors(mol)
    for mol in analysis_df["rdkit_mol"]
]

descriptor_df = pd.DataFrame(
    descriptor_records
)

analysis_df = pd.concat(
    [
        analysis_df.reset_index(drop=True),
        descriptor_df.reset_index(drop=True),
    ],
    axis=1,
)


# ============================================================
# REMOVE RDKit MOLECULE OBJECT
# ============================================================

analysis_df.drop(
    columns=["rdkit_mol"],
    inplace=True
)


# ============================================================
# DESCRIPTOR VALIDATION
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


missing_summary = (
    analysis_df[descriptor_columns]
    .isna()
    .sum()
    .sort_values(ascending=False)
)

summary = (
    analysis_df[descriptor_columns]
    .describe()
    .T
)

summary["missing"] = missing_summary

summary["missing_pct"] = (
    summary["missing"]
    / len(analysis_df)
    * 100
)


# ============================================================
# COMPARE ORIGINAL AND RDKit DESCRIPTORS
# ============================================================

comparison_pairs = {
    "MolWt": "rdkit_molwt",
    "MolLogP": "rdkit_mollogp",
    "TPSA": "rdkit_tpsa",
    "NumHDonors": "rdkit_hbd",
    "NumHAcceptors": "rdkit_hba",
    "NumRotatableBonds": "rdkit_rotatable_bonds",
    "HeavyAtomCount": "rdkit_heavy_atom_count",
    "NumHeteroatoms": "rdkit_heteroatom_count",
    "NumAromaticRings": "rdkit_aromatic_rings",
    "NumSaturatedRings": "rdkit_saturated_rings",
    "NumAliphaticRings": "rdkit_aliphatic_rings",
    "RingCount": "rdkit_ring_count",
    "MolMR": "rdkit_molmr",
    "LabuteASA": "rdkit_labute_asa",
    "BalabanJ": "rdkit_balaban_j",
    "BertzCT": "rdkit_bertz_ct",
}


comparisons = []

for existing, calculated in comparison_pairs.items():

    if existing not in analysis_df.columns:
        continue

    comparison = (
        analysis_df[
            [existing, calculated]
        ]
        .dropna()
    )

    if len(comparison) == 0:
        continue

    difference = (
        comparison[existing]
        - comparison[calculated]
    )

    comparisons.append(
        {
            "existing_descriptor":
                existing,

            "calculated_descriptor":
                calculated,

            "n_compared":
                len(comparison),

            "mean_absolute_difference":
                difference.abs().mean(),

            "median_absolute_difference":
                difference.abs().median(),

            "max_absolute_difference":
                difference.abs().max(),
        }
    )


comparison_df = pd.DataFrame(
    comparisons
)


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

OUTPUT_DATA.parent.mkdir(
    parents=True,
    exist_ok=True
)

analysis_df.to_csv(
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
        "MOLECULAR FEATURE ENGINEERING REPORT\n"
    )

    report.write(
        "=" * 70 + "\n\n"
    )

    report.write(
        f"Full dataset rows: "
        f"{len(df):,}\n"
    )

    report.write(
        f"Standard analytical-domain rows: "
        f"{len(analysis_df):,}\n"
    )

    report.write(
        f"MolWt 99th percentile: "
        f"{mw_99:.6f}\n"
    )

    report.write(
        f"MolLogP 99th percentile: "
        f"{logp_99:.6f}\n"
    )

    report.write(
        f"Valid RDKit structures: "
        f"{valid_count:,}\n"
    )

    report.write(
        f"Invalid RDKit structures: "
        f"{invalid_count:,}\n\n"
    )

    report.write(
        "=" * 70 + "\n"
    )

    report.write(
        "DESCRIPTOR SUMMARY\n"
    )

    report.write(
        "=" * 70 + "\n\n"
    )

    report.write(
        summary.to_string(
            float_format=lambda x: f"{x:.6f}"
        )
    )

    report.write("\n\n")

    report.write(
        "=" * 70 + "\n"
    )

    report.write(
        "COMPARISON WITH ORIGINAL DESCRIPTORS\n"
    )

    report.write(
        "=" * 70 + "\n\n"
    )

    if not comparison_df.empty:

        report.write(
            comparison_df.to_string(
                index=False,
                float_format=lambda x: f"{x:.10f}"
            )
        )

    else:

        report.write(
            "No comparable descriptor columns found.\n"
        )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 70)

print(
    f"\nAnalytical rows processed: "
    f"{len(analysis_df):,}"
)

print(
    f"Valid structures: "
    f"{valid_count:,}"
)

print(
    f"Invalid structures: "
    f"{invalid_count:,}"
)

print("\nDescriptor missingness:")

print(
    missing_summary.to_string()
)

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 70)

print("\nDataset saved to:")
print(OUTPUT_DATA)

print("\nReport saved to:")
print(OUTPUT_REPORT)