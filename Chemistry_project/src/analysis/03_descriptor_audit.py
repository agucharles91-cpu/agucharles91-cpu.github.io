import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from scipy import stats


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESIDUAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "baseline_residuals.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "descriptor_audit.txt"
)


# ---------------------------------------------------------------------
# DATABASE CONNECTION
# ---------------------------------------------------------------------

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "postgres",
    "user": "postgres",
    "password": os.getenv("POSTGRES_PASSWORD"),
}


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def fmt(x):
    if pd.isna(x):
        return "NA"
    return f"{x:.4f}"


# ---------------------------------------------------------------------
# START
# ---------------------------------------------------------------------

section("DESCRIPTOR DISCREPANCY AUDIT")

print("\nLoading baseline residuals...")

residuals = pd.read_csv(RESIDUAL_PATH)

print(f"Residual rows: {len(residuals):,}")


# ---------------------------------------------------------------------
# LOAD COMPLETE DATASET FROM POSTGRESQL
# ---------------------------------------------------------------------

print("\nConnecting to PostgreSQL...")

conn = psycopg2.connect(**DB_CONFIG)

query = """
SELECT
    c.compound_id AS "ID",
    c.name AS "Name",
    c.smiles AS "SMILES",
    c.inchikey AS "InChIKey",

    md.mol_wt AS "MolWt",
    md.mol_logp AS "MolLogP",
    md.mol_mr AS "MolMR",
    md.heavy_atom_count AS "HeavyAtomCount",
    md.num_h_acceptors AS "NumHAcceptors",
    md.num_h_donors AS "NumHDonors",
    md.num_heteroatoms AS "NumHeteroatoms",
    md.num_rotatable_bonds AS "NumRotatableBonds",
    md.num_valence_electrons AS "NumValenceElectrons",
    md.num_aromatic_rings AS "NumAromaticRings",
    md.num_saturated_rings AS "NumSaturatedRings",
    md.num_aliphatic_rings AS "NumAliphaticRings",
    md.ring_count AS "RingCount",
    md.tpsa AS "TPSA",
    md.labute_asa AS "LabuteASA",
    md.balaban_j AS "BalabanJ",
    md.bertz_ct AS "BertzCT"

FROM solubility.compounds c
JOIN solubility.molecular_descriptors md
    ON c.compound_id = md.compound_id
"""

df = pd.read_sql_query(query, conn)

conn.close()

print(f"Descriptor rows: {len(df):,}")


# ---------------------------------------------------------------------
# MERGE WITH BASELINE RESIDUALS
# ---------------------------------------------------------------------

df = df.merge(
    residuals[
        [
            "ID",
            "Solubility",
            "PredictedSolubility",
            "Residual",
            "Ocurrences",
            "Group",
            "SD",
        ]
    ],
    on="ID",
    how="inner",
)

print(f"Merged rows: {len(df):,}")


# ---------------------------------------------------------------------
# DESCRIPTOR DISTRIBUTION AUDIT
# ---------------------------------------------------------------------

descriptors = [
    "MolWt",
    "MolLogP",
    "MolMR",
    "HeavyAtomCount",
    "NumHAcceptors",
    "NumHDonors",
    "NumHeteroatoms",
    "NumRotatableBonds",
    "NumValenceElectrons",
    "NumAromaticRings",
    "NumSaturatedRings",
    "NumAliphaticRings",
    "RingCount",
    "TPSA",
    "LabuteASA",
    "BalabanJ",
    "BertzCT",
]

section("DESCRIPTOR DISTRIBUTIONS")

summary = df[descriptors].describe().T

summary["missing"] = df[descriptors].isna().sum()
summary["zero"] = (df[descriptors] == 0).sum()
summary["negative"] = (df[descriptors] < 0).sum()

summary = summary[
    [
        "count",
        "missing",
        "zero",
        "negative",
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
    ]
]

print(
    summary.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


# ---------------------------------------------------------------------
# RDKit STRUCTURE VALIDATION
# ---------------------------------------------------------------------

section("RDKit STRUCTURE VALIDATION")

print("\nParsing SMILES with RDKit...")

df["mol"] = df["SMILES"].apply(Chem.MolFromSmiles)

invalid = df["mol"].isna()

print(f"Total structures: {len(df):,}")
print(f"Valid structures: {(~invalid).sum():,}")
print(f"Invalid structures: {invalid.sum():,}")

if invalid.any():
    print("\nInvalid structures:")
    print(
        df.loc[
            invalid,
            ["ID", "Name", "SMILES"]
        ].to_string(index=False)
    )


# ---------------------------------------------------------------------
# RECOMPUTE KEY DESCRIPTORS
# ---------------------------------------------------------------------

valid = df.loc[~invalid].copy()

print("\nRecomputing key descriptors with RDKit...")

valid["RDKit_MolWt"] = valid["mol"].apply(
    Descriptors.MolWt
)

valid["RDKit_MolLogP"] = valid["mol"].apply(
    Crippen.MolLogP
)

valid["RDKit_TPSA"] = valid["mol"].apply(
    rdMolDescriptors.CalcTPSA
)

valid["RDKit_NumHAcceptors"] = valid["mol"].apply(
    Lipinski.NumHAcceptors
)

valid["RDKit_NumHDonors"] = valid["mol"].apply(
    Lipinski.NumHDonors
)

valid["RDKit_NumRotatableBonds"] = valid["mol"].apply(
    Lipinski.NumRotatableBonds
)


# ---------------------------------------------------------------------
# CALCULATE DISCREPANCIES
# ---------------------------------------------------------------------

valid["MolWt_diff"] = (
    valid["MolWt"] - valid["RDKit_MolWt"]
).abs()

valid["MolLogP_diff"] = (
    valid["MolLogP"] - valid["RDKit_MolLogP"]
).abs()

valid["TPSA_diff"] = (
    valid["TPSA"] - valid["RDKit_TPSA"]
).abs()

valid["HBA_diff"] = (
    valid["NumHAcceptors"]
    - valid["RDKit_NumHAcceptors"]
).abs()

valid["HBD_diff"] = (
    valid["NumHDonors"]
    - valid["RDKit_NumHDonors"]
).abs()

valid["Rotatable_diff"] = (
    valid["NumRotatableBonds"]
    - valid["RDKit_NumRotatableBonds"]
).abs()


# ---------------------------------------------------------------------
# DISCREPANCY SUMMARY
# ---------------------------------------------------------------------

section("RDKit vs DATASET DESCRIPTOR DISCREPANCIES")

comparison = {
    "MolWt": "MolWt_diff",
    "MolLogP": "MolLogP_diff",
    "TPSA": "TPSA_diff",
    "NumHAcceptors": "HBA_diff",
    "NumHDonors": "HBD_diff",
    "NumRotatableBonds": "Rotatable_diff",
}

print(
    f"\n{'Descriptor':<22}"
    f"{'Mean abs diff':<18}"
    f"{'Median abs diff':<20}"
    f"{'Max abs diff':<18}"
)

print("-" * 78)

for descriptor, diff_col in comparison.items():

    print(
        f"{descriptor:<22}"
        f"{valid[diff_col].mean():<18.6f}"
        f"{valid[diff_col].median():<20.6f}"
        f"{valid[diff_col].max():<18.6f}"
    )


# ---------------------------------------------------------------------
# RELATIONSHIP BETWEEN DISCREPANCY AND RESIDUAL MAGNITUDE
# ---------------------------------------------------------------------

section("DESCRIPTOR DISCREPANCY vs RESIDUAL MAGNITUDE")

valid["abs_residual"] = valid["Residual"].abs()

print(
    f"\n{'Descriptor discrepancy':<25}"
    f"{'Spearman rho':<18}"
    f"{'p-value':<18}"
)

print("-" * 61)

correlations = []

for descriptor, diff_col in comparison.items():

    temp = valid[[diff_col, "abs_residual"]].dropna()

    rho, p = stats.spearmanr(
        temp[diff_col],
        temp["abs_residual"]
    )

    correlations.append(
        {
            "descriptor": descriptor,
            "rho": rho,
            "p": p,
        }
    )

    print(
        f"{descriptor:<25}"
        f"{rho:<18.4f}"
        f"{p:<18.6g}"
    )


# ---------------------------------------------------------------------
# TOP DISCREPANCY COMPOUNDS
# ---------------------------------------------------------------------

section("LARGEST DESCRIPTOR DISCREPANCIES")

for descriptor, diff_col in comparison.items():

    print(f"\n--- {descriptor} ---")

    cols = [
        "ID",
        "Name",
        descriptor,
        diff_col,
        "Solubility",
        "Residual",
        "Ocurrences",
        "Group",
    ]

    print(
        valid.nlargest(5, diff_col)[cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


# ---------------------------------------------------------------------
# HIGH RESIDUAL + HIGH DISCREPANCY
# ---------------------------------------------------------------------

section("HIGH RESIDUAL + HIGH DESCRIPTOR DISCREPANCY")

# Use the 95th percentile as an objective threshold.
residual_threshold = valid["abs_residual"].quantile(0.95)

print(
    f"\n95th percentile |Residual| threshold: "
    f"{residual_threshold:.4f}"
)

for descriptor, diff_col in comparison.items():

    discrepancy_threshold = valid[diff_col].quantile(0.95)

    candidates = valid[
        (valid["abs_residual"] >= residual_threshold)
        & (valid[diff_col] >= discrepancy_threshold)
    ].copy()

    print(
        f"\n{descriptor}: "
        f"{len(candidates)} compounds"
        f" (95th percentile discrepancy >= "
        f"{discrepancy_threshold:.4f})"
    )

    if len(candidates) > 0:

        print(
            candidates.nlargest(
                10,
                "abs_residual"
            )[
                [
                    "ID",
                    "Name",
                    "Solubility",
                    "PredictedSolubility",
                    "Residual",
                    "abs_residual",
                    descriptor,
                    diff_col,
                    "Ocurrences",
                    "Group",
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}"
            )
        )


# ---------------------------------------------------------------------
# SAVE PROCESSED DISCREPANCIES
# ---------------------------------------------------------------------

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "descriptor_discrepancies.csv"
)

save_columns = [
    "ID",
    "InChIKey",
    "Name",
    "SMILES",
    "Solubility",
    "PredictedSolubility",
    "Residual",
    "abs_residual",
    "Ocurrences",
    "Group",
    "SD",
    "MolWt_diff",
    "MolLogP_diff",
    "TPSA_diff",
    "HBA_diff",
    "HBD_diff",
    "Rotatable_diff",
]

valid[save_columns].to_csv(
    OUTPUT_PATH,
    index=False
)


# ---------------------------------------------------------------------
# SAVE REPORT
# ---------------------------------------------------------------------

with open(REPORT_PATH, "w", encoding="utf-8") as f:

    f.write("DESCRIPTOR DISCREPANCY AUDIT\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Rows analyzed: {len(valid):,}\n")
    f.write(f"Invalid SMILES: {invalid.sum():,}\n\n")

    f.write("DESCRIPTOR DISTRIBUTIONS\n")
    f.write("-" * 70 + "\n\n")

    f.write(
        summary.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    f.write("\n\nRDKit VS DATASET DISCREPANCIES\n")
    f.write("-" * 70 + "\n\n")

    for descriptor, diff_col in comparison.items():

        f.write(
            f"{descriptor}: "
            f"mean={valid[diff_col].mean():.6f}, "
            f"median={valid[diff_col].median():.6f}, "
            f"max={valid[diff_col].max():.6f}\n"
        )

    f.write("\n\nDISCREPANCY VS ABSOLUTE RESIDUAL\n")
    f.write("-" * 70 + "\n\n")

    for item in correlations:

        f.write(
            f"{item['descriptor']}: "
            f"rho={item['rho']:.4f}, "
            f"p={item['p']:.6g}\n"
        )

    f.write("\n\nINVALID STRUCTURES\n")
    f.write("-" * 70 + "\n\n")

    if invalid.any():

        f.write(
            df.loc[
                invalid,
                ["ID", "Name", "SMILES"]
            ].to_string(index=False)
        )

    else:
        f.write("None\n")


# ---------------------------------------------------------------------
# COMPLETE
# ---------------------------------------------------------------------

section("AUDIT COMPLETE")

print(f"\nReport saved to:")
print(REPORT_PATH)

print(f"\nProcessed discrepancy data saved to:")
print(OUTPUT_PATH)