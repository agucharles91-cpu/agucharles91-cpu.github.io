import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from rdkit import Chem


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "baseline_residuals.csv"
)

DISCREPANCY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "descriptor_discrepancies.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "chemical_domain_audit.txt"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chemical_domain_audit.csv"
)


# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------

print("=" * 70)
print("CHEMICAL DOMAIN / RESIDUAL AUDIT")
print("=" * 70)

print("\nLoading baseline residuals...")
df = pd.read_csv(INPUT_PATH)

print(f"Residual rows: {len(df):,}")

if DISCREPANCY_PATH.exists():
    discrepancy = pd.read_csv(DISCREPANCY_PATH)

    # Keep only columns that are not already present.
    extra_columns = [
        c for c in discrepancy.columns
        if c not in df.columns
    ]

    if extra_columns:
        df = df.merge(
            discrepancy[["ID"] + extra_columns],
            on="ID",
            how="left",
            validate="one_to_one",
        )

print(f"Analysis rows: {len(df):,}")


# ---------------------------------------------------------------------
# REQUIRED COLUMNS
# ---------------------------------------------------------------------

required = [
    "ID",
    "Name",
    "SMILES",
    "Solubility",
    "PredictedSolubility",
    "Residual",
    "MolWt",
    "MolLogP",
    "Group",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        f"Required columns missing from input: {missing}"
    )


# ---------------------------------------------------------------------
# BASIC RESIDUAL METRICS
# ---------------------------------------------------------------------

df["abs_residual"] = df["Residual"].abs()


# ---------------------------------------------------------------------
# RDKit STRUCTURE ANALYSIS
# ---------------------------------------------------------------------

print("\nParsing structures with RDKit...")

molecules = []
valid_flags = []

for smiles in df["SMILES"].fillna(""):
    mol = Chem.MolFromSmiles(smiles)

    molecules.append(mol)
    valid_flags.append(mol is not None)

df["rdkit_valid"] = valid_flags


# ---------------------------------------------------------------------
# CHEMICAL DOMAIN FLAGS
# ---------------------------------------------------------------------

# Common metal elements.
# This is intentionally explicit rather than relying on atomic-number
# ranges because metalloids and unusual inorganic chemistry require
# separate interpretation.
METALS = {
    "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V",
    "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Rb", "Sr",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm",
    "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
    "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb",
    "Bi", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"
}


def has_metal(mol):
    if mol is None:
        return False

    return any(
        atom.GetSymbol() in METALS
        for atom in mol.GetAtoms()
    )


def formal_charge(mol):
    if mol is None:
        return np.nan

    return sum(
        atom.GetFormalCharge()
        for atom in mol.GetAtoms()
    )


def atom_count(mol):
    if mol is None:
        return np.nan

    return mol.GetNumAtoms()


def heavy_atom_count(mol):
    if mol is None:
        return np.nan

    return mol.GetNumHeavyAtoms()


def component_count(mol):
    if mol is None:
        return np.nan

    return len(Chem.GetMolFrags(mol))


def carbon_count(mol):
    if mol is None:
        return np.nan

    return sum(
        atom.GetSymbol() == "C"
        for atom in mol.GetAtoms()
    )


df["has_metal"] = [
    has_metal(mol)
    for mol in molecules
]

df["formal_charge"] = [
    formal_charge(mol)
    for mol in molecules
]

df["is_charged"] = (
    df["formal_charge"].fillna(0) != 0
)

df["atom_count"] = [
    atom_count(mol)
    for mol in molecules
]

df["heavy_atom_count_rdkit"] = [
    heavy_atom_count(mol)
    for mol in molecules
]

df["component_count"] = [
    component_count(mol)
    for mol in molecules
]

df["is_multicomponent"] = (
    df["component_count"].fillna(1) > 1
)

df["carbon_count"] = [
    carbon_count(mol)
    for mol in molecules
]


# ---------------------------------------------------------------------
# STRUCTURAL / SIZE FLAGS
# ---------------------------------------------------------------------

# Use dataset-derived thresholds rather than arbitrary chemistry
# thresholds where possible.

mw_95 = df["MolWt"].quantile(0.95)
mw_99 = df["MolWt"].quantile(0.99)

logp_95 = df["MolLogP"].quantile(0.95)
logp_99 = df["MolLogP"].quantile(0.99)

atom_95 = df["heavy_atom_count_rdkit"].quantile(0.95)
atom_99 = df["heavy_atom_count_rdkit"].quantile(0.99)

df["high_mw_95"] = df["MolWt"] >= mw_95
df["high_mw_99"] = df["MolWt"] >= mw_99

df["high_logp_95"] = df["MolLogP"] >= logp_95
df["high_logp_99"] = df["MolLogP"] >= logp_99

df["large_structure_95"] = (
    df["heavy_atom_count_rdkit"] >= atom_95
)

df["large_structure_99"] = (
    df["heavy_atom_count_rdkit"] >= atom_99
)


# ---------------------------------------------------------------------
# MIXTURE / MULTICOMPONENT FLAG
# ---------------------------------------------------------------------

# Multiple disconnected molecular components in the SMILES are treated
# as a structural indicator of a mixture/salt/multicomponent record.
#
# We retain the broader term "multicomponent" rather than automatically
# calling every such structure a mixture.

df["multicomponent_record"] = df["is_multicomponent"]


# ---------------------------------------------------------------------
# RESIDUAL EXTREME FLAG
# ---------------------------------------------------------------------

residual_95 = df["abs_residual"].quantile(0.95)
residual_99 = df["abs_residual"].quantile(0.99)

df["extreme_residual_95"] = (
    df["abs_residual"] >= residual_95
)

df["extreme_residual_99"] = (
    df["abs_residual"] >= residual_99
)


# ---------------------------------------------------------------------
# DUPLICATE / REPEATED OBSERVATION INFORMATION
# ---------------------------------------------------------------------

id_counts = df["ID"].value_counts()

df["id_occurrences"] = (
    df["ID"].map(id_counts)
)

df["repeated_id"] = (
    df["id_occurrences"] > 1
)

smiles_counts = df["SMILES"].value_counts()

df["smiles_occurrences"] = (
    df["SMILES"].map(smiles_counts)
)

df["repeated_smiles"] = (
    df["smiles_occurrences"] > 1
)


# ---------------------------------------------------------------------
# DOMAIN SUMMARY FUNCTION
# ---------------------------------------------------------------------

domain_columns = {
    "RDKit invalid": ~df["rdkit_valid"],
    "Metal-containing": df["has_metal"],
    "Charged": df["is_charged"],
    "Multicomponent": df["multicomponent_record"],
    "High MW (95th percentile)": df["high_mw_95"],
    "Very high MW (99th percentile)": df["high_mw_99"],
    "High MolLogP (95th percentile)": df["high_logp_95"],
    "Very high MolLogP (99th percentile)": df["high_logp_99"],
    "Large structure (95th percentile)": df["large_structure_95"],
    "Very large structure (99th percentile)": df["large_structure_99"],
    "Repeated ID": df["repeated_id"],
    "Repeated SMILES": df["repeated_smiles"],
    "Extreme residual (95th percentile)": df["extreme_residual_95"],
    "Extreme residual (99th percentile)": df["extreme_residual_99"],
}


def summarize_domain(mask):

    subset = df.loc[mask]

    if len(subset) == 0:
        return {
            "n": 0,
            "pct": 0,
            "mean_abs_residual": np.nan,
            "median_abs_residual": np.nan,
            "mean_residual": np.nan,
            "median_residual": np.nan,
            "extreme95_pct": np.nan,
            "extreme99_pct": np.nan,
        }

    return {
        "n": len(subset),
        "pct": len(subset) / len(df) * 100,
        "mean_abs_residual": subset["abs_residual"].mean(),
        "median_abs_residual": subset["abs_residual"].median(),
        "mean_residual": subset["Residual"].mean(),
        "median_residual": subset["Residual"].median(),
        "extreme95_pct": subset["extreme_residual_95"].mean() * 100,
        "extreme99_pct": subset["extreme_residual_99"].mean() * 100,
    }


domain_summary = []

for name, mask in domain_columns.items():

    result = summarize_domain(mask)
    result["domain"] = name

    domain_summary.append(result)

domain_summary = pd.DataFrame(domain_summary)

domain_summary = domain_summary[
    [
        "domain",
        "n",
        "pct",
        "mean_abs_residual",
        "median_abs_residual",
        "mean_residual",
        "median_residual",
        "extreme95_pct",
        "extreme99_pct",
    ]
]


# ---------------------------------------------------------------------
# GROUP SUMMARY
# ---------------------------------------------------------------------

group_summary = (
    df.groupby("Group")
    .agg(
        n=("ID", "size"),
        mean_abs_residual=("abs_residual", "mean"),
        median_abs_residual=("abs_residual", "median"),
        mean_residual=("Residual", "mean"),
        median_residual=("Residual", "median"),
    )
    .reset_index()
)


# ---------------------------------------------------------------------
# KRUSKAL-WALLIS TEST ACROSS GROUPS
# ---------------------------------------------------------------------

group_samples = [
    group["abs_residual"].values
    for _, group in df.groupby("Group")
]

if len(group_samples) >= 2:
    kw_stat, kw_p = stats.kruskal(*group_samples)
else:
    kw_stat, kw_p = np.nan, np.nan


# ---------------------------------------------------------------------
# CORRELATIONS WITH RESIDUAL MAGNITUDE
# ---------------------------------------------------------------------

correlation_rows = []

numeric_features = [
    "MolWt",
    "MolLogP",
    "atom_count",
    "heavy_atom_count_rdkit",
    "component_count",
    "carbon_count",
    "id_occurrences",
    "smiles_occurrences",
]

for column in numeric_features:

    valid = df[[column, "abs_residual"]].dropna()

    if len(valid) < 3:
        rho, p = np.nan, np.nan
    elif valid[column].nunique() < 2:
        rho, p = np.nan, np.nan
    else:
        rho, p = stats.spearmanr(
            valid[column],
            valid["abs_residual"],
        )

    correlation_rows.append(
        {
            "feature": column,
            "spearman_rho": rho,
            "p_value": p,
        }
    )

correlations = pd.DataFrame(correlation_rows)


# ---------------------------------------------------------------------
# EXTREME RESIDUAL RECORDS
# ---------------------------------------------------------------------

top_residuals = df.nlargest(
    25,
    "abs_residual",
)[
    [
        "ID",
        "Name",
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
]


# ---------------------------------------------------------------------
# PRINT DOMAIN SUMMARY
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("CHEMICAL DOMAIN SUMMARY")
print("=" * 70)

print(
    domain_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ---------------------------------------------------------------------
# PRINT GROUP SUMMARY
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("RESIDUAL MAGNITUDE BY GROUP")
print("=" * 70)

print(
    group_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print(
    f"\nKruskal-Wallis test across groups:"
    f" H = {kw_stat:.4f}, p = {kw_p:.6g}"
)


# ---------------------------------------------------------------------
# PRINT CORRELATIONS
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("STRUCTURAL FEATURES vs RESIDUAL MAGNITUDE")
print("=" * 70)

print(
    correlations.to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}",
    )
)


# ---------------------------------------------------------------------
# PRINT EXTREME RESIDUALS
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("TOP 25 ABSOLUTE RESIDUALS")
print("=" * 70)

print(
    top_residuals.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ---------------------------------------------------------------------
# SAVE ENRICHED DATA
# ---------------------------------------------------------------------

df.to_csv(
    OUTPUT_PATH,
    index=False,
)


# ---------------------------------------------------------------------
# SAVE REPORT
# ---------------------------------------------------------------------

with open(REPORT_PATH, "w", encoding="utf-8") as f:

    f.write("CHEMICAL DOMAIN / RESIDUAL AUDIT\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Rows analyzed: {len(df):,}\n")
    f.write(f"RDKit-valid structures: {df['rdkit_valid'].sum():,}\n")
    f.write(
        f"RDKit-invalid structures: "
        f"{(~df['rdkit_valid']).sum():,}\n\n"
    )

    f.write("THRESHOLDS\n")
    f.write("-" * 70 + "\n")
    f.write(f"MW 95th percentile: {mw_95:.4f}\n")
    f.write(f"MW 99th percentile: {mw_99:.4f}\n")
    f.write(f"MolLogP 95th percentile: {logp_95:.4f}\n")
    f.write(f"MolLogP 99th percentile: {logp_99:.4f}\n")
    f.write(f"Heavy atoms 95th percentile: {atom_95:.4f}\n")
    f.write(f"Heavy atoms 99th percentile: {atom_99:.4f}\n")
    f.write(f"|Residual| 95th percentile: {residual_95:.4f}\n")
    f.write(f"|Residual| 99th percentile: {residual_99:.4f}\n\n")

    f.write("CHEMICAL DOMAIN SUMMARY\n")
    f.write("-" * 70 + "\n")
    f.write(
        domain_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    f.write("\n\n")

    f.write("RESIDUAL MAGNITUDE BY GROUP\n")
    f.write("-" * 70 + "\n")
    f.write(
        group_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    f.write("\n\n")

    f.write("GROUP DIFFERENCE TEST\n")
    f.write("-" * 70 + "\n")
    f.write(
        f"Kruskal-Wallis H = {kw_stat:.6f}\n"
        f"Kruskal-Wallis p-value = {kw_p:.8g}\n\n"
    )

    f.write("STRUCTURAL FEATURES vs RESIDUAL MAGNITUDE\n")
    f.write("-" * 70 + "\n")
    f.write(
        correlations.to_string(
            index=False,
            float_format=lambda x: f"{x:.6g}",
        )
    )
    f.write("\n\n")

    f.write("TOP 25 ABSOLUTE RESIDUALS\n")
    f.write("-" * 70 + "\n")
    f.write(
        top_residuals.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    f.write("\n")


print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)

print("\nReport saved to:")
print(REPORT_PATH)

print("\nProcessed data saved to:")
print(OUTPUT_PATH)