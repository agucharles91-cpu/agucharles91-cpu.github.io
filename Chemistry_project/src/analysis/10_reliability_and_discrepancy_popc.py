import pandas as pd
from pathlib import Path
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "molecular_features.csv"
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "curated-solubility-dataset.csv"
RESIDUALS_PATH = PROJECT_ROOT / "data" / "processed" / "baseline_residuals_popc.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "reliability_discrepancy_popc.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "reliability_discrepancy_popc.txt"


def main():
    print("=" * 70)
    print("RELIABILITY + DESCRIPTOR DISCREPANCY — POPULATION C")
    print("=" * 70)

    residuals = pd.read_csv(RESIDUALS_PATH)
    features = pd.read_csv(FEATURES_PATH)
    raw = pd.read_csv(RAW_PATH)

    print(f"\nResidual rows: {len(residuals):,}")
    print(f"Feature rows:  {len(features):,}")
    print(f"Raw rows:      {len(raw):,}")

    # Bring in original TPSA/HBA/HBD/RotatableBonds — these never made it
    # past script 01, which only carried MolWt and MolLogP forward.
    features = features.merge(
        raw[["ID", "TPSA", "NumHAcceptors", "NumHDonors", "NumRotatableBonds"]],
        on="ID",
        how="left",
        validate="one_to_one",
    )
    # Merge residuals with original vs RDKit-recomputed descriptor pairs
    df = residuals.merge(
        features[[
            "ID", "MolWt", "MolLogP", "TPSA", "NumHAcceptors",
            "NumHDonors", "NumRotatableBonds", "rdkit_tpsa",
            "rdkit_hba", "rdkit_hbd", "rdkit_rotatable_bonds",
        ]],
        on="ID",
        how="inner",
        validate="one_to_one",
    )

    print(f"Merged rows:   {len(df):,}")

    df["abs_residual"] = df["Residual"].abs()

    # ------------------------------------------------------------
    # PART 1: RELIABILITY (Group / Occurrence / SD)
    # ------------------------------------------------------------

    group_summary = (
        df.groupby("Group")
          .agg(
              compounds=("ID", "count"),
              mean_abs_residual=("abs_residual", "mean"),
              median_abs_residual=("abs_residual", "median"),
          )
          .sort_index()
    )

    occurrence_summary = (
        df.groupby("Ocurrences")
          .agg(
              compounds=("ID", "count"),
              mean_abs_residual=("abs_residual", "mean"),
              median_abs_residual=("abs_residual", "median"),
          )
          .sort_index()
    )

    occurrence_rho, occurrence_p = stats.spearmanr(df["Ocurrences"], df["abs_residual"])

    group_samples = [g["abs_residual"].values for _, g in df.groupby("Group")]
    kruskal_stat, kruskal_p = stats.kruskal(*group_samples)

    sd_df = df.dropna(subset=["SD"])
    sd_rho, sd_p = stats.spearmanr(sd_df["SD"], sd_df["abs_residual"])

    print("\n" + "=" * 70)
    print("RESIDUALS BY GROUP")
    print("=" * 70)
    print(group_summary.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 70)
    print("RESIDUALS BY OCCURRENCE COUNT")
    print("=" * 70)
    print(occurrence_summary.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 70)
    print("RELIABILITY TESTS")
    print("=" * 70)
    print(f"Occurrences vs |Residual|: Spearman rho={occurrence_rho:.4f}, p={occurrence_p:.6g}")
    print(f"Group vs |Residual|:       Kruskal-Wallis H={kruskal_stat:.4f}, p={kruskal_p:.6g}")
    print(f"SD vs |Residual|:          Spearman rho={sd_rho:.4f}, p={sd_p:.6g}")

    # ------------------------------------------------------------
    # PART 2: DESCRIPTOR DISCREPANCY (original vs RDKit-recomputed)
    # ------------------------------------------------------------

    df["MolWt_diff"] = (df["MolWt"] - df["rdkit_molwt"]).abs()
    df["MolLogP_diff"] = (df["MolLogP"] - df["rdkit_mollogp"]).abs()
    df["TPSA_diff"] = (df["TPSA"] - df["rdkit_tpsa"]).abs()
    df["HBA_diff"] = (df["NumHAcceptors"] - df["rdkit_hba"]).abs()
    df["HBD_diff"] = (df["NumHDonors"] - df["rdkit_hbd"]).abs()
    df["Rotatable_diff"] = (df["NumRotatableBonds"] - df["rdkit_rotatable_bonds"]).abs()

    comparison = {
        "MolWt": "MolWt_diff",
        "MolLogP": "MolLogP_diff",
        "TPSA": "TPSA_diff",
        "NumHAcceptors": "HBA_diff",
        "NumHDonors": "HBD_diff",
        "NumRotatableBonds": "Rotatable_diff",
    }

    print("\n" + "=" * 70)
    print("DESCRIPTOR DISCREPANCY vs RESIDUAL MAGNITUDE")
    print("=" * 70)

    discrepancy_results = []

    for descriptor, diff_col in comparison.items():
        temp = df[[diff_col, "abs_residual"]].dropna()
        rho, p = stats.spearmanr(temp[diff_col], temp["abs_residual"])
        discrepancy_results.append({"descriptor": descriptor, "rho": rho, "p": p})
        print(f"{descriptor:<20} rho={rho:.4f}  p={p:.6g}")

    # ------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------

    df.to_csv(OUTPUT_PATH, index=False)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("RELIABILITY + DESCRIPTOR DISCREPANCY — POPULATION C\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Rows analyzed: {len(df):,}\n\n")

        f.write("RESIDUALS BY GROUP\n" + "-" * 70 + "\n")
        f.write(group_summary.to_string(float_format=lambda x: f"{x:.4f}") + "\n\n")

        f.write("RESIDUALS BY OCCURRENCE COUNT\n" + "-" * 70 + "\n")
        f.write(occurrence_summary.to_string(float_format=lambda x: f"{x:.4f}") + "\n\n")

        f.write("RELIABILITY TESTS\n" + "-" * 70 + "\n")
        f.write(f"Occurrences vs |Residual|: rho={occurrence_rho:.4f}, p={occurrence_p:.6g}\n")
        f.write(f"Group vs |Residual|: H={kruskal_stat:.4f}, p={kruskal_p:.6g}\n")
        f.write(f"SD vs |Residual|: rho={sd_rho:.4f}, p={sd_p:.6g}\n\n")

        f.write("DESCRIPTOR DISCREPANCY vs RESIDUAL MAGNITUDE\n" + "-" * 70 + "\n")
        for item in discrepancy_results:
            f.write(f"{item['descriptor']}: rho={item['rho']:.4f}, p={item['p']:.6g}\n")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nData saved to:\n{OUTPUT_PATH}")
    print(f"\nReport saved to:\n{REPORT_PATH}")


if __name__ == "__main__":
    main()