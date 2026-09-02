import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "reliability_discrepancy_popc.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "outlier_triage_popc.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "outlier_triage_popc.txt"


def main():
    print("=" * 70)
    print("OUTLIER TRIAGE — POPULATION C")
    print("=" * 70)

    df = pd.read_csv(INPUT_PATH)
    print(f"\nRows: {len(df):,}")

    # ------------------------------------------------------------
    # THRESHOLDS
    # ------------------------------------------------------------

    residual_95 = df["abs_residual"].quantile(0.95)

    # Only MolLogP_diff and TPSA_diff carried real variance in
    # Population C (MolWt/HBD/Rotatable were effectively constant —
    # see script 10). Use their 95th percentile as a discrepancy flag.
    nonzero_logp_diff = df.loc[df["MolLogP_diff"] > 0, "MolLogP_diff"]
    nonzero_tpsa_diff = df.loc[df["TPSA_diff"] > 0, "TPSA_diff"]

    logp_diff_95 = nonzero_logp_diff.quantile(0.95) if len(nonzero_logp_diff) > 0 else float("inf")
    tpsa_diff_95 = nonzero_tpsa_diff.quantile(0.95) if len(nonzero_tpsa_diff) > 0 else float("inf")
    sd_95 = df["SD"].quantile(0.95)

    print(f"\n|Residual| 95th percentile:     {residual_95:.4f}")
    print(f"MolLogP_diff 95th percentile:   {logp_diff_95:.4f}")
    print(f"TPSA_diff 95th percentile:      {tpsa_diff_95:.4f}")
    print(f"SD 95th percentile:             {sd_95:.4f}")

    # ------------------------------------------------------------
    # RELIABILITY FLAG
    # ------------------------------------------------------------
    # A compound is flagged "low reliability" if:
    #   - it has only 1 occurrence (no cross-source verification), OR
    #   - its SD is unusually high among multi-occurrence compounds, OR
    #   - its LogP or TPSA discrepancy (RDKit vs dataset) is in the
    #     top 5%, indicating computational ambiguity in the inputs
    #     the baseline model actually used.

    df["flag_single_occurrence"] = df["Ocurrences"] == 1
    df["flag_high_sd"] = df["SD"] >= sd_95
    df["flag_high_logp_discrepancy"] = (df["MolLogP_diff"] > 0) & (df["MolLogP_diff"] >= logp_diff_95)
    df["flag_high_tpsa_discrepancy"] = (df["TPSA_diff"] > 0) & (df["TPSA_diff"] >= tpsa_diff_95)
    df["low_reliability"] = (
        df["flag_single_occurrence"]
        | df["flag_high_sd"]
        | df["flag_high_logp_discrepancy"]
        | df["flag_high_tpsa_discrepancy"]
    )

    # ------------------------------------------------------------
    # BUCKET ASSIGNMENT
    # ------------------------------------------------------------

    df["extreme_residual"] = df["abs_residual"] >= residual_95

    def assign_bucket(row):
        if row["extreme_residual"] and not row["low_reliability"]:
            return "2_high_reliability_surprising"
        elif row["extreme_residual"] and row["low_reliability"]:
            return "1_low_reliability"
        else:
            return "3_not_extreme"

    df["bucket"] = df.apply(assign_bucket, axis=1)

    # ------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------

    bucket_summary = (
        df.groupby("bucket")
          .agg(
              n=("ID", "count"),
              mean_abs_residual=("abs_residual", "mean"),
              median_abs_residual=("abs_residual", "median"),
          )
          .reset_index()
    )

    print("\n" + "=" * 70)
    print("BUCKET SUMMARY")
    print("=" * 70)
    print(bucket_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ------------------------------------------------------------
    # BUCKET 2 — the real candidates for structural investigation
    # ------------------------------------------------------------

    bucket2 = (
        df[df["bucket"] == "2_high_reliability_surprising"]
        .sort_values("abs_residual", ascending=False)
    )

    print("\n" + "=" * 70)
    print(f"BUCKET 2 — HIGH RELIABILITY, STILL SURPRISING (n={len(bucket2)})")
    print("=" * 70)
    print(
        bucket2[[
            "ID", "Solubility", "PredictedSolubility", "Residual",
            "Ocurrences", "Group", "SD", "MolLogP_diff", "TPSA_diff",
        ]]
        .head(20)
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    # ------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------

    df.to_csv(OUTPUT_PATH, index=False)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("OUTLIER TRIAGE — POPULATION C\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Rows analyzed: {len(df):,}\n\n")

        f.write("THRESHOLDS\n" + "-" * 70 + "\n")
        f.write(f"|Residual| 95th percentile: {residual_95:.4f}\n")
        f.write(f"MolLogP_diff 95th percentile: {logp_diff_95:.4f}\n")
        f.write(f"TPSA_diff 95th percentile: {tpsa_diff_95:.4f}\n")
        f.write(f"SD 95th percentile: {sd_95:.4f}\n\n")

        f.write("BUCKET SUMMARY\n" + "-" * 70 + "\n")
        f.write(bucket_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        f.write("\n\n")

        f.write(f"BUCKET 2 — HIGH RELIABILITY, STILL SURPRISING (n={len(bucket2)})\n")
        f.write("-" * 70 + "\n")
        f.write(
            bucket2[[
                "ID", "Solubility", "PredictedSolubility", "Residual",
                "Ocurrences", "Group", "SD", "MolLogP_diff", "TPSA_diff",
            ]]
            .to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )
        f.write("\n\n")

        f.write("INTERPRETATION\n" + "-" * 70 + "\n")
        f.write(
            "Bucket 2 compounds show large deviation from the obvious-chemistry "
            "baseline (MolWt + MolLogP) that cannot be attributed to single-"
            "occurrence measurement, high measurement variability, or "
            "descriptor computational ambiguity. These are the strongest "
            "candidates for structural/chemical investigation. Bucket 1 "
            "compounds show similarly large residuals but carry a specific "
            "reliability caveat and should be interpreted cautiously rather "
            "than treated as confirmed chemical findings.\n"
        )

    print("\n" + "=" * 70)
    print("TRIAGE COMPLETE")
    print("=" * 70)
    print(f"\nData saved to:\n{OUTPUT_PATH}")
    print(f"\nReport saved to:\n{REPORT_PATH}")


if __name__ == "__main__":
    main()