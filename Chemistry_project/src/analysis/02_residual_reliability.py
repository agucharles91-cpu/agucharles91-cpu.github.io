import pandas as pd
from pathlib import Path
from scipy import stats


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

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "residual_reliability.txt"
)


# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------

print("=" * 70)
print("RESIDUAL RELIABILITY ANALYSIS")
print("=" * 70)

df = pd.read_csv(INPUT_PATH)

print(f"\nRows loaded: {len(df):,}")


# ---------------------------------------------------------------------
# PREPARE VARIABLES
# ---------------------------------------------------------------------

df["abs_residual"] = df["Residual"].abs()


# ---------------------------------------------------------------------
# GROUP ANALYSIS
# ---------------------------------------------------------------------

group_summary = (
    df.groupby("Group")
      .agg(
          compounds=("ID", "count"),
          mean_residual=("Residual", "mean"),
          median_residual=("Residual", "median"),
          mean_abs_residual=("abs_residual", "mean"),
          median_abs_residual=("abs_residual", "median"),
          sd_abs_residual=("abs_residual", "std"),
      )
      .sort_index()
)


# ---------------------------------------------------------------------
# OCCURRENCE ANALYSIS
# ---------------------------------------------------------------------

occurrence_summary = (
    df.groupby("Ocurrences")
      .agg(
          compounds=("ID", "count"),
          mean_abs_residual=("abs_residual", "mean"),
          median_abs_residual=("abs_residual", "median"),
          max_abs_residual=("abs_residual", "max"),
      )
      .sort_index()
)


# ---------------------------------------------------------------------
# SPEARMAN CORRELATION
# ---------------------------------------------------------------------

occurrence_corr, occurrence_p = stats.spearmanr(
    df["Ocurrences"],
    df["abs_residual"]
)


# ---------------------------------------------------------------------
# KRUSKAL-WALLIS TEST
# ---------------------------------------------------------------------

group_residuals = [
    group["abs_residual"].values
    for _, group in df.groupby("Group")
]

kruskal_stat, kruskal_p = stats.kruskal(*group_residuals)


# ---------------------------------------------------------------------
# SD ANALYSIS
# ---------------------------------------------------------------------

sd_df = df.dropna(subset=["SD"]).copy()

if len(sd_df) > 1:
    sd_corr, sd_p = stats.spearmanr(
        sd_df["SD"],
        sd_df["abs_residual"]
    )
else:
    sd_corr, sd_p = float("nan"), float("nan")


# ---------------------------------------------------------------------
# TOP OUTLIERS
# ---------------------------------------------------------------------

top_outliers = (
    df.nlargest(20, "abs_residual")[
        [
            "ID",
            "Solubility",
            "PredictedSolubility",
            "Residual",
            "abs_residual",
            "Ocurrences",
            "Group",
            "SD",
        ]
    ]
)


# ---------------------------------------------------------------------
# PRINT RESULTS
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("RESIDUALS BY GROUP")
print("=" * 70)

print(group_summary.to_string(float_format=lambda x: f"{x:.4f}"))


print("\n" + "=" * 70)
print("RESIDUALS BY OCCURRENCE COUNT")
print("=" * 70)

print(occurrence_summary.to_string(float_format=lambda x: f"{x:.4f}"))


print("\n" + "=" * 70)
print("STATISTICAL TESTS")
print("=" * 70)

print(
    f"\nSpearman correlation:"
    f"\n  Occurrences vs |Residual|"
    f"\n  rho = {occurrence_corr:.4f}"
    f"\n  p   = {occurrence_p:.6g}"
)

print(
    f"\nKruskal-Wallis:"
    f"\n  Group vs |Residual|"
    f"\n  H   = {kruskal_stat:.4f}"
    f"\n  p   = {kruskal_p:.6g}"
)

print(
    f"\nSpearman correlation:"
    f"\n  SD vs |Residual|"
    f"\n  rho = {sd_corr:.4f}"
    f"\n  p   = {sd_p:.6g}"
)


print("\n" + "=" * 70)
print("TOP 20 ABSOLUTE RESIDUALS")
print("=" * 70)

print(
    top_outliers.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ---------------------------------------------------------------------
# SAVE REPORT
# ---------------------------------------------------------------------

with open(REPORT_PATH, "w", encoding="utf-8") as f:

    f.write("RESIDUAL RELIABILITY ANALYSIS\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Rows analyzed: {len(df):,}\n\n")

    f.write("RESIDUALS BY GROUP\n")
    f.write("-" * 70 + "\n")
    f.write(
        group_summary.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )
    f.write("\n\n")

    f.write("RESIDUALS BY OCCURRENCE COUNT\n")
    f.write("-" * 70 + "\n")
    f.write(
        occurrence_summary.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )
    f.write("\n\n")

    f.write("STATISTICAL TESTS\n")
    f.write("-" * 70 + "\n")

    f.write(
        f"Occurrences vs |Residual|: "
        f"Spearman rho={occurrence_corr:.4f}, "
        f"p={occurrence_p:.6g}\n"
    )

    f.write(
        f"Group vs |Residual|: "
        f"Kruskal-Wallis H={kruskal_stat:.4f}, "
        f"p={kruskal_p:.6g}\n"
    )

    f.write(
        f"SD vs |Residual|: "
        f"Spearman rho={sd_corr:.4f}, "
        f"p={sd_p:.6g}\n"
    )

    f.write("\nTOP 20 ABSOLUTE RESIDUALS\n")
    f.write("-" * 70 + "\n")
    f.write(
        top_outliers.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )
    f.write("\n")


print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(f"\nReport saved to:")
print(REPORT_PATH)