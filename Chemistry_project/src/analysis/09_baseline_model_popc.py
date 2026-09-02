import pandas as pd
import statsmodels.api as sm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "molecular_features.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "baseline_residuals_popc.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "baseline_model_popc.txt"


def main():
    print("=" * 70)
    print("BASELINE SOLUBILITY MODEL — POPULATION C")
    print("=" * 70)

    df = pd.read_csv(INPUT_PATH)
    print(f"\nRows: {len(df):,}")

    assert len(df) == 8643, f"Expected 8,643 rows, got {len(df):,}"

    model_df = df[[
        "ID", "InChIKey", "Solubility", "SD", "Ocurrences", "Group",
        "rdkit_molwt", "rdkit_mollogp",
    ]].copy()

    X = sm.add_constant(model_df[["rdkit_molwt", "rdkit_mollogp"]])
    y = model_df["Solubility"]

    model = sm.OLS(y, X).fit()

    model_df["PredictedSolubility"] = model.predict(X)
    model_df["Residual"] = model_df["Solubility"] - model_df["PredictedSolubility"]
    model_df["AbsoluteResidual"] = model_df["Residual"].abs()

    model_df = model_df.sort_values("AbsoluteResidual", ascending=False)
    model_df.to_csv(OUTPUT_PATH, index=False)

    print("\nMODEL RESULTS")
    print("-" * 70)
    print(f"R-squared:    {model.rsquared:.4f}")
    print(f"Adjusted R²:  {model.rsquared_adj:.4f}")
    print(f"RMSE:         {model.mse_resid ** 0.5:.4f}")
    print(f"Observations: {int(model.nobs):,}")

    print("\nCOEFFICIENTS")
    print("-" * 70)
    print(model.params.to_string())

    print("\nTOP 20 ABSOLUTE RESIDUALS")
    print("-" * 70)
    print(
        model_df[["ID", "Solubility", "PredictedSolubility",
                   "Residual", "Ocurrences", "Group"]]
        .head(20).to_string(index=False)
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("BASELINE SOLUBILITY MODEL — POPULATION C\n")
        f.write("=" * 70 + "\n\n")
        f.write("Model: Solubility ~ rdkit_molwt + rdkit_mollogp\n")
        f.write(f"Population: standard analytical domain (n={len(df):,})\n\n")
        f.write(model.summary().as_text())
        f.write("\n\n" + "=" * 70 + "\n")
        f.write("TOP 50 ABSOLUTE RESIDUALS\n")
        f.write("=" * 70 + "\n\n")
        f.write(
            model_df[["ID", "Solubility", "PredictedSolubility",
                       "Residual", "AbsoluteResidual", "Ocurrences", "Group"]]
            .head(50).to_string(index=False)
        )

    print("\n" + "=" * 70)
    print("BASELINE MODEL COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to:\n{OUTPUT_PATH}")
    print(f"\nModel report saved to:\n{REPORT_PATH}")


if __name__ == "__main__":
    main()