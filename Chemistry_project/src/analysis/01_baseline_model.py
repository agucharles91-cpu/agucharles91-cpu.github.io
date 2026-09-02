import pandas as pd
import statsmodels.api as sm
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "curated-solubility-dataset.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "baseline_residuals.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "baseline_model.txt"
)


def main():

    print("=" * 70)
    print("BASELINE SOLUBILITY MODEL")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------

    print("\nReading dataset...")

    df = pd.read_csv(INPUT_PATH)

    print(f"Rows: {len(df):,}")

    # ---------------------------------------------------------
    # 2. Select variables
    # ---------------------------------------------------------

    model_df = df[
        [
            "ID",
            "InChIKey",
            "Solubility",
            "SD",
            "Ocurrences",
            "Group",
            "MolWt",
            "MolLogP",
        ]
    ].copy()

    # ---------------------------------------------------------
    # 3. Build baseline model
    # ---------------------------------------------------------

    X = model_df[["MolWt", "MolLogP"]]
    X = sm.add_constant(X)

    y = model_df["Solubility"]

    model = sm.OLS(y, X).fit()

    # ---------------------------------------------------------
    # 4. Predictions and residuals
    # ---------------------------------------------------------

    model_df["PredictedSolubility"] = model.predict(X)

    model_df["Residual"] = (
        model_df["Solubility"]
        - model_df["PredictedSolubility"]
    )

    model_df["AbsoluteResidual"] = model_df["Residual"].abs()

    # ---------------------------------------------------------
    # 5. Save compound-level results
    # ---------------------------------------------------------

    model_df = model_df.sort_values(
        "AbsoluteResidual",
        ascending=False
    )

    model_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ---------------------------------------------------------
    # 6. Print model results
    # ---------------------------------------------------------

    print("\nMODEL RESULTS")
    print("-" * 70)

    print(f"R-squared:       {model.rsquared:.4f}")
    print(f"Adjusted R²:     {model.rsquared_adj:.4f}")
    print(f"RMSE:            {model.mse_resid ** 0.5:.4f}")
    print(f"Observations:    {int(model.nobs):,}")

    print("\nCOEFFICIENTS")
    print("-" * 70)

    print(model.params.to_string())

    print("\nTOP 20 ABSOLUTE RESIDUALS")
    print("-" * 70)

    print(
        model_df[
            [
                "ID",
                "Solubility",
                "PredictedSolubility",
                "Residual",
                "Ocurrences",
                "Group",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # 7. Save full statistical report
    # ---------------------------------------------------------

    with open(REPORT_PATH, "w", encoding="utf-8") as f:

        f.write("=" * 70 + "\n")
        f.write("BASELINE SOLUBILITY MODEL\n")
        f.write("=" * 70 + "\n\n")

        f.write(
            "Model: Solubility ~ Molecular Weight + MolLogP\n\n"
        )

        f.write(model.summary().as_text())

        f.write("\n\n")
        f.write("=" * 70 + "\n")
        f.write("TOP 50 ABSOLUTE RESIDUALS\n")
        f.write("=" * 70 + "\n\n")

        f.write(
            model_df[
                [
                    "ID",
                    "Solubility",
                    "PredictedSolubility",
                    "Residual",
                    "AbsoluteResidual",
                    "Ocurrences",
                    "Group",
                ]
            ]
            .head(50)
            .to_string(index=False)
        )

    print("\n" + "=" * 70)
    print("BASELINE MODEL COMPLETE")
    print("=" * 70)

    print(f"\nResults saved to:")
    print(OUTPUT_PATH)

    print(f"\nModel report saved to:")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()