"""
32_final_model_synthesis.py

Final model synthesis and evidence integration for the AqSolDB
aqueous-solubility project.

Purpose
-------
Consolidate the established modelling and diagnostic evidence into one
final analytical comparison of the principal candidate models.

Models
------
M0:
    Solubility ~ MolWt + MolLogP

M3:
    M0 + RingCount + AromaticRings

M4:
    M3 + RotatableBonds + FractionCSP3

GB:
    Gradient Boosting using the six M4 descriptors.

Evidence integrated
-------------------
1. Repeated scaffold-aware held-out performance.
2. Nonlinear model improvement.
3. Permutation feature importance.
4. SHAP global importance.
5. Residual/error heterogeneity.
6. Prediction uncertainty.
7. Model failure modes.
8. Applicability-domain diagnostics.
9. Calibration.
10. Methodological limitations.

Important methodological rule
------------------------------
This script does not fit, tune, or modify any model.

It synthesizes previously completed analyses and therefore does not
constitute a new model evaluation.

Population C remains locked at 8,643 compounds.
"""

from pathlib import Path

import pandas as pd


# =====================================================================
# PATHS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_FILE = DATA_DIR / "final_model_synthesis.csv"
REPORT_FILE = REPORT_DIR / "final_model_synthesis.txt"


# =====================================================================
# INPUT FILES
# =====================================================================

INPUT_FILES = {
    "repeated_scaffold": DATA_DIR / "repeated_scaffold_evaluation.csv",
    "nonlinear_model": DATA_DIR / "nonlinear_model_evaluation.csv",
    "feature_importance": DATA_DIR / "nonlinear_feature_importance.csv",
    "shap": DATA_DIR / "shap_feature_summary.csv",
    "residual": DATA_DIR / "nonlinear_residual_analysis.csv",
    "calibration_region": DATA_DIR / "calibration_by_solubility_region.csv",
    "calibration_uncertainty": DATA_DIR / "calibration_by_uncertainty.csv",
}


# =====================================================================
# HEADER
# =====================================================================

print("=" * 70)
print("SCRIPT 32 — FINAL MODEL SYNTHESIS")
print("=" * 70)

print("\nProject root:")
print(PROJECT_ROOT)


# =====================================================================
# VALIDATE INPUT FILES
# =====================================================================

print("\n" + "=" * 70)
print("INPUT VALIDATION")
print("=" * 70)

missing = []

for name, path in INPUT_FILES.items():

    if path.exists():
        print(f"[OK] {name}: {path}")

    else:
        print(f"[MISSING] {name}: {path}")
        missing.append(name)

if missing:
    raise FileNotFoundError(
        "Required analytical outputs are missing: "
        + ", ".join(missing)
    )


# =====================================================================
# LOAD REPORT-LEVEL PERFORMANCE DATA
# =====================================================================

print("\n" + "=" * 70)
print("LOADING MODEL PERFORMANCE")
print("=" * 70)

repeated = pd.read_csv(INPUT_FILES["repeated_scaffold"])

nonlinear = pd.read_csv(INPUT_FILES["nonlinear_model"])

print(f"Repeated scaffold rows: {len(repeated):,}")
print(f"Nonlinear model rows: {len(nonlinear):,}")


# =====================================================================
# DISPLAY AVAILABLE STRUCTURE
# =====================================================================

print("\nRepeated scaffold columns:")
for column in repeated.columns:
    print(f"  {column}")

print("\nNonlinear model columns:")
for column in nonlinear.columns:
    print(f"  {column}")


# =====================================================================
# POPULATION C
# =====================================================================

POPULATION_C = 8643

print("\n" + "=" * 70)
print("ANALYTICAL POPULATION")
print("=" * 70)

print(f"Locked Population C: {POPULATION_C:,}")


# =====================================================================
# LOAD FEATURE IMPORTANCE
# =====================================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

importance = pd.read_csv(INPUT_FILES["feature_importance"])

print(f"Feature-importance rows: {len(importance):,}")

print("\nColumns:")
for column in importance.columns:
    print(f"  {column}")


# =====================================================================
# LOAD SHAP
# =====================================================================

print("\n" + "=" * 70)
print("SHAP")
print("=" * 70)

shap = pd.read_csv(INPUT_FILES["shap"])

print(f"SHAP rows: {len(shap):,}")

print("\nColumns:")
for column in shap.columns:
    print(f"  {column}")


# =====================================================================
# LOAD RESIDUAL DATA
# =====================================================================

print("\n" + "=" * 70)
print("HELD-OUT PREDICTION DATA")
print("=" * 70)

residual = pd.read_csv(INPUT_FILES["residual"])

print(f"Prediction rows: {len(residual):,}")

if "ID" in residual.columns:
    print(
        "Unique compounds:",
        residual["ID"].nunique()
    )


# =====================================================================
# LOAD CALIBRATION DATA
# =====================================================================

print("\n" + "=" * 70)
print("CALIBRATION")
print("=" * 70)

calibration_region = pd.read_csv(
    INPUT_FILES["calibration_region"]
)

calibration_uncertainty = pd.read_csv(
    INPUT_FILES["calibration_uncertainty"]
)

print(
    "Solubility calibration rows:",
    len(calibration_region)
)

print(
    "Uncertainty calibration rows:",
    len(calibration_uncertainty)
)


# =====================================================================
# MODEL SUMMARY
# =====================================================================

print("\n" + "=" * 70)
print("ESTABLISHED MODEL BENCHMARK")
print("=" * 70)

model_summary = pd.DataFrame(
    [
        {
            "model": "M0_baseline",
            "description": "MolWt + MolLogP",
            "mean_r2": 0.646520,
            "mean_rmse": 1.279021,
            "mean_mae": 0.980394,
        },
        {
            "model": "M3_ring_model",
            "description": (
                "MolWt + MolLogP + RingCount + AromaticRings"
            ),
            "mean_r2": 0.663819,
            "mean_rmse": 1.245696,
            "mean_mae": 0.950100,
        },
        {
            "model": "M4_selective_model",
            "description": (
                "M3 + RotatableBonds + FractionCSP3"
            ),
            "mean_r2": 0.663201,
            "mean_rmse": 1.242220,
            "mean_mae": 0.943376,
        },
        {
            "model": "GB_gradient_boosting",
            "description": (
                "Gradient Boosting using six M4 descriptors"
            ),
            "mean_r2": 0.737576,
            "mean_rmse": 1.096018,
            "mean_mae": 0.814707,
        },
    ]
)

print(model_summary.to_string(index=False))


# =====================================================================
# CALCULATE IMPROVEMENT OVER BASELINE
# =====================================================================

baseline = model_summary[
    model_summary["model"] == "M0_baseline"
].iloc[0]

model_summary["delta_r2_vs_m0"] = (
    model_summary["mean_r2"] - baseline["mean_r2"]
)

model_summary["delta_rmse_vs_m0"] = (
    model_summary["mean_rmse"] - baseline["mean_rmse"]
)

model_summary["delta_mae_vs_m0"] = (
    model_summary["mean_mae"] - baseline["mean_mae"]
)


print("\n" + "=" * 70)
print("IMPROVEMENT OVER M0")
print("=" * 70)

print(
    model_summary[
        [
            "model",
            "delta_r2_vs_m0",
            "delta_rmse_vs_m0",
            "delta_mae_vs_m0",
        ]
    ].to_string(index=False)
)


# =====================================================================
# NONLINEAR IMPROVEMENT
# =====================================================================

print("\n" + "=" * 70)
print("NONLINEAR MODEL EVIDENCE")
print("=" * 70)

gb = model_summary[
    model_summary["model"] == "GB_gradient_boosting"
].iloc[0]

m4 = model_summary[
    model_summary["model"] == "M4_selective_model"
].iloc[0]

gb_vs_m4 = {
    "delta_r2": gb["mean_r2"] - m4["mean_r2"],
    "delta_rmse": gb["mean_rmse"] - m4["mean_rmse"],
    "delta_mae": gb["mean_mae"] - m4["mean_mae"],
}

print(
    f"GB ΔR² vs M4:   {gb_vs_m4['delta_r2']:+.6f}"
)

print(
    f"GB ΔRMSE vs M4: {gb_vs_m4['delta_rmse']:+.6f}"
)

print(
    f"GB ΔMAE vs M4:  {gb_vs_m4['delta_mae']:+.6f}"
)


# =====================================================================
# CALIBRATION SUMMARY
# =====================================================================

print("\n" + "=" * 70)
print("CALIBRATION SUMMARY")
print("=" * 70)

calibration_summary = {
    "row_level_slope": 0.7617,
    "row_level_intercept": -0.6990,
    "compound_level_slope": 0.7466,
    "compound_level_intercept": -0.7336,
    "row_level_mae": 0.8164,
    "row_level_rmse": 1.1086,
    "row_level_r2": 0.7600,
}

for key, value in calibration_summary.items():
    print(f"{key}: {value}")


# =====================================================================
# REGIONAL CALIBRATION
# =====================================================================

print("\n" + "=" * 70)
print("REGIONAL CALIBRATION")
print("=" * 70)

print(calibration_region.to_string(index=False))


# =====================================================================
# UNCERTAINTY SUMMARY
# =====================================================================

print("\n" + "=" * 70)
print("UNCERTAINTY / ERROR RELATIONSHIP")
print("=" * 70)

print(calibration_uncertainty.to_string(index=False))


# =====================================================================
# FINAL EVIDENCE TABLE
# =====================================================================

evidence = pd.DataFrame(
    [
        {
            "evidence_area": "Scaffold-aware performance",
            "finding": (
                "Gradient Boosting produced the strongest mean "
                "held-out performance across repeated scaffold splits."
            ),
            "strength": "Strong",
        },
        {
            "evidence_area": "Nonlinear improvement",
            "finding": (
                "Gradient Boosting improved R², RMSE and MAE over "
                "the linear M4 model in all 10 repetitions."
            ),
            "strength": "Strong",
        },
        {
            "evidence_area": "Feature importance",
            "finding": (
                "MolLogP was consistently the dominant predictive "
                "descriptor, followed by MolWt."
            ),
            "strength": "Strong",
        },
        {
            "evidence_area": "SHAP",
            "finding": (
                "Global SHAP importance supported the same broad "
                "descriptor hierarchy."
            ),
            "strength": "Strong",
        },
        {
            "evidence_area": "Error heterogeneity",
            "finding": (
                "Prediction error varies substantially across "
                "observed solubility regimes."
            ),
            "strength": "Strong",
        },
        {
            "evidence_area": "Calibration",
            "finding": (
                "Calibration slope below 1 demonstrates prediction "
                "compression toward the central range."
            ),
            "strength": "Strong",
        },
        {
            "evidence_area": "Uncertainty",
            "finding": (
                "The highest prediction-variability group has "
                "higher prediction error than lower-variability groups."
            ),
            "strength": "Moderate",
        },
        {
            "evidence_area": "Applicability domain",
            "finding": (
                "Prediction reliability depends partly on representation "
                "within the evaluated chemical space."
            ),
            "strength": "Diagnostic",
        },
        {
            "evidence_area": "Causality",
            "finding": (
                "Descriptor importance and model behaviour do not "
                "establish causal physicochemical effects."
            ),
            "strength": "Limitation",
        },
    ]
)

print(evidence.to_string(index=False))


# =====================================================================
# SAVE SYNTHESIS TABLE
# =====================================================================

model_summary.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved model synthesis:")
print(OUTPUT_FILE)


# =====================================================================
# WRITE REPORT
# =====================================================================

report_lines = []

report_lines.append(
    "SCRIPT 32 — FINAL MODEL SYNTHESIS"
)

report_lines.append("=" * 70)

report_lines.append(
    "\nPURPOSE\n"
    "-------\n"
    "Synthesize the established modelling and diagnostic evidence "
    "for the AqSolDB aqueous-solubility project without fitting or "
    "tuning any new model."
)

report_lines.append(
    "\nDATA DESIGN\n"
    "-----------\n"
    f"Population C: {POPULATION_C:,} compounds\n"
    f"Repeated held-out prediction rows: {len(residual):,}\n"
    f"Unique compounds represented: {residual['ID'].nunique():,}\n"
    "Validation design: repeated scaffold-aware holdout\n"
)

report_lines.append(
    "\nMODEL COMPARISON\n"
    "----------------\n"
)

for _, row in model_summary.iterrows():

    report_lines.append(
        f"{row['model']}\n"
        f"  Description: {row['description']}\n"
        f"  Mean R²: {row['mean_r2']:.6f}\n"
        f"  Mean RMSE: {row['mean_rmse']:.6f}\n"
        f"  Mean MAE: {row['mean_mae']:.6f}\n"
        f"  ΔR² vs M0: {row['delta_r2_vs_m0']:+.6f}\n"
        f"  ΔRMSE vs M0: {row['delta_rmse_vs_m0']:+.6f}\n"
        f"  ΔMAE vs M0: {row['delta_mae_vs_m0']:+.6f}\n"
    )

report_lines.append(
    "\nNONLINEAR MODEL CONCLUSION\n"
    "--------------------------\n"
    "Gradient Boosting provides the strongest mean predictive "
    "performance among the evaluated models. Its improvement over "
    "M4 was positive for R² and negative for RMSE and MAE in every "
    "one of the ten scaffold-aware repetitions."
)

report_lines.append(
    "\nFEATURE IMPORTANCE CONCLUSION\n"
    "-----------------------------\n"
    "MolLogP is the dominant predictive descriptor within the "
    "evaluated six-feature set. MolWt provides the next-largest "
    "contribution, while FractionCSP3, RingCount, RotatableBonds "
    "and AromaticRings contribute smaller amounts of predictive "
    "information."
)

report_lines.append(
    "\nCALIBRATION CONCLUSION\n"
    "----------------------\n"
    "The row-level calibration slope is 0.7617 and the compound-level "
    "calibration slope is 0.7466. Both indicate prediction compression: "
    "the model reproduces central solubility values more closely than "
    "the extreme observed values."
)

report_lines.append(
    "\nREGIONAL ERROR CONCLUSION\n"
    "-------------------------\n"
    "The model systematically overestimates solubility for the most "
    "insoluble compounds and underestimates solubility for the most "
    "soluble compounds. Near-zero global mean residual therefore "
    "does not imply uniform calibration."
)

report_lines.append(
    "\nUNCERTAINTY CONCLUSION\n"
    "----------------------\n"
    "Repeated-prediction variability provides a useful diagnostic "
    "signal. The highest-uncertainty group has materially higher "
    "prediction error than the lower-uncertainty groups. This "
    "variability measure should not be interpreted as a formal "
    "confidence interval or probabilistic prediction interval."
)

report_lines.append(
    "\nOVERALL SCIENTIFIC CONCLUSION\n"
    "-----------------------------\n"
    "Aqueous solubility in this dataset is strongly associated with "
    "molecular lipophilicity and molecular size, while additional "
    "structural descriptors provide smaller incremental information. "
    "The substantially better performance of nonlinear models indicates "
    "that the relationship between the available molecular descriptors "
    "and solubility is not adequately represented by a simple linear "
    "form. Gradient Boosting provides the strongest predictive model "
    "among those evaluated under repeated scaffold-aware validation."
)

report_lines.append(
    "\nHowever, the model is not uniformly reliable across the "
    "solubility domain. Calibration analysis demonstrates systematic "
    "compression toward the central range, with increased error at "
    "both extreme insolubility and high-solubility regions. Prediction "
    "uncertainty and applicability-domain diagnostics further indicate "
    "that prediction reliability varies across compounds."
)

report_lines.append(
    "\nLIMITATIONS\n"
    "-----------\n"
    "1. Results are specific to the curated AqSolDB dataset and the "
    "defined Population C.\n"
    "2. Scaffold-aware validation reduces structural leakage but does "
    "not establish universal external validity.\n"
    "3. Descriptor importance is predictive rather than causal.\n"
    "4. Repeated-prediction standard deviation is an empirical "
    "uncertainty proxy rather than a formal predictive interval.\n"
    "5. Calibration and regional error patterns identify systematic "
    "behaviour but do not establish the underlying chemical mechanism.\n"
    "6. The evaluated models use a relatively small descriptor set "
    "rather than the full molecular representation available from "
    "modern cheminformatics methods."
)

report_lines.append(
    "\nFINAL MODEL SELECTION\n"
    "---------------------\n"
    "For predictive modelling within this project, Gradient Boosting "
    "is the preferred model because it achieved the strongest mean "
    "held-out performance under repeated scaffold-aware validation."
)

report_lines.append(
    "\nThis selection is based on predictive performance and robustness "
    "within the established experimental design, not on a claim that "
    "Gradient Boosting is universally optimal for aqueous-solubility "
    "prediction."
)

REPORT_FILE.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

print("\nSaved final synthesis report:")
print(REPORT_FILE)


# =====================================================================
# COMPLETE
# =====================================================================

print("\n" + "=" * 70)
print("SCRIPT 32 COMPLETE")
print("=" * 70)

print(f"Population C: {POPULATION_C:,}")
print(
    f"Unique compounds in repeated predictions: "
    f"{residual['ID'].nunique():,}"
)
print("Preferred predictive model: Gradient Boosting")
print(f"Mean GB R²: {gb['mean_r2']:.4f}")
print(f"Mean GB RMSE: {gb['mean_rmse']:.4f}")
print(f"Mean GB MAE: {gb['mean_mae']:.4f}")
print("Calibration slope: 0.7617")
print("Compound-level calibration slope: 0.7466")
