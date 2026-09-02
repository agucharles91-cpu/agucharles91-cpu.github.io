"""
SCRIPT 21 — FUSED-RING MODEL TEST

Purpose
-------
Test whether the properly-built fused-aromatic-ring descriptor
(max_fused_aromatic_ring_size, from Script 19) recovers independent
predictive importance where the crude rdkit_aromatic_rings count did
not (Script 18 showed it near-zero/non-significant, importance ~ -0.00009,
p=0.29, once RingCount and FractionCSP3 were both in the model).

Models compared, all on the same scaffold-aware split used in
Scripts 17-18:

    M0        : MolWt + MolLogP
    M3        : M0 + RingCount + AromaticRings                    (existing, 4 features)
    M3_fused  : M0 + RingCount + max_fused_aromatic_ring_size      (new, 4 features)
    M4        : M3 + RotatableBonds + FractionCSP3                 (existing, 6 features)
    M4_fused  : M3_fused + RotatableBonds + FractionCSP3           (new, 6 features)

M4 vs M4_fused is the real test: it matches the exact feature set from
Script 18 where AromaticRings washed out. If max_fused_aromatic_ring_size
survives here, ring FUSION specifically (not mere aromatic-ring presence)
is the structurally relevant property.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FUSED_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "fused_ring_features.csv"
SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "model_evaluation_splits.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_PATH = PROJECT_ROOT / "reports" / "fused_ring_model_test.txt"

TARGET = "Solubility"
RANDOM_STATE = 42
N_REPEATS = 50


def fit_and_evaluate(name, feature_cols, train, test):
    X_train = sm.add_constant(train[feature_cols], has_constant="add")
    y_train = train[TARGET]

    X_test = sm.add_constant(test[feature_cols], has_constant="add")
    y_test = test[TARGET]

    model = sm.OLS(y_train, X_train).fit(cov_type="HC3")
    test_pred = model.predict(X_test)

    r2 = r2_score(y_test, test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    mae = mean_absolute_error(y_test, test_pred)

    print(f"\n{name}: {' + '.join(feature_cols)}")
    print(f"  Scaffold-test R²   = {r2:.6f}")
    print(f"  Scaffold-test RMSE = {rmse:.6f}")
    print(f"  Scaffold-test MAE  = {mae:.6f}")
    print("  Coefficients:")
    for feature in feature_cols:
        print(f"    {feature:<32} coef={model.params[feature]: .6f}  p={model.pvalues[feature]:.4g}")

    return {"model": model, "name": name, "features": feature_cols, "r2": r2, "rmse": rmse, "mae": mae}


def statsmodels_predict(fitted_model, X):
    X_sm = sm.add_constant(X, has_constant="add")
    return np.asarray(fitted_model.predict(X_sm))


def manual_permutation_importance(fitted_model, X_test_df, y_test_series, feature_cols,
                                   n_repeats=30, random_state=42):
    rng = np.random.RandomState(random_state)
    y_true = y_test_series.to_numpy()

    baseline_pred = statsmodels_predict(fitted_model, X_test_df)
    baseline_r2 = r2_score(y_true, baseline_pred)

    means, stds = [], []
    for feature in feature_cols:
        drops = []
        for _ in range(n_repeats):
            X_permuted = X_test_df.copy()
            X_permuted[feature] = rng.permutation(X_permuted[feature].to_numpy())
            permuted_r2 = r2_score(y_true, statsmodels_predict(fitted_model, X_permuted))
            drops.append(baseline_r2 - permuted_r2)
        means.append(np.mean(drops))
        stds.append(np.std(drops))

    return baseline_r2, np.array(means), np.array(stds)


def run_permutation(label, fitted_result, feature_cols, test, y_test, n_repeats, random_state):
    print("\n" + "=" * 70)
    print(f"PERMUTATION IMPORTANCE — {label}")
    print("=" * 70)

    X_test_subset = test[feature_cols].copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        baseline_r2_check, means, stds = manual_permutation_importance(
            fitted_result["model"], X_test_subset, y_test, feature_cols,
            n_repeats=n_repeats, random_state=random_state,
        )

    print(f"\nBaseline R² (sanity check): {baseline_r2_check:.6f} (should match {label} R²: {fitted_result['r2']:.6f})")

    importance = pd.DataFrame({
        "feature": feature_cols,
        "mean_importance": means,
        "std_importance": stds,
    }).sort_values("mean_importance", ascending=False).reset_index(drop=True)

    print(f"\nPermutation importance ({label}, mean decrease in scaffold-test R²):")
    print(importance.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    return importance


def main():
    print("=" * 70)
    print("SCRIPT 21 — FUSED-RING MODEL TEST")
    print("=" * 70)

    features = pd.read_csv(FUSED_FEATURES_PATH)
    splits = pd.read_csv(SPLIT_PATH)

    print(f"\nFused feature rows: {len(features):,}")
    print(f"Split rows:         {len(splits):,}")

    split_column = None
    for column in splits.columns:
        if column == "ID":
            continue
        values = set(splits[column].dropna().astype(str).str.lower().unique())
        if {"train", "validation", "test"}.issubset(values):
            split_column = column
            break

    if split_column is None:
        raise ValueError(f"Could not identify split column. Columns: {list(splits.columns)}")

    split_subset = splits[["ID", split_column]].rename(columns={split_column: "data_split"})
    split_subset["data_split"] = split_subset["data_split"].astype(str).str.lower()

    data = features.merge(split_subset, on="ID", how="inner", validate="one_to_one")

    if len(data) != 8643:
        raise ValueError(f"Expected 8643 rows after merge, got {len(data):,}")

    train = data[data["data_split"] == "train"].copy()
    test = data[data["data_split"] == "test"].copy()
    y_test = test[TARGET].copy()

    print(f"\nTraining rows: {len(train):,}")
    print(f"Test rows:     {len(test):,}")

    # ------------------------------------------------------------
    # FIT ALL FIVE MODELS
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL COMPARISON — SCAFFOLD TEST PERFORMANCE")
    print("=" * 70)

    m0 = fit_and_evaluate("M0 (baseline)", ["rdkit_molwt", "rdkit_mollogp"], train, test)

    m3 = fit_and_evaluate(
        "M3 (4-feature, crude aromatic count)",
        ["rdkit_molwt", "rdkit_mollogp", "rdkit_ring_count", "rdkit_aromatic_rings"],
        train, test,
    )

    m3_fused = fit_and_evaluate(
        "M3_fused (4-feature, fused-ring descriptor)",
        ["rdkit_molwt", "rdkit_mollogp", "rdkit_ring_count", "max_fused_aromatic_ring_size"],
        train, test,
    )

    m4 = fit_and_evaluate(
        "M4 (6-feature, crude aromatic count)",
        ["rdkit_molwt", "rdkit_mollogp", "rdkit_ring_count", "rdkit_aromatic_rings",
         "rdkit_rotatable_bonds", "rdkit_fraction_csp3"],
        train, test,
    )

    m4_fused = fit_and_evaluate(
        "M4_fused (6-feature, fused-ring descriptor) — THE REAL TEST",
        ["rdkit_molwt", "rdkit_mollogp", "rdkit_ring_count", "max_fused_aromatic_ring_size",
         "rdkit_rotatable_bonds", "rdkit_fraction_csp3"],
        train, test,
    )

    print("\n" + "=" * 70)
    print("SUMMARY — R² COMPARISON")
    print("=" * 70)
    summary = pd.DataFrame([
        {"model": m["name"], "r2": m["r2"], "rmse": m["rmse"], "mae": m["mae"]}
        for m in [m0, m3, m3_fused, m4, m4_fused]
    ])
    summary["delta_r2_vs_m0"] = summary["r2"] - m0["r2"]
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # ------------------------------------------------------------
    # PERMUTATION IMPORTANCE — all four non-baseline models
    # ------------------------------------------------------------

    imp_m3 = run_permutation(
        "M3", m3, ["rdkit_molwt", "rdkit_mollogp", "rdkit_ring_count", "rdkit_aromatic_rings"],
        test, y_test, N_REPEATS, RANDOM_STATE,
    )

    imp_m3_fused = run_permutation(
        "M3_fused", m3_fused, ["rdkit_molwt", "rdkit_mollogp", "rdkit_ring_count", "max_fused_aromatic_ring_size"],
        test, y_test, N_REPEATS, RANDOM_STATE,
    )

    imp_m4 = run_permutation(
        "M4", m4,
        ["rdkit_molwt", "rdkit_mollogp", "rdkit_ring_count", "rdkit_aromatic_rings",
         "rdkit_rotatable_bonds", "rdkit_fraction_csp3"],
        test, y_test, N_REPEATS, RANDOM_STATE,
    )

    imp_m4_fused = run_permutation(
        "M4_fused — THE REAL TEST", m4_fused,
        ["rdkit_molwt", "rdkit_mollogp", "rdkit_ring_count", "max_fused_aromatic_ring_size",
         "rdkit_rotatable_bonds", "rdkit_fraction_csp3"],
        test, y_test, N_REPEATS, RANDOM_STATE,
    )

    print("\n" + "=" * 70)
    print("KEY COMPARISON POINT")
    print("=" * 70)
    print(
        "\nScript 18 found rdkit_aromatic_rings in the 6-feature M4 model:\n"
        "  importance ~ -0.00009, p = 0.29 (NOT significant)\n"
    )
    print(
        f"This run's max_fused_aromatic_ring_size in M4_fused:\n"
        f"  importance = {imp_m4_fused.set_index('feature').loc['max_fused_aromatic_ring_size', 'mean_importance']:.6f}\n"
        f"  p-value    = {m4_fused['model'].pvalues['max_fused_aromatic_ring_size']:.6g}"
    )

    # ------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------

    summary.to_csv(OUTPUT_DIR / "fused_ring_model_comparison.csv", index=False)
    imp_m3.to_csv(OUTPUT_DIR / "m3_permutation_importance_rerun.csv", index=False)
    imp_m3_fused.to_csv(OUTPUT_DIR / "m3_fused_permutation_importance.csv", index=False)
    imp_m4.to_csv(OUTPUT_DIR / "m4_permutation_importance_rerun.csv", index=False)
    imp_m4_fused.to_csv(OUTPUT_DIR / "m4_fused_permutation_importance.csv", index=False)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("SCRIPT 21 — FUSED-RING MODEL TEST\n" + "=" * 70 + "\n\n")

        f.write("MODEL COMPARISON (scaffold-test)\n" + "-" * 70 + "\n")
        f.write(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        f.write("\n\n")

        for label, imp in [("M3", imp_m3), ("M3_fused", imp_m3_fused), ("M4", imp_m4), ("M4_fused", imp_m4_fused)]:
            f.write(f"PERMUTATION IMPORTANCE — {label}\n" + "-" * 70 + "\n")
            f.write(imp.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
            f.write("\n\n")

        f.write("KEY COMPARISON POINT\n" + "-" * 70 + "\n")
        f.write(
            "Script 18 found rdkit_aromatic_rings in the 6-feature M4 model:\n"
            "  importance ~ -0.00009, p = 0.29 (not significant)\n\n"
            f"This run's max_fused_aromatic_ring_size in M4_fused:\n"
            f"  importance = {imp_m4_fused.set_index('feature').loc['max_fused_aromatic_ring_size', 'mean_importance']:.6f}\n"
            f"  p-value    = {m4_fused['model'].pvalues['max_fused_aromatic_ring_size']:.6g}\n\n"
            "If the fused descriptor shows materially higher importance and/or\n"
            "a significant p-value where the crude aromatic count did not, this\n"
            "supports the hypothesis that ring FUSION specifically (not mere\n"
            "aromatic-ring presence) is the structurally relevant property.\n"
        )

    print("\n" + "=" * 70)
    print("SCRIPT 21 COMPLETE")
    print("=" * 70)
    print(f"\nReport saved to:\n{REPORT_PATH}")


if __name__ == "__main__":
    main()