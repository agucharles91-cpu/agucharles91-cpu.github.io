# Chemical Property Analytics & Aqueous Solubility Prediction

## Overview

This project investigates whether molecular structure and physicochemical properties can be used to predict aqueous solubility, and which molecular characteristics are most strongly associated with solubility.

The analysis uses the **AqSolDB curated aqueous-solubility dataset**, containing 9,982 compounds. The project combines chemistry-aware data preparation, PostgreSQL data modeling, exploratory analysis, interpretable statistical modeling, nonlinear machine learning, scaffold-aware validation, model explainability, residual analysis, uncertainty diagnostics, applicability-domain analysis, and calibration.

The emphasis is not only on predictive performance, but also on understanding **where the model works, where it fails, and how chemical-space coverage affects prediction reliability**.

---

## Research Question

> **Can molecular structure and physicochemical properties be used to predict aqueous solubility, and which molecular characteristics are most strongly associated with solubility?**

---

## Key Findings

The final analysis produced four main model families.

| Model                 | Features                           |    Mean R² |  Mean RMSE |   Mean MAE |
| --------------------- | ---------------------------------- | ---------: | ---------: | ---------: |
| M0 Baseline           | MolWt + MolLogP                    |     0.6465 |     1.2790 |     0.9804 |
| M3 Ring               | M0 + RingCount + AromaticRings     |     0.6638 |     1.2457 |     0.9501 |
| M4 Selective          | M3 + RotatableBonds + FractionCSP3 |     0.6632 |     1.2422 |     0.9434 |
| **Gradient Boosting** | Six descriptors                    | **0.7376** | **1.0960** | **0.8147** |

Performance was evaluated using **10 repeated scaffold-aware validation splits**.

The main conclusions are:

* **MolLogP is the dominant predictor** among the evaluated descriptors.
* **Molecular weight is the second strongest predictor.**
* Ring count, aromatic ring count, rotatable bonds, and FractionCSP3 provide smaller incremental information.
* The substantially better performance of Gradient Boosting demonstrates that the descriptor–solubility relationship contains meaningful **nonlinearity**.
* Prediction error is heterogeneous across the solubility range and chemical space.
* Model performance deteriorates as compounds become more novel relative to their corresponding training populations.
* The model tends to **compress extreme predictions toward the center**.
* Highly insoluble compounds tend to be overpredicted, while highly soluble compounds tend to be underpredicted.
* Prediction variability is useful as a reliability diagnostic, but it is only weakly associated with actual prediction error.
* Applicability-domain analysis indicates that chemical-space representation contributes to prediction reliability, but does not completely explain model error.

---

## Dataset

The project uses the curated AqSolDB aqueous-solubility dataset.

The original dataset contains:

* **9,982 compounds**
* **26 columns**

The raw source data is intentionally **not included in this repository**.

The raw dataset is retained locally for reproducibility, while the repository contains the analytical source code, selected derived outputs, reports, and figures.

### Data provenance

AqSolDB should be cited using its original publication and associated data repository.

The project does not redistribute the original CSV or source archive.

---

## Analytical Population

After dataset and chemical-domain auditing, the principal modeling population was defined as:

**Population C: 8,643 compounds**

The repeated scaffold-aware evaluation produced:

* **22,872 held-out prediction observations**
* **8,160 unique compounds represented**
* **10 validation repetitions**

The distinction between observations and unique compounds is important because compounds can appear in multiple validation repetitions.

---

## Methodology

The project follows a staged analytical workflow.

```text
AqSolDB
   │
   ▼
Dataset inspection
   │
   ▼
PostgreSQL data model
   │
   ▼
Analytical dataset audit
   │
   ▼
Molecular feature engineering
   │
   ▼
Feature quality audit
   │
   ▼
Exploratory analysis
   │
   ├───────────────┐
   ▼               ▼
Structural       Descriptor
analysis         analysis
   │               │
   └───────┬───────┘
           ▼
      Baseline models
           │
           ▼
Repeated scaffold-aware validation
           │
      ┌────┴─────┐
      ▼          ▼
   Linear     Gradient
   models     Boosting
      │          │
      └────┬─────┘
           ▼
      Model explainability
           │
           ▼
      Residual analysis
           │
           ▼
    Error heterogeneity
           │
           ▼
     Uncertainty analysis
           │
           ▼
      Failure modes
           │
           ▼
   Applicability domain
           │
           ▼
       Calibration
           │
           ▼
     Final synthesis
```

---

## Molecular Features

The final model family uses six interpretable molecular descriptors:

* `MolWt`
* `MolLogP`
* `RingCount`
* `AromaticRings`
* `RotatableBonds`
* `FractionCSP3`

The relatively compact feature set was intentional. The objective was to investigate interpretable relationships between molecular properties and aqueous solubility rather than maximize performance using an opaque high-dimensional molecular representation.

---

## Validation Strategy

A major design decision was to avoid relying solely on a conventional random train/test split.

The principal evaluation uses **repeated scaffold-aware holdout validation**.

The validation was repeated ten times to examine whether conclusions were stable across different scaffold-level partitions.

This reduces the risk of overly optimistic performance caused by structurally related compounds appearing in both training and validation populations.

The final Gradient Boosting model improved on the M4 linear model in all ten scaffold-aware repetitions.

---

## Structural Analysis

The project includes dedicated analyses of molecular ring structure and structural similarity.

These investigate whether ring topology contributes information beyond basic physicochemical descriptors.

The project also includes nearest-neighbor and structure-pair analyses to connect statistical model errors with actual molecular structures.

This is important because an isolated residual value does not necessarily explain why a prediction failed.

---

## Model Explainability

Two complementary approaches were used.

### Linear models

Standardized coefficients and permutation importance were used to assess the relative contribution of the molecular descriptors.

### Gradient Boosting

Feature importance and SHAP-based analyses were used to examine nonlinear model behavior and individual feature contributions.

These analyses are interpreted as explanations of **model behavior**, not causal chemical mechanisms.

---

## Error and Reliability Analysis

Average predictive performance alone does not describe how the model behaves across the chemical domain.

The project therefore examines:

* residual distributions;
* error heterogeneity;
* error by solubility region;
* scaffold-level behavior;
* repeated-prediction variability;
* model failure modes;
* applicability-domain novelty;
* calibration.

### Failure modes

Using the 75th percentiles of absolute error and prediction variability as descriptive thresholds, compounds were categorized into four groups:

| Failure category              | Compounds | Percentage |
| ----------------------------- | --------: | ---------: |
| Low-error + stable            |     4,699 |     57.59% |
| High-error + stable           |     1,420 |     17.40% |
| Low-error + high-uncertainty  |     1,421 |     17.41% |
| High-error + high-uncertainty |       620 |      7.60% |

The **high-error + high-uncertainty** group represents a particularly important reliability warning because these compounds are both inaccurate and unstable across repeated validation.

However, the high-error + stable group demonstrates that some systematic errors remain even when repeated predictions are relatively stable.

---

## Applicability Domain

Applicability-domain analysis evaluates how well each held-out compound is represented by its corresponding training population.

The authoritative analysis calculates the Euclidean distance from each held-out compound to its nearest training compound in standardized six-descriptor space.

Lower distance indicates greater representation by the training population; higher distance indicates greater novelty.

Across novelty strata:

| Novelty stratum |    MAE |   RMSE |
| --------------- | -----: | -----: |
| Lowest          | 0.7139 | 0.9402 |
| Low-moderate    | 0.8078 | 1.0799 |
| Moderate-high   | 0.8398 | 1.1306 |
| Highest         | 0.9042 | 1.2599 |

Prediction error therefore increases progressively with descriptor-space novelty.

However, the compound-level correlations remain weak:

* Pearson: **0.1103**
* Spearman: **0.0740**

The appropriate conclusion is that chemical-space representation contributes to prediction reliability, but does not fully explain model error.

---

## Calibration

The model does not reproduce the full observed range equally well.

Calibration slopes were:

* Row-level: **0.7617**
* Compound-level: **0.7466**

Both are below 1, indicating compression of predicted values toward the center.

The regional residual pattern is consistent with this:

| Solubility region | Mean residual |
| ----------------- | ------------: |
| < -6              |       -1.1992 |
| -6 to < -4        |       -0.3915 |
| -4 to < -2        |       -0.0363 |
| -2 to < 0         |       +0.3914 |
| >= 0              |       +1.0341 |

Using residual = observed − predicted:

* Negative residuals indicate overprediction.
* Positive residuals indicate underprediction.

The model therefore tends to:

* **overpredict highly insoluble compounds**
* **underpredict highly soluble compounds**

---

## Uncertainty

Repeated scaffold-aware predictions were used to estimate prediction variability.

This variability is treated as an **empirical reliability diagnostic**, not as a formal confidence interval or prediction interval.

The relationship between prediction variability and absolute error was weak:

* Pearson correlation: **0.1300**
* Spearman correlation: **0.0693**

Consequently, uncertainty should not be treated as a standalone measure of expected prediction accuracy.

---

## Repository Structure

```text
Chemistry_project/
│
├── data/
│   ├── raw/                         # local source data; not committed
│   └── processed/                   # selected derived outputs
│
├── notebooks/                       # reserved for future notebook work
│
├── reports/
│   ├── figures/
│   │   ├── eda/
│   │   ├── model_explainability/
│   │   ├── ring_annulation/
│   │   └── structure_pairs/
│   └── *.txt                         # analytical reports
│
├── sql/
│   └── schema/
│       └── 001_create_schema.sql
│
└── src/
    ├── ingestion/
    │   ├── inspect_dataset.py
    │   └── load_to_postgres.py
    │
    └── analysis/
        ├── 01_baseline_model.py
        ├── 02_residual_reliability.py
        ├── ...
        ├── 32_final_model_synthesis.py
        └── 33_applicability_domain_analysis.py
```

---

## PostgreSQL Data Model

The database uses a dedicated schema:

```text
solubility
```

with three tables:

```text
solubility.compounds
solubility.solubility_measurements
solubility.molecular_descriptors
```

The ingestion process validates the expected dataset size and performs transactional loading.

The PostgreSQL password is supplied through the environment variable:

```text
POSTGRES_PASSWORD
```

Credentials are not stored in the source code.

---

## Reproducibility

The project is designed around script-based reproducibility rather than notebook-only execution.

The source code contains separate stages for:

1. Dataset inspection
2. Data loading
3. Analytical auditing
4. Feature engineering
5. Exploratory analysis
6. Modeling
7. Validation
8. Explainability
9. Reliability analysis
10. Final synthesis

Raw data and large intermediate datasets are excluded from Git to keep the repository focused on reproducible source code and meaningful derived results.

The raw AqSolDB data must therefore be obtained separately from its original source before reproducing the complete pipeline.

---

## Key Outputs

Important derived outputs include:

```text
data/processed/
├── final_model_synthesis.csv
├── model_evaluation_results.csv
├── model_evaluation_splits.csv
├── repeated_scaffold_evaluation.csv
├── nonlinear_model_evaluation.csv
├── nonlinear_feature_importance.csv
├── shap_feature_summary.csv
├── shap_feature_effects.csv
├── prediction_uncertainty_correlations.csv
├── failure_mode_correlations.csv
├── applicability_domain_summary.csv
├── applicability_domain_by_novelty.csv
├── calibration_by_prediction_bin.csv
├── calibration_by_solubility_region.csv
└── ...
```

The repository also contains analytical reports and selected visualizations documenting the reasoning behind the final conclusions.

---

## Limitations

### Dataset scope

The results are specific to the AqSolDB population and should not be interpreted as universal relationships across all chemical space.

### Validation scope

Scaffold-aware validation reduces structural leakage but does not establish external validity on independent datasets.

### Descriptor scope

The final model uses six molecular descriptors. More expressive molecular representations, including graph-based approaches, could produce different results.

### Interpretation

Feature importance and SHAP values describe predictive model behavior. They do not establish causal chemical mechanisms.

### Uncertainty

Repeated-prediction variability is an empirical diagnostic and should not be interpreted as a formally calibrated prediction interval.

### Applicability domain

The novelty metric is based on standardized six-descriptor space. It does not capture every dimension of molecular structural similarity.

---

## Final Conclusion

The analysis shows that aqueous solubility can be predicted reasonably well from a compact set of molecular descriptors, but predictive performance depends on both the molecular properties and the chemical space represented by the training data.

MolLogP and molecular weight provide the strongest predictive signal among the evaluated descriptors. Structural descriptors provide smaller incremental improvements, while Gradient Boosting captures nonlinear relationships that substantially improve predictive performance over linear models.

The final model achieved:

**Mean R² = 0.7376**
**Mean RMSE = 1.0960**
**Mean MAE = 0.8147**

under repeated scaffold-aware validation.

However, the reliability analysis demonstrates why aggregate performance metrics are insufficient. Prediction error increases for chemically novel compounds, the model performs poorly at the extremes of the solubility range, and some compounds exhibit high errors despite apparently stable predictions.

The resulting project therefore treats aqueous-solubility prediction as both a **prediction problem and a model-reliability problem**.

---

## Citation

The dataset should be cited using the original AqSolDB publication and associated data repository.

The project does not redistribute the original AqSolDB source files.

---

## Status

**Analytical workflow: Complete**

**Model evaluation: Complete**

**Explainability: Complete**

**Reliability analysis: Complete**

**Applicability-domain analysis: Complete**

**Calibration analysis: Complete**

**Documentation: Complete**

**Repository integration: Final review**