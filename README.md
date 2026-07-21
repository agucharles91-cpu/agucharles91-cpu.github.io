# Nairobi Runners — Training, Sleep & Recovery Analytics

**MSc Data Science Capstone** | 14 participants | Garmin wearables | 2022–2026

A Shiny dashboard and Quarto report analysing longitudinal wearable data from
14 recreational runners in Nairobi, Kenya. Covers training load, sleep architecture,
recovery physiology, and their temporal relationships.

---

## Quick start

### 1. Clone the repository
```bash
git clone https://github.com/agucharles91-cpu/agucharles91-cpu.github.io.git
cd agucharles91-cpu.github.io
```

### 2. Restore the R environment (recommended)
This project uses `renv` to pin all package versions. Run once in R:
```r
install.packages("renv")
renv::restore()   # installs every package at the exact recorded version
```

If you prefer to install manually, see **Dependencies** below.

### 3. Place the data file
Put `garmin_merged_14p.csv` in the `app/data/` subdirectory:
```
app/
└── data/
    └── garmin_merged_14p.csv
```
The file is not included in this repository (contains personal health data).
Contact the author for access under a data sharing agreement.

### 4. Launch the Shiny app
```r
shiny::runApp("app/app.R")
```
The app opens on `http://127.0.0.1:<port>` in your default browser.
Expected load time on first run: ~5–10 seconds for data preparation.

### 5. Render the Quarto report
```r
quarto::quarto_render("nairobi_runners_report.qmd")
```
Output: `nairobi_runners_report.html` (self-contained, ~8 MB).

---

## Project structure

```
.
├── app/                              # Shiny application
│   ├── app.R                         # Main application (single-file)
│   ├── data_dictionary.md            # Column definitions for the CSV
│   ├── renv_setup.R                  # Run once to initialise renv
│   ├── renv.lock                     # Pinned package versions — commit this
│   ├── session_info.txt              # Recorded R session for reproducibility
│   ├── data/
│   │   └── garmin_merged_14p.csv     # Raw data (not in repo — see above)
│   └── tests/
│       └── test_smoke.R              # Automated smoke tests (testthat)
├── docs/                             # GitHub Pages rendered output
├── nairobi_runners_report.qmd        # Quarto report source
├── nairobi_runners_report.html       # Rendered report
├── references.bib                    # Bibliography
├── _quarto.yml                       # Quarto site configuration
├── index.qmd                         # Site homepage
├── README.md                         # This file
└── LICENSE                           # MIT licence
```

---

## Dependencies

All packages are pinned in `renv.lock`. The core requirements are:

| Package | Version tested | Purpose |
|---|---|---|
| shiny | ≥ 1.8.0 | Web app framework |
| bslib | ≥ 0.7.0 | Bootstrap 5 theme |
| tidyverse | ≥ 2.0.0 | Data wrangling (includes lubridate, dplyr, ggplot2) |
| lubridate | ≥ 1.9.3 | **Must be loaded explicitly** — `lubridate::isoweek()` is called directly to avoid conflicts with `data.table` or `tsibble` if present |
| plotly | ≥ 4.10.4 | Interactive charts |
| DT | ≥ 0.33 | Interactive tables |
| shinycssloaders | ≥ 1.0.0 | Spinner widgets |
| zoo | ≥ 1.8.12 | Rolling windows |
| scales | ≥ 1.3.0 | Axis formatting |
| mgcv | ≥ 1.9.1 | GAM / LOESS |
| ggcorrplot | ≥ 0.1.4 | Correlation heatmap |
| lme4 | ≥ 1.1.35 | Mixed-effects models |
| lmerTest | ≥ 3.1.3 | Satterthwaite p-values for LME |
| testthat | ≥ 3.2.1 | Automated tests |

Install manually (if not using renv):
```r
install.packages(c(
  "shiny","bslib","tidyverse","plotly","DT","shinycssloaders",
  "zoo","scales","mgcv","ggcorrplot","lme4","lmerTest","testthat"
))
```

---

## Running the automated tests

```r
setwd("app")                               # must be inside app/ for paths to resolve
testthat::test_file("tests/test_smoke.R")
```

Tests cover: CSV loading, date parsing, derived variable computation
(RecoveryScore, BB_delta, SleepQuality_pct), and empty-filter edge cases.
All tests must pass before committing changes.

---

## Dashboard tabs

| Tab | Contents |
|---|---|
| **Foundation** | Data coverage heatmap, participant overview table, KPIs |
| **Training** | Running calendar, load distribution, economy, pace, RI |
| **Sleep & Circadian** | Stage trends, quality distribution, social jetlag, variability |
| **Recovery & Interactions** | Recovery timeline, respiration, weekday rhythm, per-participant means |
| **Relationships** | Training→Sleep, Training→Recovery, Sleep→Recovery, Statistical Analysis |
| **Fitness & Individuals** | VO₂max trajectories, resilience scatter, individual deep-dive |

### Global filters
A filter bar at the top of every page controls:
- **Date range** — defaults to full study period (2022-01-01 to 2026-02-14)
- **Sex** — Male / Female / both

---

## Key variable definitions

See `data_dictionary.md` for all raw columns. Key derived variables:

- **RecoveryScore** = −(daily RHR − 7-day rolling mean RHR per participant). Positive = well-recovered.
- **BB_delta** = `charged(t+1) − drained(t)`, clamped to ±20. Aligns with RecoveryScore(t+1).
- **SleepQuality_pct** = (DeepSleepProp + REMSleepProp_final) × 100. Values > 75% excluded as artefacts.
- **SL_Intensity** = Light / Moderate / High based on 33rd and 67th percentile of non-zero SessionLoad.
- **RE** (Running Economy) = ActivitiesAvgSpeed_kmh / ActivitiesAvgHr.

---

## Analysis cohorts

| Cohort | Participants | Used for |
|---|---|---|
| `full` | 11 (excludes P006, P013, P015) | Any figure using sleep staging (DeepSleepProp, REM, SleepQuality_pct) |
| all | 14 | Training load, RHR, Body Battery, respiration |

P006, P013, P015 lack 4-stage sleep data due to firmware limitations.
P008 has 0% Body Battery coverage (device not supported).

---

## Data provenance

Data were collected from personal Garmin devices worn continuously by volunteer
recreational runners based in Nairobi, Kenya. Data were exported via the Garmin
Connect API and combined into a single CSV file. All participant identifiers are
pseudonymised (P001–P015). No names, locations, or other direct identifiers are
present in the dataset.

Data collection period: 1 January 2022 – 14 February 2026.
Ethics: collected under informed consent for MSc research purposes.

---

## Known limitations

1. **n = 14** — between-person statistical comparisons are underpowered. All
   modelling is treated as descriptive or within-person.
2. **LOCF imputation** for VO₂max creates horizontal plateaus between observed
   readings — these are not real fitness changes.
3. **RecoveryScore** is an RHR proxy. HRV-based measures would be more direct.
4. **No ground-truth performance** — no race times or laboratory test results to
   validate wearable-derived metrics.
5. **Sleep staging** absent for 3 of 14 participants due to firmware version.

---

## Reproducing the report

After the app runs cleanly:
```r
quarto::quarto_render("nairobi_runners_report.qmd", output_format = "html")
```
The rendered report uses the same data prep code as the app. Key numbers
(mean RecoveryScore, Spearman ρ values, model coefficients) are computed inline
in code chunks and will match the dashboard exactly.

---

## sessionInfo snapshot

Recorded on the development machine (update this after any package upgrade):

```
R version 4.5.0 (2025-04-11)
Platform: x86_64-w64-mingw32/x64
Running under: Windows 11 x64

Attached packages:
shiny_1.8.1        bslib_0.7.0        tidyverse_2.0.0    lubridate_1.9.3
plotly_4.10.4      DT_0.33            shinycssloaders_1.0.0  zoo_1.8.12
scales_1.3.0       mgcv_1.9.1         ggcorrplot_0.1.4   lme4_1.1.35
lmerTest_3.1.3     testthat_3.2.1
```

Run `sessionInfo()` in your R console and paste the output here after restoring
from `renv.lock` to confirm environment parity.

---

## Licence

MIT — see `LICENSE`.
