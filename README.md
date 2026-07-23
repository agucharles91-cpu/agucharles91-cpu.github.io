# Nairobi Runners — Training, Sleep & Recovery Analytics

**MSc Data Science Capstone** | 14 participants | Garmin wearables | 2022–2026

A Shiny dashboard and Quarto report analysing longitudinal wearable data from
14 recreational runners in Nairobi, Kenya. Covers training load, sleep architecture,
recovery physiology, and their temporal relationships.

## Live demo

🔗 **[https://agucharles.shinyapps.io/nairobi-runners/](https://agucharles.shinyapps.io/nairobi-runners/)**

> Hosted on shinyapps.io free tier (25 active hours/month).  
> If the app shows "not available", the monthly limit has been reached — clone the repo and run locally instead using `shiny::runApp("app/app.R")`.
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
R version 4.5.1 (2025-06-13 ucrt)
Platform: x86_64-w64-mingw32/x64
Running under: Windows 11 x64 (build 26200)

Matrix products: default
  LAPACK version 3.12.1

locale:
[1] LC_COLLATE=English_United States.utf8  LC_CTYPE=English_United States.utf8    LC_MONETARY=English_United States.utf8
[4] LC_NUMERIC=C                           LC_TIME=English_United States.utf8    

time zone: Africa/Nairobi
tzcode source: internal

attached base packages:
[1] stats     graphics  grDevices utils     datasets  methods   base     

other attached packages:
 [1] testthat_3.3.2        lmerTest_3.2-1        lme4_2.0-1            Matrix_1.7-3          ggcorrplot_0.1.4.1   
 [6] mgcv_1.9-3            nlme_3.1-168          scales_1.4.0          zoo_1.8-15            shinycssloaders_1.1.0
[11] DT_0.34.0             plotly_4.12.0         lubridate_1.9.5       forcats_1.0.0         stringr_1.6.0        
[16] dplyr_1.2.1           purrr_1.1.0           readr_2.1.5           tidyr_1.3.1           tibble_3.3.0         
[21] ggplot2_4.0.2         tidyverse_2.0.0       bslib_0.9.0           shiny_1.13.0         

loaded via a namespace (and not attached):
 [1] tidyselect_1.2.1    viridisLite_0.4.3   farver_2.1.2        S7_0.2.0            fastmap_1.2.0       lazyeval_0.2.2     
 [7] promises_1.5.0      digest_0.6.39       timechange_0.4.0    mime_0.13           lifecycle_1.0.5     magrittr_2.0.4     
[13] compiler_4.5.1      rlang_1.2.0         sass_0.4.10         tools_4.5.1         yaml_2.3.10         data.table_1.17.8  
[19] knitr_1.50          htmlwidgets_1.6.4   bit_4.6.0           RColorBrewer_1.1-3  rsconnect_1.8.0     withr_3.0.2        
[25] numDeriv_2016.8-1.1 grid_4.5.1          xtable_1.8-4        MASS_7.3-65         cli_3.6.5           crayon_1.5.3       
[31] reformulas_0.4.3.1  generics_0.1.4      otel_0.2.0          rstudioapi_0.18.0   httr_1.4.7          tzdb_0.5.0         
[37] minqa_1.2.8         cachem_1.1.0        splines_4.5.1       parallel_4.5.1      vctrs_0.7.3         boot_1.3-31        
[43] jsonlite_2.0.0      hms_1.1.3           bit64_4.6.0-1       archive_1.1.12.1    crosstalk_1.2.2     jquerylib_0.1.4    
[49] glue_1.8.0          nloptr_2.2.1        stringi_1.8.7       gtable_0.3.6        later_1.4.4         pillar_1.11.1      
[55] brio_1.1.5          htmltools_0.5.8.1   R6_2.6.1            Rdpack_2.6.5        vroom_1.6.5         evaluate_1.0.5     
[61] lattice_0.22-7      rbibutils_2.4.1     memoise_2.0.1       httpuv_1.6.17       Rcpp_1.1.1-1.1      xfun_0.53          
[67] fs_1.6.6            pkgconfig_2.0.3    

```

Run `sessionInfo()` in your R console and paste the output here after restoring
from `renv.lock` to confirm environment parity.

---

## Licence

MIT — see `LICENSE`.
