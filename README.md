# Garmin Wearable Analytics — Shiny App v3.0

## Overview

14-participant longitudinal wearable analytics dashboard.  
**Data**: `garmin_merged_14p.csv` (12,760 rows × 69 cols)  
**Engine**: R Shiny + shinydashboard, fully modular

---

## Quick start

```r
# 1. Install required packages (run once)
install.packages(c(
  "shiny", "shinydashboard",
  "tidyverse", "lubridate",
  "DT", "scales", "zoo",
  "cowplot", "ggridges", "GGally"
))

# 2. Run the app
shiny::runApp("garmin_shiny/")
```

---

## File structure

```
garmin_shiny/
├── app.R                   # Main: ui + server + global reactives
├── data/
│   ├── garmin_merged_14p.csv
│   └── data_prep.R         # Load, cast, derive columns
├── modules/
│   ├── mod_overview.R      # Tab 1: KPI cards, completeness heatmap, timeline
│   ├── mod_sleep.R         # Tab 2: TSD/SE/SF time series, debt, heatmap
│   ├── mod_architecture.R  # Tab 3: Stage props + circadian midpoint + SRI
│   ├── mod_training.R      # Tab 4: SessionLoad, PMC (ATL/CTL/TSB), heatmap
│   ├── mod_recovery.R      # Tab 5: BB, RHR, Respiration × Sleep
│   └── mod_extra.R         # Tabs 6–8: VO2max, Individual deep dive, Explorer
├── helpers/
│   ├── plot_themes.R       # Shared ggplot2 theme + palette (14 colours)
│   └── stat_helpers.R      # Rolling windows, ATL/CTL/TSB, SRI, anomaly flags
└── www/
    └── custom.css          # Bootstrap overrides, dark navy navbar
```

---

## Tabs

| Tab | Module | Key inputs | Key outputs |
|-----|--------|------------|-------------|
| Overview | mod_overview | Participants, date range, cohort | KPI cards, completeness heatmap, timeline |
| Sleep Analysis | mod_sleep | Metric, smoother, sleep debt target | Time series, distribution, weekday box, calendar heatmap |
| Sleep Architecture | mod_architecture | Group-by, chart type, SRI window | Stage stacked bar/radar/scatter, midpoint violin, SRI rolling |
| Training Load | mod_training | Activity types, window, PMC uid | SessionLoad LOESS, PMC (ATL/CTL/TSB), activity heatmap |
| VO₂max | mod_vo2 | Source filter, smoothing | Trajectories, annual delta table, VO2 × TSD scatter |
| Recovery | mod_recovery | Metrics, lag, BB exclusion | BB net, RHR heatmap, respiration, RHR × lagged TSD |
| Individual Dive | mod_individual | Participant, primary/secondary metrics | Dual-axis, stage area, activity profile, anomaly detector |
| Cohort Comparison | inline | Metric selection, normalisation | Correlation heatmap, ridge plots, parallel coords, pairwise scatter |
| Data Explorer | mod_explorer | Column groups, filters | Filterable DT table, missingness bar, CSV/RDS download |

---

## analysis_cohort filter logic

```r
# The global sidebar "Analysis cohort" dropdown maps to:
full           → filter(analysis_cohort == "full")                          # n = 9,737
full_legacy    → filter(analysis_cohort %in% c("full","legacy_staging"))    # n = 9,883
full_legacy_imp→ filter(analysis_cohort %in% c("full","legacy_staging","tsd_imputed"))  # n = 11,183
all            → all rows                                                    # n = 12,760
```

The `show_imputed_global` checkbox additionally surfaces P006 wake-bucket imputed TSD (n=1,300)  
in sleep-specific plots that would otherwise exclude them.

---

## P006 staging notes

P006 has three staging modes in the data — all transparently flagged:

| staging_mode | n | Valid for |
|---|---|---|
| `4stage_rem` | 37 | All sleep metrics (cross-cohort comparable) |
| `legacy_3stage` | 146 | TSD/SE/SF/Midpoint — NOT stage proportions |
| `tsd_imputed` | 1,300 | TSD (±57min MAE), Midpoint only |

Filter `sleep_prop_source == "4stage_rem_comparable"` for cross-cohort stage analyses.
