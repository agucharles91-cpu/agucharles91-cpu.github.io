# Data Dictionary — garmin_merged_14p.csv

One row = one participant × one calendar day.
**Rows:** ~12,760 | **Participants:** 14 (P001–P015, excluding P007)
**Date range:** 2022-01-01 to 2026-02-14

---

## Identifiers

| Column | Type | Description | Notes |
|---|---|---|---|
| `User_id` | character | Pseudonymised participant ID | P001–P015, P007 absent |
| `calendar_date` | date (YYYY-MM-DD) | Observation date | Parsed with `as.Date()` |
| `Sex` | character | Biological sex | "M" or "F" |
| `Age` | integer | Age in years at study start | |
| `analysis_cohort` | character | Sleep staging availability | "full" = 4-stage data; "no_sleep_stages" = P006, P013, P015 |

---

## Sleep variables

| Column | Type | Unit | Missingness | Description |
|---|---|---|---|---|
| `TSD` | numeric | hours | ~8% | Total sleep duration. Rows with TSD < 3h or > 12h are excluded as device artefacts. |
| `SE` | numeric | proportion (0–1) | ~9% | Sleep efficiency = time asleep / time in bed. Values ≥ 1.0 are excluded. |
| `AwakeSleepProp` | numeric | proportion (0–1) | ~9% | Proportion of time in bed spent awake. Multiply × 100 for Sleep Fragmentation %. |
| `DeepSleepProp` | numeric | proportion (0–1) | ~30% | Proportion of sleep in deep (slow-wave) stage. Absent for P006, P013, P015. |
| `LightSleepProp` | numeric | proportion (0–1) | ~30% | Proportion of sleep in light stage. Absent for P006, P013, P015. |
| `REMSleepProp_final` | numeric | proportion (0–1) | ~38% | REM sleep proportion. LOCF-imputed within participant where possible. Absent for P006, P013, P015. |
| `SleepMidpoint` | numeric | hours (decimal, 0–24) | ~10% | Clock time of sleep midpoint. Used for social jetlag calculation. |
| `SleepAwakeSleepSeconds` | numeric | seconds | ~9% | Total seconds spent awake during the sleep period. |
| `SocialJetlag` | numeric | hours | ~15% | Difference between weekday and weekend sleep midpoint per participant per week. |

### Derived sleep variables (computed in app.R)
| Variable | Formula | Notes |
|---|---|---|
| `SleepQuality_pct` | `(DeepSleepProp + REMSleepProp_final) × 100` | Capped at 75% (values above are physiologically implausible device artefacts). Full cohort only. |

---

## Recovery & physiology variables

| Column | Type | Unit | Missingness | Description |
|---|---|---|---|---|
| `DailyRestingHeartRate_clean` | numeric | bpm | ~5% | Resting heart rate. Cleaned (outliers removed by Garmin firmware). |
| `DailyRespiration.avgWakingRespirationValue` | numeric | breaths/min | ~20% | Average waking respiration rate. Normal range in fit adults: 12–16. |
| `DailyBodyBattery.chargedValue` | numeric | 0–100 | ~25% | Garmin Body Battery charged during sleep. P008 = 0% coverage (unsupported device). |
| `DailyBodyBattery.drainedValue` | numeric | 0–100 | ~25% | Body Battery drained during the waking day. P008 = 0% coverage. |

### Derived recovery variables (computed in app.R)
| Variable | Formula | Notes |
|---|---|---|
| `rhr_roll7` | 7-day rolling mean of `DailyRestingHeartRate_clean`, per participant | Right-aligned window |
| `RHR_dev_7d` | `DailyRestingHeartRate_clean − rhr_roll7` | Positive = RHR elevated above personal baseline |
| `RecoveryScore` | `−RHR_dev_7d` | Positive = well-recovered; negative = under physiological stress |
| `rec_roll7` | 7-day rolling mean of RecoveryScore | Used for trend line in timeline figures |
| `BB_delta` | `pmin(pmax(lead(chargedValue,1) − drainedValue, −20), 20)` | Net Body Battery change, clamped ±20. Aligns with RecoveryScore(t+1). |
| `BB_delta_lag1` | `lag(BB_delta, 1)` | Used in weekday rhythm chart — aligns with same-morning RecoveryScore(t) |

---

## Training / activity variables

| Column | Type | Unit | Missingness | Description |
|---|---|---|---|---|
| `activity_day` | logical | — | ~0% | TRUE if any activity was recorded that day |
| `ActivitiesType` | character | — | ~48% | Activity type string from Garmin (e.g. "running", "cycling", "strength_training") |
| `SessionLoad` | numeric | arbitrary units | ~48% | Session load = ActivitiesAvgHr × session duration in minutes. Structural missingness on rest days. |
| `TV` | numeric | km | ~48% | Distance covered in the session (used for running distance). |
| `ActivitiesAvgHr` | numeric | bpm | ~48% | Average heart rate during the session. |
| `ActivitiesMaxHr` | numeric | bpm | ~48% | Maximum heart rate recorded during the session. |
| `ActivitiesAvgSpeed_kmh` | numeric | km/h | ~50% | Average speed during the session, converted to km/h. |
| `RelativeIntensity` | numeric | proportion (0–1) | ~50% | ActivitiesAvgHr / ActivitiesMaxHr. Multiply × 100 for RI%. |
| `DailyTotalSteps` | numeric | steps | ~5% | Total step count for the day (all movement, not just sessions). |

### Derived training variables (computed in app.R)
| Variable | Formula | Notes |
|---|---|---|
| `SL_Intensity` | Tertile split of non-zero SessionLoad | "Rest Day" / "Light" / "Moderate" / "High" |
| `ActivityGroup` | Mapped from ActivitiesType | "Running" / "Cycling" / "Strength-HIIT" / "Swimming-Rowing" / "Mind-Body" / "Other" / "Rest Day" |
| `RE` | `ActivitiesAvgSpeed_kmh / ActivitiesAvgHr` | Running economy — higher = more efficient. Running sessions only. |
| `Load_roll7` | 7-day rolling sum of SessionLoad (rest days = 0) | Cumulative training load |
| `overtraining_flag` | SessionLoad > p75 AND RHR_dev_7d > 1.5 AND (TSD < median OR SE < 0.85) | Composite risk flag |

---

## Fitness variables

| Column | Type | Unit | Missingness | Description |
|---|---|---|---|---|
| `VO2max_imputed_final` | numeric | ml/kg/min | ~3% | VO₂max estimated by Garmin algorithm. **Use this column** — LOCF-imputed for gaps ≤28 days. |
| `ActivitiesVO2MaxValue` | numeric | ml/kg/min | ~70% | Raw Garmin VO₂max reading. Very sparse — do not use for trend analysis. |

---

## Temporal / calendar variables

| Column | Type | Description |
|---|---|---|
| `weekday` | character | Full weekday name (e.g. "Monday"). Pre-computed in source data. |
| `is_weekend` | logical | TRUE for Saturday and Sunday |

**Note:** `lubridate::isoweek()` and `lubridate::isoyear()` are called explicitly
with the namespace prefix throughout the app to avoid conflicts with any other
package that exports a `wday()` or `isoweek()` function (e.g. `data.table`, `tsibble`).

---

## Lag variables (all computed within participant groups)

All lag/lead operations use `dplyr::lag()` and `dplyr::lead()` within
`group_by(User_id)` to prevent cross-participant leakage.

| Variable | Description |
|---|---|
| `TSD_lag1`, `TSD_lag2`, `TSD_lag3` | TSD shifted 1–3 days forward |
| `SE_lag1`, `SE_lag2`, `SE_lag3` | SE shifted 1–3 days forward |
| `SessionLoad_lag1/2/3` | SessionLoad shifted 1–3 days forward |
| `RHR_dev_7d_lag1` | RHR deviation shifted 1 day forward |
| `TSD_next`, `SE_next2`, `SF_next`, `SQ_next2` | Next-night sleep metrics (lead 1) |
| `Rec_next`, `BB_delta_next` | Next-day recovery and BB (lead 1) |

---

## Missing data summary

| Variable | Approx. % missing | Cause |
|---|---|---|
| SessionLoad, ActivitiesType, TV, ActivitiesAvgHr | ~48% | Structural — rest days have no activity |
| REMSleepProp_final, DeepSleepProp, LightSleepProp | ~30–38% | Structural — P006, P013, P015 lack 4-stage firmware |
| DailyBodyBattery.* | ~25% | P008 = 0% coverage; P009/P010 = 18–24% missing |
| DailyRespiration.* | ~20% | Device and wear-time variation |
| TSD, SE, AwakeSleepProp | ~8–9% | Random (device not worn, bad signal) |
| VO2max_imputed_final | ~3% | LOCF imputation used; gaps > 28 days remain NA |
| DailyRestingHeartRate_clean | ~5% | Random |
