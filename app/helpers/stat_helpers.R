# helpers/stat_helpers.R  — rolling windows, ATL/CTL/TSB, anomaly detection

library(dplyr)
library(zoo)

# ── Rolling helpers ──────────────────────────────────────────────────────────
roll_mean <- function(x, k = 7, fill = NA) {
  zoo::rollmean(x, k = k, fill = fill, align = "right")
}
roll_sd <- function(x, k = 14, fill = NA) {
  zoo::rollapply(x, width = k, FUN = sd, fill = fill, align = "right", na.rm = TRUE)
}

# ── ATL / CTL / TSB (Performance Management Chart) ──────────────────────────
# ATL: Acute Training Load  (7-day EWM)
# CTL: Chronic Training Load (42-day EWM)
# TSB: Training Stress Balance = CTL - ATL
compute_pmc <- function(df, load_col = "SessionLoad") {
  df <- df %>%
    arrange(calendar_date) %>%
    mutate(
      load_val  = ifelse(is.na(.data[[load_col]]), 0, .data[[load_col]]),
      ATL       = stats::filter(load_val, filter = 1/7,  method = "recursive", sides = 1),
      CTL       = stats::filter(load_val, filter = 1/42, method = "recursive", sides = 1),
      TSB       = as.numeric(CTL) - as.numeric(ATL)
    )
  df
}

# ── Anomaly detection (2 SD from rolling mean) ───────────────────────────────
flag_anomalies <- function(x, window = 30, threshold = 2) {
  rm  <- roll_mean(x, k = window, fill = NA)
  rsd <- roll_sd(x, k = window, fill = NA)
  abs(x - rm) > (threshold * rsd)
}

# ── Sleep Regularity Index (SRI) ─────────────────────────────────────────────
# SRI: % of time in same sleep/wake state as 24h ago (-100 to 100)
# Approximated using SleepMidpoint consistency
compute_sri <- function(midpoints) {
  n   <- length(midpoints)
  if (n < 2) return(NA_real_)
  diffs <- abs(diff(midpoints))
  diffs[diffs > 12] <- 24 - diffs[diffs > 12]  # circular distance
  100 * (1 - mean(diffs, na.rm = TRUE) / 12)
}

# ── Cohort filter lists ───────────────────────────────────────────────────────
COHORT_LEVELS <- list(
  "Full sleep metrics"     = c("full"),
  "All participants"       = c("full", "no_sleep_stages")
)

# ── Standard cohort disclaimer (use as plot subtitle) ─────────────────────────
COHORT_NOTE <- "N=14 recreational runners, Nairobi — cohort-level description, not population inference"

# ── Median vs mean guidance ────────────────────────────────────────────────────
# Skewed → median + IQR:  TSD, SE, SF, SessionLoad, SleepMidpoint, REMSleepProp
# ~Normal → mean + SD:    RHR, VO2max, RHR_dev_7d, RelativeIntensity
USE_MEDIAN <- c("TSD","SE","SF","SessionLoad","SleepMidpoint","REMSleepProp_final",
                "SleepMidpoint_sd","TIB_h")

central <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(NA_real_)
  median(x)
}
spread <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(NA_real_)
  IQR(x)
}

# ── Label helpers ─────────────────────────────────────────────────────────────
metric_labels <- c(
  TSD                        = "Total Sleep Duration (h)",
  SE                         = "Sleep Efficiency (0–1)",
  SF                         = "Sleep Fragmentation (0–1)",
  time_in_bed                = "Time In Bed (s)",
  SleepMidpoint              = "Sleep Midpoint (h from midnight)",
  SleepMidpoint_sd           = "Sleep Midpoint SD [28d]",
  SleepScore                 = "Sleep Score (composite)",
  DeepSleepProp              = "Deep Sleep Proportion",
  LightSleepProp             = "Light Sleep Proportion",
  REMSleepProp_final         = "REM Sleep Proportion",
  AwakeSleepProp             = "Awake Proportion",
  DailyRestingHeartRate_clean= "RHR (bpm)",
  RHR_dev                    = "RHR Deviation (bpm)",
  VO2max_imputed_final       = "VO₂max (ml/kg/min)",
  SessionLoad                = "Session Load (HR × min)",
  RelativeIntensity          = "Relative Intensity (AvgHR / MaxHR)",
  EfficiencyIndex            = "Efficiency Index (km / HR·min)",
  ActivitiesAvgSpeed_kmh     = "Speed (km/h)",
  DailyTotalSteps            = "Daily Steps",
  DailyTotalKilocalories     = "Daily Kilocalories",
  `DailyBodyBattery.chargedValue`  = "Body Battery Charged",
  `DailyBodyBattery.drainedValue`  = "Body Battery Drained",
  `DailyRespiration.avgWakingRespirationValue` = "Waking Respiration (brpm)"
)
