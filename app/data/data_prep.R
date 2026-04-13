# data/data_prep.R  — load garmin_merged_14p.csv, type-cast, derive ready columns

library(readr)
library(dplyr)
library(lubridate)

load_garmin <- function(path = "data/garmin_merged_14p.csv") {
  raw <- read_csv(path, show_col_types = FALSE, na = c("NA","","nan","NA NA"))

  df <- raw %>%
    mutate(
      calendar_date  = as.Date(calendar_date),
      year           = year(calendar_date),
      month          = month(calendar_date, label = TRUE, abbr = TRUE),
      week           = isoweek(calendar_date),
      weekday        = factor(weekday,
                              levels = c("Monday","Tuesday","Wednesday","Thursday",
                                         "Friday","Saturday","Sunday")),
      is_weekend     = (weekday %in% c("Saturday","Sunday")),
      activity_day   = (activity_day == "TRUE"),

      # Numeric casts
      across(c(TSD, SE, SF, SleepMidpoint, SleepMidpoint_sd, SleepScore,
               DeepSleepProp, LightSleepProp, REMSleepProp_final, AwakeSleepProp,
               DailyRestingHeartRate_clean, DailyMinHeartRate_clean, DailyMaxHeartRate_clean,
               RHR_dev, VO2max_imputed_final, SessionLoad, RelativeIntensity,
               EfficiencyIndex, ActivitiesAvgSpeed_kmh, ActivitiesAvgSpeed_ms,
               DailyTotalSteps, DailyTotalKilocalories, DailyTotalDistanceMeters,
               `DailyBodyBattery.chargedValue`, `DailyBodyBattery.drainedValue`,
               `DailyRespiration.avgWakingRespirationValue`,
               ActivitiesAvgHr, ActivitiesMaxHr, ActivitiesVO2MaxValue,
               ActivitiesDurationMinutes, ActivitiesDistance_km, HR_range,
               SleepDeepSleepSeconds, SleepLightSleepSeconds,
               SleepRemSleepSeconds, SleepTotalSleepSeconds, SleepAwakeSleepSeconds,
               time_in_bed), as.numeric),

      # Derived convenience columns
      TSD_h         = TSD,
      TIB_h         = time_in_bed / 3600,
      BB_net        = `DailyBodyBattery.chargedValue` - `DailyBodyBattery.drainedValue`,
      DeepSleep_h   = SleepDeepSleepSeconds / 3600,
      LightSleep_h  = SleepLightSleepSeconds / 3600,
      REMSleep_h    = SleepRemSleepSeconds / 3600,
      AwakeSleep_h  = SleepAwakeSleepSeconds / 3600,

      # Simplified two-tier cohort
      # full            → 9,700 rows: all sleep metrics valid
      # no_sleep_stages → 3,060 rows: P006 + P013 + P015 (sleep excluded)
      analysis_cohort = factor(analysis_cohort,
        levels = c("full", "no_sleep_stages")),
      staging_mode    = factor(staging_mode,
        levels = c("native_4stage","4stage_rem","legacy_3stage","tsd_imputed")),

      # RHR_dev_7d: deviation from personal 7-day rolling mean
      # More interpretable than long-term baseline — captures acute recovery state
      # Computed below via group_by after mutate chain
      rhr_dev_available = !is.na(RHR_dev),
      stage_comparable  = sleep_prop_source %in%
          c("4stage_rem_comparable","original","recalculated",
            "original_recalculated","rem_recalculated")
    ) %>%
    # RHR_dev_7d: per-participant 7-day rolling mean deviation
    # Captures acute recovery state; generalizable across participants
    group_by(User_id) %>%
    arrange(calendar_date) %>%
    mutate(
      rhr_7d_mean  = zoo::rollmean(DailyRestingHeartRate_clean, k = 7,
                                   fill = NA, align = "right", na.rm = TRUE),
      RHR_dev_7d   = DailyRestingHeartRate_clean - rhr_7d_mean
    ) %>%
    ungroup() %>%
    arrange(User_id, calendar_date)

  df
}
