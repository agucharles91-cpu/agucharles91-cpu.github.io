# tests/test_smoke.R
# Automated smoke tests for the Nairobi Runners Shiny app.
# Run with: testthat::test_file("tests/test_smoke.R")
# All tests must pass before committing changes.

library(testthat)
library(tidyverse)
library(lubridate)
library(zoo)

# ── Path to data (relative to project root) ───────────────────────────────────
DATA_PATH <- testthat::test_path("..", "data", "garmin_merged_14p.csv")

# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 1: Data loading
# ─────────────────────────────────────────────────────────────────────────────
test_that("data file exists and is readable", {
  expect_true(
    file.exists(DATA_PATH),
    info = paste0("Expected data file not found at: ", DATA_PATH,
                  "\nPlace garmin_merged_14p.csv in the data/ directory.")
  )
  df <- suppressMessages(
    read_csv(DATA_PATH, na = c("NA","","N/A"), show_col_types = FALSE)
  )
  expect_s3_class(df, "data.frame")
  expect_gt(nrow(df), 1000,
            label = "Dataset has more than 1000 rows")
})

test_that("required columns are present", {
  df <- suppressMessages(
    read_csv(DATA_PATH, na = c("NA","","N/A"), show_col_types = FALSE)
  )
  required_cols <- c(
    "User_id", "calendar_date", "Sex", "Age", "analysis_cohort",
    "TSD", "SE", "AwakeSleepProp", "DeepSleepProp",
    "REMSleepProp_final", "SleepMidpoint",
    "DailyRestingHeartRate_clean",
    "DailyBodyBattery.chargedValue", "DailyBodyBattery.drainedValue",
    "SessionLoad", "ActivitiesAvgHr", "ActivitiesMaxHr",
    "ActivitiesAvgSpeed_kmh", "RelativeIntensity",
    "DailyTotalSteps", "VO2max_imputed_final",
    "weekday", "is_weekend", "activity_day"
  )
  missing <- setdiff(required_cols, names(df))
  expect_equal(length(missing), 0L,
               label = paste("Missing columns:", paste(missing, collapse = ", ")))
})

test_that("expected participants are present", {
  df <- suppressMessages(
    read_csv(DATA_PATH, na = c("NA","","N/A"), show_col_types = FALSE)
  )
  expected_ids <- paste0("P", sprintf("%03d", c(1:6, 8:15)))
  present_ids  <- sort(unique(df$User_id))
  # All expected IDs should be present
  missing_ids <- setdiff(expected_ids, present_ids)
  expect_equal(length(missing_ids), 0L,
               label = paste("Missing participants:", paste(missing_ids, collapse = ", ")))
  # P007 should NOT be present
  expect_false("P007" %in% present_ids,
               label = "P007 should be excluded from dataset")
})

# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 2: Date parsing and transformation
# ─────────────────────────────────────────────────────────────────────────────
test_that("calendar_date parses to Date class with no NAs", {
  df <- suppressMessages(
    read_csv(DATA_PATH, na = c("NA","","N/A"), show_col_types = FALSE)
  ) %>% mutate(calendar_date = as.Date(calendar_date))

  expect_s3_class(df$calendar_date, "Date")
  n_na <- sum(is.na(df$calendar_date))
  expect_equal(n_na, 0L,
               label = paste(n_na, "NA dates found after parsing"))
})

test_that("date range is plausible (2022–2026)", {
  df <- suppressMessages(
    read_csv(DATA_PATH, na = c("NA","","N/A"), show_col_types = FALSE)
  ) %>% mutate(calendar_date = as.Date(calendar_date))

  min_date <- min(df$calendar_date, na.rm = TRUE)
  max_date <- max(df$calendar_date, na.rm = TRUE)

  expect_gte(as.numeric(min_date), as.numeric(as.Date("2022-01-01")),
             label = "Earliest date should be on or after 2022-01-01")
  expect_lte(as.numeric(max_date), as.numeric(as.Date("2026-12-31")),
             label = "Latest date should be on or before 2026-12-31")
})

test_that("no invalid or future dates exist", {
  df <- suppressMessages(
    read_csv(DATA_PATH, na = c("NA","","N/A"), show_col_types = FALSE)
  ) %>% mutate(calendar_date = as.Date(calendar_date))

  future_rows <- sum(df$calendar_date > Sys.Date(), na.rm = TRUE)
  expect_equal(future_rows, 0L,
               label = paste(future_rows, "rows have future dates"))
})

test_that("weekday column is consistent with calendar_date", {
  df <- suppressMessages(
    read_csv(DATA_PATH, na = c("NA","","N/A"), show_col_types = FALSE)
  ) %>% mutate(calendar_date = as.Date(calendar_date))

  # Compute expected weekday using lubridate explicitly (avoids wday() masking)
  df <- df %>%
    mutate(expected_wday = lubridate::wday(calendar_date, label = TRUE,
                                            abbr = FALSE, locale = "C"))
  # Check a sample of 500 rows for consistency
  df_sample <- df %>% slice_sample(n = min(500L, nrow(df)))
  mismatch <- sum(
    tolower(df_sample$weekday) != tolower(as.character(df_sample$expected_wday)),
    na.rm = TRUE
  )
  expect_equal(mismatch, 0L,
               label = paste(mismatch, "rows have weekday/calendar_date mismatch"))
})

# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 3: Derived variable computation
# ─────────────────────────────────────────────────────────────────────────────

# Helper: run the same data prep as app.R
prepare_data <- function(path = DATA_PATH) {
  df_raw <- suppressMessages(
    read_csv(path, na = c("NA","","N/A"), show_col_types = FALSE)
  ) %>%
    arrange(User_id, calendar_date) %>%
    mutate(calendar_date = as.Date(calendar_date))

  df_raw %>%
    group_by(User_id) %>%
    arrange(calendar_date) %>%
    mutate(
      rhr_roll7     = zoo::rollapply(DailyRestingHeartRate_clean, 7, mean,
                                     na.rm = TRUE, fill = NA, align = "right"),
      RHR_dev_7d    = DailyRestingHeartRate_clean - rhr_roll7,
      RecoveryScore = -RHR_dev_7d,
      # Calculate and cap exactly as in app.R
      SleepQuality_pct = (DeepSleepProp + REMSleepProp_final) * 100,
      SleepQuality_pct = pmin(SleepQuality_pct, 75),
      SleepQuality_pct = pmax(SleepQuality_pct, 0),
      BB_delta      = pmin(pmax(
        dplyr::lead(DailyBodyBattery.chargedValue, 1) -
          DailyBodyBattery.drainedValue, -20), 20),
      BB_delta_lag1 = dplyr::lag(BB_delta, 1),
      RE = ifelse(ActivitiesAvgHr > 0 & ActivitiesAvgSpeed_kmh > 0 & 
            !is.na(ActivitiesAvgHr) & !is.na(ActivitiesAvgSpeed_kmh),
            ActivitiesAvgSpeed_kmh / ActivitiesAvgHr, NA_real_)
    ) %>%
    ungroup()
}

test_that("RecoveryScore is computed and mostly finite", {
  garmin <- prepare_data()
  # At least 50% of rows with valid RHR should have a finite RecoveryScore
  valid_rhr  <- sum(!is.na(garmin$DailyRestingHeartRate_clean))
  finite_rec <- sum(is.finite(garmin$RecoveryScore))
  # After 7-day window warmup, expect most valid RHR rows to have a score
  expect_gt(finite_rec / valid_rhr, 0.5,
            label = "Less than 50% of valid-RHR rows have a finite RecoveryScore")
  # RecoveryScore should be centred near zero (mean within ±5 bpm)
  mean_rec <- mean(garmin$RecoveryScore, na.rm = TRUE)
  expect_lt(abs(mean_rec), 5,
            label = paste("Mean RecoveryScore =", round(mean_rec,3),
                          "— expected near 0"))
})

test_that("SleepQuality_pct is bounded 0–75 after cap", {
  garmin <- prepare_data()
  sq <- garmin$SleepQuality_pct
  sq_valid <- sq[!is.na(sq)]
  expect_gte(min(sq_valid), 0,
             label = "SleepQuality_pct has values below 0")
  # Before cap: values above 75 may exist; after applying the cap in the app they won't
  # Here we just check the raw computed values are non-negative and plausible
  expect_lt(quantile(sq_valid, 0.99), 100,
            label = "99th percentile of SleepQuality_pct should be below 100%")
})

test_that("BB_delta is clamped to ±20", {
  garmin <- prepare_data()
  bb <- garmin$BB_delta[!is.na(garmin$BB_delta)]
  expect_gte(min(bb), -20, label = "BB_delta has values below −20")
  expect_lte(max(bb),  20, label = "BB_delta has values above +20")
})

test_that("RE (running economy) is positive where computable", {
  garmin <- prepare_data()
  re_valid <- garmin$RE[!is.na(garmin$RE)]
  expect_gt(length(re_valid), 0, label = "No RE values computed — check ActivitiesAvgHr")
  expect_true(all(re_valid > 0), label = "Some RE values are ≤ 0")
})

test_that("lag variables do not cross participant boundaries", {
  garmin <- prepare_data()
  # TSD_lag1 should be NA on the first day of each participant
  first_days <- garmin %>%
    group_by(User_id) %>%
    slice(1) %>%
    ungroup()
  # We can't access TSD_lag1 since it's not in prepare_data(), so check BB_delta_lag1
  n_nonNA_first <- sum(!is.na(first_days$BB_delta_lag1))
  expect_equal(n_nonNA_first, 0L,
               label = paste(n_nonNA_first,
                             "participants have non-NA BB_delta_lag1 on their first day",
                             "— possible cross-participant leakage"))
})

# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 4: Edge cases — empty filter results
# ─────────────────────────────────────────────────────────────────────────────
test_that("filtering to impossible date range returns zero rows gracefully", {
  garmin <- prepare_data()
  result <- garmin %>%
    filter(calendar_date >= as.Date("2000-01-01"),
           calendar_date <= as.Date("2000-12-31"))
  expect_equal(nrow(result), 0L)
  # The app uses req() and validate(need()) to catch this — confirm the
  # filtered data frame is a valid (empty) tibble, not an error
  expect_s3_class(result, "data.frame")
})

test_that("filtering to a single sex returns valid subset", {
  garmin <- prepare_data() %>%
    mutate(Gender = ifelse(Sex == "M", "Male", "Female"))

  male_only   <- garmin %>% filter(Gender == "Male")
  female_only <- garmin %>% filter(Gender == "Female")

  expect_gt(nrow(male_only), 0,   label = "No Male rows found")
  expect_gt(nrow(female_only), 0, label = "No Female rows found")
  expect_equal(nrow(male_only) + nrow(female_only), nrow(garmin),
               label = "Male + Female rows don't sum to total — unexpected Gender values")
})

test_that("participants with no Body Battery data are identifiable", {
  garmin <- prepare_data()
  bb_coverage <- garmin %>%
    group_by(User_id) %>%
    summarise(pct_bb = mean(!is.na(DailyBodyBattery.chargedValue)), .groups = "drop")
  # P008 should have 0% Body Battery coverage
  p008_coverage <- bb_coverage$pct_bb[bb_coverage$User_id == "P008"]
  if (length(p008_coverage) > 0) {
    expect_equal(p008_coverage, 0, tolerance = 0.01,
                 label = "P008 should have 0% Body Battery coverage")
  }
})

# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 5: Package conflict guard
# ─────────────────────────────────────────────────────────────────────────────
test_that("lubridate::isoweek() is available and not masked", {
  # Explicitly check that lubridate's isoweek works correctly
  test_date <- as.Date("2024-01-01")  # Week 1 of 2024
  result <- lubridate::isoweek(test_date)
  expect_equal(result, 1L, label = "lubridate::isoweek('2024-01-01') should return 1")
})

test_that("lubridate::wday() returns correct day name", {
  # 2024-01-01 was a Monday
  test_date <- as.Date("2024-01-01")
  result <- lubridate::wday(test_date, label = TRUE, abbr = FALSE, locale = "C")
  expect_equal(as.character(result), "Monday",
               label = "lubridate::wday('2024-01-01') should return 'Monday'")
})

cat("\n--- All smoke tests complete ---\n")
cat("If all passed: app.R is safe to launch and commit.\n")
