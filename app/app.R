# ==============================================================================
# Training, Sleep & Recovery Analytics — Nairobi Runners (Definitive Edition)
# MSc Data Science Capstone  |  14 participants  |  2022–2026
# Run: shiny::runApp(".")
# Packages: shiny bslib tidyverse plotly DT shinycssloaders zoo scales lubridate mgcv ggcorrplot
# ==============================================================================

library(shiny)
library(bslib)
library(tidyverse)
library(plotly)
library(DT)
library(shinycssloaders)
library(zoo)
library(scales)
library(lubridate)
library(mgcv)
library(ggcorrplot)

# ==============================================================================
# 1. DATA LOADING & PREPARATION
# ==============================================================================

df_raw <- read_csv("app/data/garmin_merged_14p.csv",
                   na = c("NA","","N/A"), show_col_types = FALSE) %>%
  arrange(User_id, calendar_date) %>%
  mutate(calendar_date = as.Date(calendar_date))

# Pre-compute global thresholds BEFORE mutate (avoids masking dplyr's n())
sl_p75   <- quantile(df_raw$SessionLoad, 0.75, na.rm = TRUE)
tsd_med  <- median(df_raw$TSD,           na.rm = TRUE)
sl_33    <- quantile(df_raw$SessionLoad[df_raw$SessionLoad > 0 & !is.na(df_raw$SessionLoad)], 0.33, na.rm = TRUE)
sl_67    <- quantile(df_raw$SessionLoad[df_raw$SessionLoad > 0 & !is.na(df_raw$SessionLoad)], 0.67, na.rm = TRUE)

# Helper: per-participant VO2max slope
slope_fn <- function(y, x) {
  tryCatch(coef(lm(y ~ x))[2] * 365, error = function(e) NA_real_)
}

garmin <- df_raw %>%
  group_by(User_id) %>%
  arrange(calendar_date) %>%
  mutate(
    # ── Primary recovery metric ──────────────────────────────────────────────
    rhr_roll7     = rollapply(DailyRestingHeartRate_clean, 7, mean,
                              na.rm = TRUE, fill = NA, align = "right"),
    RHR_dev_7d    = DailyRestingHeartRate_clean - rhr_roll7,
    RecoveryScore = -RHR_dev_7d,
    rec_roll7     = rollapply(RecoveryScore, 7, mean,
                              na.rm = TRUE, fill = NA, align = "right"),
    # ── 7-day cumulative training load ───────────────────────────────────────
    Load_roll7    = rollapply(replace_na(SessionLoad, 0), 7, sum,
                              fill = NA, align = "right"),
    # ── Lag variables (1–3 days within participant) ──────────────────────────
    TSD_lag1          = lag(TSD, 1),
    TSD_lag2          = lag(TSD, 2),
    TSD_lag3          = lag(TSD, 3),
    SE_lag1           = lag(SE, 1),
    SE_lag2           = lag(SE, 2),
    SE_lag3           = lag(SE, 3),
    SessionLoad_lag1  = lag(SessionLoad, 1),
    SessionLoad_lag2  = lag(SessionLoad, 2),
    SessionLoad_lag3  = lag(SessionLoad, 3),
    RHR_dev_7d_lag1   = lag(RHR_dev_7d, 1),
    NextDayRecovery   = lead(RecoveryScore, 1),
    # ── Body Battery delta (net energy change per day) ────────────────────────
    BB_delta          = pmin(pmax(lead(DailyBodyBattery.chargedValue, 1) -
                                   DailyBodyBattery.drainedValue, -20), 20),
    BB_delta_lag1     = lag(BB_delta, 1),
    # BB_delta(t) = charged(t+1) - drained(t): overnight recharge from night t
    # minus waking drain of day t. Pairs with RecoveryScore(t+1). ────────────
    # ── Sleep quality — must be here so SQ_next2 lead works ──────────────────
    SleepQuality_pct  = (DeepSleepProp + REMSleepProp_final) * 100,
    # ── Next-day lead variables for Relationships tab ─────────────────────────
    TSD_next          = lead(TSD, 1),
    SE_next2          = lead(SE, 1),
    SF_next           = lead(AwakeSleepProp * 100, 1),
    SQ_next2          = lead(SleepQuality_pct, 1),
    Rec_next          = lead(RecoveryScore, 1),
    BB_delta_next     = lead(BB_delta, 1),
    # ── Running economy (same-day) ────────────────────────────────────────────
    RE                = ifelse(ActivitiesAvgHr > 0 & !is.na(ActivitiesAvgHr),
                               ActivitiesAvgSpeed_kmh / ActivitiesAvgHr, NA_real_)
  ) %>%
  ungroup() %>%
  mutate(
    # ── Sleep quality level labels (uses SleepQuality_pct computed above) ────
    SleepQualityLevel = case_when(
      SleepQuality_pct >= 56 ~ "Rebound",
      SleepQuality_pct >= 48 ~ "Excellent",
      SleepQuality_pct >= 38 ~ "Normal",
      SleepQuality_pct >= 30 ~ "Fair",
      !is.na(SleepQuality_pct) ~ "Poor",
      TRUE ~ NA_character_
    ),
    # ── Training intensity category ──────────────────────────────────────────
    SL_Intensity = case_when(
      is.na(SessionLoad) | SessionLoad == 0 ~ "Rest Day",
      SessionLoad <= sl_33 ~ "Light",
      SessionLoad <= sl_67 ~ "Moderate",
      TRUE ~ "High"
    ),
    SL_Intensity = factor(SL_Intensity,
                          levels = c("Rest Day","Light","Moderate","High")),
    # ── Overtraining signal: high load + elevated RHR + poor sleep ───────────
    overtraining_flag = !is.na(SessionLoad) & SessionLoad > sl_p75 &
                        !is.na(RHR_dev_7d)  & RHR_dev_7d  > 1.5  &
                        ((!is.na(TSD) & TSD < tsd_med) |
                         (!is.na(SE)  & SE  < 0.85)),
    # ── Demographics ────────────────────────────────────────────────────────
    Gender    = ifelse(Sex == "M", "Male", "Female"),
    Weekday   = factor(weekday,
                       levels = c("Monday","Tuesday","Wednesday",
                                  "Thursday","Friday","Saturday","Sunday")),
    IsWeekend = ifelse(is_weekend, "Weekend", "Weekday"),
    # ── Activity grouping ────────────────────────────────────────────────────
    ActivityGroup = case_when(
      ActivitiesType %in% c("running","track_running","trail_running",
                            "treadmill_running")           ~ "Running",
      ActivitiesType %in% c("cycling","indoor_cycling",
                            "e_bike_fitness")              ~ "Cycling",
      ActivitiesType %in% c("strength_training","hiit")   ~ "Strength/HIIT",
      ActivitiesType %in% c("lap_swimming","open_water_swimming",
                            "indoor_rowing")              ~ "Swimming/Rowing",
      ActivitiesType %in% c("yoga","pilates","meditation",
                            "breathwork")                 ~ "Mind-Body",
      is.na(ActivitiesType)                               ~ "Rest Day",
      TRUE                                                ~ "Other"
    ),
    ActivityGroup = factor(ActivityGroup,
                           levels = c("Running","Cycling","Strength/HIIT",
                                      "Swimming/Rowing","Mind-Body",
                                      "Other","Rest Day")),
    # ── Sleep midpoint — trim artefacts at 1st / 99th percentile ───────────
    SleepMidpoint_clean = {
      lo <- quantile(SleepMidpoint, 0.01, na.rm = TRUE)
      hi <- quantile(SleepMidpoint, 0.99, na.rm = TRUE)
      ifelse(!is.na(SleepMidpoint) & SleepMidpoint >= lo & SleepMidpoint <= hi,
             SleepMidpoint, NA_real_)
    }
  )

# ── Participant VO2max trajectory slopes ──────────────────────────────────────
vo2_summary <- garmin %>%
  filter(!is.na(VO2max_imputed_final)) %>%
  group_by(User_id) %>%
  filter(n() >= 90) %>%
  summarise(
    vo2_slope     = slope_fn(VO2max_imputed_final, as.numeric(calendar_date)),
    vo2_mean      = mean(VO2max_imputed_final,  na.rm = TRUE),
    mean_recovery = mean(RecoveryScore,          na.rm = TRUE),
    rhr_dev_sd    = sd(RHR_dev_7d,               na.rm = TRUE),
    load_mean     = mean(SessionLoad,            na.rm = TRUE),
    Gender        = first(Gender),
    Age           = first(Age),
    .groups       = "drop"
  )

# ── Data coverage matrix ──────────────────────────────────────────────────────
COV_VARS <- c("TSD","SE","RHR_dev_7d","SessionLoad",
              "REMSleepProp_final","VO2max_imputed_final",
              "DailyBodyBattery.chargedValue",
              "DailyRespiration.avgWakingRespirationValue",
              "SleepQuality_pct","SleepMidpoint")
COV_LABELS <- c("Sleep Duration","Sleep Efficiency","RHR Deviation 7d",
                "Session Load","REM Proportion","VO2max",
                "Body Battery","Respiration Rate",
                "Sleep Quality %","Sleep Midpoint")

coverage_df <- garmin %>%
  group_by(User_id) %>%
  summarise(across(all_of(COV_VARS), ~round(100 * mean(!is.na(.)), 1)),
            .groups = "drop") %>%
  pivot_longer(-User_id, names_to = "Variable", values_to = "Coverage") %>%
  mutate(Variable = factor(Variable, levels = COV_VARS,
                            labels = COV_LABELS))

# ── Static participant info ───────────────────────────────────────────────────
USERS     <- sort(unique(garmin$User_id))
MIN_DATE  <- min(garmin$calendar_date, na.rm = TRUE)
MAX_DATE  <- max(garmin$calendar_date, na.rm = TRUE)

PART_COLORS <- setNames(
  colorRampPalette(c("#2C5F8A","#8A4A2C","#3A6B4A","#8A3A5F","#B07D2C",
                     "#4A6B8A","#6B4A8A","#8A6B2C","#2C8A5F","#8A2C5F",
                     "#5F8A2C","#2C4A8A","#8A5F2C","#4A8A6B"))(length(USERS)),
  USERS
)
ACT_COLORS <- c("Running"="#2C5F8A","Cycling"="#3A6B4A","Strength/HIIT"="#8A2C2C",
                "Swimming/Rowing"="#2C7A8A","Mind-Body"="#6B4A8A",
                "Other"="#B07D2C","Rest Day"="#C8C4BC")

# ==============================================================================
# 2. COLOUR & THEME CONSTANTS
# ==============================================================================

COL_P  <- "#2C5F8A"  # primary steel blue
COL_S  <- "#8A4A2C"  # secondary terracotta
COL_G  <- "#3A6B4A"  # good / success
COL_W  <- "#B07D2C"  # warning amber
COL_D  <- "#8A2C2C"  # danger red
COL_N  <- "#6B7280"  # neutral grey
DIV_PAL <- list(c(0,"#8A2C2C"), c(0.5,"#F5F5F0"), c(1,"#2C5F8A"))

acad_theme <- bs_theme(
  version      = 5,
  bg           = "#F5F4F0", fg = "#1A1A1A",
  primary      = "#2C5F8A", secondary = "#6B7280",
  success      = "#3A6B4A", warning   = "#B07D2C", danger = "#8A2C2C",
  base_font    = font_google("Lora"),
  heading_font = font_google("Libre Baskerville"),
  font_scale   = 0.95
)

acad_css <- "
body { background:#EEECE6; color:#1A1A1A; }
body::before { content:''; position:fixed; inset:0;
  background-image:repeating-linear-gradient(transparent,transparent 27px,rgba(180,170,150,0.18) 28px);
  pointer-events:none; z-index:0; }
.navbar { background:#1A1A1A !important; border-bottom:3px solid #2C5F8A; padding:0 28px;
  font-family:'Libre Baskerville',Georgia,serif; }
.navbar-brand { color:#F5F4F0 !important; font-size:16px; font-weight:700;
  letter-spacing:0.03em; padding:14px 0; }
.nav-link { color:#BBBBAA !important; font-size:13px; letter-spacing:0.06em;
  padding:14px 16px !important; border-bottom:3px solid transparent;
  font-family:'Libre Baskerville',Georgia,serif; }
.nav-link:hover { color:#F5F4F0 !important; }
.nav-link.active { color:#F5F4F0 !important; border-bottom:3px solid #2C5F8A !important;
  background:transparent !important; }
.page-body { background:#F5F4F0; padding:28px 32px; min-height:calc(100vh - 60px); }
.sec-head { font-family:'Libre Baskerville',Georgia,serif; font-size:20px; font-weight:700;
  color:#1A1A1A; margin:0 0 4px; padding-bottom:8px; border-bottom:2px solid #2C5F8A; }
.sec-sub  { font-size:12.5px; color:#6B7280; margin:0 0 22px; font-style:italic;
  font-family:'Lora',Georgia,serif; line-height:1.5; }
.acad-card { background:#FAFAF8; border:1px solid #DDD9D0; border-radius:3px;
  padding:20px 22px; margin-bottom:18px; box-shadow:0 1px 3px rgba(0,0,0,0.06);
  position:relative; }
.acad-card::before { content:''; position:absolute; left:0; top:14px; bottom:14px;
  width:3px; background:#2C5F8A; border-radius:0 2px 2px 0; }
.card-lbl { font-family:'Libre Baskerville',serif; font-size:10.5px; font-weight:700;
  text-transform:uppercase; letter-spacing:0.14em; color:#6B7280;
  margin-bottom:12px; padding-bottom:7px; border-bottom:1px solid #E8E4DE; }
.kpi-row  { display:flex; gap:14px; margin-bottom:22px; flex-wrap:wrap; }
.kpi-box  { flex:1; min-width:130px; background:#FAFAF8; border:1px solid #DDD9D0;
  border-top:3px solid #2C5F8A; border-radius:2px; padding:16px 18px 12px;
  box-shadow:0 1px 3px rgba(0,0,0,0.05); }
.kpi-box:nth-child(2){border-top-color:#3A6B4A}
.kpi-box:nth-child(3){border-top-color:#8A4A2C}
.kpi-box:nth-child(4){border-top-color:#B07D2C}
.kpi-box:nth-child(5){border-top-color:#8A2C2C}
.kpi-val  { font-family:'Libre Baskerville',serif; font-size:30px; font-weight:700;
  color:#1A1A1A; line-height:1.1; margin:3px 0 5px; }
.kpi-lbl  { font-size:10.5px; text-transform:uppercase; letter-spacing:0.12em;
  color:#6B7280; font-family:'Lora',serif; }
.fil-panel{ background:#FAFAF8; border:1px solid #DDD9D0; border-radius:3px;
  padding:18px; font-size:12.5px; }
.fil-panel h5 { font-family:'Libre Baskerville',serif; font-size:11px; font-weight:700;
  text-transform:uppercase; letter-spacing:0.12em; color:#444; margin-bottom:14px;
  padding-bottom:7px; border-bottom:1px solid #DDD9D0; }
.fil-panel label { font-size:12px; color:#555; font-family:'Lora',serif; }
.anote { background:#F0EDE6; border-left:4px solid #2C5F8A; border-radius:0 3px 3px 0;
  padding:11px 15px; margin-bottom:18px; font-size:12.5px; color:#444;
  line-height:1.6; font-family:'Lora',Georgia,serif; font-style:italic; }
.anote strong { font-style:normal; color:#1A1A1A; }
.awarn { background:#FDF3E3; border-left:4px solid #B07D2C; border-radius:0 3px 3px 0;
  padding:11px 15px; margin-bottom:14px; font-size:12.5px; color:#5C3D0E;
  line-height:1.6; font-family:'Lora',serif; font-style:italic; }
.awarn strong { font-style:normal; }
.adanger{ background:#FAE8E8; border-left:4px solid #8A2C2C; border-radius:0 3px 3px 0;
  padding:11px 15px; margin-bottom:14px; font-size:12.5px; color:#4A0E0E;
  font-family:'Lora',serif; }
.tab-content .nav-tabs { border-bottom:2px solid #2C5F8A; margin-bottom:16px; }
.tab-content .nav-tabs .nav-link { color:#444 !important; background:#F0EDE6;
  border:1px solid #DDD9D0; border-bottom:none; margin-right:4px;
  padding:7px 16px !important; border-radius:3px 3px 0 0; font-size:12px;
  letter-spacing:0.04em; font-family:Lora,Georgia,serif; }
.tab-content .nav-tabs .nav-link:hover { color:#1A1A1A !important; background:#E8E4DE; }
.tab-content .nav-tabs .nav-link.active { color:#FAFAF8 !important;
  background:#2C5F8A !important; border-color:#2C5F8A; }
.fig-cap { font-size:11px; color:#6B7280; margin-top:7px; line-height:1.5;
  font-style:italic; font-family:'Lora',serif; padding-top:7px;
  border-top:1px solid #E8E4DE; }
pre.shiny-text-output { background:#F0EDE6!important; border:1px solid #DDD9D0!important;
  border-radius:2px!important; font-size:12px!important; color:#2B2B2B!important;
  padding:10px 13px!important; }
table.dataTable { font-family:'Lora',Georgia,serif; font-size:12px;
  border-collapse:collapse!important; }
table.dataTable thead th { background:#1A1A1A; color:#F5F4F0;
  font-family:'Libre Baskerville',serif; font-size:10.5px; text-transform:uppercase;
  letter-spacing:0.1em; font-weight:700; border:none; padding:9px 11px; }
table.dataTable tbody tr { border-bottom:1px solid #E8E4DE; }
table.dataTable tbody tr:hover td { background:#F0EDE6!important; }
table.dataTable tbody tr.odd  td { background:#FAFAF8; }
table.dataTable tbody tr.even td { background:#F5F4F0; }
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:#EEECE6; }
::-webkit-scrollbar-thumb { background:#BBBBAA; border-radius:3px; }
.global-filters { background:#1A1A1A; border-bottom:1px solid #333; padding:7px 28px; }
.gf-inner { display:flex; align-items:center; gap:32px; flex-wrap:wrap; }
.gf-lbl { font-family:'Libre Baskerville',serif; font-size:10px; font-weight:700;
  text-transform:uppercase; letter-spacing:0.12em; color:#BBBBAA;
  margin-right:8px; white-space:nowrap; }
.gf-item { display:flex; align-items:center; gap:4px; }
.gf-item .shiny-input-container { margin:0 !important; }
.gf-item .form-control { background:#2A2A2A; border:1px solid #444; color:#F0EDE6;
  font-family:'Lora',serif; font-size:12px; padding:3px 8px; border-radius:2px;
  height:28px; }
.gf-item .checkbox-inline { color:#CCCCBB; font-family:'Lora',serif;
  font-size:12px; margin-left:0; padding-left:18px; }
.gf-item .checkbox-inline input { margin-top:2px; }
.lag-selector { display:flex; gap:6px; padding:4px 0 8px 0; flex-wrap:wrap; }
.lag-selector .radio-inline { margin:0; }
.lag-selector input[type=radio] { display:none; }
.lag-selector input[type=radio]+span {
  display:inline-block; padding:5px 12px; border-radius:3px;
  border:1.5px solid #2C5F8A; color:#2C5F8A; cursor:pointer;
  font-family:Lora,Georgia,serif; font-size:11.5px;
  background:#FAFAF8; transition:all .15s; white-space:nowrap; }
.lag-selector input[type=radio]:checked+span {
  background:#2C5F8A; color:#FAFAF8; border-color:#2C5F8A; }
.lag-selector input[type=radio]+span:hover { background:#E8EFF6; }
"

# ==============================================================================
# 3. HELPERS
# ==============================================================================

acad_layout <- function(p, xlab = "", ylab = "") {
  p %>% layout(
    font          = list(family = "Georgia,'Times New Roman',serif",
                         size = 12, color = "#2B2B2B"),
    paper_bgcolor = "#FAFAF8", plot_bgcolor = "#FAFAF8",
    xaxis = list(title = list(text = xlab, font = list(size = 12)),
                 showgrid = TRUE, gridcolor = "#E8E4DE", tickfont = list(size = 11),
                 linecolor = "#CCCCCC", mirror = TRUE, zeroline = FALSE),
    yaxis = list(title = list(text = ylab, font = list(size = 12)),
                 showgrid = TRUE, gridcolor = "#E8E4DE", tickfont = list(size = 11),
                 linecolor = "#CCCCCC", mirror = TRUE, zeroline = FALSE),
    margin = list(l = 55, r = 20, t = 28, b = 52),
    legend = list(bgcolor = "rgba(250,250,248,0.92)", bordercolor = "#CCCCCC",
                  borderwidth = 1, font = list(size = 11))
  ) %>% config(displayModeBar = FALSE)
}

fig_card <- function(lbl, pid, ht = "300px", cap = NULL) {
  div(class = "acad-card",
    div(class = "card-lbl", lbl),
    withSpinner(plotlyOutput(pid, height = ht), type = 6,
                color = COL_P, size = 0.5),
    if (!is.null(cap)) div(class = "fig-cap", cap)
  )
}

dtbl_card <- function(lbl, pid, cap = NULL) {
  div(class = "acad-card",
    div(class = "card-lbl", lbl),
    withSpinner(DTOutput(pid), type = 6, color = COL_P, size = 0.5),
    if (!is.null(cap)) div(class = "fig-cap", cap)
  )
}

kpis <- function(...) div(class = "kpi-row", ...)
kpi  <- function(lbl, oid) {
  div(class = "kpi-box",
    div(class = "kpi-lbl", lbl),
    div(class = "kpi-val", textOutput(oid, inline = TRUE))
  )
}

ph <- function(title, sub = NULL) {
  tagList(div(class = "sec-head", title),
          if (!is.null(sub)) div(class = "sec-sub", sub))
}

an <- function(...) div(class = "anote", ...)
aw <- function(...) div(class = "awarn", ...)
ad <- function(...) div(class = "adanger", ...)

fil <- function(...) div(class = "fil-panel", h5("Controls"), ...)

spear <- function(x, y) {
  ct <- cor.test(x, y, method = "spearman", use = "complete.obs", exact = FALSE)
  list(r = round(ct$estimate, 3), p = round(ct$p.value, 4))
}

fmt_p <- function(p) if (is.na(p)) "NA" else if (p < 0.001) "< 0.001" else as.character(round(p, 3))

safe_lm <- function(f, d) tryCatch(lm(as.formula(f), d, na.action = na.omit), error = function(e) NULL)

coef_tbl <- function(m) {
  if (is.null(m)) return(data.frame(Error = "Model could not be fitted"))
  ct <- summary(m)$coefficients
  ci <- confint(m)
  data.frame(
    Term       = rownames(ct),
    Estimate   = round(ct[,1], 4),
    `Std.Err`  = round(ct[,2], 4),
    `CI 2.5%`  = round(ci[,1], 4),
    `CI 97.5%` = round(ci[,2], 4),
    `p`        = sapply(ct[,4], fmt_p),
    check.names = FALSE
  )
}

interp_ui <- function(m, term, x_lbl, y_lbl) {
  if (is.null(m)) return(NULL)
  ct <- summary(m)$coefficients
  ci <- confint(m)
  if (!term %in% rownames(ct)) return(NULL)
  est <- round(ct[term,1], 4); lo <- round(ci[term,1], 4); hi <- round(ci[term,2], 4)
  p   <- ct[term,4]
  sig <- if (p < 0.05) "statistically significant (p < 0.05)" else "not statistically significant (p ≥ 0.05)"
  an(tags$strong("Plain-English: "),
     paste0("A one-unit increase in ", x_lbl, " is associated with a ",
            abs(est), " ", if (est > 0) "increase" else "decrease",
            " in ", y_lbl, " (95% CI [", lo, ", ", hi, "], p ", fmt_p(p), "). ",
            "This is ", sig, "."))
}

# ==============================================================================
# 4. UI
# ==============================================================================

ui <- navbarPage(
  title = "Garmin Training, Sleep & Recovery Analytics",
  theme = acad_theme,
  collapsible = TRUE,
  header = tagList(
    tags$head(tags$style(HTML(acad_css))),
    div(class = "global-filters",
      div(class = "gf-inner",
        div(class = "gf-item",
          tags$label("Date range", class = "gf-lbl"),
          dateRangeInput("gf_dates", NULL,
                         start = MIN_DATE, end = MAX_DATE,
                         min   = MIN_DATE, max = MAX_DATE,
                         format = "d M yyyy", separator = "→",
                         width  = "100%")
        ),
        div(class = "gf-item",
          tags$label("Sex", class = "gf-lbl"),
          checkboxGroupInput("gf_sex", NULL,
                             choices  = c("Male","Female"),
                             selected = c("Male","Female"),
                             inline   = TRUE)
        )
      )
    )
  ),
  footer = div(style = "text-align:center;padding:12px;font-size:11px;
                        color:#94A3B8;font-family:'Lora',serif;
                        border-top:1px solid #DDD9D0;background:#F5F4F0;",
               "MSc Data Science Capstone · Nairobi Runners · 14 participants · 2022–2026"),

  # ── TAB 1: FOUNDATION ────────────────────────────────────────────────────────
  tabPanel("Foundation",
    div(class = "page-body",
      ph("Cohort Foundation",
         "Data availability, participant characteristics, and observation structure — the essential context for interpreting all subsequent analyses."),
      kpis(
        kpi("Participants",       "kpi_n"),
        kpi("Total Days",         "kpi_days"),
        kpi("Training Days",      "kpi_train"),
        kpi("Full-Staging Cohort","kpi_full"),
        kpi("Date Range",         "kpi_range")
      ),
      fluidRow(
        column(7,
          div(class = "acad-card",
            div(class = "card-lbl", "Figure 1 — Which variables have complete data for each participant?"),
            an(tags$strong("How to read: "),
               "Each cell shows the percentage of days for which that variable has a valid value. ",
               "Red = high missingness. Structural gaps (e.g. SessionLoad on rest days, ",
               "REM Proportion for P006/P013/P015) are expected — not data quality failures."),
            withSpinner(plotlyOutput("coverage_heatmap", height = "340px"),
                        type = 6, color = COL_P, size = 0.5),
            div(class = "fig-cap",
                "Body Battery: P008 has 0% coverage (device unsupported). ",
                "Sleep staging (REM Prop, Sleep Quality): P006, P013, P015 absent due to firmware limitations.")
          )
        ),
        column(5,
          dtbl_card("Table 1 — Participant overview", "cohort_tbl",
                    "VO2max shown as range across observation period. 'Full' cohort includes all sleep stage metrics.")
        )
      )
    )
  ),

  # ── TAB 2: TRAINING ──────────────────────────────────────────────────────────
  tabPanel("Training",
    div(class = "page-body",
      ph("Running Training Behaviour",
         "Session frequency, load distribution, distance, intensity, and economy for running sessions across the study period. All panels reflect the date range and sex selection above."),
      fluidRow(
        column(12,
          div(class = "acad-card",
            div(class = "card-lbl", "Figure 2 — When and how hard did each athlete run?"),
            an("Each tile = one day. Colour intensity shows running session load (AvgHR × Duration). ",
               "Grey = rest or non-running day."),
            withSpinner(plotlyOutput("activity_calendar", height = "300px"),
                        type = 6, color = COL_P, size = 0.5),
            div(class = "fig-cap",
                "Only running sessions shown. Colour scale capped at 95th percentile of running SessionLoad.")
          )
        )
      ),
      fluidRow(
        column(6,
          fig_card("Figure 3 — Do higher-load sessions tend to be longer runs?", "dist_dist", "280px",
                   HTML("Histogram of running session distances (km) with a density curve, coloured by session load intensity (Light / Moderate / High). The x-axis is the same across all three groups so you can directly compare where each intensity cluster sits on the distance scale. If Light sessions cluster at short distances and High at long, the cohort is using distance as a proxy for intensity. If the distributions overlap heavily, some athletes run fast over short distances at high load and slow over long distances at lower load."))
        ),
        column(6,
          fig_card("Figure 4 — Which days of the week carry the highest training load?", "load_weekday", "280px",
                   "Boxplots of running SessionLoad by day of week. Y-axis capped at 65,000. Identifies preferred high-load training days across the cohort.")
        )
      ),
      fluidRow(
        column(6,
          fig_card("Figure 5 — Does running efficiency change across session distances?", "re_vs_dist", "270px",
                   HTML("<strong>Running economy</strong> = average speed (km/h) ÷ average heart rate (bpm). Higher values = covering more ground per heartbeat = more efficient. X = session distance. Coloured by session load intensity. A rising trend with distance would suggest the cohort becomes more efficient on longer runs (pace management). A falling trend suggests accumulated fatigue."))
        ),
        column(6,
          fig_card("Figure 6 — How active are the athletes on a typical day?", "steps_dist", "270px",
                   HTML("Histogram of all recorded daily step counts (days with > 0 and ≤ 60,000 steps). The <strong style='color:#8A2C2C;'>dashed red line = mean</strong>; the <strong style='color:#8A4A2C;'>dotted amber line = median</strong>. Steps include all movement, not just running — rest days with commuting, walking, and daily activity show up here too."))
        )
      ),
      fluidRow(
        column(6,
          fig_card("Figure 7 — Who is the fastest over 10 km?", "speed_10k", "270px",
                   HTML("Each box = one participant's distribution of average speeds on runs between 9.5 and 10.5 km — a rough equivalent-distance comparison. Ordered by median speed (slowest left, fastest right). Wider boxes = more variable pacing across the study period."))
        ),
        column(6,
          fig_card("Figure 8 — Do longer runs stay at lower intensity?", "ri_vs_dist", "270px",
                   HTML("X = session distance (km). Y = Relative Intensity (AvgHR / MaxHR %). Points coloured by session load category. The dashed red line marks the 90% RI threshold. The LOESS trend reveals whether longer runs are done at lower intensity (classic periodised training) or whether intensity stays high regardless of distance (overreaching risk)."))
        )
      )
    )
  ),

  # ── TAB 3: SLEEP & CIRCADIAN ─────────────────────────────────────────────────
  tabPanel("Sleep & Circadian",
    div(class = "page-body",
      ph("Sleep & Circadian Patterns",
         "Cohort-level analyses of sleep duration, quality, architecture, and timing. All panels show population distributions across all participants matching the date range and sex filters above. Stage-level figures (9, 10, 13, 14) are restricted to the 11 participants with 4-stage firmware."),
      fluidRow(
        column(6,
          fig_card("Figure 9 — Do athletes sleep longer and better on certain days?", "tsd_violin", "300px",
                   HTML("Bars = mean total sleep duration per day of the week across the cohort (left axis). Line = mean sleep efficiency (SE) for the same days (right axis). Reveals whether athletes sleep longer or better on particular days — and whether rest days translate to higher sleep quality."))
        ),
        column(6,
          fig_card("Figure 10 — Which nights are the most disrupted and least restorative?", "sleep_weekday_dual", "300px",
                   HTML("Amber bars = mean sleep fragmentation, measured as the percentage of time in bed spent awake (higher = more disrupted). Navy line = mean restorative sleep quality (Deep + REM %). Both axes vary across the week. Full cohort only (11 participants with staging data)."))
        )
      ),
      aw(tags$strong("Data quality note: "),
         "Sleep nights below 3 h or above 12 h are excluded globally as implausible device recordings. ",
         "Figures 9, 10, 13 and 14 are further restricted to the 'full' cohort — P006, P013, and P015 are excluded ",
         "due to absent sleep staging data (firmware limitation)."),
      fluidRow(
        column(12,
          fig_card("Figure 11 — Are sleep stages shifting across the study period?", "stage_area", "300px",
                   HTML("Three trend lines show how the average proportion of each sleep stage has shifted across the study period. <strong>Deep sleep</strong> (navy, solid) is the physically restorative stage — muscle repair, immune function. <strong>Light sleep</strong> (grey, dashed) fills the majority of the night. <strong>REM</strong> (dark green, dotted) is the dreaming stage linked to memory and emotional recovery. Faint dots = daily cohort averages; thick lines = smoothed trends. Rising Deep or REM over time may reflect fitness adaptation. Full cohort only."))
        )
      ),
      fluidRow(
        column(5,
          fig_card("Figure 12 — How much of the night is spent in restorative sleep?", "sleep_qual_hist", "280px",
                   HTML("<strong>SleepQuality %</strong> = proportion of the night spent in Deep or REM sleep — the two stages the body uses for physical repair and mental restoration. 100% is physiologically impossible: the brain always cycles through Light sleep between restorative phases, which occupies at minimum ~45% of any night. Realistic values run from ~25% (poor) to ~55% (exceptional). Values above 75% are excluded as device artefacts. Bars are coloured by quality band: Poor (&lt;30%), Fair (30–40%), Normal (40–50%), Excellent (50–60%), Rebound (&gt;60%). Full cohort only."))
        ),
        column(7,
          fig_card("Figure 13 — How variable is each athlete's sleep duration?", "midpoint_density", "280px",
                   HTML("Bar height = <strong>sleep duration variability</strong>: the standard deviation (SD) of nightly sleep across the study period. Think of it as the average gap between that athlete's best and worst nights. An SD of 1.5h means the athlete routinely swings by 1.5 hours either side of their average — one night they sleep 6h, the next they sleep 9h. An SD close to 0 means their sleep is clockwork-consistent. Bars run red→green from most to least variable. The dashed line is the cohort average. Consistent sleep is associated with better hormonal regulation, mood stability, and training adaptation in the sports science literature; irregular sleep can mask genuine recovery deficits even when the average looks fine."))
        )
      ),
      fluidRow(
        column(6,
          fig_card("Figure 14 — Are athletes shifting their sleep schedule at weekends?", "social_jetlag_box", "280px",
                   HTML("Social jetlag = the difference between an athlete's weekday and weekend sleep midpoint. A value of 2 h means the athlete effectively crosses two time zones every weekend. The dashed line marks the 1-hour clinical threshold. <strong>Red dots</strong> = above threshold. Full cohort."))
        ),
        column(6,
          fig_card("Figure 15 — How many nights fall below the recommended 7 hours?", "sleep_perf_scatter", "280px",
                   HTML("Pie chart showing the proportion of nights in each sleep duration band across the full cohort (3–12 h window). The 7-hour mark is the minimum recommended for athletic populations."))
        )
      ),
      fluidRow(
        column(6,
          fig_card("Figure 16 — Does sleeping longer also mean sleeping better?", "tsd_vs_deep", "260px",
                   HTML("X axis: hours of sleep. Y axis: percentage of that sleep in Deep or REM stages. These are independent — a 9-hour night in shallow light sleep is not restorative. If the trend is flat, quantity and quality are decoupled. Full cohort only."))
        ),
        column(6,
          fig_card("Figure 17 — Are more disrupted nights also shorter nights?", "se_sf_scatter", "260px",
                   HTML("Each point = one night. X = total minutes spent awake during the sleep period. Y = total sleep duration (h). A downward slope means more disrupted nights are also shorter — disruption and insufficient duration co-occur. Outliers below 3 h excluded."))
        )
      )
    )
  ),

  # ── TAB 4: RECOVERY & INTERACTIONS ───────────────────────────────────────────
  tabPanel("Recovery & Interactions",
    div(class = "page-body",
      ph("Recovery Physiology",
         "Cohort-level recovery signals over time and across individuals. All panels reflect the date range and sex filters above."),
      an(tags$strong("Recovery Score = "), "−(today's RHR − personal 7-day rolling mean RHR). Positive = well-recovered; negative = under stress. ",
         tags$strong("Net BB change(t) = charged(t+1) − drained(t)"), " — aligns with RecoveryScore(t+1) as both reflect the same overnight window."),
      fluidRow(
        column(12,
          div(class = "acad-card",
            div(class = "card-lbl", "Figure 18 — How has cohort recovery trended across the study period?"),
            an("Blue line = daily cohort-mean Recovery Score. Orange dotted line = 7-day rolling mean — smooths day-to-day noise to reveal sustained fatigue periods. ",
               "Grey bars = mean overnight Body Battery recharge (right axis, 0–100). ",
               "Body Battery is Garmin's proprietary estimate of energy reserves based on HRV, stress, and sleep. ",
               "When the recovery score drops and Body Battery fails to recharge, that is a compounding stress signal worth examining."),
            withSpinner(plotlyOutput("rhr_timeline", height = "360px"),
                        type = 6, color = COL_P, size = 0.5),
            div(class = "fig-cap",
                "Cohort-mean across all participants matching the date and sex filters. The zero line (dashed) separates well-recovered (above) from under-recovered (below) days.")
          )
        )
      ),
      fluidRow(
        column(6,
          fig_card("Figure 19 — Is resting respiration rate elevated during stress periods?", "resp_timeline", "280px",
                   HTML("Cohort-mean waking respiration rate (breaths per minute) over time, with 7-day rolling mean. Normal resting respiration in fit adults is 12–16 breaths/min. The shaded red zone above 16 breaths/min may indicate physiological stress — either from cumulative training load, illness, or inadequate recovery. Respiration responds to the same autonomic nervous system disruptions as RHR, making it an independent confirmation signal."))
        ),
        column(6,
          fig_card("Figure 20 — Which days of the week show the worst recovery?", "rec_weekday", "280px",
                   HTML("Bars = mean Recovery Score per weekday across the full study period (green = positive/well-recovered, red = negative/under-stress). The amber line (right axis) shows mean Net BB change for the same day-of-week. Both axes are zeroed at the midpoint so the zero line is shared. The dashed horizontal line marks the zero threshold for recovery score."))
        )
      ),
      fluidRow(
        column(6,
          fig_card("Figure 21 — How often is the cohort well-recovered vs under-recovered?", "rec_hist", "270px",
                   HTML("Histogram of all daily Recovery Scores across the cohort. Positive values = days when RHR was below personal baseline. Negative values = days when RHR was elevated. The amber spline line shows mean Net BB change per bin as cross-validation."))
        ),
        column(6,
          fig_card("Figure 22 — Which athletes are chronically under-recovered?", "rec_part_bar", "270px",
                   HTML("Each bar = one athlete's average Recovery Score across the full study period. Green = net positive recovery. Red = chronic under-recovery. Ordered from best to worst. The spread between participants is as important as the average."))
        )
      )
    )
  ),

  # ── TAB 5: RELATIONSHIPS ─────────────────────────────────────────────────────
  tabPanel("Relationships",
    div(class = "page-body",
      ph("Cross-Domain Relationships",
         "Temporal and concurrent associations between training behaviour, sleep architecture, and recovery physiology. All panels use Spearman ρ (rank-based, no normality assumption). Temporal direction is noted at the top of each sub-tab."),
      tabsetPanel(type = "tabs",

        # ── Sub-tab A: Training → Sleep ─────────────────────────────────────────
        tabPanel("Training → Sleep",
          br(),
          an(tags$strong("Temporal direction: "),
             "Training metric on day t paired with sleep metric on night t+1. Tests whether how you train today predicts how you sleep tonight."),
          fluidRow(
            column(12,
              fig_card("Figure 23 — Does training harder today disrupt sleep tonight?", "sl_vs_sleep_merged", "320px",
                       HTML("Four-panel boxplot. X = session load category including Rest Days. Panels: <strong>TSD</strong> (total sleep duration), <strong>SE</strong> (sleep efficiency), <strong>SQ</strong> (restorative sleep quality — Deep + REM %), <strong>SF</strong> (sleep fragmentation — awake %). Rest Day is included as the baseline. If High-load sessions consistently show worse outcomes than Rest Days, training intensity is directly costing sleep quality. Full cohort for SQ panel."))
            )
          ),
          fluidRow(
            column(6,
              fig_card("Figure 24 — Does running near maximum heart rate hurt next-night sleep?", "ri_vs_sleep_merged", "310px",
                       HTML("Three-panel scatter: RI (AvgHR / MaxHR %) on day t vs <strong>SE</strong>, <strong>SQ</strong>, and <strong>SF</strong> the following night. The 90% RI threshold is marked in each panel. Running near maximum heart rate may deplete the autonomic nervous system in ways that disrupt both sleep architecture and continuity. Spearman ρ shown per panel. Full cohort for SQ."))
            ),
            column(6,
              fig_card("Figure 25 — Do longer runs lead to worse sleep quality the following night?", "dist_vs_sleep_merged", "310px",
                       HTML("Two-panel scatter: distance (km) on day t vs <strong>SQ</strong> (Deep + REM %, top) and <strong>SF</strong> (awake %, bottom) the following night. Points coloured by session load intensity. Longer runs at high load in the top-right of the SF panel represent the highest disruption risk. LOESS trend + Spearman ρ per panel. Full cohort for SQ."))
            )
          ),
          fluidRow(
            column(6,
              fig_card("Figure 26 — Does total daily movement affect sleep that night?", "steps_vs_sleep", "300px",
                       HTML("Total steps on day t (x) vs sleep duration and fragmentation on night t+1. Steps capture whole-day activity load — not just structured training. A negative relationship with duration or a positive relationship with fragmentation would suggest cumulative movement disrupts sleep beyond planned sessions."))
            ),
            column(6,
              fig_card("Figure 27 — Does sleeping better last night improve running efficiency today?", "re_vs_sleep", "300px",
                       HTML("<strong>Running economy (RE)</strong> = average speed ÷ average heart rate during the session — higher values mean the athlete covers more ground per heartbeat, i.e. more efficient movement. <strong>Left panel</strong>: RE plotted against Sleep Efficiency the previous night (the fraction of time in bed actually asleep; ≥85% is the clinical benchmark). <strong>Right panel</strong>: RE against Sleep Quality the previous night (Deep + REM % of total sleep). A positive slope in either panel would suggest that better sleep the night before translates to more efficient running the following day. Full cohort only for the Sleep Quality panel (11 participants with stage data)."))
            )
          )
        ),

        # ── Sub-tab B: Training → Recovery ──────────────────────────────────────
        tabPanel("Training → Recovery",
          br(),
          an(tags$strong("Temporal direction: "),
             "Training metric on day t paired with recovery score on morning t+1. ",
             "Net BB change on day t also reflects the same overnight window (charged t+1 − drained t) and is shown as a secondary axis where relevant."),
          fluidRow(
            column(6,
              fig_card("Figure 28 — How much does session intensity cost next-day recovery?", "sl_vs_rec", "300px",
                       HTML("Boxplots of next-day recovery score by session load category (Light / Moderate / High / Rest). The amber diamond markers show mean Net BB change (right axis, −20 to +20). If both signals align — High sessions show lower recovery and lower BB change — the evidence is convergent."))
            ),
            column(6,
              fig_card("Figure 29 — Does high daily movement suppress next-morning recovery?", "steps_vs_rec", "300px",
                       HTML("Total steps on day t (x) vs recovery score on morning t+1 (y). Net BB change as binned secondary. A negative LOESS slope would confirm that high daily movement — even outside structured sessions — elevates RHR the following morning."))
            )
          ),
          fluidRow(
            column(6,
              fig_card("Figure 30 — Do near-maximal sessions depress recovery the following morning?", "ri_vs_rec", "300px",
                       HTML("RI on day t (x) vs recovery score on morning t+1 (y). Near-maximal sessions (RI > 0.90) may produce measurable HRV suppression and RHR elevation the following morning. The 90% RI threshold is marked. Net BB change as binned secondary."))
            ),
            column(6,
              fig_card("Figure 31 — Does running further mean recovering worse the next day?", "dist_vs_rec", "300px",
                       HTML("Distance on day t (x) vs recovery score on morning t+1 (y). Points coloured by session load intensity. This separates the effect of volume (distance) from intensity (load colour)."))
            )
          )
        ),

        # ── Sub-tab C: Sleep → Recovery ──────────────────────────────────────────
        tabPanel("Sleep → Recovery",
          br(),
          an(tags$strong("Temporal direction: "),
             "Sleep metric on night t paired with recovery score on the same morning t. Both reflect the same overnight window. ",
             "Net BB change (BB_delta_lag1, same overnight window as RecoveryScore) is shown as a secondary line."),
          fluidRow(
            column(6,
              fig_card("Figure 32 — Does sleeping longer produce better morning recovery?", "tsd_vs_rec_scatter", "290px",
                       HTML("Total sleep duration (x) vs recovery score (y). BB overnight recharge overlaid as binned mean (right axis). If flat or negative, sleep quality may matter more than quantity."))
            ),
            column(6,
              fig_card("Figure 33 — Does more efficient sleep mean better recovery?", "se_vs_rec_scatter", "290px",
                       HTML("Sleep efficiency (x) vs recovery score (y). Dashed line at 85% SE — below this, fragmented sleep is clinically relevant. BB recharge as secondary."))
            )
          ),
          fluidRow(
            column(6,
              fig_card("Figure 34 — Does a more disrupted night hurt morning recovery?", "sf_vs_rec_scatter", "290px",
                       HTML("Awake proportion % (x) vs recovery score (y). Fragmentation and SE tell related but different stories — an athlete can have high SE but still be fragmented if they have many short awakenings."))
            ),
            column(6,
              fig_card("Figure 35 — Does deeper, more restorative sleep improve recovery?", "sq_vs_rec_scatter", "290px",
                       HTML("SQ = Deep + REM % of total sleep time. X-axis shows the proportion of the night spent in the two restorative stages. If this is the strongest predictor among the four sleep metrics, sleep architecture matters more than duration or continuity. Full cohort only (11 participants with staging data)."))
            )
          )
        ),

        # ── Sub-tab D: Statistical Analysis ─────────────────────────────────────
        tabPanel("Statistical Analysis",
          br(),
          an(tags$strong("What this sub-tab contains: "),
             "A Spearman correlation matrix, lag-structure correlations, overtraining risk, a lagged OLS regression, a mixed-effects model (LME4), ",
             "training monotony index, and autocorrelation of Recovery Score. ",
             "These quantify the patterns seen in the other sub-tabs."),
          fluidRow(
            column(12,
              fig_card("Figure 36 — How do all key training, sleep, and recovery variables correlate with each other?",
                       "corr_heatmap", "420px",
                       HTML("Spearman ρ heatmap. Each cell = pairwise correlation between two cohort-level variables. ",
                            "Vivid blue = strong positive. Vivid red = strong negative. White = near-zero. ",
                            "Cells with |ρ| < 0.05 are left blank. Hover for exact values."))
            )
          ),
          fluidRow(
            column(12,
              fig_card("Figure 37 — How persistent is the recovery signal across days?",
                       "acf_plot", "360px",
                       tagList(
                         tags$strong("What is this plot? "),
                         "Each bar = the Spearman correlation between today's cohort Recovery Score and the score N days later (lag). ",
                         "Bars coloured blue exceed the 95% confidence bounds (±1.96/√N) and are statistically significant. ",
                         tags$br(),
                         tags$strong("What to look for: "),
                         "Many consecutive significant lags = recovery compounds — a bad week tends to stay bad without deliberate rest. ",
                         "Only lag 1 significant = each day is largely independent. ",
                         "A spike at lag 7 = weekly training rhythm visible in the recovery signal."))
            )
          ),
          fluidRow(
            column(6,
              div(class = "acad-card",
                div(class = "card-lbl", "Figure 38 — How many days does the training or sleep effect persist?"),
                an(tags$strong("How to read: "),
                   "Each bar = Spearman ρ at a given time gap. Green = positive correlation. Red = negative. ",
                   "The lag where the bar is tallest tells you how long the effect persists. ",
                   "If 'Training Load → Recovery' peaks at lag 1, hard training today costs recovery tomorrow."),
                div(class = "lag-selector",
                    radioButtons("lag_pair", NULL,
                                 choices = c("Training Load → Sleep"    = "sl_tsd",
                                             "Training Load → Recovery" = "sl_rec",
                                             "Sleep → Recovery"         = "tsd_rec"),
                                 selected = "tsd_rec", inline = TRUE)),
                withSpinner(plotlyOutput("lag_bars", height = "260px"),
                            type = 6, color = COL_P, size = 0.5),
                div(class = "fig-cap", "Spearman ρ at lags 0–3 days. Switch the pair above to explore all three pathways.")
              )
            ),
            column(6,
              div(class = "acad-card",
                div(class = "card-lbl", "Figure 39 — Which sessions pushed athletes into overtraining territory?"),
                an(tags$strong("Quadrant guide: "),
                   "Vertical dashed line = 75th percentile session load. Horizontal dashed line = recovery threshold (−1.5). ",
                   tags$strong("Bottom-right: "), "high load + poor recovery = overtraining risk zone. ",
                   tags$strong("Top-right: "), "high load, recovery intact = well-adapted. ",
                   tags$strong("Bottom-left: "), "low load, poor recovery = non-training stressor (illness, life stress)."),
                withSpinner(plotlyOutput("overtraining_scatter", height = "260px"),
                            type = 6, color = COL_P, size = 0.5),
                div(class = "fig-cap",
                    "Red flags: SessionLoad > 75th pct AND Recovery Score < −1.5 AND (TSD < median OR SE < 0.85).")
              )
            )
          ),
          fluidRow(
            column(12,
              dtbl_card("Table 4 — Training monotony index by participant (weekly load mean ÷ SD)",
                        "monotony_tbl",
                        tagList(
                          tags$strong("What this table measures: "),
                          "Training monotony quantifies how repetitive an athlete's week-to-week loading pattern is. ",
                          "Developed by Foster (1998), it flags a common but underappreciated injury and burnout risk: ",
                          "not training too hard, but training the same way every day.",
                          tags$br(), tags$br(),
                          tags$strong("Column guide:"), tags$br(),
                          tags$strong("Weeks — "),
                          "Number of ISO calendar weeks included for that participant (weeks with fewer than 3 session-load observations are excluded).",
                          tags$br(),
                          tags$strong("Mean monotony — "),
                          "Average of (weekly mean load ÷ weekly SD of load) across all included weeks. ",
                          "A value of 1.0 means the average day's load equals one standard deviation — good variation. ",
                          "A value of 3.0 means the SD is only a third of the mean — almost every session is the same intensity.",
                          tags$br(),
                          tags$strong("Max monotony — "),
                          "The single worst week for that participant. A high max alongside a low mean suggests isolated monotonous blocks, not a chronic problem.",
                          tags$br(),
                          tags$strong("Weeks > 2 / % weeks > 2 — "),
                          "How many weeks (and what share) crossed the Foster threshold of 2.0. ",
                          "Athletes above 25% of weeks in the red zone are at chronic elevated risk.",
                          tags$br(),
                          tags$strong("Mean strain — "),
                          "Weekly total load × monotony, averaged across weeks. ",
                          "Strain compounds volume with uniformity: a heavy monotonous week scores far higher than a heavy varied one. ",
                          "It is a composite overtraining exposure metric.",
                          tags$br(), tags$br(),
                          tags$strong("Colour coding: "),
                          "Green < 1.5 (healthy variation). Amber 1.5–2.0 (borderline). Red > 2.0 (high risk, Foster 1998). ",
                          "Applied to both Mean monotony and % weeks > 2 columns."))
            )
          ),
          fluidRow(
            column(7,
              dtbl_card("Table 2 — Lagged OLS regression: Recovery ~ yesterday's sleep + training",
                        "lag_reg_tbl",
                        "Green p-values < 0.05. OLS on pooled data — treat as descriptive only.")
            ),
            column(5,
              div(class = "acad-card",
                div(class = "card-lbl", "Plain-English interpretation"),
                uiOutput("lag_reg_interp")
              )
            )
          ),
          aw(tags$strong("Why the OLS above is insufficient: "),
             "Each participant contributes hundreds of observations. OLS treats these as independent, inflating degrees of freedom and distorting standard errors. ",
             "The mixed-effects model below adds random intercepts per participant — the minimum correction for repeated-measures data."),
          fluidRow(
            column(7,
              div(class = "acad-card",
                div(class = "card-lbl", "Table 3 — Mixed-effects model: Recovery ~ sleep + training (random intercept per participant)"),
                an(tags$strong("lme4 lmer(). "),
                   "Fixed effects = population-level associations. ",
                   "Random intercept = each participant's baseline recovery is allowed to vary. ",
                   "This is the minimum statistically appropriate model for this data structure."),
                withSpinner(DTOutput("lme_tbl"), type = 6, color = COL_P, size = 0.5),
                div(class = "fig-cap",
                    "Std.Err and p-values from lmerTest (Satterthwaite df). Compare with OLS above — coefficients should be similar but SEs will typically be larger.")
              )
            ),
            column(5,
              div(class = "acad-card",
                div(class = "card-lbl", "LME plain-English interpretation"),
                uiOutput("lme_interp")
              )
            )
          )
        )
      )
    )
  ),

  # ── TAB 6: FITNESS & INDIVIDUALS ─────────────────────────────────────────────
  tabPanel("Fitness & Individuals",
    div(class = "page-body",
      ph("Fitness Trajectories & Individual Profiles",
         "VO₂max trends over time, resilient athlete identification, and a personal deep-dive for any participant."),
      # ── Compact inline selector ──────────────────────────────────────────────
      div(class = "fil-panel", style = "margin-bottom:16px; padding:8px 18px;",
        div(style = "display:flex; align-items:center; gap:16px; flex-wrap:wrap;",
          tags$span("Individual participant:", class = "gf-lbl",
                    style = "color:#555; font-family:Lora,serif;"),
          div(style = "width:160px;",
            selectInput("pid_indiv", NULL, USERS, USERS[1])
          ),
          tags$span(style = "font-size:11px; color:#6B7280; font-family:Lora,serif;",
            "VO₂max uses LOCF imputation. Figures 42–47 update for the selected participant.")
        )
      ),
      # ── Fig 38: full-width VO₂max ────────────────────────────────────────────
      fluidRow(
        column(12,
          div(class = "acad-card",
            div(class = "card-lbl", "Figure 40 — Does fitness improve over time?"),
            an("Each coloured line = one participant's VO₂max trajectory. ",
               "Bold black line = cohort mean (LOESS-smoothed). ",
               "Faint horizontal plateaus = LOCF imputation between observed readings."),
            withSpinner(plotlyOutput("vo2_traj", height = "340px"),
                        type = 6, color = COL_P, size = 0.5),
            div(class = "fig-cap",
                "Only participants with ≥ 90 valid VO₂max observations shown.")
          )
        )
      ),
      # ── Fig 41: resilience (full width) ──────────────────────────────────────
      fluidRow(
        column(12,
          div(class = "acad-card",
            div(class = "card-lbl", "Figure 41 — Who is improving fitness while staying physiologically stable?"),
            an(tags$strong("How to read: "),
               "X = annualised VO₂max slope (positive = improving fitness). ",
               "Y = SD of RHR deviation (lower = physiologically more stable). ",
               "Bottom-right = resilient. Point size = mean training load."),
            withSpinner(plotlyOutput("resilience_scatter", height = "360px"),
                        type = 6, color = COL_P, size = 0.5),
            div(class = "fig-cap", "Ideal quadrant: high positive slope + low RHR variability.")
          )
        )
      ),
      # ── Fig 42: individual timeline (full width) ──────────────────────────────
      fluidRow(
        column(12,
          div(class = "acad-card",
            div(class = "card-lbl", "Figure 42 — How did this athlete's recovery and sleep evolve over time?"),
            an("Recovery score (coloured line) and sleep duration (grey bars) for the selected participant. ",
               "Triangles = running sessions. Hover for daily detail."),
            withSpinner(plotlyOutput("indiv_timeline", height = "400px"),
                        type = 6, color = COL_P, size = 0.5)
          )
        )
      ),
      # ── Fig 43: Individual lag correlation bars ─────────────────────────────
      fluidRow(
        column(12,
          fig_card("Figure 43 — How long does sleep or training affect this athlete's recovery?",
                   "indiv_lag_bars", "320px",
                   HTML("Individual-level lag correlation: Spearman ρ between training load (or sleep) today and recovery score 0–3 days later, computed only for this participant. Compare with the cohort-level Figure 38 — divergence means this athlete responds differently to training stress than the average."))
        )
      ),
      # ── Fig 44: Sleep debt tracker ───────────────────────────────────────────
      fluidRow(
        column(12,
          fig_card("Figure 44 — Is this athlete accumulating sleep debt?",
                   "indiv_sleep_debt", "300px",
                   HTML("Weekly cumulative sleep deficit: total hours slept below 7h per week (negative = debt, zero = sufficient). Overlaid on a 4-week rolling mean Recovery Score (right axis). If recovery dips consistently lag sleep debt by 1–2 weeks, it suggests the athlete needs a minimum of 7h to maintain physiological baseline."))
        )
      ),
      # ── Fig 45: Effort vs outcome ────────────────────────────────────────────
      fluidRow(
        column(6,
          fig_card("Figure 45 — Does training harder cost this athlete recovery the next day?",
                   "indiv_effort_outcome", "300px",
                   HTML("Session load (x) vs next-day recovery score (y) for this athlete only. Points coloured by session type. A downward trend means harder sessions reliably suppress recovery; a flat trend means this athlete tolerates load well."))
        ),
        column(6,
          fig_card("Figure 46 — Does sleeping longer help this athlete's recovery?",
                   "indiv_lag", "300px",
                   HTML("Each point = one day for the selected participant. X = last night's sleep duration. Y = today's Recovery Score. Points coloured by training intensity. A positive slope means more sleep preceded better recovery for this person."))
        )
      ),
      # ── Fig 47: Monthly recovery (full width) ────────────────────────────────
      fluidRow(
        column(12,
          fig_card("Figure 47 — How has this athlete's monthly recovery trended?",
                   "indiv_monthly_recovery", "300px",
                   HTML("Mean Recovery Score by calendar month for the selected participant. Green bars = net positive recovery months. Red bars = net negative. Hover for the monthly mean and observation count. Useful for identifying training blocks, illness periods, or seasonal patterns."))
        )
      ),
      # ── Personal summary (full width, at end) ────────────────────────────────
      fluidRow(
        column(12,
          div(class = "acad-card",
            div(class = "card-lbl", "Personal summary statistics"),
            withSpinner(uiOutput("indiv_summary"), type = 6, color = COL_P, size = 0.5)
          )
        )
      )
    )
  )

)

# ==============================================================================
# 5. SERVER
# ==============================================================================

server <- function(input, output, session) {

  # ── Reactive: global date + sex filter ───────────────────────────────────
  d_global <- reactive({
    req(input$gf_dates, input$gf_sex)
    garmin %>%
      filter(calendar_date >= input$gf_dates[1],
             calendar_date <= input$gf_dates[2],
             Gender %in% input$gf_sex,
             is.na(TSD) | (TSD >= 3 & TSD <= 12))   # implausible sleep nights excluded
  })

  # ── Reactive: training filter (running sessions only) ─────────────────────
  d_train <- reactive({
    d <- d_global() %>% filter(ActivityGroup == "Running" | !activity_day)
    d
  })

  d_sleep <- reactive({
    d_global()
  })

  d_rec <- reactive({
    d_global()
  })

  # ── Tab 1: Foundation KPIs ────────────────────────────────────────────────
  output$kpi_n      <- renderText({ n_distinct(garmin$User_id) })
  output$kpi_days   <- renderText({ format(nrow(garmin), big.mark = ",") })
  output$kpi_train  <- renderText({
    paste0(round(100 * mean(garmin$activity_day, na.rm = TRUE), 0), "%")
  })
  output$kpi_full   <- renderText({
    n <- n_distinct(garmin$User_id[garmin$analysis_cohort == "full"])
    paste0(n, " / 14")
  })
  output$kpi_range  <- renderText({
    paste0(format(MIN_DATE, "%b %Y"), " – ", format(MAX_DATE, "%b %Y"))
  })

  # ── Tab 1: Coverage heatmap ───────────────────────────────────────────────
  output$coverage_heatmap <- renderPlotly({
    mat <- coverage_df %>%
      mutate(Variable = as.character(Variable)) %>%
      pivot_wider(names_from = Variable, values_from = Coverage)
    vars <- COV_LABELS
    uids <- mat$User_id
    z    <- as.matrix(mat[, -1])
    plot_ly(x = vars, y = uids, z = z, type = "heatmap",
            colorscale = list(c(0,"#8A2C2C"), c(0.5,"#F5F3DC"), c(1,"#3A6B4A")),
            zmin = 0, zmax = 100,
            hovertemplate = "%{y} · %{x}<br>Coverage: %{z:.0f}%<extra></extra>",
            colorbar = list(title = "% Available", len = 0.8,
                            tickfont = list(family = "Lora,serif"))) %>%
      add_annotations(x = rep(vars, each = length(uids)),
                      y = rep(uids, times = length(vars)),
                      text = as.vector(z),
                      showarrow = FALSE,
                      font = list(size = 10, color = "#1A1A1A",
                                  family = "Lora,serif")) %>%
      layout(font = list(family = "Lora,Georgia,serif", size = 11),
             paper_bgcolor = "#FAFAF8", plot_bgcolor = "#FAFAF8",
             xaxis = list(title = "", tickangle = -35, linecolor = "#CCCCCC"),
             yaxis = list(title = "", autorange = "reversed",
                          linecolor = "#CCCCCC"),
             margin = list(l = 50, r = 10, t = 10, b = 100)) %>%
      config(displayModeBar = FALSE)
  })

  # ── Tab 1: Cohort table ───────────────────────────────────────────────────
  output$cohort_tbl <- renderDT({
    tbl <- garmin %>%
      group_by(User_id) %>%
      summarise(
        Sex      = first(Sex),
        Age      = first(Age),
        `VO₂max` = {
          v <- VO2max_imputed_final[!is.na(VO2max_imputed_final)]
          if (length(v)) paste0(round(min(v)), "–", round(max(v))) else "N/A"
        },
        Cohort = first(analysis_cohort),
        Days   = n(),
        .groups = "drop"
      )
    datatable(tbl, rownames = FALSE, class = "compact stripe",
              options = list(dom = "t", pageLength = 14,
                             scrollX = FALSE,
                             autoWidth = TRUE,
                             columnDefs = list(
                               list(width = "55px", targets = 0),
                               list(width = "45px", targets = c(1, 2)),
                               list(width = "75px", targets = 3),
                               list(width = "80px", targets = 4),
                               list(width = "50px", targets = 5)
                             ))) %>%
      formatStyle("Cohort",
        backgroundColor = styleEqual(c("full","no_sleep_stages"),
                                     c("#DCF0DC","#FEF3D0")))
  })

  # ── Tab 2: Activity calendar (SL_Intensity categories) ───────────────────
  output$activity_calendar <- renderPlotly({
    wday_order <- c("Monday","Tuesday","Wednesday","Thursday",
                    "Friday","Saturday","Sunday")
    cat_cols <- c("Rest" = "#E8E4DE", "Light" = "#93C5FD",
                  "Moderate" = "#2C5F8A", "High" = "#1B3A6B")

    d <- d_train() %>%
      mutate(
        wk_start  = floor_date(calendar_date, "week", week_start = 1),
        wday_fac  = factor(weekday, levels = wday_order),
        cat_label = case_when(
          ActivityGroup == "Running" & SL_Intensity == "High"     ~ "High",
          ActivityGroup == "Running" & SL_Intensity == "Moderate" ~ "Moderate",
          ActivityGroup == "Running" & SL_Intensity == "Light"    ~ "Light",
          TRUE                                                     ~ "Rest"
        )
      ) %>%
      group_by(wk_start, wday_fac) %>%
      summarise(
        cat_label = {
          # pick hardest category that day
          lvls <- c("High","Moderate","Light","Rest")
          hit  <- lvls[lvls %in% cat_label][1]
          if (is.na(hit)) "Rest" else hit
        },
        n_active  = sum(ActivityGroup == "Running", na.rm = TRUE),
        load_top  = {
          v <- SessionLoad[ActivityGroup == "Running" & !is.na(SessionLoad)]
          if (length(v) == 0) NA_real_ else max(v)
        },
        .groups = "drop"
      ) %>%
      mutate(
        z_num = match(cat_label, c("Rest","Light","Moderate","High")) - 1L,
        hover = paste0("Week of ", format(wk_start, "%d %b %Y"),
                       "<br>Day: ",      wday_fac,
                       "<br>Category: ", cat_label,
                       "<br>Sessions: ", n_active,
                       "<br>Peak load: ",ifelse(is.na(load_top), "—", round(load_top)))
      )

    x_vals <- sort(unique(d$wk_start))
    y_vals <- wday_order

    z_mat <- matrix(NA_real_, nrow = 7L, ncol = length(x_vals))
    h_mat <- matrix("",       nrow = 7L, ncol = length(x_vals))
    for (i in seq_len(nrow(d))) {
      xi <- match(d$wk_start[i], x_vals)
      yi <- match(as.character(d$wday_fac[i]), y_vals)
      if (!is.na(xi) && !is.na(yi)) {
        z_mat[yi, xi] <- d$z_num[i]
        h_mat[yi, xi] <- d$hover[i]
      }
    }

    plot_ly(
      x = x_vals, y = y_vals,
      z = z_mat, text = h_mat,
      type      = "heatmap",
      hoverinfo = "text",
      colorscale = list(
        c(0,   "#DEDAD4"),   # Rest  — warm grey
        c(0.33,"#E8A838"),   # Light — amber
        c(0.67,"#2DA87A"),   # Moderate — teal-green
        c(1,   "#C0392B")    # High  — red
      ),
      zmin = 0, zmax = 3,
      showscale = FALSE,
      hoverongaps = FALSE,
      xgap = 1, ygap = 1
    ) %>%
      layout(
        paper_bgcolor = "#FAFAF8", plot_bgcolor = "#FAFAF8",
        font    = list(family = "Lora,Georgia,serif", size = 11),
        xaxis   = list(title = "", type = "date",
                       tickformat = "%b %Y", nticks = 12,
                       tickangle = -35, linecolor = "#CCCCCC"),
        yaxis   = list(title = "", autorange = "reversed",
                       linecolor = "#CCCCCC", tickfont = list(size = 11)),
        margin  = list(l = 80, r = 20, t = 50, b = 40),
        annotations = list(
          list(x = 0.08, y = 1.13, xref = "paper", yref = "paper",
               text = "<b>■</b> Rest",
               showarrow = FALSE, xanchor = "center",
               font = list(color = "#888880", size = 12, family = "Lora,serif")),
          list(x = 0.32, y = 1.13, xref = "paper", yref = "paper",
               text = "<b>■</b> Light",
               showarrow = FALSE, xanchor = "center",
               font = list(color = "#E8A838", size = 12, family = "Lora,serif")),
          list(x = 0.58, y = 1.13, xref = "paper", yref = "paper",
               text = "<b>■</b> Moderate",
               showarrow = FALSE, xanchor = "center",
               font = list(color = "#2DA87A", size = 12, family = "Lora,serif")),
          list(x = 0.84, y = 1.13, xref = "paper", yref = "paper",
               text = "<b>■</b> High",
               showarrow = FALSE, xanchor = "center",
               font = list(color = "#C0392B", size = 12, family = "Lora,serif"))
        )
      ) %>%
      config(displayModeBar = FALSE)
  })

  # ── Tab 2: Running load by weekday (capped at 65k) ───────────────────────
  output$load_weekday <- renderPlotly({
    d <- d_train() %>%
      filter(!is.na(SessionLoad), SessionLoad > 0,
             ActivityGroup == "Running")
    plot_ly(d, x = ~Weekday, y = ~SessionLoad, type = "box",
            color = ~Weekday,
            colors = setNames(
              colorRampPalette(c(COL_P, COL_G))(7),
              levels(garmin$Weekday)
            ),
            boxmean = TRUE, showlegend = FALSE,
            marker = list(size = 3, opacity = 0.3),
            line   = list(width = 1.5)) %>%
      acad_layout("", "Session Load") %>%
      layout(yaxis = list(range = c(0, 65000)))
  })

  # ── Tab 2: Distance distribution coloured by SL intensity ───────────────
  output$dist_dist <- renderPlotly({
    cat_cols <- c("Light" = COL_G, "Moderate" = COL_W, "High" = COL_D)
    d <- d_train() %>%
      filter(ActivityGroup == "Running", !is.na(TV), TV > 0, TV <= 60,
             !is.na(SL_Intensity), SL_Intensity != "Rest Day") %>%
      mutate(SL_Intensity = factor(SL_Intensity, levels = c("Light","Moderate","High")))
    p <- ggplot(d, aes(x = TV, fill = SL_Intensity, colour = SL_Intensity)) +
      geom_histogram(aes(y = after_stat(density)), binwidth = 1,
                     alpha = 0.65, position = "identity",
                     linewidth = 0.2, colour = "#FAFAF8") +
      geom_density(aes(fill = NULL), linewidth = 0.9, alpha = 0) +
      scale_fill_manual(values = cat_cols, name = "Session load") +
      scale_colour_manual(values = cat_cols, name = "Session load") +
      scale_x_continuous(limits = c(0, 60), breaks = seq(0, 60, 10)) +
      labs(x = "Running distance (km)", y = "Density",
           subtitle = paste0("Running sessions only · n = ", nrow(d))) +
      theme_minimal(base_family = "serif") +
      theme(legend.position  = "right",
            panel.grid.major = element_line(colour = "#E8E4DE"),
            plot.background  = element_rect(fill = "#FAFAF8", colour = NA),
            panel.background = element_rect(fill = "#FAFAF8", colour = NA))
    ggplotly(p) %>%
      layout(paper_bgcolor = "#FAFAF8", plot_bgcolor = "#FAFAF8",
             font   = list(family = "Lora,Georgia,serif", size = 11),
             legend = list(title = list(text = "Session load"),
                           font = list(size = 10))) %>%
      config(displayModeBar = FALSE)
  })

  # ── Tab 2: Running economy vs distance coloured by SL ────────────────────
  output$re_vs_dist <- renderPlotly({
    cat_cols <- c("Light" = COL_G, "Moderate" = COL_W, "High" = COL_D)
    d <- d_train() %>%
      filter(ActivityGroup == "Running",
             !is.na(TV), TV > 0, TV <= 60,
             !is.na(ActivitiesAvgSpeed_kmh), !is.na(ActivitiesAvgHr),
             ActivitiesAvgHr > 0,
             !is.na(SL_Intensity), SL_Intensity != "Rest Day") %>%
      mutate(RE = ActivitiesAvgSpeed_kmh / ActivitiesAvgHr,
             SL_Intensity = factor(SL_Intensity, levels = c("Light","Moderate","High")))
    d_samp <- slice_sample(d, n = min(3000L, nrow(d)))
    sp <- spear(d_samp$TV, d_samp$RE)
    p <- ggplot(d_samp,
                aes(x = TV, y = RE, colour = SL_Intensity,
                    text = paste0(User_id, "<br>Date: ", calendar_date,
                                  "<br>Distance: ", round(TV,1), " km",
                                  "<br>Economy: ", round(RE,3), " km·h⁻¹·bpm⁻¹",
                                  "<br>Intensity: ", SL_Intensity))) +
      geom_point(alpha = 0.28, size = 1.4) +
      geom_smooth(aes(group = 1), method = "loess", formula = y ~ x,
                  colour = "#1A1A1A", linewidth = 1.2, se = TRUE,
                  fill = "#CCCCCC", alpha = 0.25) +
      scale_colour_manual(values = cat_cols, name = "Session load") +
      labs(x = "Running distance (km)",
           y = "Economy (km/h per bpm)",
           subtitle = paste0("Spearman ρ = ", sp$r, "  ·  p ", fmt_p(sp$p),
                             "  ·  n = ", nrow(d_samp))) +
      theme_minimal(base_family = "serif") +
      theme(legend.position  = "right",
            panel.grid.major = element_line(colour = "#E8E4DE"),
            plot.background  = element_rect(fill = "#FAFAF8", colour = NA),
            panel.background = element_rect(fill = "#FAFAF8", colour = NA))
    ggplotly(p, tooltip = "text") %>%
      layout(paper_bgcolor = "#FAFAF8", plot_bgcolor = "#FAFAF8",
             font   = list(family = "Lora,Georgia,serif", size = 11),
             legend = list(title = list(text = "Session load"),
                           x = 1.02, y = 0.5, font = list(size = 10))) %>%
      config(displayModeBar = FALSE)
  })

  # ── Tab 2: Daily steps distribution ──────────────────────────────────────
  output$steps_dist <- renderPlotly({
    d <- d_global() %>%
      filter(!is.na(DailyTotalSteps), DailyTotalSteps > 0,
             DailyTotalSteps <= 60000)
    mn  <- mean(d$DailyTotalSteps,   na.rm = TRUE)
    med <- median(d$DailyTotalSteps, na.rm = TRUE)
    # Build histogram bins manually so we can use pure plotly (no ggplotly)
    # This guarantees the vlines and their hover labels render correctly
    bw    <- 1000  # 1 k steps per bin
    brks  <- seq(0, 60000, by = bw)
    mids  <- brks[-length(brks)] + bw / 2
    cnts  <- tabulate(findInterval(d$DailyTotalSteps, brks, rightmost.closed = TRUE),
                      nbins = length(mids))
    dens  <- cnts / (sum(cnts) * bw / 1000)  # density in per-thousand-steps units
    # KDE
    kde   <- density(d$DailyTotalSteps / 1000, bw = "nrd0", n = 512)
    plot_ly() %>%
      add_bars(x = mids / 1000, y = dens,
               name = "Daily steps",
               marker = list(color = COL_P, opacity = 0.75,
                             line = list(color = "#FAFAF8", width = 0.4)),
               hovertemplate = "Steps: %{x:.0f}k<br>Density: %{y:.4f}<extra></extra>",
               showlegend = FALSE) %>%
      add_lines(x = kde$x, y = kde$y,
                name = "Density", showlegend = FALSE,
                line = list(color = COL_G, width = 1.8),
                hoverinfo = "none") %>%
      add_lines(x = c(mn/1000, mn/1000), y = c(0, max(dens) * 1.15),
                name = paste0("Mean: ", round(mn/1000,1), "k"),
                line = list(color = COL_D, dash = "dash", width = 2),
                hovertemplate = paste0("<b>Mean</b>: ", round(mn/1000,1),
                                       "k steps<extra></extra>")) %>%
      add_lines(x = c(med/1000, med/1000), y = c(0, max(dens) * 1.15),
                name = paste0("Median: ", round(med/1000,1), "k"),
                line = list(color = COL_S, dash = "dot", width = 2),
                hovertemplate = paste0("<b>Median</b>: ", round(med/1000,1),
                                       "k steps<extra></extra>")) %>%
      layout(
        paper_bgcolor = "#FAFAF8", plot_bgcolor = "#FAFAF8",
        font    = list(family = "Lora,Georgia,serif", size = 11),
        xaxis   = list(title = "Daily steps (thousands)", gridcolor = "#E8E4DE",
                       tickvals = seq(0, 60, 10), linecolor = "#CCCCCC"),
        yaxis   = list(title = "Density", gridcolor = "#E8E4DE",
                       linecolor = "#CCCCCC"),
        barmode = "overlay",
        bargap  = 0.05,
        legend  = list(orientation = "h", y = 1.08, font = list(size = 10))
      ) %>%
      config(displayModeBar = FALSE)
  })

  # ── Tab 2: Speed at 9.5–10.5 km per participant ──────────────────────────
  output$speed_10k <- renderPlotly({
    d <- d_train() %>%
      filter(ActivityGroup == "Running",
             !is.na(TV), TV >= 9.5, TV <= 10.5,
             !is.na(ActivitiesAvgSpeed_kmh), ActivitiesAvgSpeed_kmh > 0) %>%
      mutate(User_id = factor(User_id, levels = sort(unique(User_id))),
             Pace = 60 / ActivitiesAvgSpeed_kmh)  # min/km
    med_by_p <- d %>%
      group_by(User_id) %>%
      summarise(med = median(Pace, na.rm = TRUE),
                n   = n(), .groups = "drop") %>%
      arrange(desc(med))  # slower pace = higher value = order ascending pace
    p <- ggplot(d, aes(x = factor(User_id, levels = med_by_p$User_id),
                       y = Pace,
                       fill = User_id,
                       text = paste0(User_id,
                                     "<br>Pace: ", round(Pace, 2), " min/km",
                                     "<br>Distance: ", round(TV, 1), " km",
                                     "<br>Date: ", calendar_date))) +
      geom_boxplot(width = 0.55, outlier.shape = NA, alpha = 0.8,
                   colour = "#1A1A1A", linewidth = 0.4) +
      geom_jitter(aes(colour = User_id), width = 0.18, size = 1,
                  alpha = 0.25, show.legend = FALSE) +
      scale_fill_manual(values = PART_COLORS, guide = "none") +
      scale_colour_manual(values = PART_COLORS, guide = "none") +
      scale_y_reverse() +
      labs(x = NULL, y = "Pace (min/km)",
           subtitle = "Runs 9.5–10.5 km only  ·  ordered by median pace (faster left)") +
      theme_minimal(base_family = "serif") +
      theme(axis.text.x    = element_text(angle = 30, hjust = 1, size = 9),
            panel.grid.major = element_line(colour = "#E8E4DE"),
            plot.background  = element_rect(fill = "#FAFAF8", colour = NA),
            panel.background = element_rect(fill = "#FAFAF8", colour = NA))
    ggplotly(p, tooltip = "text") %>%
      layout(paper_bgcolor = "#FAFAF8", plot_bgcolor = "#FAFAF8",
             font = list(family = "Lora,Georgia,serif", size = 11),
             showlegend = FALSE) %>%
      config(displayModeBar = FALSE)
  })

  # ── Tab 2: RI by training distance coloured by SL ─────────────────────────
  output$ri_vs_dist <- renderPlotly({
    cat_cols <- c("Light" = COL_G, "Moderate" = COL_W, "High" = COL_D)
    d <- d_train() %>%
      filter(ActivityGroup == "Running",
             !is.na(TV), TV > 0, TV <= 60,
             !is.na(RelativeIntensity), RelativeIntensity > 0, RelativeIntensity <= 1,
             !is.na(SL_Intensity), SL_Intensity != "Rest Day") %>%
      mutate(SL_Intensity = factor(SL_Intensity, levels = c("Light","Moderate","High")))
    d_samp <- slice_sample(d, n = min(3000L, nrow(d)))
    sp <- spear(d_samp$TV, d_samp$RelativeIntensity)
    p <- ggplot(d_samp,
                aes(x = TV, y = RelativeIntensity * 100,
                    colour = SL_Intensity,
                    text = paste0(User_id, "<br>Date: ", calendar_date,
                                  "<br>Distance: ", round(TV,1), " km",
                                  "<br>RI: ", round(RelativeIntensity*100), "%",
                                  "<br>Intensity: ", SL_Intensity))) +
      geom_point(alpha = 0.28, size = 1.4) +
      geom_smooth(aes(group = 1), method = "loess", formula = y ~ x,
                  colour = "#1A1A1A", linewidth = 1.2, se = TRUE,
                  fill = "#CCCCCC", alpha = 0.25) +
      scale_colour_manual(values = cat_cols, name = "Session load") +
      scale_x_continuous(breaks = seq(0, 60, 5)) +
      scale_x_continuous(breaks = seq(0, 60, 10)) +
      scale_y_continuous(labels = function(x) paste0(x, "%")) +
      geom_hline(yintercept = 90, linetype = "dashed",
                 colour = COL_D, linewidth = 0.6) +
      labs(x = "Running distance (km)",
           y = "Relative Intensity (%)",
           subtitle = paste0("Spearman ρ = ", sp$r, "  ·  p ", fmt_p(sp$p),
                             "  ·  dashed = 90% RI threshold")) +
      theme_minimal(base_family = "serif") +
      theme(legend.position  = "right",
            panel.grid.major = element_line(colour = "#E8E4DE"),
            plot.background  = element_rect(fill = "#FAFAF8", colour = NA),
            panel.background = element_rect(fill = "#FAFAF8", colour = NA))
    ggplotly(p, tooltip = "text") %>%
      layout(paper_bgcolor = "#FAFAF8", plot_bgcolor = "#FAFAF8",
             font   = list(family = "Lora,Georgia,serif", size = 11),
             legend = list(title = list(text = "Session load"),
                           x = 1.02, y = 0.5, font = list(size = 10))) %>%
      config(displayModeBar = FALSE)
  })

  # ── Tab 3: TSD boxplot by participant ─────────────────────────────────────
  # ── Fig 17: Mean TSD by weekday (bars) + mean SE (line) ───────────────────
  output$tsd_violin <- renderPlotly({
    wday_ord <- c("Monday","Tuesday","Wednesday","Thursday",
                  "Friday","Saturday","Sunday")
    d <- d_sleep() %>%
      filter(!is.na(TSD), !is.na(SE), SE < 1.0, TSD >= 3) %>%
      mutate(Weekday = factor(weekday, levels = wday_ord)) %>%
      group_by(Weekday) %>%
      summarise(mean_TSD = mean(TSD, na.rm = TRUE),
                mean_SE  = mean(SE,  na.rm = TRUE) * 100,
                n        = n(), .groups = "drop")
    plot_ly() %>%
      add_bars(data = d, x = ~Weekday, y = ~mean_TSD,
               name = "Mean sleep duration (h)",
               marker = list(color = COL_P, opacity = 0.8,
                             line = list(color = "#FAFAF8", width = 0.5)),
               hovertemplate = "<b>%{x}</b><br>Mean TSD: %{y:.2f}h<br>n = %{customdata}<extra></extra>",
               customdata = ~n) %>%
      add_lines(data = d, x = ~Weekday, y = ~mean_SE,
                name = "Mean sleep efficiency (%)",
                yaxis = "y2",
                line = list(color = COL_S, width = 2.5),
                marker = list(size = 7, color = COL_S),
                hovertemplate = "<b>%{x}</b><br>Mean SE: %{y:.1f}%<extra></extra>") %>%
      layout(
        paper_bgcolor = "#FAFAF8", plot_bgcolor = "#FAFAF8",
        font    = list(family = "Lora,Georgia,serif", size = 11),
        xaxis   = list(title = "", gridcolor = "#E8E4DE", linecolor = "#CCCCCC"),
        yaxis   = list(title = "Mean sleep duration (h)",
                       gridcolor = "#E8E4DE", linecolor = "#CCCCCC",
                       range = c(0, max(d$mean_TSD, na.rm=TRUE) * 1.15)),
        yaxis2  = list(title = "Mean sleep efficiency (%)",
                       overlaying = "y", side = "right", showgrid = FALSE,
                       linecolor = "#CCCCCC",
                       range = c(min(d$mean_SE, na.rm=TRUE) * 0.97,
                                 max(d$mean_SE, na.rm=TRUE) * 1.03)),
        legend  = list(orientation = "h", y = 1.08,
                       font = list(size = 10)),
        margin  = list(l = 55, r = 55, t = 35, b = 45),
        barmode = "group"
      ) %>%
      config(displayModeBar = FALSE)
  })

  # ── Fig 18: Awakenings (min) vs total sleep duration ──────────────────────
  output$se_sf_scatter <- renderPlotly({
    d <- d_sleep() %>%
      mutate(awake_min = SleepAwakeSleepSeconds / 60) %>%
      filter(!is.na(awake_min), !is.na(TSD), TSD >= 3,
             awake_min >= 0, awake_min <= 120)
    d_samp <- slice_sample(d, n = min(3000L, nrow(d)))
    sp <- spear(d_samp$TSD, d_samp$awake_min)
    p <- ggplot(d_samp,
                aes(x = TSD, y = awake_min, colour = User_id,
                    text = paste0(User_id, "<br>Date: ", calendar_date,
                                  "<br>Total sleep: ", round(TSD, 1), "h",
                                  "<br>Awake during night: ",
                                  round(awake_min), " min"))) +
      geom_point(alpha = 0.18, size = 1.1) +
      geom_smooth(aes(group = 1), method = "loess", formula = y ~ x,
                  colour = "#1A1A1A", linewidth = 1.2, se = TRUE,
                  fill = "#CCCCCC", alpha = 0.25) +
      scale_colour_manual(values = PART_COLORS) +
      labs(x = "Sleep Duration (h)",
           y = "Time awake during night (minutes)",
           subtitle = paste0("Spearman ρ = ", sp$r, "  ·  p ", fmt_p(sp$p),
                             "  ·  n = ", nrow(d_samp))) +
      theme_minimal(base_family = "serif") +
      theme(legend.position = "none",
            panel.grid.major = element_line(colour = "#E8E4DE"),
            plot.background  = element_rect(fill = "#FAFAF8", colour = NA),
            panel.background = element_rect(fill = "#FAFAF8", colour = NA))
    ggplotly(p, tooltip = "text") %>%
      layout(paper_bgcolor = "#FAFAF8", plot_bgcolor = "#FAFAF8",
             font = list(family = "Lora,Georgia,serif", size = 11)) %>%
      config(displayModeBar = FALSE)
  })

  # ── Fig 19: Cohort-mean sleep stage trends over time ───────────────────────
  output$stage_area <- renderPlotly({
    d <- d_global() %>%
      filter(analysis_cohort == "full",
             !is.na(DeepSleepProp), !is.na(LightSleepProp),
             !is.na(REMSleepProp_final)) %>%
      group_by(calendar_date) %>%
      summarise(Deep  = mean(DeepSleepProp,     na.rm = TRUE),
                Light = mean(LightSleepProp,     na.rm = TRUE),
                REM   = mean(REMSleepProp_final, na.rm = TRUE),
                .groups = "drop") %>%
      arrange(calendar_date)
    if (nrow(d) < 10) return(plotly_empty() %>%
      add_annotations(text = "Insufficient stage data for selected filters.",
                      x = 0.5, y = 0.5, xref = "paper", yref = "paper",
                      showarrow = FALSE,
                      font = list(size = 12, color = "#6B7280",
                                  family = "Lora,serif")) %>%
      layout(paper_bgcolor = "#FAFAF8") %>% config(displayModeBar = FALSE))

    # Compute LOESS smooths per stage in R (span = 0.25 for sensitivity)
    loess_fit <- function(dates, vals, span = 0.25) {
      ok  <- !is.na(vals)
      fit <- loess(vals[ok] ~ as.numeric(dates[ok]), span = span)
      pred <- rep(NA_real_, length(dates))
      pred[ok] <- predict(fit)
      pred
    }
    d$Deep_s  <- loess_fit(d$calendar_date, d$Deep)
    d$Light_s <- loess_fit(d$calendar_date, d$Light)
    d$REM_s   <- loess_fit(d$calendar_date, d$REM)

    stage_cfg <- list(
      Deep  = list(raw = "#7BAFD4", trend = "#1A3A6B",
                   dash = "solid",    name = "Deep"),
      Light = list(raw = "#B0C4DE", trend = "#5A5A5A",
                   dash = "dash",     name = "Light"),
      REM   = list(raw = "#7EC8A0", trend = "#1A5C35",
                   dash = "dot",      name = "REM")
    )

    fig <- plot_ly()
    for (st in c("Deep","Light","REM")) {
      cfg <- stage_cfg[[st]]
      fig <- fig %>%
        add_markers(data = d, x = ~calendar_date,
                    y = as.formula(paste0("~", st)),
                    name = paste0(cfg$name, " (daily)"),
                    marker = list(color = cfg$raw, size = 3, opacity = 0.25),
                    showlegend = FALSE,
                    hoverinfo = "skip") %>%
        add_lines(data = d, x = ~calendar_date,
                  y = as.formula(paste0("~", st, "_s")),
                  name = cfg$name,
                  line = list(color = cfg$trend, width = 2.8,
                              dash = cfg$dash),
                  hovertemplate = paste0("<b>%{x|%b %Y}</b><br>",
                                         cfg$name, ": %{y:.1%}<extra></extra>"))
    }
    fig %>% layout(
      paper_bgcolor = "#FAFAF8", plot_bgcolor = "#FAFAF8",
      font   = list(family = "Lora,Georgia,serif", size = 11),
      xaxis  = list(title = "", gridcolor = "#E8E4DE", linecolor = "#CCCCCC"),
      yaxis  = list(title = "Cohort-mean proportion",
                    tickformat = ".0%",
                    gridcolor = "#E8E4DE", linecolor = "#CCCCCC"),
      legend = list(orientation = "h", y = -0.18,
                    font = list(size = 11)),
      hovermode = "x unified",
      margin = list(l = 60, r = 30, t = 20, b = 60)
    ) %>% config(displayModeBar = FALSE)
  })

  # ── Fig 20: Sleep quality histogram ──────────────────────────────────────
  output$sleep_qual_hist <- renderPlotly({
    d <- d_sleep() %>%
      filter(!is.na(SleepQuality_pct), analysis_cohort == "full",
             SleepQuality_pct <= 75) %>%           # >75% physiologically implausible
      mutate(SleepQualityLevel = factor(SleepQualityLevel,
             levels = c("Poor","Fair","Normal","Excellent","Rebound")))
    clrs <- c("Poor"=COL_D,"Fair"=COL_W,"Normal"=COL_N,
              "Excellent"=COL_G,"Rebound"=COL_P)
    plot_ly(d, x = ~SleepQuality_pct, type = "histogram",
            color = ~SleepQualityLevel, colors = clrs, nbinsx = 30,
            marker = list(line = list(width = 0.4, color = "#FAFAF8"))) %>%
      acad_layout("Restorative sleep % (Deep + REM)", "Count") %>%
      layout(barmode = "stack",
             xaxis  = list(range = c(0, 75)),
             legend = list(orientation = "h", y = -0.32,
                           font = list(size = 10)),
             margin = list(l = 50, r = 20, t = 20, b = 100))
  })

  # ── Fig 21: Sleep duration variability (SD of TSD per participant) ────────
  output$midpoint_density <- renderPlotly({
    d <- d_sleep() %>%
      filter(!is.na(TSD), TSD >= 3) %>%
      group_by(User_id) %>%
      summarise(sd_TSD  = sd(TSD, na.rm = TRUE),
                mean_TSD = mean(TSD, na.rm = TRUE),
                n        = n(), .groups = "drop") %>%
      arrange(desc(sd_TSD))
    if (nrow(d) < 2) return(plotly_empty() %>%
      add_annotations(text = "Insufficient data.",
                      x = 0.5, y = 0.5, xref = "paper", yref = "paper",
                      showarrow = FALSE,
                      font = list(size = 12, color = "#6B7280",
                                  family = "Lora,serif")) %>%
      layout(paper_bgcolor = "#FAFAF8") %>% config(displayModeBar = FALSE))
    cohort_sd <- round(mean(d$sd_TSD, na.rm = TRUE), 2)
    plot_ly(d,
            x = ~reorder(User_id, -sd_TSD), y = ~sd_TSD,
            type  = "bar",
            marker = list(
              color     = d$sd_TSD,
              colorscale = list(c(0, COL_G), c(0.5, COL_W), c(1, COL_D)),
              cmin      = min(d$sd_TSD, na.rm = TRUE),
              cmax      = max(d$sd_TSD, na.rm = TRUE),
              cauto     = FALSE,
              showscale = FALSE,
              line = list(color = "#FAFAF8", width = 0.5)
            ),
            text  = ~paste0(round(sd_TSD, 2), "h"),
            textposition = "outside",
            textfont = list(size = 9, family = "Lora,serif"),
            hovertemplate = paste0("<b>%{x}</b><br>",
                                   "SD of TSD: %{y:.2f}h<br>",
                                   "Mean TSD: ",
                                   round(d$mean_TSD, 1), "h<br>",
                                   "n = ", d$n, " nights<extra></extra>")) %>%
      acad_layout("Participant", "Sleep duration variability (SD, hours)") %>%
      layout(
        shapes = list(list(
          type = "line", x0 = -0.5, x1 = nrow(d) - 0.5,
          y0 = cohort_sd, y1 = cohort_sd, xref = "x", yref = "y",
          line = list(dash = "dash", color = COL_P, width = 1.5)
        )),
        annotations = list(list(
          x = nrow(d) - 1, y = cohort_sd + 0.03,
          text = paste0("Cohort mean SD: ", cohort_sd, "h"),
          showarrow = FALSE, xref = "x", yref = "y",
          font = list(size = 9, color = COL_P, family = "Lora,serif"),
          xanchor = "right"
        ))
      )
  })

  # ── Fig 17: Social jetlag — per-participant mean (lollipop) ──────────────
  output$social_jetlag_box <- renderPlotly({
    d <- d_global() %>%
      filter(!is.na(SocialJetlag)) %>%
      group_by(User_id) %>%
      summarise(mean_sj = mean(abs(SocialJetlag), na.rm = TRUE),
                n       = n(), .groups = "drop") %>%
      arrange(desc(mean_sj)) %>%
      mutate(exceeds = mean_sj > 1,
             col     = ifelse(exceeds, COL_D, COL_G))
    # Lollipop: one scatter trace per participant drawn as a line from 0 → value
    # then a dot marker at the value. Pure plotly, no lapply, no formula in list().
    d_ord <- d %>% arrange(mean_sj) %>% mutate(y_pos = seq_len(n()))

    fig <- plot_ly()
    for (i in seq_len(nrow(d_ord))) {
      fig <- fig %>%
        add_lines(
          x = c(0, d_ord$mean_sj[i]),
          y = c(d_ord$y_pos[i], d_ord$y_pos[i]),
          line = list(color = d_ord$col[i], width = 2.2),
          showlegend = FALSE, hoverinfo = "none"
        )
    }
    fig %>%
      add_markers(
        data   = d_ord,
        x      = ~mean_sj,
        y      = ~y_pos,
        marker = list(
          color = d_ord$col,          # direct vector, not formula
          size  = 11,
          line  = list(color = "#FAFAF8", width = 1.5)
        ),
        text = ~paste0(
          User_id, "<br>Mean SJ: ", round(mean_sj, 2), "h<br>",
          ifelse(exceeds, "\u26a0 Exceeds 1h threshold", "Within 1h")
        ),
        hoverinfo = "text",
        showlegend = FALSE
      ) %>%
      layout(
        paper_bgcolor = "#FAFAF8", plot_bgcolor = "#FAFAF8",
        font  = list(family = "Lora,Georgia,serif", size = 11),
        xaxis = list(
          title     = "Mean social jetlag (hours)",
          range     = c(0, max(d_ord$mean_sj) * 1.18),
          gridcolor = "#E8E4DE", linecolor = "#CCCCCC",
          zeroline  = FALSE
        ),
        yaxis = list(
          title     = "",
          tickvals  = d_ord$y_pos,
          ticktext  = d_ord$User_id,
          range     = c(0.3, nrow(d_ord) + 0.7),
          gridcolor = "#E8E4DE", linecolor = "#CCCCCC"
        ),
        shapes = list(list(
          type = "line", x0 = 1, x1 = 1,
          y0 = 0.3, y1 = nrow(d_ord) + 0.7,
          xref = "x", yref = "y",
          line = list(dash = "dash", color = COL_D, width = 1.5)
        )),
        annotations = list(list(
          x = 1.03, y = nrow(d_ord) + 0.5,
          xref = "x", yref = "y",
          text = "1h threshold", showarrow = FALSE,
          font = list(size = 9, color = COL_D, family = "Lora,serif"),
          xanchor = "left", yanchor = "top"
        )),
        margin = list(l = 70, r = 30, t = 20, b = 55)
      ) %>%
      config(displayModeBar = FALSE)
  })

  # ── Fig 19: TSD vs restorative sleep quality ─────────────────────────────
  output$tsd_vs_deep <- renderPlotly({
    d <- d_global() %>%
      filter(!is.na(TSD), !is.na(SleepQuality_pct),
             TSD >= 3, SleepQuality_pct <= 75,
             analysis_cohort == "full")
    sp <- spear(d$TSD, d$SleepQuality_pct)
    p <- ggplot(d, aes(x = TSD, y = SleepQuality_pct,
                       colour = User_id,
                       text = paste0(User_id, "<br>Date: ", calendar_date,
                                     "<br>Sleep duration: ", round(TSD, 1), "h",
                                     "<br>Sleep quality: ",
                                     round(SleepQuality_pct, 1), "%"))) +
      geom_point(alpha = 0.18, size = 1.1) +
      geom_smooth(aes(group = 1), method = "loess", formula = y ~ x,
                  colour = "#1A1A1A", linewidth = 1.2, se = TRUE,
                  fill = "#CCCCCC", alpha = 0.25) +
      scale_colour_manual(values = PART_COLORS) +
      scale_y_continuous(limits = c(NA, 75)) +
      labs(x = "Sleep Duration (h)",
           y = "Sleep quality (%)",
           subtitle = paste0("Spearman ρ = ", sp$r, "  ·  p ", fmt_p(sp$p),
                             "  ·  full cohort")) +
      theme_minimal(base_family = "serif") +
      theme(legend.position = "none",
            panel.grid.major = element_line(colour = "#E8E4DE"),
            plot.background  = element_rect(fill = "#FAFAF8", colour = NA),
            panel.background = element_rect(fill = "#FAFAF8", colour = NA))
    ggplotly(p, tooltip = "text") %>%
      layout(paper_bgcolor = "#FAFAF8", plot_bgcolor = "#FAFAF8",
             font = list(family = "Lora,Georgia,serif", size = 11)) %>%
      config(displayModeBar = FALSE)
  })

  # ── Fig 20: TSD distribution histogram ───────────────────────────────────
  output$sleep_perf_scatter <- renderPlotly({
    d <- d_sleep() %>% filter(!is.na(TSD))
    # Bin into duration bands for pie
    d <- d %>% mutate(band = case_when(
      TSD < 5              ~ "< 5h (severe)",
      TSD >= 5 & TSD < 6   ~ "5–6h",
      TSD >= 6 & TSD < 7   ~ "6–7h (insufficient)",
      TSD >= 7 & TSD < 8   ~ "7–8h (adequate)",
      TSD >= 8 & TSD < 9   ~ "8–9h (good)",
      TSD >= 9              ~ "≥ 9h (long)"
    ))
    pie_d <- d %>%
      group_by(band) %>%
      summarise(n = n(), .groups = "drop") %>%
      mutate(band = factor(band, levels = c("< 5h (severe)", "5–6h",
                           "6–7h (insufficient)", "7–8h (adequate)",
                           "8–9h (good)", "≥ 9h (long)"))) %>%
      arrange(band)
    pie_cols <- c("#8A2C2C","#B07D2C","#C9A84C","#3A6B4A","#2C5F8A","#5B8DB8")
    plot_ly(pie_d,
            labels = ~band, values = ~n,
            type   = "pie",
            marker = list(colors = pie_cols,
                          line   = list(color = "#FAFAF8", width = 1.5)),
            textinfo = "label+percent",
            hovertemplate = "<b>%{label}</b><br>Nights: %{value}<br>Share: %{percent}<extra></extra>",
            sort = FALSE) %>%
      layout(
        paper_bgcolor = "#FAFAF8",
        font   = list(family = "Lora,Georgia,serif", size = 11),
        legend = list(orientation = "v", x = 1.02, y = 0.5,
                      font = list(size = 10)),
        margin = list(l = 10, r = 10, t = 20, b = 10),
        annotations = list(list(
          text = paste0("n = ", nrow(d), " nights"),
          x = 0.5, y = -0.06, xref = "paper", yref = "paper",
          showarrow = FALSE, font = list(size = 10, color = "#6B7280",
                                         family = "Lora,serif")
        ))
      ) %>% config(displayModeBar = FALSE)
  })
  # ── Fig 21: SF bars + Sleep Quality line by weekday (dual-axis) ──────────
  output$sleep_weekday_dual <- renderPlotly({
    wday_ord <- c("Monday","Tuesday","Wednesday","Thursday",
                  "Friday","Saturday","Sunday")
    d <- d_sleep() %>%
      filter(!is.na(SleepQuality_pct), !is.na(AwakeSleepProp),
             SleepQuality_pct <= 75, analysis_cohort == "full") %>%
      mutate(Weekday = factor(weekday, levels = wday_ord)) %>%
      group_by(Weekday) %>%
      summarise(mean_sq   = mean(SleepQuality_pct, na.rm = TRUE),
                mean_frag = mean(AwakeSleepProp * 100, na.rm = TRUE),
                n         = n(), .groups = "drop")

    plot_ly() %>%
      add_bars(data = d,
               x    = ~Weekday, y = ~mean_frag,
               name = "Sleep fragmentation (awake %)",
               marker = list(color = COL_W, opacity = 0.82,
                             line = list(color = "#FAFAF8", width = 0.5)),
               hovertemplate = "<b>%{x}</b><br>Awake %: %{y:.1f}%<extra></extra>",
               yaxis = "y") %>%
      add_lines(data = d,
                x    = ~Weekday, y = ~mean_sq,
                name = "Sleep quality (Deep + REM %)",
                line = list(color = COL_P, width = 2.8),
                marker = list(size = 8, color = COL_P,
                              line = list(color = "#FAFAF8", width = 1.5)),
                hovertemplate = "<b>%{x}</b><br>Sleep quality: %{y:.1f}%<extra></extra>",
                yaxis = "y2") %>%
      layout(
        paper_bgcolor = "#FAFAF8", plot_bgcolor = "#FAFAF8",
        font    = list(family = "Lora,Georgia,serif", size = 11),
        xaxis   = list(title = "", gridcolor = "#E8E4DE",
                       linecolor = "#CCCCCC", tickangle = -25),
        yaxis   = list(title = "Sleep Fragmentation (%)",
                       gridcolor = "#E8E4DE", linecolor = "#CCCCCC",
                       rangemode = "tozero"),
        yaxis2  = list(title = "Sleep Quality (%)",
                       overlaying = "y", side = "right",
                       range = c(34, 36),
                       showgrid = FALSE, linecolor = "#CCCCCC"),
        legend  = list(orientation = "h", y = 1.08,
                       font = list(size = 10)),
        barmode = "group",
        margin  = list(l = 60, r = 70, t = 35, b = 60)
      ) %>%
      config(displayModeBar = FALSE)
  })

  output$rhr_timeline <- renderPlotly({
    d <- d_rec() %>%
      arrange(calendar_date) %>%
      group_by(calendar_date) %>%
      summarise(RecoveryScore = mean(RecoveryScore, na.rm = TRUE),
                BB_delta      = mean(BB_delta, na.rm = TRUE),
                .groups = "drop") %>%
      mutate(rec_roll7 = rollapply(RecoveryScore, 7, mean, na.rm = TRUE,
                                   fill = NA, align = "right"))
    rec_col <- ifelse(is.na(d$RecoveryScore) | d$RecoveryScore >= 0,
                      COL_G, COL_D)
    bb_col  <- ifelse(is.na(d$BB_delta) | d$BB_delta >= 0,
                      "rgba(58,107,74,0.28)", "rgba(138,44,44,0.28)")
    plot_ly() %>%
      add_bars(data = d, x = ~calendar_date, y = ~BB_delta,
               name = "Net BB change",
               yaxis = "y2",
               marker = list(color = bb_col),
               hovertemplate = "<b>%{x}</b><br>Net BB change: %{y:.0f}<extra></extra>") %>%
      add_bars(data = d, x = ~calendar_date, y = ~RecoveryScore,
               name = "Recovery score",
               marker = list(color = rec_col, opacity = 0.85,
                             line = list(color = "#FAFAF8", width = 0.3)),
               hovertemplate = "<b>%{x}</b><br>Recovery score: %{y:.2f}<extra></extra>") %>%
      add_lines(data = d, x = ~calendar_date, y = ~rec_roll7,
                name = "7-day trend",
                line = list(color = COL_S, width = 2.2, dash = "dot"),
                hovertemplate = "<b>%{x}</b><br>7-day mean: %{y:.2f}<extra></extra>") %>%
      layout(
        paper_bgcolor = "#FAFAF8", plot_bgcolor = "#FAFAF8",
        font = list(family = "Lora,Georgia,serif", size = 11),
        xaxis  = list(title = "", gridcolor = "#E8E4DE", linecolor = "#CCCCCC"),
        yaxis  = list(title = "Recovery score",
                      gridcolor = "#E8E4DE", linecolor = "#CCCCCC",
                      zeroline = TRUE, zerolinecolor = COL_D, zerolinewidth = 1.5),
        yaxis2 = list(title = "Net BB change",
                      overlaying = "y", side = "right",
                      range = c(-20, 20), showgrid = FALSE, linecolor = "#CCCCCC"),
        hovermode = "x unified",
        legend = list(orientation = "h", y = 1.08,
                      bgcolor = "rgba(250,250,248,0.9)",
                      bordercolor = "#DDD9D0", borderwidth = 1,
                      font = list(size = 10)),
        margin = list(l = 65, r = 70, t = 20, b = 45),
        shapes = list(list(type = "line", x0 = 0, x1 = 1, xref = "paper",
                           y0 = 0, y1 = 0, yref = "y",
                           line = list(dash = "dash", color = "#AAAAAA", width = 1)))
      ) %>% config(displayModeBar = FALSE)
  })

  # ── Tab 4: Lag correlation bars ───────────────────────────────────────────
  output$lag_bars <- renderPlotly({
    d <- d_rec()
    pair_map <- list(
      sl_tsd  = list(x_base = "SessionLoad", y = "TSD",
                     lbl = "Training Load → Sleep Duration"),
      sl_rec  = list(x_base = "SessionLoad", y = "RecoveryScore",
                     lbl = "Training Load → Recovery"),
      tsd_rec = list(x_base = "TSD",         y = "RecoveryScore",
                     lbl = "Sleep Duration → Recovery")
    )
    pr <- pair_map[[input$lag_pair]]
    res <- lapply(0:3, function(lg) {
      x_col <- if (lg == 0) pr$x_base else paste0(pr$x_base, "_lag", lg)
      if (!x_col %in% names(d)) return(NULL)
      x <- d[[x_col]]; y <- d[[pr$y]]
      ok <- complete.cases(x, y)
      if (sum(ok) < 10) return(NULL)
      r <- cor(x[ok], y[ok], method = "spearman")
      data.frame(Lag = lg, r = r)
    }) %>% bind_rows()
    if (nrow(res) == 0) return(plotly_empty())
    bar_col <- ifelse(res$r >= 0, COL_G, COL_D)
    plot_ly(res, x = ~factor(Lag), y = ~r, type = "bar",
            marker = list(color = bar_col, opacity = 0.85,
                          line = list(color = "#FAFAF8", width = 0.5)),
            text  = ~paste0("ρ = ", round(r, 3)),
            hoverinfo = "x+text",
            textposition = "outside",
            textfont = list(size = 11, family = "Lora,serif")) %>%
      acad_layout("Lag (days)", "Spearman ρ") %>%
      layout(
        title = list(text = pr$lbl, x = 0, font = list(size = 12)),
        xaxis = list(tickvals = c(0,1,2,3),
                     ticktext = c("Same day","1 day prior",
                                  "2 days prior","3 days prior")),
        yaxis = list(zeroline = TRUE, zerolinecolor = "#999",
                     zerolinewidth = 1.5, range = c(-0.1, 0.1)),
        shapes = list(list(type = "line", x0 = -0.5, x1 = 3.5,
                           y0 = 0, y1 = 0, xref = "x", yref = "y",
                           line = list(color = "#999", width = 1)))
      )
  })

  # ── Tab 4: Overtraining scatter ───────────────────────────────────────────
  output$overtraining_scatter <- renderPlotly({
    d <- d_rec() %>%
      filter(!is.na(SessionLoad), !is.na(RecoveryScore), SessionLoad > 0)
    d_samp <- slice_sample(d, n = min(3000L, nrow(d)))
    d_samp <- d_samp %>% filter(SessionLoad <= 60000)
    flag_lbl <- ifelse(d_samp$overtraining_flag, "At risk", "No flag")
    plot_ly(d_samp, x = ~SessionLoad, y = ~RecoveryScore, type = "scattergl",
            mode = "markers",
            color = ~flag_lbl,
            colors = c("No flag" = COL_P, "At risk" = COL_D),
            marker = list(size = 5, opacity = 0.35, line = list(width = 0)),
            text = ~paste0(User_id, "<br>Date: ", calendar_date,
                           "<br>Load: ", round(SessionLoad),
                           "<br>Recovery: ", round(RecoveryScore, 2),
                           "<br>TSD: ", round(TSD, 1), "h"),
            hoverinfo = "text") %>%
      acad_layout("Session Load", "Recovery score") %>%
      layout(
        showlegend = TRUE,
        xaxis = list(range = c(0, 60000)),
        legend = list(title = list(text = "Overtraining"),
                      orientation = "h", y = 1.06, font = list(size = 10)),
        annotations = list(
          list(text = "⚠ Overtraining risk",
               x = 60000, y = -1.5, xref = "x", yref = "y",
               xanchor = "right", yanchor = "top",
               showarrow = FALSE,
               font = list(size = 10, color = COL_D, family = "Lora,serif")),
          list(text = "Well-adapted",
               x = 60000, y = max(d_samp$RecoveryScore, na.rm=TRUE) * 0.85,
               xref = "x", yref = "y",
               xanchor = "right", yanchor = "top",
               showarrow = FALSE,
               font = list(size = 10, color = COL_G, family = "Lora,serif"))
        ),
        shapes = list(
          list(type = "line", x0 = sl_p75, x1 = sl_p75,
               y0 = min(d_samp$RecoveryScore, na.rm=TRUE),
               y1 = max(d_samp$RecoveryScore, na.rm=TRUE),
               xref = "x", yref = "y",
               line = list(dash = "dot", color = COL_W, width = 1.5)),
          list(type = "line", x0 = 0, x1 = 60000,
               y0 = -1.5, y1 = -1.5, xref = "x", yref = "y",
               line = list(dash = "dot", color = COL_D, width = 1.5))
        )
      )
  })

  # ── Tab 4: Respiration timeline ───────────────────────────────────────────
  output$resp_timeline <- renderPlotly({
    d <- d_rec() %>%
      filter(!is.na(DailyRespiration.avgWakingRespirationValue)) %>%
      arrange(calendar_date) %>%
      group_by(calendar_date) %>%
      summarise(resp = mean(DailyRespiration.avgWakingRespirationValue, na.rm = TRUE),
                .groups = "drop") %>%
      mutate(roll7 = rollapply(resp, 7, mean, na.rm = TRUE, fill = NA, align = "right"))
    plot_ly(d, x = ~calendar_date) %>%
      add_markers(y = ~resp, name = "Daily respiration (cohort mean)",
                  marker = list(size = 3, color = COL_P, opacity = 0.4)) %>%
      add_lines(y = ~roll7, name = "7-day rolling mean",
                line = list(color = COL_S, width = 2)) %>%
      acad_layout("", "Respiration (breaths/min)") %>%
      layout(
        hovermode = "x unified",
        yaxis = list(range = c(12, 18), gridcolor = "#E8E4DE", linecolor = "#CCCCCC"),
        legend = list(orientation = "h", y = 1.06, font = list(size = 10)),
        shapes = list(list(type = "rect", x0 = 0, x1 = 1, xref = "paper",
                           y0 = 16, y1 = 18, yref = "y",
                           fillcolor = "rgba(138,44,44,0.07)",
                           line = list(width = 0)))
      )
  })

  # ── Tab 4: Fig 18 – Weekly recovery rhythm + Body Battery ────────────────
  output$rec_weekday <- renderPlotly({
    wday_ord <- c("Monday","Tuesday","Wednesday","Thursday",
                  "Friday","Saturday","Sunday")
    d <- d_rec() %>%
      mutate(Weekday = factor(weekday, levels = wday_ord)) %>%
      group_by(Weekday) %>%
      summarise(mean_rec = mean(RecoveryScore, na.rm = TRUE),
                mean_bb  = mean(BB_delta_lag1, na.rm = TRUE),  # lag1: charged(t)-drained(t-1) = same overnight window as RecoveryScore(t)
                n = n(), .groups = "drop")
    bar_col <- ifelse(d$mean_rec >= 0, COL_G, COL_D)
    plot_ly() %>%
      add_bars(data = d, x = ~Weekday, y = ~mean_rec,
               name = "Mean recovery score",
               marker = list(color = bar_col, opacity = 0.85,
                             line = list(color = "#FAFAF8", width = 0.5)),
               hovertemplate = "<b>%{x}</b><br>Recovery score: %{y:.2f}<extra></extra>",
               yaxis = "y") %>%
      add_lines(data = d, x = ~Weekday, y = ~mean_bb,
                name = "Mean Body Battery recharge",
                line = list(color = COL_W, width = 2.4),
                marker = list(size = 8, color = COL_W,
                              line = list(color = "#FAFAF8", width = 1.5)),
                hovertemplate = "<b>%{x}</b><br>Body Battery: %{y:.0f}<extra></extra>",
                yaxis = "y2") %>%
      layout(
        paper_bgcolor = "#FAFAF8", plot_bgcolor = "#FAFAF8",
        font  = list(family = "Lora,Georgia,serif", size = 11),
        xaxis = list(title = "", gridcolor = "#E8E4DE", linecolor = "#CCCCCC",
                     tickangle = -20),
        yaxis = list(title = "Recovery score",
                     range = c(-0.6, 0.6),
                     gridcolor = "#E8E4DE", linecolor = "#CCCCCC",
                     zeroline = TRUE, zerolinecolor = "#888", zerolinewidth = 1.2),
        yaxis2 = list(title = "Net BB change",
                      overlaying = "y", side = "right",
                      range = c(-20, 20), showgrid = FALSE, linecolor = "#CCCCCC"),
        legend = list(orientation = "h", y = 1.08, font = list(size = 10)),
        shapes = list(list(type = "line", x0 = 0, x1 = 1, xref = "paper",
                           y0 = 0, y1 = 0, yref = "y",
                           line = list(dash = "dash", color = "#888", width = 1))),
        margin = list(l = 60, r = 70, t = 30, b = 55)
      ) %>% config(displayModeBar = FALSE)
  })

  # ── Tab 4: Fig 19 – Recovery score histogram + BB mean per bin ───────────
  output$rec_hist <- renderPlotly({
    d <- d_rec() %>%
      filter(!is.na(RecoveryScore)) %>%
      mutate(rec_bin = cut(RecoveryScore,
                           breaks = seq(floor(min(RecoveryScore, na.rm=TRUE)) - 1,
                                        ceiling(max(RecoveryScore, na.rm=TRUE)) + 1,
                                        by = 0.5),
                           include.lowest = TRUE))
    hist_d <- d %>%
      group_by(rec_bin) %>%
      summarise(n       = n(),
                bin_mid = mean(RecoveryScore, na.rm = TRUE),
                # Clip bins with few obs to reduce outlier spikes
                mean_bb = {
                  vals <- BB_delta[!is.na(BB_delta)]
                  if (length(vals) < 3) NA_real_
                  else mean(vals[vals >= quantile(vals, 0.10) &
                                 vals <= quantile(vals, 0.90)], na.rm = TRUE)
                },
                .groups = "drop") %>%
      filter(!is.na(bin_mid), n >= 3) %>%
      arrange(bin_mid)
    bar_col <- ifelse(hist_d$bin_mid >= 0, COL_G, COL_D)
    pct_positive <- round(100 * mean(d$RecoveryScore >= 0, na.rm = TRUE), 1)
    plot_ly() %>%
      add_bars(data = hist_d, x = ~bin_mid, y = ~n,
               name = "Recovery Score",
               marker = list(color = bar_col, opacity = 0.82,
                             line = list(color = "#FAFAF8", width = 0.4)),
               hovertemplate = paste0("Recovery score: %{x:.1f}<br>",
                                      "Days: %{y}<extra></extra>"),
               yaxis = "y") %>%
      add_lines(data = hist_d, x = ~bin_mid, y = ~mean_bb,
                name = "Net BB change",
                line = list(color = COL_W, width = 2.5),
                marker = list(size = 6, color = COL_W,
                              line = list(color = "#FAFAF8", width = 1)),
                hovertemplate = "Recovery score: %{x:.1f}<br>Net BB: %{y:.1f}<extra></extra>",
                yaxis = "y2") %>%
      layout(
        paper_bgcolor = "#FAFAF8", plot_bgcolor = "#FAFAF8",
        font  = list(family = "Lora,Georgia,serif", size = 11),
        xaxis = list(title = "Recovery score", gridcolor = "#E8E4DE",
                     linecolor = "#CCCCCC",
                     zeroline = TRUE, zerolinecolor = "#888", zerolinewidth = 1.5),
        yaxis  = list(title = "Days (count)",
                      gridcolor = "#E8E4DE", linecolor = "#CCCCCC"),
        yaxis2 = list(title = "Net BB change",
                      overlaying = "y", side = "right",
                      range = c(-20, 20), showgrid = FALSE, linecolor = "#CCCCCC"),
        legend = list(orientation = "h", y = -0.32, font = list(size = 10)),
        annotations = list(list(
          text = paste0(pct_positive, "% of days above zero (well-recovered)"),
          x = 0.5, y = 1.06, xref = "paper", yref = "paper",
          showarrow = FALSE, xanchor = "center",
          font = list(size = 10, color = COL_G, family = "Lora,serif")
        )),
        margin = list(l = 60, r = 70, t = 45, b = 90)
      ) %>% config(displayModeBar = FALSE)
  })

  # ── Tab 4: Fig 20 – Per-participant mean recovery score bar chart ─────────
  output$rec_part_bar <- renderPlotly({
    d <- d_rec() %>%
      filter(!is.na(RecoveryScore), User_id != "P013") %>%
      group_by(User_id) %>%
      summarise(mean_rec = mean(RecoveryScore, na.rm = TRUE),
                n        = n(), .groups = "drop") %>%
      arrange(desc(mean_rec))
    bar_col <- ifelse(d$mean_rec >= 0, COL_G, COL_D)
    plot_ly(d, x = ~reorder(User_id, mean_rec), y = ~mean_rec,
            type = "bar",
            marker = list(color = bar_col, opacity = 0.88,
                          line = list(color = "#FAFAF8", width = 0.5)),
            text  = ~paste0("n = ", n, " days"),
            hovertemplate = paste0("<b>%{x}</b><br>Mean recovery: %{y:.3f}",
                                   "<br>%{text}<extra></extra>")) %>%
      layout(
        paper_bgcolor = "#FAFAF8", plot_bgcolor = "#FAFAF8",
        font  = list(family = "Lora,Georgia,serif", size = 11),
        xaxis = list(title = "", gridcolor = "#E8E4DE", linecolor = "#CCCCCC",
                     tickangle = -30),
        yaxis = list(title = "Recovery score",
                     range = c(-0.2, 0.6),
                     gridcolor = "#E8E4DE", linecolor = "#CCCCCC",
                     zeroline = TRUE, zerolinecolor = "#AAAAAA", zerolinewidth = 1.5),
        shapes = list(list(type = "line", x0 = 0, x1 = 1, xref = "paper",
                           y0 = 0, y1 = 0, yref = "y",
                           line = list(dash = "dash", color = "#AAAAAA", width = 1))),
        margin = list(l = 65, r = 20, t = 15, b = 60)
      ) %>% config(displayModeBar = FALSE)
  })

  # ── Tab 4: Lagged regression ──────────────────────────────────────────────
  m_lag <- reactive({
    safe_lm("RecoveryScore ~ TSD_lag1 + SE_lag1 + SessionLoad_lag1", d_rec())
  })

  output$lag_reg_tbl <- renderDT({
    tbl <- coef_tbl(m_lag())
    datatable(tbl, rownames = FALSE, class = "compact stripe",
              options = list(dom = "t", pageLength = 10)) %>%
      formatStyle("p",
        backgroundColor = styleInterval(c(0.05, 0.10),
                                        c("#DCF0DC","#FEF3D0","#FAEAEA")))
  })

  output$lag_reg_interp <- renderUI({
    interp_ui(m_lag(), "TSD_lag1", "TSD_lag1 (last night's sleep hours)", "RecoveryScore")
  })

  # ── Tab 5: VO2max trajectories ────────────────────────────────────────────
  output$vo2_traj <- renderPlotly({
    d <- garmin %>%
      filter(!is.na(VO2max_imputed_final)) %>%
      group_by(User_id) %>% filter(n() >= 90) %>% ungroup()
    # Cohort daily mean + LOESS smooth for the bold overlay
    d_mean <- d %>%
      group_by(calendar_date) %>%
      summarise(vo2_cohort = mean(VO2max_imputed_final, na.rm = TRUE),
                .groups = "drop") %>%
      arrange(calendar_date) %>%
      mutate(vo2_smooth = {
        fit <- loess(vo2_cohort ~ as.numeric(calendar_date), span = 0.15)
        predict(fit)
      })
    fig <- plot_ly()
    for (pid in sort(unique(d$User_id))) {
      di <- d %>% filter(User_id == pid)
      fig <- fig %>%
        add_lines(data = di, x = ~calendar_date, y = ~VO2max_imputed_final,
                  name = pid, legendgroup = pid,
                  line = list(color = PART_COLORS[pid], width = 1.1),
                  opacity = 0.4,
                  text = ~paste0(pid, "<br>", calendar_date,
                                 "<br>VO₂max: ", round(VO2max_imputed_final,1)),
                  hoverinfo = "text", showlegend = TRUE)
    }
    fig %>%
      add_lines(data = d_mean, x = ~calendar_date, y = ~vo2_smooth,
                name = "Cohort mean",
                line = list(color = "#1A1A1A", width = 3),
                hovertemplate = "<b>Cohort mean</b><br>%{x|%b %Y}<br>VO₂max: %{y:.1f}<extra></extra>",
                showlegend = TRUE) %>%
      layout(
        paper_bgcolor = "#FAFAF8", plot_bgcolor = "#FAFAF8",
        font  = list(family = "Lora,Georgia,serif", size = 11),
        xaxis = list(title = "", gridcolor = "#E8E4DE", linecolor = "#CCCCCC",
                     tickformat = "%b %Y"),
        yaxis = list(title = "VO₂max (ml/kg/min)",
                     gridcolor = "#E8E4DE", linecolor = "#CCCCCC"),
        legend = list(orientation = "h", y = -0.22, font = list(size = 9),
                      title = list(text = "Participant")),
        hovermode = "closest",
        margin = list(l = 55, r = 20, t = 20, b = 80)
      ) %>%
      config(displayModeBar = FALSE)
  })

  # ── Tab 5: Resilience scatter ─────────────────────────────────────────────
  output$resilience_scatter <- renderPlotly({
    d <- vo2_summary %>% drop_na(vo2_slope)
    if (nrow(d) < 3) return(plotly_empty())
    plot_ly(d, x = ~vo2_slope, y = ~rhr_dev_sd,
            type = "scatter", mode = "markers+text",
            color = ~Gender,
            colors = c("Male" = COL_P, "Female" = "#8A3A5F"),
            size  = ~load_mean, sizes = c(10, 60),
            text  = ~User_id,
            textposition = "top center",
            textfont = list(size = 10, family = "Lora,serif"),
            marker = list(opacity = 0.82,
                          line = list(color = "white", width = 1)),
            hovertemplate = paste0("<b>%{text}</b><br>",
                                   "VO₂max slope: %{x:.2f} ml/kg/min/yr<br>",
                                   "RHR dev SD: %{y:.2f} bpm<br>",
                                   "<extra></extra>")) %>%
      acad_layout("VO₂max slope (ml/kg/min per year)", "RHR deviation SD (bpm)") %>%
      layout(
        legend = list(title = list(text = "Sex"), font = list(size = 11)),
        shapes = list(
          list(type = "line", x0 = 0, x1 = 0, y0 = 0,
               y1 = max(d$rhr_dev_sd, na.rm=TRUE) + 1,
               xref = "x", yref = "y",
               line = list(dash = "dot", color = "#AAAAAA", width = 1.5))
        ),
        annotations = list(
          list(x = max(d$vo2_slope, na.rm=TRUE) * 0.8,
               y = min(d$rhr_dev_sd, na.rm=TRUE) + 0.3,
               text = "Resilient", showarrow = FALSE,
               font = list(size = 10, color = COL_G, family = "Lora,serif")),
          list(x = min(d$vo2_slope, na.rm=TRUE) + 0.1,
               y = max(d$rhr_dev_sd, na.rm=TRUE) - 0.3,
               text = "Struggling", showarrow = FALSE,
               font = list(size = 10, color = COL_D, family = "Lora,serif"))
        )
      )
  })

    # ── Tab 5: Individual timeline ────────────────────────────────────────────
  output$indiv_timeline <- renderPlotly({
    pid <- input$pid_indiv
    d   <- garmin %>% filter(User_id == pid) %>% arrange(calendar_date)
    col <- PART_COLORS[pid]
    fig <- plot_ly() %>%
      add_bars(data = d, x = ~calendar_date, y = ~TSD, name = "Sleep (h)",
               yaxis = "y2",
               marker = list(color = "rgba(107,114,128,0.28)"),
               hovertemplate = "<b>%{x}</b><br>TSD: %{y:.1f}h<extra></extra>") %>%
      add_lines(data = d, x = ~calendar_date, y = ~RecoveryScore,
                name = "Recovery",
                line = list(color = col, width = 1.6),
                hovertemplate = "<b>%{x}</b><br>Recovery: %{y:.2f}<extra></extra>") %>%
      add_lines(data = d, x = ~calendar_date, y = ~rec_roll7,
                name = "7d mean",
                line = list(color = COL_S, width = 2, dash = "dot"))
    run_d <- d %>% filter(ActivityGroup == "Running", activity_day == TRUE)
    if (nrow(run_d) > 0) {
      fig <- fig %>%
        add_markers(data = run_d, x = ~calendar_date, y = ~RecoveryScore,
                    name = "Running session", yaxis = "y",
                    marker = list(symbol = "triangle-up", size = 7,
                                  color = COL_P, opacity = 0.85))
    }
    fig %>% layout(
      paper_bgcolor = "#FAFAF8", plot_bgcolor = "#FAFAF8",
      font = list(family = "Lora,Georgia,serif", size = 11),
      title = list(text = paste0(pid, " — Personal Timeline"), x = 0,
                   font = list(size = 12)),
      xaxis = list(title = "", gridcolor = "#E8E4DE", linecolor = "#CCCCCC"),
      yaxis = list(title = "Recovery score", zeroline = TRUE,
                   zerolinecolor = COL_D, zerolinewidth = 1.5,
                   gridcolor = "#E8E4DE", linecolor = "#CCCCCC"),
      yaxis2 = list(title = "Sleep (h)", overlaying = "y", side = "right",
                    showgrid = FALSE),
      hovermode = "x unified",
      legend = list(orientation = "h", y = -0.22, font = list(size = 10)),
      margin = list(l = 55, r = 55, t = 30, b = 50),
      shapes = list(list(type = "line", x0 = 0, x1 = 1, xref = "paper",
                         y0 = 0, y1 = 0, yref = "y",
                         line = list(dash = "dash", color = "#AAAAAA", width = 1)))
    ) %>% config(displayModeBar = FALSE)
  })

  # ── Tab 5: Individual lag scatter ─────────────────────────────────────────
  output$indiv_lag <- renderPlotly({
    pid <- input$pid_indiv
    d   <- garmin %>% filter(User_id == pid) %>% drop_na(TSD_lag1, RecoveryScore)
    if (nrow(d) < 15) return(plotly_empty() %>%
      add_annotations(text = "Insufficient data.", x = 0.5, y = 0.5,
                       xref = "paper", yref = "paper", showarrow = FALSE))
    sp  <- spear(d$TSD_lag1, d$RecoveryScore)
    col <- PART_COLORS[pid]
    p <- ggplot(d, aes(x = TSD_lag1, y = RecoveryScore,
                       colour = SL_Intensity,
                       text = paste0("Date: ", calendar_date,
                                     "<br>Last night: ", round(TSD_lag1,1), "h",
                                     "<br>Recovery: ", round(RecoveryScore,2),
                                     "<br>Training: ", SL_Intensity))) +
      geom_point(alpha = 0.4, size = 1.8) +
      geom_smooth(aes(group = 1), method = "lm", se = TRUE, colour = "#1A1A1A",
                  linewidth = 1, formula = y~x) +
      geom_hline(yintercept = 0, linetype = "dashed", colour = COL_N) +
      scale_colour_manual(values = c("Rest Day"=COL_N,"Light"=COL_G,
                                     "Moderate"=COL_W,"High"=COL_D)) +
      labs(x = "Last night's sleep duration (h)", y = "Today's Recovery Score",
           colour = "Intensity",
           subtitle = paste0("Spearman ρ = ", sp$r, "  ·  p ", fmt_p(sp$p),
                             "  ·  n = ", nrow(d))) +
      theme_minimal(base_family = "serif") +
      theme(panel.grid.major = element_line(colour = "#E8E4DE"),
            plot.background  = element_rect(fill = "#FAFAF8", colour = NA),
            panel.background = element_rect(fill = "#FAFAF8", colour = NA),
            legend.position  = "none")       # suppress ggplot legend; use plotly below
    ggplotly(p, tooltip = "text") %>%
      layout(paper_bgcolor = "#FAFAF8", plot_bgcolor = "#FAFAF8",
             font = list(family = "Lora,Georgia,serif", size = 11),
             legend = list(orientation = "h", y = 1.12, x = 0,
                           xanchor = "left", font = list(size = 10)),
             margin = list(l = 55, r = 20, t = 55, b = 55)) %>%
      config(displayModeBar = FALSE)
  })

  # ── Tab 5: Individual summary ─────────────────────────────────────────────
  output$indiv_summary <- renderUI({
    pid <- input$pid_indiv
    d   <- garmin %>% filter(User_id == pid)

    best_wk <- d %>%
      mutate(wk = paste0(year(calendar_date), "-W",
                         sprintf("%02d", isoweek(calendar_date)))) %>%
      group_by(wk) %>%
      summarise(mr = mean(RecoveryScore, na.rm=TRUE), .groups="drop") %>%
      drop_na() %>%
      { if (nrow(.) > 0) .$wk[which.max(.$mr)] else "N/A" }

    worst_wk <- d %>%
      mutate(wk = paste0(year(calendar_date), "-W",
                         sprintf("%02d", isoweek(calendar_date)))) %>%
      group_by(wk) %>%
      summarise(mr = mean(RecoveryScore, na.rm=TRUE), .groups="drop") %>%
      drop_na() %>%
      { if (nrow(.) > 0) .$wk[which.min(.$mr)] else "N/A" }

    tsd_act  <- round(mean(d$TSD[d$activity_day == TRUE],  na.rm=TRUE), 1)
    tsd_rest <- round(mean(d$TSD[d$activity_day == FALSE], na.rm=TRUE), 1)
    n_flag   <- sum(d$overtraining_flag, na.rm=TRUE)
    pct_flag <- round(100 * n_flag / nrow(d), 1)
    v_mean   <- round(mean(d$VO2max_imputed_final, na.rm=TRUE), 1)

    tagList(
      div(class = "kpi-row",
        div(class = "kpi-box", div(class = "kpi-lbl", "Best recovery week"),
            div(class = "kpi-val", style="font-size:18px;", best_wk)),
        div(class = "kpi-box", div(class = "kpi-lbl", "Worst recovery week"),
            div(class = "kpi-val", style="font-size:18px;", worst_wk)),
        div(class = "kpi-box", div(class = "kpi-lbl", "Mean VO₂max"),
            div(class = "kpi-val", style="font-size:22px;",
                paste0(v_mean, " ml/kg/min")))
      ),
      div(class = "kpi-row",
        div(class = "kpi-box", div(class = "kpi-lbl", "Sleep: training vs rest"),
            div(class = "kpi-val", style="font-size:20px;",
                paste0(tsd_act, " vs ", tsd_rest, "h"))),
        div(class = "kpi-box", div(class = "kpi-lbl", "Overtraining risk days"),
            div(class = "kpi-val",
                style = paste0("font-size:22px; color:",
                               if (pct_flag > 5) COL_D else COL_G, ";"),
                paste0(n_flag, " (", pct_flag, "%)"))
            )
      )
    )
  })

  # ── Fig 44: Individual lag correlation bars ──────────────────────────────────
  output$indiv_lag_bars <- renderPlotly({
    pid <- input$pid_indiv
    d   <- garmin %>% filter(User_id == pid)
    pairs <- list(
      list(x = "SessionLoad", y = "RecoveryScore", lbl = "Training Load → Recovery"),
      list(x = "TSD",         y = "RecoveryScore", lbl = "Sleep Duration → Recovery")
    )
    res <- lapply(pairs, function(pr) {
      lapply(0:3, function(lg) {
        x_col <- if (lg == 0) pr$x else paste0(pr$x, "_lag", lg)
        if (!x_col %in% names(d)) return(NULL)
        x <- d[[x_col]]; y <- d[[pr$y]]
        ok <- complete.cases(x, y)
        if (sum(ok) < 8) return(NULL)
        r <- cor(x[ok], y[ok], method = "spearman")
        data.frame(Lag = lg, r = round(r, 3), Pair = pr$lbl)
      }) %>% bind_rows()
    }) %>% bind_rows()
    if (nrow(res) == 0) return(plotly_empty() %>%
      add_annotations(text = "Insufficient data for this participant.",
        x = 0.5, y = 0.5, xref = "paper", yref = "paper",
        showarrow = FALSE, font = list(size = 12, color = "#6B7280")) %>%
      layout(paper_bgcolor = "#FAFAF8") %>% config(displayModeBar = FALSE))
    bar_col <- ifelse(res$r >= 0, COL_G, COL_D)
    plot_ly(res, x = ~factor(Lag), y = ~r, color = ~Pair,
            colors = c(COL_P, COL_S),
            type = "bar",
            marker = list(opacity = 0.85, line = list(color = "#FAFAF8", width = 0.5)),
            text  = ~paste0("ρ = ", r),
            hoverinfo = "x+text",
            textposition = "outside",
            textfont = list(size = 10, family = "Lora,serif")) %>%
      acad_layout("Lag (days)", "Spearman ρ") %>%
      layout(
        barmode = "group",
        xaxis = list(tickvals = c(0,1,2,3),
                     ticktext = c("Same day","Lag 1","Lag 2","Lag 3")),
        yaxis = list(zeroline = TRUE, zerolinecolor = "#999",
                     zerolinewidth = 1.5, range = c(-0.1, 0.1)),
        legend = list(orientation = "h", y = 1.08, font = list(size = 10)),
        shapes = list(list(type = "line", x0 = -0.5, x1 = 3.5,
                           y0 = 0, y1 = 0, xref = "x", yref = "y",
                           line = list(color = "#999", width = 1))),
        annotations = list(list(
          text = paste0("Individual: ", pid),
          x = 0.02, y = 0.97, xref = "paper", yref = "paper",
          showarrow = FALSE, xanchor = "left",
          font = list(size = 10, color = "#555", family = "Lora,serif")))
      )
  })

  # ── Fig 45: Weekly sleep debt tracker ────────────────────────────────────────
  output$indiv_sleep_debt <- renderPlotly({
    pid <- input$pid_indiv
    d   <- garmin %>%
      filter(User_id == pid, !is.na(TSD), TSD >= 3) %>%
      mutate(
        iso_week  = paste0(isoyear(calendar_date), "-W",
                           sprintf("%02d", isoweek(calendar_date))),
        wk_start  = floor_date(calendar_date, "week", week_start = 1),
        deficit   = pmax(0, 7 - TSD)          # hours below 7h per night
      ) %>%
      group_by(wk_start) %>%
      summarise(
        sleep_debt  = -sum(deficit, na.rm = TRUE),   # negative = debt
        mean_rec    = mean(RecoveryScore, na.rm = TRUE),
        n           = n(),
        .groups = "drop"
      ) %>%
      filter(n >= 3) %>%
      arrange(wk_start) %>%
      mutate(rec_roll4 = rollapply(mean_rec, 4, mean, na.rm=TRUE,
                                   fill=NA, align="right"))
    if (nrow(d) < 4) return(plotly_empty() %>%
      add_annotations(text = "Insufficient data.", x=0.5, y=0.5,
        xref="paper", yref="paper", showarrow=FALSE) %>%
      layout(paper_bgcolor="#FAFAF8") %>% config(displayModeBar=FALSE))
    bar_col <- ifelse(d$sleep_debt >= 0, COL_P, COL_D)
    plot_ly() %>%
      add_bars(data = d, x = ~wk_start, y = ~sleep_debt,
               name = "Weekly sleep debt (h)",
               marker = list(color = bar_col, opacity = 0.75,
                             line = list(color = "#FAFAF8", width = 0.4)),
               hovertemplate = "<b>%{x|%d %b %Y}</b><br>Sleep debt: %{y:.1f}h<extra></extra>") %>%
      add_lines(data = d, x = ~wk_start, y = ~rec_roll4,
                name = "4-week rolling recovery",
                yaxis = "y2",
                line = list(color = COL_S, width = 2.2, dash = "dot"),
                hovertemplate = "<b>%{x|%d %b %Y}</b><br>Recovery (4-wk avg): %{y:.2f}<extra></extra>") %>%
      layout(
        paper_bgcolor = "#FAFAF8", plot_bgcolor = "#FAFAF8",
        font  = list(family = "Lora,Georgia,serif", size = 11),
        xaxis = list(title = "", gridcolor = "#E8E4DE", linecolor = "#CCCCCC"),
        yaxis = list(title = "Weekly sleep debt (h)",
                     gridcolor = "#E8E4DE", linecolor = "#CCCCCC",
                     zeroline = TRUE, zerolinecolor = "#888", zerolinewidth = 1.2),
        yaxis2 = list(title = "Recovery score (4-wk avg)",
                      overlaying = "y", side = "right",
                      showgrid = FALSE, linecolor = "#CCCCCC"),
        legend = list(orientation = "h", y = 1.08, font = list(size = 10)),
        shapes = list(list(type = "line", x0 = 0, x1 = 1, xref = "paper",
                           y0 = 0, y1 = 0, yref = "y",
                           line = list(dash = "dash", color = "#888", width = 1))),
        margin = list(l = 60, r = 70, t = 20, b = 50)
      ) %>% config(displayModeBar = FALSE)
  })

  # ── Fig 46: Individual effort vs outcome ─────────────────────────────────────
  output$indiv_effort_outcome <- renderPlotly({
    pid <- input$pid_indiv
    d   <- garmin %>%
      filter(User_id == pid) %>%
      arrange(calendar_date) %>%
      mutate(Rec_next = lead(RecoveryScore, 1)) %>%
      filter(!is.na(SessionLoad), SessionLoad > 0, !is.na(Rec_next))
    if (nrow(d) < 10) return(plotly_empty() %>%
      add_annotations(text = "Insufficient training data.", x=0.5, y=0.5,
        xref="paper", yref="paper", showarrow=FALSE) %>%
      layout(paper_bgcolor="#FAFAF8") %>% config(displayModeBar=FALSE))
    sp <- spear(d$SessionLoad, d$Rec_next)
    p <- ggplot(d, aes(x = SessionLoad, y = Rec_next,
                       colour = SL_Intensity,
                       text = paste0("Date: ", calendar_date,
                                     "<br>Load: ", round(SessionLoad),
                                     "<br>Recovery next day: ", round(Rec_next, 2),
                                     "<br>Type: ", SL_Intensity))) +
      geom_point(alpha = 0.45, size = 2) +
      geom_smooth(aes(group = 1), method = "lm", formula = y ~ x,
                  colour = "#1A1A1A", linewidth = 1.1, se = TRUE,
                  fill = "#CCCCCC", alpha = 0.22) +
      geom_hline(yintercept = 0, linetype = "dashed", colour = "#AAAAAA") +
      scale_colour_manual(
        values = c("Rest Day"=COL_N,"Light"=COL_G,"Moderate"=COL_W,"High"=COL_D)) +
      labs(x = "Session Load", y = "Next-day Recovery Score",
           colour = "Intensity",
           subtitle = paste0("Spearman ρ = ", sp$r, "  ·  p ", fmt_p(sp$p),
                             "  ·  n = ", nrow(d))) +
      theme_minimal(base_family = "serif") +
      theme(panel.grid.major = element_line(colour = "#E8E4DE"),
            plot.background  = element_rect(fill = "#FAFAF8", colour = NA),
            panel.background = element_rect(fill = "#FAFAF8", colour = NA),
            legend.position  = "none")
    ggplotly(p, tooltip = "text") %>%
      layout(paper_bgcolor = "#FAFAF8", plot_bgcolor = "#FAFAF8",
             font = list(family = "Lora,Georgia,serif", size = 11),
             legend = list(orientation = "h", y = 1.10, x = 0,
                           xanchor = "left", font = list(size = 10)),
             margin = list(l = 55, r = 20, t = 50, b = 55)) %>%
      config(displayModeBar = FALSE)
  })

  # ── Fig 47: Monthly recovery bar chart ───────────────────────────────────────
  output$indiv_monthly_recovery <- renderPlotly({
    pid <- input$pid_indiv
    d   <- garmin %>%
      filter(User_id == pid, !is.na(RecoveryScore)) %>%
      mutate(month_lbl = format(calendar_date, "%Y-%m")) %>%
      group_by(month_lbl) %>%
      summarise(mean_rec = mean(RecoveryScore, na.rm = TRUE),
                n        = n(), .groups = "drop") %>%
      filter(n >= 5) %>%
      arrange(month_lbl)
    if (nrow(d) < 2) return(plotly_empty() %>%
      add_annotations(text = "Insufficient data.", x=0.5, y=0.5,
        xref="paper", yref="paper", showarrow=FALSE) %>%
      layout(paper_bgcolor="#FAFAF8") %>% config(displayModeBar=FALSE))
    bar_col <- ifelse(d$mean_rec >= 0, COL_G, COL_D)
    plot_ly(d, x = ~month_lbl, y = ~mean_rec,
            type = "bar",
            marker = list(color = bar_col, opacity = 0.85,
                          line = list(color = "#FAFAF8", width = 0.5)),
            text  = ~paste0("n = ", n, " days"),
            hovertemplate = paste0("<b>%{x}</b><br>Mean recovery: %{y:.3f}",
                                   "<br>%{text}<extra></extra>")) %>%
      layout(
        paper_bgcolor = "#FAFAF8", plot_bgcolor = "#FAFAF8",
        font  = list(family = "Lora,Georgia,serif", size = 11),
        xaxis = list(title = "", tickangle = -35,
                     gridcolor = "#E8E4DE", linecolor = "#CCCCCC"),
        yaxis = list(title = "Mean Recovery Score",
                     gridcolor = "#E8E4DE", linecolor = "#CCCCCC",
                     zeroline = TRUE, zerolinecolor = "#888", zerolinewidth = 1.2),
        shapes = list(list(type = "line", x0 = 0, x1 = 1, xref = "paper",
                           y0 = 0, y1 = 0, yref = "y",
                           line = list(dash = "dash", color = "#888", width = 1))),
        margin = list(l = 60, r = 20, t = 15, b = 70)
      ) %>% config(displayModeBar = FALSE)
  })

  # Helper: BB net change clamped to ±20
  # NOTE: for inline use only where precomputed BB_delta column is unavailable.
  # Correct formula: charged(t+1) - drained(t). When called inline,
  # pass lead(charged,1) as the first argument.
  bb_delta_fn <- function(charged_next, drained) {
    pmin(pmax(charged_next - drained, -20), 20)
  }

  # Helper: binned mean of precomputed BB_delta for scatter secondary axis
  bb_bin_line <- function(d, x_var, n_bins = 15) {
    rng  <- range(d[[x_var]], na.rm = TRUE)
    cuts <- seq(rng[1], rng[2], length.out = n_bins + 1)
    d$bin <- cut(d[[x_var]], breaks = cuts, include.lowest = TRUE)
    d %>%
      group_by(bin) %>%
      summarise(bin_mid = mean(.data[[x_var]], na.rm = TRUE),
                n_obs   = sum(!is.na(BB_delta)),
                mean_bb = {
                  vals <- BB_delta[!is.na(BB_delta)]
                  if (length(vals) < 5) NA_real_
                  else mean(vals[vals >= quantile(vals, 0.10) &
                                 vals <= quantile(vals, 0.90)], na.rm = TRUE)
                },
                .groups = "drop") %>%
      filter(!is.na(bin_mid), n_obs >= 5, !is.na(mean_bb))
  }

  # ── Relationships: Fig 27 — SL category vs next-night sleep (4 outcomes, merged) ──
  output$sl_vs_sleep_merged <- renderPlotly({
    sl_cols <- c("Rest Day" = "#DEDAD4", "Light" = COL_G,
                 "Moderate" = COL_W, "High" = COL_D)
    d <- d_global() %>%
      arrange(User_id, calendar_date) %>%
      group_by(User_id) %>%
      mutate(TSD_next = lead(TSD, 1),
             SE_next  = lead(SE, 1),
             SQ_next  = lead(SleepQuality_pct, 1),
             SF_next  = lead(AwakeSleepProp * 100, 1)) %>%
      ungroup() %>%
      filter(!is.na(SL_Intensity), !is.na(TSD_next), TSD_next >= 3)
    mk_box <- function(df, y_var, y_lbl, extra_filter = TRUE) {
      df2 <- df %>% filter({{ extra_filter }}, !is.na(.data[[y_var]]))
      if (y_var == "SE_next") df2 <- df2 %>% filter(SE_next < 1.0)
      if (y_var == "SQ_next") df2 <- df2 %>% filter(SQ_next <= 75, analysis_cohort == "full")
      ggplot(df2, aes(x = SL_Intensity, y = .data[[y_var]],
                      fill = SL_Intensity,
                      text = paste0(User_id, "<br>", SL_Intensity,
                                    "<br>", y_lbl, ": ", round(.data[[y_var]], 1)))) +
        geom_boxplot(width = 0.5, outlier.shape = NA, alpha = 0.82,
                     colour = "#333", linewidth = 0.4) +
        scale_fill_manual(values = sl_cols, guide = "none") +
        labs(x = NULL, y = y_lbl) +
        theme_minimal(base_family = "serif") +
        theme(legend.position = "none", axis.text.x = element_text(size = 7),
              axis.title.y = element_text(size = 10, face = "bold"),
              panel.grid.major = element_line(colour = "#E8E4DE"),
              plot.background  = element_rect(fill = "#FAFAF8", colour = NA),
              panel.background = element_rect(fill = "#FAFAF8", colour = NA))
    }
    p1 <- mk_box(d, "TSD_next", "Sleep Duration — next (h)")
    p2 <- mk_box(d, "SE_next",  "Sleep Efficiency — next (%)")
    p3 <- mk_box(d, "SQ_next",  "Sleep Quality — next (%)")
    p4 <- mk_box(d, "SF_next",  "Sleep Fragmentation — next (%)")
    subplot(ggplotly(p1, tooltip="text"),
            ggplotly(p2, tooltip="text"),
            ggplotly(p3, tooltip="text"),
            ggplotly(p4, tooltip="text"),
            nrows = 1, shareX = FALSE, margin = 0.06) %>%
      layout(paper_bgcolor = "#FAFAF8", plot_bgcolor = "#FAFAF8",
             font = list(family = "Lora,Georgia,serif", size = 10),
             showlegend = FALSE,
             yaxis  = list(title = list(text="Sleep Duration — next (h)",
                                        font=list(size=11,family="Lora,serif")),
                           side="left"),
             yaxis2 = list(title = list(text="Sleep Efficiency — next (%)",
                                        font=list(size=11,family="Lora,serif")),
                           side="left"),
             yaxis3 = list(title = list(text="Sleep Quality — next (%)",
                                        font=list(size=11,family="Lora,serif")),
                           side="left"),
             yaxis4 = list(title = list(text="Sleep Fragmentation — next (%)",
                                        font=list(size=11,family="Lora,serif")),
                           side="left"),
             margin = list(t=20, l=65)) %>%
      config(displayModeBar = FALSE)
  })

  # ── Relationships: Fig 28 — RI vs next-night sleep (3 outcomes, merged) ──────
  output$ri_vs_sleep_merged <- renderPlotly({
    d <- d_global() %>%
      arrange(User_id, calendar_date) %>%
      group_by(User_id) %>%
      mutate(SE_next = lead(SE, 1),
             SQ_next = lead(SleepQuality_pct, 1),
             SF_next = lead(AwakeSleepProp * 100, 1)) %>%
      ungroup() %>%
      filter(ActivityGroup == "Running",
             !is.na(RelativeIntensity), RelativeIntensity > 0, RelativeIntensity <= 1)
    d_samp <- slice_sample(d, n = min(3000L, nrow(d)))
    mk_ri <- function(df, y_var, y_lbl, extra = NULL) {
      df2 <- df %>% filter(!is.na(.data[[y_var]]))
      if (!is.null(extra)) df2 <- df2 %>% filter({{ extra }})
      if (y_var == "SE_next") df2 <- df2 %>% filter(SE_next < 1.0)
      if (y_var == "SQ_next") df2 <- df2 %>%
            filter(SQ_next <= 75, analysis_cohort == "full")
      sp <- spear(df2$RelativeIntensity, df2[[y_var]])
      ggplot(df2, aes(x = RelativeIntensity * 100, y = .data[[y_var]],
                      colour = User_id,
                      text = paste0(User_id, "<br>RI: ",
                                    round(RelativeIntensity*100), "%",
                                    "<br>", y_lbl, ": ",
                                    round(.data[[y_var]], 1)))) +
        geom_point(alpha = 0.20, size = 1.1) +
        geom_smooth(aes(group = 1), method = "loess", formula = y ~ x,
                    colour = "#1A1A1A", linewidth = 1.1, se = TRUE,
                    fill = "#CCCCCC", alpha = 0.25) +
        geom_vline(xintercept = 90, linetype = "dashed",
                   colour = COL_D, linewidth = 0.5) +
        scale_colour_manual(values = PART_COLORS, guide = "none") +
        scale_x_continuous(labels = function(x) paste0(x, "%")) +
        labs(x = NULL, y = y_lbl,
             subtitle = paste0("ρ=", sp$r, " p", fmt_p(sp$p))) +
        theme_minimal(base_family = "serif") +
        theme(legend.position = "none",
              axis.title.y = element_text(size = 10, face = "bold"),
              plot.subtitle = element_text(size = 7.5, colour = "#555"),
              panel.grid.major = element_line(colour = "#E8E4DE"),
              plot.background  = element_rect(fill = "#FAFAF8", colour = NA),
              panel.background = element_rect(fill = "#FAFAF8", colour = NA))
    }
    p1 <- mk_ri(d_samp, "SE_next", "Sleep Efficiency — next (%)")
    p2 <- mk_ri(d_samp, "SQ_next", "Sleep Quality — next (%)")
    subplot(ggplotly(p1, tooltip="text"),
            ggplotly(p2, tooltip="text"),
            nrows = 1, shareX = TRUE, margin = 0.10) %>%
      layout(paper_bgcolor = "#FAFAF8", plot_bgcolor = "#FAFAF8",
             font = list(family = "Lora,Georgia,serif", size = 10),
             showlegend = FALSE,
             xaxis  = list(title = "", tickfont = list(size=9)),
             xaxis2 = list(title = "", tickfont = list(size=9)),
             yaxis  = list(title = list(text="Sleep Efficiency — next (%)",
                                        font=list(size=11,family="Lora,serif")),
                           side="left"),
             yaxis2 = list(title = list(text="Sleep Quality — next (%)",
                                        font=list(size=11,family="Lora,serif")),
                           side="left"),
             margin = list(t=20, b=45, l=65)) %>%
      config(displayModeBar = FALSE)
  })

  # ── Relationships: Fig 29 — Distance vs next-night SQ + SF (merged) ──────────
  output$dist_vs_sleep_merged <- renderPlotly({
    cat_cols <- c("Light" = COL_G, "Moderate" = COL_W, "High" = COL_D)
    d <- d_global() %>%
      arrange(User_id, calendar_date) %>%
      group_by(User_id) %>%
      mutate(SQ_next = lead(SleepQuality_pct, 1),
             SF_next = lead(AwakeSleepProp * 100, 1)) %>%
      ungroup() %>%
      filter(ActivityGroup == "Running",
             !is.na(TV), TV > 0, TV <= 60,
             !is.na(SL_Intensity), SL_Intensity != "Rest Day") %>%
      mutate(SL_Intensity = factor(SL_Intensity,
                                   levels = c("Light","Moderate","High")))
    d_sq <- d %>% filter(!is.na(SQ_next), SQ_next <= 75,
                          analysis_cohort == "full")
    sp_sq <- spear(d_sq$TV, d_sq$SQ_next)
    d_sf <- d %>% filter(!is.na(SF_next))
    sp_sf <- spear(d_sf$TV, d_sf$SF_next)
    p1 <- ggplot(slice_sample(d_sq, n = min(2000L, nrow(d_sq))),
                 aes(x = TV, y = SQ_next, colour = SL_Intensity,
                     text = paste0(User_id, "<br>", round(TV,1), " km",
                                   "<br>Sleep Quality next: ", round(SQ_next,1), "%",
                                   "<br>", SL_Intensity))) +
      geom_point(alpha = 0.25, size = 1.2) +
      geom_smooth(aes(group = 1), method = "loess", formula = y ~ x,
                  colour = "#1A1A1A", linewidth = 1.1, se = TRUE,
                  fill = "#CCCCCC", alpha = 0.25) +
      scale_colour_manual(values = cat_cols, name = "Load") +
      scale_x_continuous(breaks = seq(0,60,10)) +
      scale_y_continuous(limits = c(NA, 75)) +
      labs(x = NULL, y = "Sleep Quality — next (%)",
           subtitle = paste0("ρ=", sp_sq$r, " p", fmt_p(sp_sq$p))) +
      theme_minimal(base_family = "serif") +
      theme(legend.position = "none",
            axis.title.y = element_text(size = 10, face = "bold"),
            plot.subtitle = element_text(size = 7.5, colour = "#555"),
            panel.grid.major = element_line(colour = "#E8E4DE"),
            plot.background  = element_rect(fill = "#FAFAF8", colour = NA),
            panel.background = element_rect(fill = "#FAFAF8", colour = NA))
    p2 <- ggplot(slice_sample(d_sf, n = min(2000L, nrow(d_sf))),
                 aes(x = TV, y = SF_next, colour = SL_Intensity,
                     text = paste0(User_id, "<br>", round(TV,1), " km",
                                   "<br>Sleep Fragmentation next: ", round(SF_next,1), "%",
                                   "<br>", SL_Intensity))) +
      geom_point(alpha = 0.25, size = 1.2) +
      geom_smooth(aes(group = 1), method = "loess", formula = y ~ x,
                  colour = "#1A1A1A", linewidth = 1.1, se = TRUE,
                  fill = "#CCCCCC", alpha = 0.25) +
      scale_colour_manual(values = cat_cols, name = "Load") +
      scale_x_continuous(breaks = seq(0,60,10)) +
      labs(x = "Distance (km)", y = "Sleep Fragmentation — next (%)",
           subtitle = paste0("ρ=", sp_sf$r, " p", fmt_p(sp_sf$p))) +
      theme_minimal(base_family = "serif") +
      theme(legend.position = "right", axis.title = element_text(size = 8),
            legend.text = element_text(size = 8),
            legend.title = element_text(size = 8),
            plot.subtitle = element_text(size = 7.5, colour = "#555"),
            panel.grid.major = element_line(colour = "#E8E4DE"),
            plot.background  = element_rect(fill = "#FAFAF8", colour = NA),
            panel.background = element_rect(fill = "#FAFAF8", colour = NA))
    subplot(ggplotly(p1, tooltip="text") %>% style(showlegend=FALSE),
            ggplotly(p2, tooltip="text"),
            nrows = 1, shareX = TRUE, margin = 0.10) %>%
      layout(paper_bgcolor = "#FAFAF8", plot_bgcolor = "#FAFAF8",
             font = list(family = "Lora,Georgia,serif", size = 10),
             xaxis  = list(title = "Distance (km)", tickfont=list(size=9)),
             xaxis2 = list(title = "Distance (km)", tickfont=list(size=9)),
             yaxis  = list(title = list(text="Sleep Quality — next (%)",
                                        font=list(size=11,family="Lora,serif")),
                           side="left"),
             yaxis2 = list(title = list(text="Sleep Fragmentation — next (%)",
                                        font=list(size=11,family="Lora,serif")),
                           side="left"),
             legend = list(title = list(text="Load"), x=1.01, y=0.5,
                           font=list(size=9)),
             margin = list(t=20, b=45, l=65)) %>%
      config(displayModeBar = FALSE)
  })
  # ── Fig 35: Daily steps vs next-night sleep ──────────────────────────────
  output$steps_vs_sleep <- renderPlotly({
    d <- d_global() %>%
      arrange(User_id, calendar_date) %>%
      group_by(User_id) %>%
      mutate(TSD_next = lead(TSD, 1),
             SF_next  = lead(AwakeSleepProp * 100, 1)) %>%
      ungroup() %>%
      filter(!is.na(DailyTotalSteps), DailyTotalSteps > 0,
             DailyTotalSteps <= 60000)
    d_samp <- slice_sample(d, n = min(3000L, nrow(d)))
    sp1 <- spear(d_samp$DailyTotalSteps, d_samp$TSD_next)
    sp2 <- spear(d_samp$DailyTotalSteps, d_samp$SF_next)
    p1 <- ggplot(d_samp %>% filter(!is.na(TSD_next), TSD_next >= 3),
                 aes(x = DailyTotalSteps / 1000, y = TSD_next, colour = User_id,
                     text = paste0(User_id, "<br>Steps: ",
                                   format(round(DailyTotalSteps), big.mark=","),
                                   "<br>Sleep Duration next: ", round(TSD_next,1), "h"))) +
      geom_point(alpha = 0.18, size = 1.1) +
      geom_smooth(aes(group = 1), method = "loess", formula = y ~ x,
                  colour = "#1A1A1A", linewidth = 1.1, se = TRUE,
                  fill = "#CCCCCC", alpha = 0.25) +
      scale_colour_manual(values = PART_COLORS, guide = "none") +
      scale_x_continuous(breaks = seq(0, 60, 10)) +
      labs(x = "Daily steps (thousands)",
           y = "Sleep Duration — next night (h)",
           subtitle = paste0("ρ = ", sp1$r, "  ·  p ", fmt_p(sp1$p))) +
      theme_minimal(base_family = "serif") +
      theme(panel.grid.major = element_line(colour = "#E8E4DE"),
            plot.background  = element_rect(fill = "#FAFAF8", colour = NA),
            panel.background = element_rect(fill = "#FAFAF8", colour = NA))
    p2 <- ggplot(d_samp %>% filter(!is.na(SF_next)),
                 aes(x = DailyTotalSteps / 1000, y = SF_next, colour = User_id,
                     text = paste0(User_id, "<br>Steps: ",
                                   format(round(DailyTotalSteps), big.mark=","),
                                   "<br>Awake % next: ", round(SF_next,1), "%"))) +
      geom_point(alpha = 0.18, size = 1.1) +
      geom_smooth(aes(group = 1), method = "loess", formula = y ~ x,
                  colour = "#1A1A1A", linewidth = 1.1, se = TRUE,
                  fill = "#CCCCCC", alpha = 0.25) +
      scale_colour_manual(values = PART_COLORS, guide = "none") +
      scale_x_continuous(breaks = seq(0, 60, 10)) +
      labs(x = "Daily steps (thousands)",
           y = "Sleep Fragmentation — next night (%)",
           subtitle = paste0("ρ = ", sp2$r, "  ·  p ", fmt_p(sp2$p))) +
      theme_minimal(base_family = "serif") +
      theme(panel.grid.major = element_line(colour = "#E8E4DE"),
            plot.background  = element_rect(fill = "#FAFAF8", colour = NA),
            panel.background = element_rect(fill = "#FAFAF8", colour = NA))
    subplot(ggplotly(p1, tooltip = "text"),
            ggplotly(p2, tooltip = "text"),
            nrows = 1, margin = 0.10) %>%
      layout(paper_bgcolor = "#FAFAF8", plot_bgcolor = "#FAFAF8",
             font = list(family = "Lora,Georgia,serif", size = 11),
             showlegend = FALSE,
             yaxis  = list(title = list(text="Sleep Duration — next (h)",
                                        font=list(size=11,family="Lora,serif")),
                           side="left"),
             yaxis2 = list(title = list(text="Sleep Fragmentation — next (%)",
                                        font=list(size=11,family="Lora,serif")),
                           side="left"),
             margin = list(t=20, l=65)) %>%
      config(displayModeBar = FALSE)
  })

  # ── Fig 37: Running distance vs next-night SF coloured by SQ ─────────────

  # ── Fig 38: Running economy vs prior-night sleep ──────────────────────────
  output$re_vs_sleep <- renderPlotly({
    d <- d_global() %>%
      arrange(User_id, calendar_date) %>%
      group_by(User_id) %>%
      mutate(SE_prev = lag(SE, 1),
             SQ_prev = lag(SleepQuality_pct, 1)) %>%
      ungroup() %>%
      filter(ActivityGroup == "Running",
             !is.na(ActivitiesAvgSpeed_kmh), !is.na(ActivitiesAvgHr),
             ActivitiesAvgHr > 0) %>%
      mutate(RE = ActivitiesAvgSpeed_kmh / ActivitiesAvgHr)
    d_samp <- slice_sample(d, n = min(3000L, nrow(d)))
    d_se <- d_samp %>% filter(!is.na(SE_prev), SE_prev > 0, SE_prev < 1)
    sp_se <- spear(d_se$SE_prev, d_se$RE)
    d_sq <- d_samp %>% filter(!is.na(SQ_prev), SQ_prev <= 75,
                               analysis_cohort == "full")
    sp_sq <- spear(d_sq$SQ_prev, d_sq$RE)
    p1 <- ggplot(d_se,
                 aes(x = SE_prev * 100, y = RE, colour = User_id,
                     text = paste0(User_id, "<br>SE prev night: ",
                                   round(SE_prev*100), "%",
                                   "<br>Economy: ", round(RE,3)))) +
      geom_point(alpha = 0.20, size = 1.2) +
      geom_smooth(aes(group = 1), method = "loess", formula = y ~ x,
                  colour = "#1A1A1A", linewidth = 1.1, se = TRUE,
                  fill = "#CCCCCC", alpha = 0.25) +
      scale_colour_manual(values = PART_COLORS, guide = "none") +
      labs(x = "Sleep Efficiency — prior night (%)",
           y = "Running economy (km·h⁻¹·bpm⁻¹)",
           subtitle = paste0("ρ = ", sp_se$r, "  ·  p ", fmt_p(sp_se$p))) +
      theme_minimal(base_family = "serif") +
      theme(panel.grid.major = element_line(colour = "#E8E4DE"),
            plot.background  = element_rect(fill = "#FAFAF8", colour = NA),
            panel.background = element_rect(fill = "#FAFAF8", colour = NA))
    p2 <- ggplot(d_sq,
                 aes(x = SQ_prev, y = RE, colour = User_id,
                     text = paste0(User_id, "<br>SQ prev night: ",
                                   round(SQ_prev,1), "%",
                                   "<br>Economy: ", round(RE,3)))) +
      geom_point(alpha = 0.20, size = 1.2) +
      geom_smooth(aes(group = 1), method = "loess", formula = y ~ x,
                  colour = "#1A1A1A", linewidth = 1.1, se = TRUE,
                  fill = "#CCCCCC", alpha = 0.25) +
      scale_colour_manual(values = PART_COLORS, guide = "none") +
      labs(x = "Sleep Quality — prior night (%)",
           y = "Running economy (km·h⁻¹·bpm⁻¹)",
           subtitle = paste0("ρ = ", sp_sq$r, "  ·  p ", fmt_p(sp_sq$p),
                             "  ·  full cohort")) +
      theme_minimal(base_family = "serif") +
      theme(panel.grid.major = element_line(colour = "#E8E4DE"),
            plot.background  = element_rect(fill = "#FAFAF8", colour = NA),
            panel.background = element_rect(fill = "#FAFAF8", colour = NA))
    subplot(ggplotly(p1, tooltip = "text"),
            ggplotly(p2, tooltip = "text"),
            nrows = 1, margin = 0.07) %>%
      layout(paper_bgcolor = "#FAFAF8", plot_bgcolor = "#FAFAF8",
             font = list(family = "Lora,Georgia,serif", size = 11),
             showlegend = FALSE,
             xaxis  = list(title = list(text="Sleep Efficiency (%)",
                                        font=list(size=10)), tickfont=list(size=9)),
             xaxis2 = list(title = list(text="Sleep Quality (%)",
                                        font=list(size=10)), tickfont=list(size=9)),
             margin = list(b = 55)) %>%
      config(displayModeBar = FALSE)
  })

  # ── Fig 39: SL categories vs next-day recovery + BB ──────────────────────
  output$sl_vs_rec <- renderPlotly({
    sl_cols <- c("Rest Day" = "#DEDAD4", "Light" = COL_G,
                 "Moderate" = COL_W,    "High"    = COL_D)
    d <- d_global() %>%
      arrange(User_id, calendar_date) %>%
      group_by(User_id) %>%
      mutate(Rec_next = lead(RecoveryScore, 1)) %>%
      ungroup() %>%
      filter(!is.na(SL_Intensity), !is.na(Rec_next))
    # BB_delta(t) = charged(t+1) - drained(t): aligns with Rec_next = RecoveryScore(t+1)
    # Both reflect the same overnight window (night t -> morning t+1)
    bb_means <- d %>%
      filter(!is.na(BB_delta)) %>%
      group_by(SL_Intensity) %>%
      summarise(mean_bb = mean(BB_delta, na.rm = TRUE), .groups = "drop")
    p <- ggplot(d, aes(x = SL_Intensity, y = Rec_next, fill = SL_Intensity,
                       text = paste0(User_id, "<br>Intensity: ", SL_Intensity,
                                     "<br>Recovery next day: ", round(Rec_next,2)))) +
      geom_boxplot(width = 0.55, outlier.shape = NA, alpha = 0.8,
                   colour = "#1A1A1A", linewidth = 0.4) +
      scale_fill_manual(values = sl_cols, guide = "none") +
      geom_hline(yintercept = 0, linetype = "dashed",
                 colour = "#AAAAAA", linewidth = 0.7) +
      labs(x = NULL, y = "Recovery Score — next day") +
      theme_minimal(base_family = "serif") +
      theme(panel.grid.major = element_line(colour = "#E8E4DE"),
            plot.background  = element_rect(fill = "#FAFAF8", colour = NA),
            panel.background = element_rect(fill = "#FAFAF8", colour = NA))
    gg <- ggplotly(p, tooltip = "text")
    # Add BB mean points as a second trace
    plot_ly() %>%
      add_trace(data = d,
                x = ~SL_Intensity, y = ~Rec_next,
                type = "box",
                color = ~SL_Intensity,
                colors = sl_cols,
                boxmean = FALSE,
                line = list(color = "#1A1A1A", width = 0.8),
                marker = list(opacity = 0),
                showlegend = FALSE,
                text = ~paste0(User_id, "<br>Recovery: ", round(Rec_next,2)),
                hoverinfo = "text") %>%
      add_markers(data = bb_means,
                  x = ~SL_Intensity, y = ~mean_bb,
                  yaxis = "y2",
                  name = "Mean Net BB change",
                  marker = list(symbol = "diamond", size = 11,
                                color = COL_W,
                                line = list(color = "#333", width = 1.5)),
                  text = ~paste0(SL_Intensity, "<br>Mean Net BB: ",
                                 round(mean_bb,1)),
                  hoverinfo = "text") %>%
      layout(
        paper_bgcolor = "#FAFAF8", plot_bgcolor = "#FAFAF8",
        font  = list(family = "Lora,Georgia,serif", size = 11),
        xaxis = list(title = "", gridcolor = "#E8E4DE", linecolor = "#CCCCCC"),
        yaxis = list(title = "Recovery score (next day)",
                     range = c(-2, 2),
                     gridcolor = "#E8E4DE", linecolor = "#CCCCCC",
                     zeroline = TRUE, zerolinecolor = "#888"),
        yaxis2 = list(title = "Net BB change",
                      overlaying = "y", side = "right",
                      range = c(-20, 20), showgrid = FALSE),
        legend = list(orientation = "h", y = 1.06, font = list(size = 10)),
        margin = list(l = 60, r = 70, t = 20, b = 50)
      ) %>% config(displayModeBar = FALSE)
  })

  # ── Fig 40: Daily steps vs next-day recovery ─────────────────────────────
  output$steps_vs_rec <- renderPlotly({
    d <- d_global() %>%
      arrange(User_id, calendar_date) %>%
      group_by(User_id) %>%
      mutate(Rec_next = lead(RecoveryScore, 1)) %>%
      ungroup() %>%
      filter(!is.na(DailyTotalSteps), DailyTotalSteps > 0,
             DailyTotalSteps <= 60000, !is.na(Rec_next))
    d_samp <- slice_sample(d, n = min(3000L, nrow(d)))
    bb_d   <- bb_bin_line(d_samp, "DailyTotalSteps")  # BB_delta(t) aligns with Rec_next(t+1)
    sp <- spear(d_samp$DailyTotalSteps, d_samp$Rec_next)
    p <- ggplot(d_samp,
                aes(x = DailyTotalSteps / 1000, y = Rec_next,
                    colour = User_id,
                    text = paste0(User_id, "<br>Steps: ",
                                  format(round(DailyTotalSteps), big.mark=","),
                                  "<br>Recovery next day: ",
                                  round(Rec_next,2)))) +
      geom_point(alpha = 0.18, size = 1.1) +
      geom_smooth(aes(group = 1), method = "loess", formula = y ~ x,
                  colour = "#1A1A1A", linewidth = 1.2, se = TRUE,
                  fill = "#CCCCCC", alpha = 0.25) +
      geom_hline(yintercept = 0, linetype = "dashed",
                 colour = "#AAAAAA", linewidth = 0.6) +
      scale_colour_manual(values = PART_COLORS, guide = "none") +
      scale_x_continuous(breaks = seq(0, 60, 10)) +
      labs(x = "Daily steps (thousands)",
           y = "Recovery Score — next day",
           subtitle = paste0("ρ = ", sp$r, "  ·  p ", fmt_p(sp$p))) +
      theme_minimal(base_family = "serif") +
      theme(panel.grid.major = element_line(colour = "#E8E4DE"),
            plot.background  = element_rect(fill = "#FAFAF8", colour = NA),
            panel.background = element_rect(fill = "#FAFAF8", colour = NA))
    ggplotly(p, tooltip = "text") %>%
      add_lines(data = bb_d, x = ~bin_mid / 1000, y = ~mean_bb,
                name = "Net BB change",
                yaxis = "y2",
                line = list(color = COL_W, width = 2.2, dash = "solid"),
                hovertemplate = "Steps: %{x:.0f}k<br>Net BB: %{y:.1f}<extra></extra>") %>%
      layout(paper_bgcolor = "#FAFAF8", plot_bgcolor = "#FAFAF8",
             font = list(family = "Lora,Georgia,serif", size = 11),
             yaxis  = list(range = c(-2, 2), zeroline = TRUE, zerolinecolor = "#888",
                           gridcolor = "#E8E4DE", linecolor = "#CCCCCC"),
             yaxis2 = list(title = "Net BB change",
                           overlaying = "y", side = "right",
                           range = c(-20, 20), showgrid = FALSE),
             legend = list(orientation = "h", y = 1.08, font = list(size = 10))) %>%
      config(displayModeBar = FALSE)
  })

  # ── Fig 41: RI vs next-day recovery ──────────────────────────────────────
  output$ri_vs_rec <- renderPlotly({
    d <- d_global() %>%
      arrange(User_id, calendar_date) %>%
      group_by(User_id) %>%
      mutate(Rec_next = lead(RecoveryScore, 1)) %>%
      ungroup() %>%
      filter(ActivityGroup == "Running",
             !is.na(RelativeIntensity), RelativeIntensity > 0, RelativeIntensity <= 1,
             !is.na(Rec_next))
    d_samp <- slice_sample(d, n = min(3000L, nrow(d)))
    bb_d   <- bb_bin_line(d_samp, "RelativeIntensity")
    sp <- spear(d_samp$RelativeIntensity, d_samp$Rec_next)
    p <- ggplot(d_samp,
                aes(x = RelativeIntensity * 100, y = Rec_next,
                    colour = User_id,
                    text = paste0(User_id, "<br>RI: ",
                                  round(RelativeIntensity*100), "%",
                                  "<br>Recovery next day: ", round(Rec_next,2)))) +
      geom_point(alpha = 0.22, size = 1.2) +
      geom_smooth(aes(group = 1), method = "loess", formula = y ~ x,
                  colour = "#1A1A1A", linewidth = 1.2, se = TRUE,
                  fill = "#CCCCCC", alpha = 0.25) +
      geom_vline(xintercept = 90, linetype = "dashed",
                 colour = COL_D, linewidth = 0.6) +
      geom_hline(yintercept = 0, linetype = "dashed",
                 colour = "#AAAAAA", linewidth = 0.6) +
      scale_colour_manual(values = PART_COLORS, guide = "none") +
      scale_x_continuous(labels = function(x) paste0(x, "%")) +
      labs(x = "Relative Intensity (%)",
           y = "Recovery Score — next day",
           subtitle = paste0("ρ = ", sp$r, "  ·  p ", fmt_p(sp$p),
                             "  ·  dashed = 90% RI threshold")) +
      theme_minimal(base_family = "serif") +
      theme(panel.grid.major = element_line(colour = "#E8E4DE"),
            plot.background  = element_rect(fill = "#FAFAF8", colour = NA),
            panel.background = element_rect(fill = "#FAFAF8", colour = NA))
    ggplotly(p, tooltip = "text") %>%
      add_lines(data = bb_d, x = ~bin_mid * 100, y = ~mean_bb,
                name = "Mean Net BB change",
                yaxis = "y2",
                line = list(color = COL_W, width = 2.2),
                hovertemplate = "RI: %{x:.0f}%<br>Net BB: %{y:.1f}<extra></extra>") %>%
      layout(paper_bgcolor = "#FAFAF8", plot_bgcolor = "#FAFAF8",
             font = list(family = "Lora,Georgia,serif", size = 11),
             yaxis  = list(range = c(-2, 2), zeroline = TRUE, zerolinecolor = "#888",
                           gridcolor = "#E8E4DE", linecolor = "#CCCCCC"),
             yaxis2 = list(title = "Net BB change",
                           overlaying = "y", side = "right",
                           range = c(-20, 20), showgrid = FALSE),
             legend = list(orientation = "h", y = -0.18, font = list(size = 10))) %>%
      config(displayModeBar = FALSE)
  })

  # ── Fig 42: Distance vs next-day recovery by SL ───────────────────────────
  output$dist_vs_rec <- renderPlotly({
    cat_cols <- c("Light" = COL_G, "Moderate" = COL_W, "High" = COL_D)
    d <- d_global() %>%
      arrange(User_id, calendar_date) %>%
      group_by(User_id) %>%
      mutate(Rec_next = lead(RecoveryScore, 1)) %>%
      ungroup() %>%
      filter(ActivityGroup == "Running",
             !is.na(TV), TV > 0, TV <= 60,
             !is.na(SL_Intensity), SL_Intensity != "Rest Day",
             !is.na(Rec_next)) %>%
      mutate(SL_Intensity = factor(SL_Intensity, levels = c("Light","Moderate","High")))
    d_samp <- slice_sample(d, n = min(3000L, nrow(d)))
    bb_d   <- bb_bin_line(d_samp, "TV")
    sp <- spear(d_samp$TV, d_samp$Rec_next)
    p <- ggplot(d_samp,
                aes(x = TV, y = Rec_next, colour = SL_Intensity,
                    text = paste0(User_id, "<br>Distance: ", round(TV,1), " km",
                                  "<br>Intensity: ", SL_Intensity,
                                  "<br>Recovery next day: ", round(Rec_next,2)))) +
      geom_point(alpha = 0.25, size = 1.3) +
      geom_smooth(aes(group = 1), method = "loess", formula = y ~ x,
                  colour = "#1A1A1A", linewidth = 1.2, se = TRUE,
                  fill = "#CCCCCC", alpha = 0.25) +
      geom_hline(yintercept = 0, linetype = "dashed",
                 colour = "#AAAAAA", linewidth = 0.6) +
      scale_colour_manual(values = cat_cols, name = "Session load") +
      scale_x_continuous(breaks = seq(0, 60, 10)) +
      labs(x = "Running distance (km)",
           y = "Recovery Score — next day",
           subtitle = paste0("ρ = ", sp$r, "  ·  p ", fmt_p(sp$p))) +
      theme_minimal(base_family = "serif") +
      theme(legend.position  = "right",
            panel.grid.major = element_line(colour = "#E8E4DE"),
            plot.background  = element_rect(fill = "#FAFAF8", colour = NA),
            panel.background = element_rect(fill = "#FAFAF8", colour = NA))
    ggplotly(p, tooltip = "text") %>%
      add_lines(data = bb_d, x = ~bin_mid, y = ~mean_bb,
                name = "Mean Net BB change",
                yaxis = "y2",
                line = list(color = COL_W, width = 2.2),
                hovertemplate = "Distance: %{x:.1f}km<br>Net BB: %{y:.1f}<extra></extra>") %>%
      layout(paper_bgcolor = "#FAFAF8", plot_bgcolor = "#FAFAF8",
             font = list(family = "Lora,Georgia,serif", size = 11),
             yaxis  = list(range = c(-2, 2), zeroline = TRUE, zerolinecolor = "#888",
                           gridcolor = "#E8E4DE", linecolor = "#CCCCCC"),
             yaxis2 = list(title = "Net BB change",
                           overlaying = "y", side = "right",
                           range = c(-20, 20), showgrid = FALSE),
             xaxis = list(title = list(text = "Running distance (km)",
                                        font = list(size = 11)),
                            gridcolor = "#E8E4DE", linecolor = "#CCCCCC"),
             legend = list(title = list(text = "Session load"),
                           orientation = "h", y = -0.22, font = list(size = 10)),
             margin = list(b = 80)) %>%
      config(displayModeBar = FALSE)
  })

  # ── Helper: scatter + BB secondary (Sleep → Recovery plots) ──────────────
  sleep_rec_scatter <- function(d_in, x_var, x_lbl,
                                 x_pct = FALSE, sq_only = FALSE,
                                 vline = NULL) {
    d <- d_in %>% filter(!is.na(.data[[x_var]]), !is.na(RecoveryScore))
    if (sq_only) d <- d %>% filter(analysis_cohort == "full",
                                    .data[[x_var]] <= 75)
    if (x_pct) d <- d %>% filter(.data[[x_var]] > 0, .data[[x_var]] < 1)
    d_samp <- slice_sample(d, n = min(3000L, nrow(d)))
    x_vals <- if (x_pct) d_samp[[x_var]] * 100 else d_samp[[x_var]]
    sp <- spear(x_vals, d_samp$RecoveryScore)
    # BB_delta(t) = charged(t+1) - drained(t): reflects same overnight window as
    # RecoveryScore(t+1). For same-morning Recovery (t), use BB_delta_lag1 which
    # = charged(t) - drained(t-1) — both reflect the same night-t.
    # (BB_delta_lag1 is precomputed in global mutate)
    d_samp_bb <- d_samp %>% rename(BB_delta_plot = BB_delta_lag1)
    d_samp_bb$BB_delta <- d_samp_bb$BB_delta_plot  # bb_bin_line uses BB_delta column
    bb_bins <- bb_bin_line(d_samp_bb, x_var)
    if (x_pct) {
      bb_bins$bin_mid <- bb_bins$bin_mid * 100
    }
    p_main <- plot_ly() %>%
      add_markers(data = d_samp,
                  x    = if (x_pct) ~get(x_var)*100 else ~get(x_var),
                  y    = ~RecoveryScore,
                  color = ~User_id,
                  colors = PART_COLORS,
                  marker = list(size = 4, opacity = 0.20),
                  text  = ~paste0(User_id, "<br>",
                                  x_lbl, ": ",
                                  if (x_pct) paste0(round(get(x_var)*100), "%")
                                  else round(get(x_var),2),
                                  "<br>Recovery: ", round(RecoveryScore,2)),
                  hoverinfo = "text",
                  showlegend = FALSE) %>%
      add_lines(data = data.frame(
                  x = if (x_pct)
                        d_samp[[x_var]][order(d_samp[[x_var]])] * 100
                      else sort(d_samp[[x_var]]),
                  y = fitted(loess(RecoveryScore ~ get(x_var),
                                   data = d_samp, span = 0.5,
                                   na.action = na.exclude))[order(d_samp[[x_var]])]),
                x = ~x, y = ~y,
                line = list(color = "#1A1A1A", width = 1.8),
                name = "LOESS", showlegend = FALSE,
                hoverinfo = "none") %>%
      add_lines(data = bb_bins, x = ~bin_mid, y = ~mean_bb,
                yaxis = "y2", name = "Net BB change (binned)",
                line = list(color = COL_W, width = 2.2),
                marker = list(size = 5, color = COL_W,
                              line = list(color = "#FAFAF8", width = 1)),
                hovertemplate = paste0(x_lbl, ": %{x:.1f}<br>",
                                       "Net BB: %{y:.1f}<extra></extra>"))
    if (!is.null(vline)) {
      vline_x <- if (x_pct) vline * 100 else vline
      p_main <- p_main %>%
        add_lines(x = c(vline_x, vline_x), y = c(-10, 10), yaxis = "y",
                  line = list(color = COL_D, dash = "dot", width = 1),
                  showlegend = FALSE, hoverinfo = "none")
    }
    p_main %>%
      layout(
        paper_bgcolor = "#FAFAF8", plot_bgcolor = "#FAFAF8",
        font  = list(family = "Lora,Georgia,serif", size = 11),
        xaxis = list(title = x_lbl, gridcolor = "#E8E4DE", linecolor = "#CCCCCC"),
        yaxis = list(title = "Recovery Score",
                     range = c(-1.5, 1.5), fixedrange = TRUE,
                     zeroline = TRUE, zerolinecolor = "#888", zerolinewidth = 1.5,
                     gridcolor = "#E8E4DE", linecolor = "#CCCCCC"),
        yaxis2 = list(title = "Net BB change",
                      overlaying = "y", side = "right",
                      range = c(-20, 20), fixedrange = TRUE,
                      zeroline = FALSE, showgrid = FALSE),
        legend = list(orientation = "h", y = -0.22, font = list(size = 10)),
        annotations = list(list(
          text = paste0("ρ = ", sp$r, "  ·  p ", fmt_p(sp$p)),
          x = 0.98, y = -0.14, xref = "paper", yref = "paper",
          showarrow = FALSE, xanchor = "right",
          font = list(size = 10, family = "Lora,serif", color = "#444")
        )),
        margin = list(l = 60, r = 70, t = 20, b = 80)
      ) %>% config(displayModeBar = FALSE)
  }

  output$tsd_vs_rec_scatter <- renderPlotly({
    sleep_rec_scatter(d_rec(), "TSD", "Sleep Duration (h)")
  })

  output$se_vs_rec_scatter <- renderPlotly({
    sleep_rec_scatter(d_rec(), "SE", "Sleep Efficiency (%)",
                      x_pct = TRUE, vline = 0.85)
  })

  output$sf_vs_rec_scatter <- renderPlotly({
    d <- d_rec() %>%
      mutate(SF_pct = AwakeSleepProp * 100)
    sleep_rec_scatter(d, "SF_pct", "Sleep Fragmentation (%)")
  })

  output$sq_vs_rec_scatter <- renderPlotly({
    sleep_rec_scatter(d_rec(), "SleepQuality_pct",
                      "Sleep Quality (%)", sq_only = TRUE)
  })


  # ── Correlation heatmap ───────────────────────────────────────────────────
  output$corr_heatmap <- renderPlotly({
    var_map <- c(
      "TSD"                                        = "Sleep Duration",
      "SE"                                         = "Sleep Efficiency",
      "SleepQuality_pct"                           = "Sleep Quality",
      "AwakeSleepProp"                             = "Sleep Fragmentation",
      "SessionLoad"                                = "Session Load",
      "RelativeIntensity"                          = "Rel. Intensity",
      "DailyTotalSteps"                            = "Daily Steps",
      "RecoveryScore"                              = "Recovery Score",
      "BB_delta"                                   = "BB Change",
      "DailyRespiration.avgWakingRespirationValue" = "Respiration"
    )
    d_c <- d_rec() %>%
      select(any_of(names(var_map))) %>%
      rename_with(~ var_map[.x], .cols = everything()) %>%
      drop_na()
    if (nrow(d_c) < 20) return(plotly_empty() %>%
      add_annotations(text = "Insufficient data for selected filters.",
                      x = 0.5, y = 0.5, xref = "paper", yref = "paper",
                      showarrow = FALSE, font = list(size = 12, color = "#6B7280")) %>%
      layout(paper_bgcolor = "#FAFAF8") %>% config(displayModeBar = FALSE))
    lbl     <- colnames(d_c)
    cor_mat <- cor(d_c, method = "spearman", use = "pairwise.complete.obs")
    cor_rnd <- round(cor_mat, 2)
    # Mask near-zero (<|0.05|) so heatmap reads cleanly
    cell_txt <- ifelse(abs(cor_rnd) < 0.05, "", as.character(cor_rnd))
    plot_ly(
      x = lbl, y = lbl, z = cor_mat,
      type      = "heatmap",
      zmin = -1, zmax = 1,
      colorscale = list(
        c(0,   "#B2182B"),   # vivid crimson  — strong negative
        c(0.2, "#EF8A62"),   # salmon
        c(0.5, "#FFFFFF"),   # pure white     — zero correlation
        c(0.8, "#67A9CF"),   # sky blue
        c(1,   "#1A5276")    # deep cobalt    — strong positive
      ),
      colorbar   = list(title = "ρ", len = 0.65,
                        tickfont = list(family = "Lora,serif", size = 10)),
      hovertemplate = "%{y} ↔ %{x}<br>ρ = %{z:.3f}<extra></extra>"
    ) %>%
      add_annotations(
        x          = rep(lbl, each = length(lbl)),
        y          = rep(lbl, times = length(lbl)),
        text       = as.vector(cell_txt),
        showarrow  = FALSE,
        font       = list(size = 9, color = "#1A1A1A", family = "Lora,serif")
      ) %>%
      layout(
        paper_bgcolor = "#FAFAF8", plot_bgcolor = "#FAFAF8",
        font  = list(family = "Lora,Georgia,serif", size = 11),
        xaxis = list(title = "", tickangle = -40, linecolor = "#CCCCCC"),
        yaxis = list(title = "", autorange = "reversed", linecolor = "#CCCCCC"),
        margin = list(l = 140, r = 20, t = 20, b = 130)
      ) %>%
      config(displayModeBar = FALSE)
  })

  # ── Mixed-effects model (lmerTest) ────────────────────────────────────────
  m_lme <- reactive({
    if (!requireNamespace("lme4",      quietly = TRUE)) return(NULL)
    if (!requireNamespace("lmerTest",  quietly = TRUE)) return(NULL)
    d <- d_rec() %>% drop_na(RecoveryScore, TSD_lag1, SE_lag1, SessionLoad_lag1)
    if (n_distinct(d$User_id) < 3) return(NULL)
    tryCatch(
      lmerTest::lmer(RecoveryScore ~ TSD_lag1 + SE_lag1 + SessionLoad_lag1 +
                       (1 | User_id), data = d, REML = TRUE),
      error = function(e) NULL
    )
  })

  output$lme_tbl <- renderDT({
    m <- m_lme()
    if (is.null(m)) {
      df <- data.frame(Note = "lme4 / lmerTest not installed. Run: install.packages(c('lme4','lmerTest'))")
      return(datatable(df, rownames = FALSE, class = "compact",
                       options = list(dom = "t")))
    }
    ct  <- as.data.frame(summary(m)$coefficients)
    ci  <- tryCatch(confint(m, method = "Wald"), error = function(e) NULL)
    tbl <- data.frame(
      Term      = rownames(ct),
      Estimate  = round(ct[,1], 4),
      `Std.Err` = round(ct[,2], 4),
      `CI 2.5%` = if (!is.null(ci)) round(ci[rownames(ct), 1], 4) else NA,
      `CI 97.5%`= if (!is.null(ci)) round(ci[rownames(ct), 2], 4) else NA,
      df        = round(ct[,3], 1),
      p         = sapply(ct[,5], fmt_p),
      check.names = FALSE
    )
    datatable(tbl, rownames = FALSE, class = "compact stripe",
              options = list(dom = "t", pageLength = 10)) %>%
      formatStyle("p",
        backgroundColor = styleInterval(c(0.05, 0.10),
                                        c("#DCF0DC","#FEF3D0","#FAEAEA")))
  })

  output$lme_interp <- renderUI({
    m <- m_lme()
    if (is.null(m)) return(an("Install lme4 and lmerTest to enable this model."))
    ct  <- summary(m)$coefficients
    term <- "TSD_lag1"
    if (!term %in% rownames(ct)) return(NULL)
    est <- round(ct[term, 1], 4)
    p   <- ct[term, 5]
    ci  <- tryCatch(confint(m, method = "Wald"), error = function(e) NULL)
    lo  <- if (!is.null(ci)) round(ci[term, 1], 4) else "—"
    hi  <- if (!is.null(ci)) round(ci[term, 2], 4) else "—"
    sig <- if (p < 0.05) "statistically significant (p < 0.05)" else "not significant (p ≥ 0.05)"
    an(tags$strong("Mixed-effects result: "),
       paste0("After accounting for each athlete's individual baseline recovery, ",
              "a 1-hour increase in last night's sleep is associated with a ",
              abs(est), if (est > 0) " unit improvement" else " unit decrease",
              " in Recovery Score (95% CI [", lo, ", ", hi, "], p ", fmt_p(p), "). ",
              "This is ", sig, ". ",
              "The random intercept captures between-participant differences in average recovery level, ",
              "so this estimate reflects the within-person effect only."))
  })

  # ── Training monotony index ───────────────────────────────────────────────
  output$monotony_tbl <- renderDT({
    d <- garmin %>%
      filter(!is.na(SessionLoad)) %>%
      mutate(iso_week = paste0(isoyear(calendar_date), "-W",
                               sprintf("%02d", isoweek(calendar_date)))) %>%
      group_by(User_id, iso_week) %>%
      summarise(
        wk_mean = mean(SessionLoad, na.rm = TRUE),
        wk_sd   = sd(SessionLoad,   na.rm = TRUE),
        wk_sum  = sum(SessionLoad,  na.rm = TRUE),
        n_days  = n(),
        .groups = "drop"
      ) %>%
      filter(n_days >= 3, wk_sd > 0) %>%   # need ≥3 obs and real variation
      mutate(monotony = wk_mean / wk_sd,
             strain   = wk_sum * monotony) %>%
      group_by(User_id) %>%
      summarise(
        Weeks          = n(),
        `Mean monotony`= round(mean(monotony, na.rm = TRUE), 2),
        `Max monotony` = round(max(monotony,  na.rm = TRUE), 2),
        `Weeks >2`     = sum(monotony > 2, na.rm = TRUE),
        `% weeks >2`   = round(100 * mean(monotony > 2, na.rm = TRUE), 1),
        `Mean strain`  = round(mean(strain,   na.rm = TRUE), 0),
        .groups = "drop"
      ) %>%
      arrange(desc(`Mean monotony`))
    datatable(d, rownames = FALSE, class = "compact stripe",
              options = list(dom = "t", pageLength = 14, scrollX = FALSE)) %>%
      formatStyle("Mean monotony",
        backgroundColor = styleInterval(c(1.5, 2.0),
                                        c("#DCF0DC","#FEF3D0","#FAEAEA"))) %>%
      formatStyle("% weeks >2",
        backgroundColor = styleInterval(c(10, 25),
                                        c("#DCF0DC","#FEF3D0","#FAEAEA")))
  })

  # ── Autocorrelation of Recovery Score ─────────────────────────────────────
  output$acf_plot <- renderPlotly({
    d_ts <- d_rec() %>%
      group_by(calendar_date) %>%
      summarise(rec = mean(RecoveryScore, na.rm = TRUE), .groups = "drop") %>%
      arrange(calendar_date)
    if (nrow(d_ts) < 30) return(plotly_empty() %>%
      add_annotations(text = "Insufficient data for ACF.",
                      x = 0.5, y = 0.5, xref = "paper", yref = "paper",
                      showarrow = FALSE, font = list(size = 12, color = "#6B7280")) %>%
      layout(paper_bgcolor = "#FAFAF8") %>% config(displayModeBar = FALSE))
    # Interpolate tiny gaps, compute ACF up to lag 21
    ts_vals <- d_ts$rec
    ts_vals[is.na(ts_vals)] <- zoo::na.approx(ts_vals, na.rm = FALSE)
    ts_vals[is.na(ts_vals)] <- 0
    acf_out  <- acf(ts_vals, lag.max = 21, plot = FALSE)
    lags     <- as.numeric(acf_out$lag)
    acf_vals <- as.numeric(acf_out$acf)
    ci_bound <- 1.96 / sqrt(length(ts_vals))
    bar_col  <- ifelse(abs(acf_vals) > ci_bound & lags > 0, COL_P, "#CCCCBB")
    plot_ly() %>%
      add_bars(x = lags, y = acf_vals,
               name = "ACF",
               marker = list(color = bar_col, opacity = 0.85,
                             line = list(color = "#FAFAF8", width = 0.4)),
               hovertemplate = "Lag %{x} days<br>ACF: %{y:.3f}<extra></extra>") %>%
      layout(
        paper_bgcolor = "#FAFAF8", plot_bgcolor = "#FAFAF8",
        font  = list(family = "Lora,Georgia,serif", size = 11),
        xaxis = list(title = "Lag (days)", gridcolor = "#E8E4DE",
                     linecolor = "#CCCCCC", dtick = 1),
        yaxis = list(title = "Autocorrelation",
                     gridcolor = "#E8E4DE", linecolor = "#CCCCCC",
                     zeroline = TRUE, zerolinecolor = "#888", zerolinewidth = 1.2),
        shapes = list(
          list(type = "line", x0 = -0.5, x1 = 21.5,
               y0 = ci_bound, y1 = ci_bound, xref = "x", yref = "y",
               line = list(dash = "dash", color = COL_P, width = 1.2)),
          list(type = "line", x0 = -0.5, x1 = 21.5,
               y0 = -ci_bound, y1 = -ci_bound, xref = "x", yref = "y",
               line = list(dash = "dash", color = COL_P, width = 1.2))
        ),
        annotations = list(list(
          text = "Bars coloured blue exceed the 95% CI bound (±1.96/√N)",
          x = 0.99, y = 0.02, xref = "paper", yref = "paper",
          showarrow = FALSE, xanchor = "right",
          font = list(size = 9, color = "#666", family = "Lora,serif")
        )),
        legend = list(orientation = "h", y = 1.06),
        margin = list(l = 60, r = 20, t = 20, b = 55)
      ) %>%
      config(displayModeBar = FALSE)
  })

} # end server

shinyApp(ui, server)
