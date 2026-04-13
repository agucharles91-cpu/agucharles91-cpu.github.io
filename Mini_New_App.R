# =============================================================================
# README
# =============================================================================
# PROJECT : Training, Sleep & Recovery — Nairobi Runners
# AUTHOR  : MSc Data Science Capstone Project
# DATA    : garmin_merged_14p.csv  (12,760 rows, 14 participants, 2022–2026)
# HOW TO RUN:
#   1. Place garmin_merged_14p.csv in the data/ subfolder
#   2. Install required packages (see below)
#   3. Run: shiny::runApp(".")
#
# PACKAGES NEEDED:
#   install.packages(c("shiny","bslib","tidyverse","plotly","DT",
#                      "corrplot","ggcorrplot","shinycssloaders",
#                      "zoo","scales","lubridate","bsicons"))
# =============================================================================

library(shiny)
library(bslib)
library(tidyverse)
library(plotly)
library(DT)
library(ggcorrplot)
library(shinycssloaders)
library(zoo)
library(scales)
library(lubridate)
library(bsicons)

# =============================================================================
# GLOBAL: Load & prepare data
# =============================================================================

df_raw <- read_csv("data/garmin_merged_14p.csv",
                   na = c("NA", "", "N/A"),
                   show_col_types = FALSE)

df <- df_raw %>%
  mutate(
    calendar_date = as.Date(calendar_date),
    is_weekend     = as.logical(is_weekend),
    activity_day   = as.logical(activity_day),
    weekday        = factor(weekday,
                            levels = c("Monday","Tuesday","Wednesday",
                                       "Thursday","Friday","Saturday","Sunday")),
    # Compute RHR_dev_7d: deviation from 7-day rolling mean per participant
    RHR_dev_7d = {
      tmp <- df_raw %>%
        group_by(User_id) %>%
        arrange(calendar_date) %>%
        mutate(rhr_roll7 = rollapply(DailyRestingHeartRate_clean,
                                     width = 7, FUN = mean,
                                     na.rm = TRUE, fill = NA, align = "right"),
               RHR_dev_7d = DailyRestingHeartRate_clean - rhr_roll7) %>%
        ungroup() %>%
        pull(RHR_dev_7d)
      tmp
    },
    # Activity grouping
    ActivityGroup = case_when(
      ActivitiesType %in% c("running","track_running","trail_running",
                            "treadmill_running") ~ "Running",
      ActivitiesType %in% c("cycling","indoor_cycling","e_bike_fitness") ~ "Cycling",
      ActivitiesType %in% c("strength_training","hiit") ~ "Strength/HIIT",
      ActivitiesType %in% c("lap_swimming","open_water_swimming",
                            "indoor_rowing") ~ "Swimming/Rowing",
      ActivitiesType %in% c("yoga","pilates","meditation","breathwork") ~ "Mind-Body",
      is.na(ActivitiesType) ~ "Rest Day",
      TRUE ~ "Other"
    ),
    ActivityGroup = factor(ActivityGroup,
                           levels = c("Running","Cycling","Strength/HIIT",
                                      "Swimming/Rowing","Mind-Body","Other","Rest Day")),
    # Build lagged variables per participant
    year_week = paste0(year(calendar_date), "-W",
                       sprintf("%02d", isoweek(calendar_date)))
  ) %>%
  group_by(User_id) %>%
  arrange(calendar_date) %>%
  mutate(
    SessionLoad_lag1  = lag(SessionLoad,  1),
    SessionLoad_lag2  = lag(SessionLoad,  2),
    SessionLoad_lag3  = lag(SessionLoad,  3),
    TSD_lag1          = lag(TSD,          1),
    TSD_lag2          = lag(TSD,          2),
    TSD_lag3          = lag(TSD,          3),
    SE_lag1           = lag(SE,           1),
    SE_lag2           = lag(SE,           2),
    SE_lag3           = lag(SE,           3),
    RHR_dev_7d_lag1   = lag(RHR_dev_7d,   1)
  ) %>%
  ungroup()

# Participant demographics lookup
demo <- df %>%
  group_by(User_id) %>%
  summarise(
    Age    = first(Age),
    Sex    = first(Sex),
    VO2max = round(mean(VO2max_imputed_final, na.rm = TRUE), 1),
    .groups = "drop"
  ) %>%
  mutate(VO2max = ifelse(is.nan(VO2max), NA_real_, VO2max))

PARTICIPANTS <- sort(unique(df$User_id))

# Fixed colour palette per participant
PART_COLORS <- setNames(
  colorRampPalette(c("#1B3A6B","#0D9488","#60A5FA","#F59E0B",
                     "#EF4444","#8B5CF6","#10B981","#F97316",
                     "#3B82F6","#EC4899","#14B8A6","#A78BFA",
                     "#FBBF24","#34D399"))(length(PARTICIPANTS)),
  PARTICIPANTS
)

# Activity type colours
ACT_COLORS <- c(
  "Running"        = "#1B3A6B",
  "Cycling"        = "#0D9488",
  "Strength/HIIT"  = "#EF4444",
  "Swimming/Rowing"= "#3B82F6",
  "Mind-Body"      = "#8B5CF6",
  "Other"          = "#F59E0B",
  "Rest Day"       = "#CBD5E1"
)

# Key analytic variables (for dropdowns / corr matrix)
KEY_VARS <- c(
  "SessionLoad"                          = "SessionLoad",
  "Relative Intensity"                   = "RelativeIntensity",
  "Total Sleep Duration (hrs)"           = "TSD",
  "Sleep Efficiency"                     = "SE",
  "Sleep Fragmentation"                  = "SF",
  "RHR Deviation 7d"                     = "RHR_dev_7d",
  "VO2max (imputed)"                     = "VO2max_imputed_final",
  "Body Battery Charged"                 = "DailyBodyBattery.chargedValue",
  "REM Sleep Proportion"                 = "REMSleepProp_final",
  "Daily Steps"                          = "DailyTotalSteps",
  "Session Duration (min)"               = "ActivitiesDurationMinutes",
  "Waking Respiration Rate"              = "DailyRespiration.avgWakingRespirationValue"
)

# =============================================================================
# DATA DICTIONARY (for Methods tab)
# =============================================================================

data_dict <- tibble(
  Column      = c("RHR_dev_7d","SessionLoad","RelativeIntensity","TSD","SE","SF",
                  "REMSleepProp_final","VO2max_imputed_final","DailyBodyBattery.chargedValue",
                  "DailyTotalSteps","ActivitiesDurationMinutes",
                  "DailyRespiration.avgWakingRespirationValue","analysis_cohort",
                  "activity_day","ActivityGroup"),
  Group       = c("Recovery","Training","Training","Sleep","Sleep","Sleep",
                  "Sleep","Fitness","Recovery","Activity","Training",
                  "Physiology","Cohort","Activity","Activity"),
  Missing_pct = c(4.3, 48.1, 48.1, 17.4, 17.5, 18.7,
                  38.2, 2.6, 17.0, 4.4, 48.1, 20.7, 0.0, 0.0, 0.0),
  Description = c(
    "★ PRIMARY RECOVERY METRIC. RHR deviation from participant's own 7-day rolling mean. Positive = under-recovered (elevated HR); Negative = well-recovered.",
    "★ PRIMARY TRAINING LOAD. Internal load = AvgHR × DurationMinutes. NA on rest days (structural).",
    "AvgHR / MaxHR during session. Proportion of HR reserve used. Comparable across participants.",
    "★ Total Sleep Duration in hours. Primary sleep outcome. Right-skewed — use median.",
    "★ Sleep Efficiency = TSD / Time-in-Bed. ⚠ Values of exactly 1.0 = device floor, not truly perfect sleep.",
    "Sleep Fragmentation = AwakeSeconds / TotalSleepSeconds. Higher = more disrupted sleep.",
    "★ REM sleep as proportion of TSD. Use this column (recalculated & validated) over REMSleepProp.",
    "★ VO2max ml/kg/min after LOCF imputation (gaps ≤28 days). Primary fitness metric.",
    "Body Battery recharged mainly during sleep. P008 = 0% coverage; P009/P010 = 18–24% missing.",
    "Total step count for the day.",
    "Session duration in minutes. NA on rest days.",
    "Average waking breathing rate (breaths/min). Elevated values may signal illness or overtraining.",
    "'full' = all sleep metrics valid (9,700 rows). 'no_sleep_stages' = P006/P013/P015 (3,060 rows).",
    "TRUE if a structured activity session was logged that day.",
    "Derived grouping: Running / Cycling / Strength-HIIT / Swimming-Rowing / Mind-Body / Other / Rest Day."
  )
)

# =============================================================================
# THEME
# =============================================================================

app_theme <- bs_theme(
  bootswatch  = "flatly",
  primary     = "#1B3A6B",
  success     = "#0D9488",
  base_font   = font_google("Inter"),
  heading_font= font_google("Inter"),
  `enable-rounded` = TRUE
)

# =============================================================================
# HELPERS
# =============================================================================

info_panel <- function(...) {
  div(style = "background:#EFF6FF; border-left:4px solid #2563EB;
               padding:14px 18px; border-radius:6px; margin-bottom:18px;
               font-size:0.88rem; color:#1E3A5F; line-height:1.6;", ...)
}

warn_panel <- function(...) {
  div(style = "background:#FFFBEB; border-left:4px solid #D97706;
               padding:12px 16px; border-radius:6px; margin-bottom:14px;
               font-size:0.85rem; color:#78350F; line-height:1.5;", ...)
}

section_card <- function(title, ...) {
  card(
    card_header(strong(title)),
    card_body(...),
    class = "mb-3",
    style = "border-radius:10px;"
  )
}

spearman_r <- function(x, y) {
  ct <- cor.test(x, y, method = "spearman", use = "complete.obs",
                 exact = FALSE)
  list(r = round(ct$estimate, 3), p = round(ct$p.value, 4))
}

fmt_p <- function(p) {
  if (is.na(p)) return("NA")
  if (p < 0.001) return("< 0.001")
  round(p, 3)
}

# =============================================================================
# UI
# =============================================================================

ui <- page_navbar(
  title = tags$span(
    tags$img(src = "https://img.icons8.com/fluency/32/heart-monitor.png",
             style = "height:24px; margin-right:8px; vertical-align:middle;"),
    "Training, Sleep & Recovery — Nairobi Runners"
  ),
  theme = app_theme,
  fillable = FALSE,

  # ── Global sidebar ─────────────────────────────────────────────────────────
  sidebar = sidebar(
    width = 260,
    bg = "#F8FAFC",
    tags$div(style = "font-weight:700; font-size:0.8rem; color:#64748B;
                      text-transform:uppercase; letter-spacing:0.05em;
                      margin-bottom:10px;", "Global Filters"),

    dateRangeInput("date_range", "Date range",
                   start = min(df$calendar_date, na.rm = TRUE),
                   end   = max(df$calendar_date, na.rm = TRUE),
                   min   = min(df$calendar_date, na.rm = TRUE),
                   max   = max(df$calendar_date, na.rm = TRUE)),

    selectInput("act_filter", "Activity type",
                choices  = c("All", levels(df$ActivityGroup)),
                selected = "All"),

    checkboxInput("act_only", "Activity days only", value = FALSE),

    hr(),

    actionButton("show_about", "ℹ About this dataset",
                 class = "btn-outline-primary btn-sm w-100"),

    hr(),

    tags$div(style = "font-size:0.75rem; color:#94A3B8; line-height:1.5;",
      "14 participants · Nairobi · 2022–2026",
      tags$br(), "Unit: one row = one participant-day",
      tags$br(), tags$br(),
      tags$b("Colour legend:"),
      uiOutput("part_legend_mini")
    )
  ),

  # ── ABOUT modal ────────────────────────────────────────────────────────────
  nav_panel(
    title = "📊 Overview",
    value = "tab_overview",

    # KPI row
    uiOutput("kpi_row"),

    layout_columns(
      col_widths = c(8, 4),

      section_card(
        "Recovery Calendar Heatmap",
        info_panel(
          "Each tile is one day. Colour shows RHR deviation from a 7-day personal baseline.",
          tags$b(" Green = well-recovered (RHR below baseline)."),
          " Red = under-recovered. Use the participant selector to explore individual patterns."
        ),
        selectInput("heatmap_pid", "Participant (or cohort mean)",
                    choices = c("Cohort mean", PARTICIPANTS), selected = "Cohort mean"),
        withSpinner(plotlyOutput("heatmap_plot", height = "340px"), color = "#1B3A6B"),
        textOutput("heatmap_caption")
      ),

      section_card(
        "Missingness Summary",
        info_panel("Variables with > 20% missing (red) require cautious interpretation.",
                   " Body Battery and sleep stage missingness is device/firmware-related, not random."),
        withSpinner(DTOutput("missing_tbl"), color = "#1B3A6B")
      )
    ),

    section_card(
      "Distribution of Key Variables",
      info_panel(
        "Density curves show the spread of sleep duration, sleep efficiency, and recovery deviation",
        " across all 14 participants. Each colour = one participant.",
        " Vertical dashed line = cohort mean."
      ),
      layout_columns(
        col_widths = c(4, 4, 4),
        withSpinner(plotlyOutput("dist_tsd",  height = "260px"), color = "#1B3A6B"),
        withSpinner(plotlyOutput("dist_se",   height = "260px"), color = "#1B3A6B"),
        withSpinner(plotlyOutput("dist_rhr",  height = "260px"), color = "#1B3A6B")
      )
    )
  ),

  # ── TAB 2: Population ──────────────────────────────────────────────────────
  nav_panel(
    title = "📈 Population Analysis",
    value = "tab_pop",

    navset_tab(
      # 2A Descriptive
      nav_panel("2A · Descriptive Stats",
        br(),
        info_panel(
          tags$b("What this section shows: "),
          "Summary statistics for key training, sleep, and recovery variables,",
          " broken down by whether a training session occurred that day.",
          " Because TSD and SE are right-skewed, focus on median values rather than means."
        ),
        warn_panel("⚠ TSD and SE are right-skewed distributions — interpret medians, not means."),
        section_card("Summary Table — Activity vs Rest Days",
          withSpinner(DTOutput("desc_tbl"), color = "#1B3A6B")
        ),
        section_card("Distributions by Activity Type",
          selectInput("boxplot_var", "Variable to plot",
                      choices = c("TSD","SE","RHR_dev_7d","SessionLoad"),
                      selected = "TSD"),
          withSpinner(plotlyOutput("boxplot_act", height = "380px"), color = "#1B3A6B"),
          textOutput("boxplot_caption")
        )
      ),

      # 2B Correlation
      nav_panel("2B · Correlation Analysis",
        br(),
        info_panel(
          tags$b("What this section shows: "),
          "Pairwise relationships between key variables across all participant-days.",
          " Spearman correlation is the default because several variables are skewed.",
          " Use the scatterplot explorer to investigate specific pairs interactively."
        ),
        warn_panel(
          "⚠ Each data point represents one participant-day.",
          " Because multiple observations come from the same person,",
          " treat these as descriptive associations — not independent observations."
        ),
        layout_columns(
          col_widths = c(5, 7),
          section_card(
            "Correlation Matrix",
            radioButtons("corr_method", "Method",
                         choices = c("Spearman (recommended)" = "spearman",
                                     "Pearson" = "pearson"),
                         selected = "spearman", inline = TRUE),
            withSpinner(plotOutput("corr_matrix", height = "420px"), color = "#1B3A6B")
          ),
          section_card(
            "Scatterplot Explorer",
            layout_columns(
              col_widths = c(6,6),
              selectInput("scatter_x", "X axis", choices = KEY_VARS, selected = "SessionLoad"),
              selectInput("scatter_y", "Y axis", choices = KEY_VARS, selected = "TSD")
            ),
            withSpinner(plotlyOutput("scatter_plot", height = "350px"), color = "#1B3A6B"),
            uiOutput("scatter_caption")
          )
        )
      ),

      # 2C Regression
      nav_panel("2C · Regression Models",
        br(),
        info_panel(
          tags$b("What this section shows: "),
          "Three ordinary least squares (OLS) regression models examining",
          " (1) how training load predicts sleep, (2) how sleep predicts recovery,",
          " and (3) how training and recovery relate to fitness over time."
        ),
        div(
          style = "background:#FEF2F2; border:1.5px solid #EF4444; border-radius:8px;
                   padding:14px 18px; margin-bottom:18px; font-size:0.85rem; color:#7F1D1D;",
          tags$b("⚠ Important caveat: "),
          "These are population-average OLS regressions. Observations are NOT independent",
          " (repeated measures per person). Coefficients describe associations, not causal effects.",
          " For causal inference, linear mixed-effects models would be required."
        ),
        tabsetPanel(
          tabPanel("Model 1: Sleep ~ Training Load",
            br(),
            info_panel("Does higher training load on a given day predict how long or efficiently a person sleeps that night?"),
            layout_columns(
              col_widths = c(7, 5),
              section_card("Coefficient Table",
                withSpinner(DTOutput("m1_tbl"), color = "#1B3A6B"),
                br(), uiOutput("m1_interp")
              ),
              section_card("Residuals vs Fitted",
                withSpinner(plotOutput("m1_resid", height = "300px"), color = "#1B3A6B")
              )
            )
          ),
          tabPanel("Model 2: Recovery ~ Sleep",
            br(),
            info_panel(
              "Does better sleep predict better next-day recovery (lower RHR deviation)?",
              tags$br(),
              tags$b("Interpretation note: "),
              "RHR_dev_7d > 0 = elevated RHR (under-recovered); < 0 = suppressed RHR (well-recovered)."
            ),
            layout_columns(
              col_widths = c(7, 5),
              section_card("Coefficient Table",
                withSpinner(DTOutput("m2_tbl"), color = "#1B3A6B"),
                br(), uiOutput("m2_interp")
              ),
              section_card("Residuals vs Fitted",
                withSpinner(plotOutput("m2_resid", height = "300px"), color = "#1B3A6B")
              )
            )
          ),
          tabPanel("Model 3: Fitness ~ Training + Recovery",
            br(),
            info_panel(
              "Does accumulated training load — combined with sleep and recovery — predict",
              " aerobic fitness (VO2max) over time?"
            ),
            warn_panel("⚠ VO2max_imputed_final uses Last-Observation-Carried-Forward (LOCF) imputation for gaps ≤28 days. Coefficients for calendar_date partially reflect data interpolation artefacts."),
            layout_columns(
              col_widths = c(7, 5),
              section_card("Coefficient Table",
                withSpinner(DTOutput("m3_tbl"), color = "#1B3A6B"),
                br(), uiOutput("m3_interp")
              ),
              section_card("Residuals vs Fitted",
                withSpinner(plotOutput("m3_resid", height = "300px"), color = "#1B3A6B")
              )
            )
          )
        )
      ),

      # 2D Lagged Effects
      nav_panel("2D · Lagged Effects",
        br(),
        info_panel(
          tags$b("What this section shows: "),
          "Lagged analysis shifts the predictor back N days to test whether",
          " past training or sleep patterns predict current recovery.",
          " This helps answer: does a hard session affect sleep tonight, tomorrow, or 2 days later?",
          tags$br(),
          tags$i("Note: this is a temporal association, not a controlled experiment.")
        ),
        section_card(
          "Spearman r Across Lag 0–3 Days",
          info_panel(
            "Each bar shows the Spearman correlation between a predictor (at lag N days ago)",
            " and today's outcome. Look for the lag where the bar is tallest — that is when",
            " the relationship is strongest."
          ),
          withSpinner(plotlyOutput("lag_bar_plot", height = "400px"), color = "#1B3A6B")
        ),
        section_card(
          "Lagged Regression: Recovery ~ Yesterday's Sleep + Training Load",
          warn_panel("⚠ Same non-independence caveat as Section 2C applies here."),
          withSpinner(DTOutput("lag_reg_tbl"), color = "#1B3A6B"),
          br(), uiOutput("lag_reg_interp")
        )
      )
    )
  ),

  # ── TAB 3: Individual ──────────────────────────────────────────────────────
  nav_panel(
    title = "🔍 Individual Explorer",
    value = "tab_indiv",

    info_panel(
      tags$b("What this section shows: "),
      "Population averages can mask meaningful individual differences.",
      " Select any participant to see their personal training-sleep-recovery story over time.",
      " Toggle the comparison mode to view two participants side by side."
    ),

    layout_columns(
      col_widths = c(4, 4, 4),
      selectInput("indiv_pid", "Participant A",
                  choices = PARTICIPANTS, selected = PARTICIPANTS[1]),
      uiOutput("indiv_demo_card_a"),
      checkboxInput("compare_on", "Compare with Participant B", value = FALSE)
    ),

    conditionalPanel(
      "input.compare_on == true",
      layout_columns(
        col_widths = c(4, 4, 4),
        selectInput("indiv_pid_b", "Participant B",
                    choices = PARTICIPANTS, selected = PARTICIPANTS[2]),
        uiOutput("indiv_demo_card_b"),
        div()
      )
    ),

    # Timeline
    section_card(
      "Personal Recovery & Sleep Timeline",
      info_panel(
        "Blue line = RHR deviation from 7-day baseline. Bars = total sleep duration.",
        " Dashed line at 0 = baseline. Points above zero = under-recovered (elevated HR).",
        " Coloured triangles mark training days."
      ),
      withSpinner(plotlyOutput("timeline_a", height = "380px"), color = "#1B3A6B"),
      conditionalPanel(
        "input.compare_on == true",
        br(),
        withSpinner(plotlyOutput("timeline_b", height = "380px"), color = "#1B3A6B")
      )
    ),

    layout_columns(
      col_widths = c(6, 6),

      section_card(
        "Sleep Stage Profile",
        info_panel(
          "Stacked area shows proportion of sleep in each stage over time.",
          " Only available for participants with 4-stage Garmin firmware."
        ),
        withSpinner(plotlyOutput("sleep_stage_a", height = "280px"), color = "#1B3A6B"),
        conditionalPanel(
          "input.compare_on == true",
          br(),
          withSpinner(plotlyOutput("sleep_stage_b", height = "280px"), color = "#1B3A6B")
        )
      ),

      section_card(
        "Personal Lag Plot: Last Night's Sleep → Today's Recovery",
        info_panel(
          "Each point = one day. X axis = how many hours you slept last night.",
          " Y axis = today's RHR deviation. Downward slope = more sleep → better recovery."
        ),
        withSpinner(plotlyOutput("lag_scatter_a", height = "280px"), color = "#1B3A6B"),
        conditionalPanel(
          "input.compare_on == true",
          br(),
          withSpinner(plotlyOutput("lag_scatter_b", height = "280px"), color = "#1B3A6B")
        )
      )
    ),

    section_card(
      "Personal Summary Statistics",
      withSpinner(uiOutput("personal_summary_a"), color = "#1B3A6B")
    ),

    conditionalPanel(
      "input.compare_on == true",
      section_card(
        "Participant Comparison Table",
        warn_panel(
          "⚠ Individual comparisons are exploratory. Differences may reflect age,",
          " fitness level, training volume, or device-specific recording artifacts",
          " rather than true physiological differences."
        ),
        withSpinner(DTOutput("compare_tbl"), color = "#1B3A6B")
      )
    )
  ),

  # ── TAB 4: Methods & Data Quality ─────────────────────────────────────────
  nav_panel(
    title = "📋 Methods & Quality",
    value = "tab_methods",

    info_panel(
      tags$b("What this section shows: "),
      "Full variable dictionary, data quality notes, and cohort definitions.",
      " This section documents every analytical decision made in this project",
      " and is intended for academic review."
    ),

    layout_columns(
      col_widths = c(12),
      section_card(
        "Variable Dictionary (searchable)",
        DTOutput("dict_tbl")
      )
    ),

    layout_columns(
      col_widths = c(6, 6),

      section_card(
        "Data Quality Notes",
        warn_panel(tags$b("SE = 1.0 values: "),
          "Sleep Efficiency values of exactly 1.0 indicate Garmin detected zero",
          " awake time — this is a device floor, not truly perfect sleep.",
          " These rows are retained but flagged for sensitivity analysis."),
        warn_panel(tags$b("Body Battery: "),
          "P008 has 0% Body Battery coverage.",
          " P009 and P010 have 18–24% missing.",
          " Treat Body Battery metrics as supplementary only."),
        warn_panel(tags$b("Sleep staging: "),
          "P006, P013, and P015 lack 4-stage sleep data (legacy firmware).",
          " Stage-level analyses (DeepSleepProp, REMSleepProp) are restricted to",
          " the 'full' analysis cohort (11 participants)."),
        warn_panel(tags$b("VO2max columns: "),
          "ActivitiesVO2MaxValue is Garmin's per-session estimate (outdoor running only, 70% missing).",
          " Always use VO2max_imputed_final for analysis."),
        warn_panel(tags$b("DailyActivityLoad: "),
          "This column mirrors step count — do not use as an independent predictor",
          " alongside DailyTotalSteps.")
      ),

      section_card(
        "Analysis Cohort Definitions",
        info_panel(
          tags$b("'full' cohort (9,700 rows): "),
          "All sleep metrics valid including sleep stage proportions.",
          " 11 participants. Use for any analysis involving DeepSleepProp,",
          " REMSleepProp_final, LightSleepProp, or SleepScore.",
          tags$br(), tags$br(),
          tags$b("'no_sleep_stages' cohort (3,060 rows): "),
          "P006, P013, P015 — no stage data available.",
          " Can be included in TSD, SE, SF, RHR_dev_7d analyses."
        ),
        section_card(
          "Cohort breakdown by participant",
          withSpinner(DTOutput("cohort_tbl"), color = "#1B3A6B")
        ),
        br(),
        section_card(
          "Session Info (Reproducibility)",
          verbatimTextOutput("session_info")
        )
      )
    )
  )
)

# =============================================================================
# SERVER
# =============================================================================

server <- function(input, output, session) {

  # ── About modal ─────────────────────────────────────────────────────────────
  observeEvent(input$show_about, {
    showModal(modalDialog(
      title = "About this Dataset",
      size  = "l",
      easyClose = TRUE,
      footer = modalButton("Close"),
      tags$p(tags$b("Cohort:"), " 14 recreational runners based in Nairobi, Kenya."),
      tags$p(tags$b("Study period:"), " 2022-01-01 to 2026-02-14 (continuous wearable monitoring)."),
      tags$p(tags$b("Device:"), " Garmin wearables — continuous HR, sleep tracking, GPS activity logging."),
      tags$hr(),
      tags$h5("Key metric explanations"),
      tags$p(tags$b("RHR_dev_7d (Primary recovery metric):"),
        " Each participant's resting heart rate (RHR) minus their own 7-day rolling mean.",
        " Positive values = RHR is elevated above recent baseline = likely under-recovered.",
        " Negative values = RHR is suppressed = well-recovered or training adaptation."),
      tags$p(tags$b("SessionLoad:"), " Internal training load calculated as",
        " Average HR × Session Duration (minutes). Higher = more physiological stress."),
      tags$p(tags$b("TSD:"), " Total Sleep Duration in hours. Primary sleep outcome."),
      tags$p(tags$b("SE:"), " Sleep Efficiency = sleep time / time in bed.",
        " Values of exactly 1.0 are a device floor artifact — not truly perfect sleep."),
      tags$hr(),
      tags$h5("Missing data caveats"),
      tags$ul(
        tags$li("Body Battery: P008 has 0% coverage; P009/P010 have 18–24% missing."),
        tags$li("Sleep stages: P006, P013, P015 have no 4-stage data (legacy firmware)."),
        tags$li("SessionLoad / activity variables: ~48% missing — structural (rest days have no session).")
      )
    ))
  })

  # ── Reactive: filtered data ─────────────────────────────────────────────────
  df_filt <- reactive({
    d <- df %>%
      filter(calendar_date >= input$date_range[1],
             calendar_date <= input$date_range[2])
    if (input$act_filter != "All")
      d <- d %>% filter(ActivityGroup == input$act_filter)
    if (input$act_only)
      d <- d %>% filter(activity_day == TRUE)
    d
  })

  # ── Participant colour legend ──────────────────────────────────────────────
  output$part_legend_mini <- renderUI({
    tags$div(
      lapply(PARTICIPANTS, function(p) {
        tags$div(style = "display:flex; align-items:center; gap:5px; margin:2px 0;",
          tags$div(style = paste0("width:10px; height:10px; border-radius:50%;
                                    background:", PART_COLORS[p], ";")),
          tags$span(style = "font-size:0.72rem;", p)
        )
      })
    )
  })

  # ── KPI row ────────────────────────────────────────────────────────────────
  output$kpi_row <- renderUI({
    d <- df_filt()
    n_part     <- n_distinct(d$User_id)
    n_days     <- nrow(d)
    mean_tsd   <- round(mean(d$TSD,       na.rm = TRUE), 1)
    sd_tsd     <- round(sd(d$TSD,         na.rm = TRUE), 1)
    mean_rhr   <- round(mean(d$RHR_dev_7d,na.rm = TRUE), 2)
    pct_act    <- round(100 * mean(d$activity_day, na.rm = TRUE), 0)

    layout_columns(
      col_widths = c(2,2,3,3,2),
      value_box(title = "Participants",      value = n_part,
                showcase = bs_icon("people-fill"), theme = "primary"),
      value_box(title = "Participant-days",  value = format(n_days, big.mark = ","),
                showcase = bs_icon("calendar3"), theme = "info"),
      value_box(title = "Mean Sleep (±SD)",
                value = paste0(mean_tsd, " hrs"),
                p(paste0("SD = ", sd_tsd, " hrs")),
                showcase = bs_icon("moon-stars-fill"), theme = "success"),
      value_box(title = "Mean RHR Deviation",
                value = ifelse(mean_rhr > 0,
                               paste0("+", mean_rhr, " bpm"),
                               paste0(mean_rhr, " bpm")),
                p(ifelse(mean_rhr > 0, "Slightly under-recovered on average",
                         "Well-recovered on average")),
                showcase = bs_icon("heart-pulse-fill"),
                theme = ifelse(mean_rhr > 0.5, "warning", "success")),
      value_box(title = "Training Days",     value = paste0(pct_act, "%"),
                showcase = bs_icon("activity"), theme = "secondary")
    )
  })

  # ── Heatmap ────────────────────────────────────────────────────────────────
  output$heatmap_plot <- renderPlotly({
    d <- df_filt()
    if (input$heatmap_pid != "Cohort mean") {
      d <- d %>% filter(User_id == input$heatmap_pid)
    } else {
      d <- d %>% group_by(calendar_date, weekday) %>%
        summarise(RHR_dev_7d = mean(RHR_dev_7d, na.rm = TRUE),
                  .groups = "drop") %>%
        mutate(User_id = "Cohort mean")
    }
    d <- d %>%
      mutate(week_num = as.numeric(format(calendar_date, "%V")),
             yr       = year(calendar_date),
             yr_week  = paste0(yr, "-W", sprintf("%02d", week_num)))

    p <- ggplot(d, aes(x = calendar_date, y = weekday,
                       fill = RHR_dev_7d,
                       text = paste0("Date: ", calendar_date,
                                     "<br>RHR dev: ", round(RHR_dev_7d, 2), " bpm"))) +
      geom_tile(colour = "white", linewidth = 0.3) +
      scale_fill_gradient2(low = "#0D9488", mid = "#F1F5F9", high = "#EF4444",
                           midpoint = 0, na.value = "#E2E8F0",
                           name = "RHR dev (bpm)") +
      scale_x_date(date_breaks = "3 months", date_labels = "%b %Y") +
      labs(x = NULL, y = NULL,
           subtitle = "Green = below baseline (recovered) · Red = above baseline (fatigued)") +
      theme_minimal(base_size = 11) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1),
            legend.position = "bottom")
    ggplotly(p, tooltip = "text") %>% layout(legend = list(orientation = "h"))
  })

  output$heatmap_caption <- renderText({
    pid <- input$heatmap_pid
    if (pid == "Cohort mean") {
      "Cohort-average RHR deviation across time. Use the dropdown to view an individual."
    } else {
      paste0(pid, "'s daily recovery state. Sustained red periods indicate accumulated fatigue.")
    }
  })

  # ── Missingness table ──────────────────────────────────────────────────────
  output$missing_tbl <- renderDT({
    d <- df_filt()
    vars <- names(KEY_VARS)
    miss <- sapply(KEY_VARS, function(v) {
      if (!v %in% names(d)) return(NA_real_)
      round(100 * mean(is.na(d[[v]])), 1)
    })
    tbl <- tibble(
      Variable    = vars,
      Column      = KEY_VARS,
      `Missing %` = miss
    ) %>% arrange(desc(`Missing %`))

    datatable(tbl, rownames = FALSE, options = list(pageLength = 12, dom = "t")) %>%
      formatStyle("Missing %",
        background = styleInterval(c(5, 20),
                                   c("#DCFCE7","#FEF9C3","#FEE2E2")),
        fontWeight = "bold")
  })

  # ── Distribution plots ─────────────────────────────────────────────────────
  dist_plot <- function(var, label, xlab) {
    d <- df_filt() %>% filter(!is.na(.data[[var]]))
    mn <- mean(d[[var]], na.rm = TRUE)
    p <- ggplot(d, aes(x = .data[[var]], colour = User_id, fill = User_id)) +
      geom_density(alpha = 0.15, linewidth = 0.7) +
      geom_vline(xintercept = mn, linetype = "dashed", colour = "#1B3A6B", linewidth = 1) +
      scale_colour_manual(values = PART_COLORS) +
      scale_fill_manual(values = PART_COLORS) +
      labs(title = label,
           subtitle = paste0("Dashed line = cohort mean (", round(mn, 2), ")"),
           x = xlab, y = "Density") +
      theme_minimal(base_size = 11) +
      theme(legend.position = "none")
    ggplotly(p) %>% layout(showlegend = FALSE)
  }

  output$dist_tsd <- renderPlotly({ dist_plot("TSD",        "Sleep Duration",    "Hours") })
  output$dist_se  <- renderPlotly({ dist_plot("SE",         "Sleep Efficiency",  "Ratio (0–1)") })
  output$dist_rhr <- renderPlotly({ dist_plot("RHR_dev_7d", "RHR Deviation 7d",  "bpm") })

  # ── Descriptive summary table ──────────────────────────────────────────────
  output$desc_tbl <- renderDT({
    d <- df_filt()
    vars_to_sum <- c("TSD","SE","SF","RHR_dev_7d","SessionLoad",
                     "RelativeIntensity","VO2max_imputed_final")
    res <- lapply(c(TRUE, FALSE), function(act) {
      sub <- d %>% filter(activity_day == act)
      sapply(vars_to_sum, function(v) {
        x <- sub[[v]]
        c(Mean   = round(mean(x,   na.rm = TRUE), 2),
          Median = round(median(x, na.rm = TRUE), 2),
          SD     = round(sd(x,     na.rm = TRUE), 2),
          IQR    = round(IQR(x,    na.rm = TRUE), 2),
          Min    = round(min(x,    na.rm = TRUE), 2),
          Max    = round(max(x,    na.rm = TRUE), 2),
          N      = sum(!is.na(x)))
      })
    })
    tbl_act  <- as.data.frame(t(res[[1]])) %>% mutate(Variable = vars_to_sum, Day = "Activity day")
    tbl_rest <- as.data.frame(t(res[[2]])) %>% mutate(Variable = vars_to_sum, Day = "Rest day")
    bind_rows(tbl_act, tbl_rest) %>%
      select(Variable, Day, everything()) %>%
      datatable(rownames = FALSE, filter = "top",
                options = list(pageLength = 14, dom = "ftp")) %>%
      formatStyle("Day",
        backgroundColor = styleEqual(c("Activity day","Rest day"),
                                     c("#EFF6FF","#F0FDF4")))
  })

  # ── Boxplots by activity type ──────────────────────────────────────────────
  output$boxplot_act <- renderPlotly({
    d <- df_filt()
    var <- input$boxplot_var
    counts <- d %>% count(ActivityGroup)
    d2 <- d %>%
      left_join(counts, by = "ActivityGroup") %>%
      mutate(label = paste0(ActivityGroup, "\n(n=", n, ")"))
    p <- ggplot(d2, aes(x = label, y = .data[[var]],
                        fill = ActivityGroup,
                        text = paste0("Group: ", ActivityGroup,
                                      "<br>", var, ": ", round(.data[[var]], 2)))) +
      geom_boxplot(alpha = 0.8, outlier.size = 0.8) +
      scale_fill_manual(values = ACT_COLORS) +
      labs(x = NULL, y = var,
           subtitle = paste0("Distribution of ", var, " by activity type")) +
      theme_minimal(base_size = 11) +
      theme(legend.position = "none",
            axis.text.x = element_text(size = 8, angle = 20, hjust = 1))
    ggplotly(p, tooltip = "text")
  })

  output$boxplot_caption <- renderText({
    paste0("Boxplots of ", input$boxplot_var, " across activity types.",
           " Rest days are structural NAs for SessionLoad — this is expected.")
  })

  # ── Correlation matrix ─────────────────────────────────────────────────────
  output$corr_matrix <- renderPlot({
    d <- df_filt()
    vars <- unname(KEY_VARS[KEY_VARS %in% names(d)])
    mat  <- d %>% select(all_of(vars)) %>%
      rename_with(~ names(KEY_VARS)[match(., KEY_VARS)])
    cm   <- cor(mat, use = "pairwise.complete.obs", method = input$corr_method)
    ggcorrplot(cm, type = "lower", lab = TRUE, lab_size = 2.8,
               colors = c("#EF4444","#F1F5F9","#0D9488"),
               outline.color = "white",
               ggtheme = theme_minimal(base_size = 10)) +
      labs(title = paste0(tools::toTitleCase(input$corr_method), " Correlation Matrix"),
           subtitle = "Only participant-days with no missing data used per pair")
  })

  # ── Scatterplot explorer ───────────────────────────────────────────────────
  output$scatter_plot <- renderPlotly({
    d  <- df_filt() %>% filter(!is.na(.data[[input$scatter_x]]),
                                !is.na(.data[[input$scatter_y]]))
    sp <- spearman_r(d[[input$scatter_x]], d[[input$scatter_y]])
    xl <- names(KEY_VARS)[KEY_VARS == input$scatter_x]
    yl <- names(KEY_VARS)[KEY_VARS == input$scatter_y]
    p  <- ggplot(d, aes(x = .data[[input$scatter_x]],
                        y = .data[[input$scatter_y]],
                        colour = User_id,
                        text = paste0("Participant: ", User_id,
                                      "<br>Date: ", calendar_date,
                                      "<br>", xl, ": ", round(.data[[input$scatter_x]], 2),
                                      "<br>", yl, ": ", round(.data[[input$scatter_y]], 2)))) +
      geom_point(alpha = 0.25, size = 0.9) +
      geom_smooth(aes(group = 1), method = "loess", colour = "#1B3A6B",
                  linewidth = 1.2, se = TRUE, formula = y~x) +
      scale_colour_manual(values = PART_COLORS) +
      labs(x = xl, y = yl,
           subtitle = paste0("Spearman r = ", sp$r, " · p ", fmt_p(sp$p))) +
      theme_minimal(base_size = 11) +
      theme(legend.position = "none")
    ggplotly(p, tooltip = "text")
  })

  output$scatter_caption <- renderUI({
    sp <- {
      d <- df_filt()
      if (input$scatter_x %in% names(d) && input$scatter_y %in% names(d))
        spearman_r(d[[input$scatter_x]], d[[input$scatter_y]])
      else list(r = NA, p = NA)
    }
    div(style = "font-size:0.8rem; color:#475569; padding-top:6px;",
        paste0("Spearman r = ", sp$r, " (p ", fmt_p(sp$p), "). ",
               "Each dot = one participant-day. ",
               "Blue line = LOESS smoother with 95% CI. ",
               "Points are coloured by participant — patterns may differ between individuals."))
  })

  # ── Regression models ──────────────────────────────────────────────────────
  run_model <- function(formula_str, data) {
    tryCatch(
      lm(as.formula(formula_str), data = data, na.action = na.omit),
      error = function(e) NULL
    )
  }

  fmt_coef_tbl <- function(m) {
    if (is.null(m)) return(data.frame(Error = "Model could not be fitted"))
    ct <- summary(m)$coefficients
    ci <- confint(m)
    tbl <- data.frame(
      Term      = rownames(ct),
      Estimate  = round(ct[,1], 4),
      Std.Error = round(ct[,2], 4),
      `CI 2.5%` = round(ci[,1], 4),
      `CI 97.5%`= round(ci[,2], 4),
      `p-value` = sapply(ct[,4], fmt_p),
      check.names = FALSE
    )
    datatable(tbl, rownames = FALSE,
              options = list(dom = "t", pageLength = 15)) %>%
      formatStyle("p-value",
        backgroundColor = styleInterval(c(0.05, 0.10),
                                        c("#DCFCE7","#FEF9C3","#FEE2E2")))
  }

  plain_interp <- function(m, term, unit_label = "", outcome_label = "") {
    if (is.null(m)) return(NULL)
    ct <- summary(m)$coefficients
    ci <- confint(m)
    if (!term %in% rownames(ct)) return(NULL)
    est  <- round(ct[term, 1], 4)
    lo   <- round(ci[term, 1], 4)
    hi   <- round(ci[term, 2], 4)
    pval <- ct[term, 4]
    dir  <- ifelse(est > 0, "increase", "decrease")
    sig  <- ifelse(pval < 0.05, "statistically significant (p < 0.05)",
                                "not statistically significant (p ≥ 0.05)")
    div(
      style = "background:#EFF6FF; border-left:4px solid #2563EB;
               padding:12px 16px; border-radius:6px; font-size:0.85rem; color:#1E3A5F;",
      tags$b("Plain-English interpretation: "),
      paste0("A one-unit increase in ", unit_label,
             " is associated with a ", abs(est), " ",
             ifelse(est > 0, "increase", "decrease"),
             " in ", outcome_label, " (95% CI: [", lo, ", ", hi, "], p ",
             fmt_p(pval), "). This association is ", sig, ".")
    )
  }

  resid_plot <- function(m) {
    if (is.null(m)) return(ggplot() + labs(title = "Model not fitted"))
    tibble(fitted = fitted(m), resid = resid(m)) %>%
      ggplot(aes(x = fitted, y = resid)) +
      geom_point(alpha = 0.2, colour = "#1B3A6B", size = 0.8) +
      geom_hline(yintercept = 0, linetype = "dashed", colour = "#EF4444") +
      geom_smooth(method = "loess", se = FALSE, colour = "#0D9488",
                  linewidth = 0.9, formula = y~x) +
      labs(x = "Fitted values", y = "Residuals",
           subtitle = "Residuals should scatter randomly around zero") +
      theme_minimal(base_size = 10)
  }

  m1 <- reactive({
    run_model("TSD ~ SessionLoad + RelativeIntensity + is_weekend + VO2max_imputed_final",
              df_filt())
  })
  m2 <- reactive({
    run_model("RHR_dev_7d ~ TSD + SE + SF + SessionLoad + VO2max_imputed_final",
              df_filt())
  })
  m3 <- reactive({
    run_model("VO2max_imputed_final ~ SessionLoad + TSD + RHR_dev_7d + as.numeric(calendar_date)",
              df_filt())
  })

  output$m1_tbl   <- renderDT({ fmt_coef_tbl(m1()) })
  output$m1_interp<- renderUI({ plain_interp(m1(), "SessionLoad",
                                              "SessionLoad", "TSD (hours)") })
  output$m1_resid <- renderPlot({ resid_plot(m1()) })

  output$m2_tbl   <- renderDT({ fmt_coef_tbl(m2()) })
  output$m2_interp<- renderUI({ plain_interp(m2(), "TSD",
                                              "TSD (hours)", "RHR_dev_7d (bpm)") })
  output$m2_resid <- renderPlot({ resid_plot(m2()) })

  output$m3_tbl   <- renderDT({ fmt_coef_tbl(m3()) })
  output$m3_interp<- renderUI({ plain_interp(m3(), "SessionLoad",
                                              "SessionLoad", "VO2max (ml/kg/min)") })
  output$m3_resid <- renderPlot({ resid_plot(m3()) })

  # ── Lagged effects ─────────────────────────────────────────────────────────
  output$lag_bar_plot <- renderPlotly({
    d <- df_filt()

    pairs <- list(
      list(x_base = "SessionLoad", y = "TSD",        label = "Training Load → Sleep Duration"),
      list(x_base = "SessionLoad", y = "RHR_dev_7d", label = "Training Load → Recovery (RHR dev)"),
      list(x_base = "TSD",        y = "RHR_dev_7d", label = "Sleep Duration → Recovery (RHR dev)")
    )

    lag_results <- lapply(pairs, function(pr) {
      lapply(0:3, function(lg) {
        x_col <- if (lg == 0) pr$x_base else paste0(pr$x_base, "_lag", lg)
        if (!x_col %in% names(d)) return(NULL)
        x <- d[[x_col]]; y <- d[[pr$y]]
        ok <- complete.cases(x, y)
        if (sum(ok) < 10) return(NULL)
        r <- cor(x[ok], y[ok], method = "spearman")
        data.frame(Pair = pr$label, Lag = lg, Spearman_r = r)
      }) %>% bind_rows()
    }) %>% bind_rows()

    p <- ggplot(lag_results, aes(x = factor(Lag), y = Spearman_r,
                                  fill = Pair,
                                  text = paste0("Pair: ", Pair,
                                                "<br>Lag: ", Lag, " day(s)",
                                                "<br>Spearman r: ", round(Spearman_r, 3)))) +
      geom_col(position = "dodge", alpha = 0.85) +
      geom_hline(yintercept = 0, linetype = "dashed") +
      scale_fill_manual(values = c("#1B3A6B","#0D9488","#EF4444")) +
      scale_x_discrete(labels = c("Same day","Lag 1d","Lag 2d","Lag 3d")) +
      labs(x = "Lag", y = "Spearman r", fill = NULL,
           subtitle = "Look for the lag where the bar is tallest — that is when the relationship peaks") +
      theme_minimal(base_size = 11) +
      theme(legend.position = "bottom")
    ggplotly(p, tooltip = "text") %>% layout(legend = list(orientation = "h"))
  })

  output$lag_reg_tbl <- renderDT({
    m <- run_model("RHR_dev_7d ~ TSD_lag1 + SE_lag1 + SessionLoad_lag1", df_filt())
    fmt_coef_tbl(m)
  })

  output$lag_reg_interp <- renderUI({
    m <- run_model("RHR_dev_7d ~ TSD_lag1 + SE_lag1 + SessionLoad_lag1", df_filt())
    plain_interp(m, "TSD_lag1", "TSD_lag1 (last night's sleep hours)", "RHR_dev_7d (bpm)")
  })

  # ── Individual: demo cards ─────────────────────────────────────────────────
  demo_card <- function(pid) {
    d <- demo %>% filter(User_id == pid)
    if (nrow(d) == 0) return(NULL)
    vo2_txt <- if (is.na(d$VO2max)) "VO2max: not available" else
      paste0("VO2max: ", d$VO2max, " ml/kg/min")
    div(
      style = paste0("background:", PART_COLORS[pid],
                     "18; border-left:4px solid ", PART_COLORS[pid],
                     "; border-radius:8px; padding:10px 14px; font-size:0.82rem;"),
      tags$b(pid),
      tags$br(),
      paste0("Age: ", d$Age, " · Sex: ", d$Sex),
      tags$br(),
      vo2_txt
    )
  }

  output$indiv_demo_card_a <- renderUI({ demo_card(input$indiv_pid) })
  output$indiv_demo_card_b <- renderUI({ demo_card(input$indiv_pid_b) })

  # ── Individual: timeline ───────────────────────────────────────────────────
  timeline_plot <- function(pid) {
    d <- df_filt() %>% filter(User_id == pid) %>% arrange(calendar_date)
    act_days <- d %>% filter(activity_day == TRUE)
    col <- PART_COLORS[pid]

    fig <- plot_ly()

    # TSD bars
    fig <- fig %>% add_bars(
      data = d, x = ~calendar_date, y = ~TSD,
      name = "Sleep Duration (hrs)", yaxis = "y2",
      marker = list(color = "#93C5FD", opacity = 0.5),
      hovertemplate = "<b>%{x}</b><br>TSD: %{y:.1f} hrs<extra></extra>"
    )

    # RHR_dev_7d line
    fig <- fig %>% add_lines(
      data = d, x = ~calendar_date, y = ~RHR_dev_7d,
      name = "RHR Deviation (bpm)", yaxis = "y",
      line = list(color = col, width = 1.8),
      hovertemplate = "<b>%{x}</b><br>RHR dev: %{y:.2f} bpm<extra></extra>"
    )

    # Activity markers
    for (ag in levels(df$ActivityGroup)) {
      sub <- act_days %>% filter(ActivityGroup == ag)
      if (nrow(sub) == 0) next
      fig <- fig %>% add_markers(
        data = sub, x = ~calendar_date, y = ~RHR_dev_7d,
        name = ag, yaxis = "y",
        marker = list(symbol = "triangle-up", size = 7,
                      color = ACT_COLORS[ag], opacity = 0.8),
        hovertemplate = paste0("<b>%{x}</b><br>", ag,
                               "<br>RHR dev: %{y:.2f} bpm<extra></extra>")
      )
    }

    # Zero reference line
    fig <- fig %>% layout(
      title     = list(text = paste0(pid, " — Recovery & Sleep Timeline"), x = 0),
      xaxis     = list(title = ""),
      yaxis     = list(title = "RHR deviation (bpm)", zeroline = TRUE,
                       zerolinecolor = "#EF4444", zerolinewidth = 1.5),
      yaxis2    = list(title = "Sleep duration (hrs)", overlaying = "y",
                       side = "right", showgrid = FALSE),
      legend    = list(orientation = "h", y = -0.2),
      hovermode = "x unified",
      shapes    = list(list(type = "line", x0 = 0, x1 = 1, xref = "paper",
                            y0 = 0, y1 = 0, yref = "y",
                            line = list(dash = "dash", color = "#94A3B8")))
    )
    fig
  }

  output$timeline_a <- renderPlotly({ timeline_plot(input$indiv_pid) })
  output$timeline_b <- renderPlotly({ timeline_plot(input$indiv_pid_b) })

  # ── Individual: sleep stage profile ───────────────────────────────────────
  sleep_stage_plot <- function(pid) {
    d <- df_filt() %>%
      filter(User_id == pid, analysis_cohort == "full") %>%
      arrange(calendar_date) %>%
      select(calendar_date, DeepSleepProp, LightSleepProp, REMSleepProp_final) %>%
      drop_na()

    if (nrow(d) < 5) {
      return(plot_ly() %>% add_annotations(
        text = "Sleep stage data not available for this participant\n(device firmware limitation).",
        x = 0.5, y = 0.5, xref = "paper", yref = "paper",
        showarrow = FALSE, font = list(size = 13, color = "#64748B")
      ))
    }

    d_long <- d %>%
      pivot_longer(cols = -calendar_date, names_to = "Stage", values_to = "Prop") %>%
      mutate(Stage = recode(Stage,
        "DeepSleepProp"    = "Deep",
        "LightSleepProp"   = "Light",
        "REMSleepProp_final" = "REM"
      ))

    p <- ggplot(d_long, aes(x = calendar_date, y = Prop, fill = Stage)) +
      geom_area(alpha = 0.75, position = "stack") +
      scale_fill_manual(values = c("Deep" = "#1B3A6B", "Light" = "#60A5FA", "REM" = "#0D9488")) +
      scale_y_continuous(labels = percent_format()) +
      labs(title = paste0(pid, " — Sleep Stage Profile"),
           subtitle = "Stacked proportion of sleep stages per night",
           x = NULL, y = "Proportion", fill = NULL) +
      theme_minimal(base_size = 10) +
      theme(legend.position = "bottom")
    ggplotly(p)
  }

  output$sleep_stage_a <- renderPlotly({ sleep_stage_plot(input$indiv_pid) })
  output$sleep_stage_b <- renderPlotly({ sleep_stage_plot(input$indiv_pid_b) })

  # ── Individual: personal lag scatter ──────────────────────────────────────
  lag_scatter_plot <- function(pid) {
    d <- df_filt() %>%
      filter(User_id == pid) %>%
      drop_na(TSD_lag1, RHR_dev_7d)

    if (nrow(d) < 10) {
      return(plot_ly() %>% add_annotations(
        text = "Not enough data to compute lag relationship.",
        x = 0.5, y = 0.5, xref = "paper", yref = "paper",
        showarrow = FALSE))
    }

    sp <- spearman_r(d$TSD_lag1, d$RHR_dev_7d)
    col <- PART_COLORS[pid]

    p <- ggplot(d, aes(x = TSD_lag1, y = RHR_dev_7d,
                        text = paste0("Date: ", calendar_date,
                                      "<br>Last night's sleep: ", round(TSD_lag1, 1), " hrs",
                                      "<br>RHR dev today: ", round(RHR_dev_7d, 2), " bpm"))) +
      geom_point(alpha = 0.4, colour = col, size = 1.5) +
      geom_smooth(method = "lm", se = TRUE, colour = "#1B3A6B",
                  linewidth = 1, formula = y~x) +
      geom_hline(yintercept = 0, linetype = "dashed", colour = "#94A3B8") +
      labs(title = paste0(pid, " — Last Night's Sleep → Today's Recovery"),
           subtitle = paste0("Spearman r = ", sp$r, " · p ", fmt_p(sp$p),
                             " · ", nrow(d), " days"),
           x = "Last night's TSD (hrs)", y = "Today's RHR deviation (bpm)") +
      theme_minimal(base_size = 10)
    ggplotly(p, tooltip = "text")
  }

  output$lag_scatter_a <- renderPlotly({ lag_scatter_plot(input$indiv_pid) })
  output$lag_scatter_b <- renderPlotly({ lag_scatter_plot(input$indiv_pid_b) })

  # ── Personal summary cards ─────────────────────────────────────────────────
  output$personal_summary_a <- renderUI({
    pid <- input$indiv_pid
    d   <- df_filt() %>% filter(User_id == pid)

    # Best/worst week by mean RHR_dev_7d
    week_sum <- d %>%
      mutate(yr_week = paste0(year(calendar_date), "-W",
                              sprintf("%02d", isoweek(calendar_date)))) %>%
      group_by(yr_week) %>%
      summarise(mean_rhr = mean(RHR_dev_7d, na.rm = TRUE), .groups = "drop") %>%
      drop_na()

    best_week  <- if (nrow(week_sum) > 0) week_sum$yr_week[which.min(week_sum$mean_rhr)]  else "N/A"
    worst_week <- if (nrow(week_sum) > 0) week_sum$yr_week[which.max(week_sum$mean_rhr)] else "N/A"

    tsd_act  <- round(mean(d$TSD[d$activity_day == TRUE],  na.rm = TRUE), 1)
    tsd_rest <- round(mean(d$TSD[d$activity_day == FALSE], na.rm = TRUE), 1)
    sl_75    <- quantile(d$SessionLoad, 0.75, na.rm = TRUE)
    rhr_hard <- round(mean(d$RHR_dev_7d[!is.na(d$SessionLoad) &
                                          d$SessionLoad > sl_75], na.rm = TRUE), 2)

    layout_columns(
      col_widths = c(3,3,3,3),
      value_box(title = "Best recovery week",   value = best_week,
                showcase = bs_icon("emoji-smile"), theme = "success"),
      value_box(title = "Worst recovery week",  value = worst_week,
                showcase = bs_icon("emoji-frown"), theme = "danger"),
      value_box(title = "Sleep: activity vs rest",
                value = paste0(tsd_act, " vs ", tsd_rest, " hrs"),
                showcase = bs_icon("moon"), theme = "info"),
      value_box(title = "RHR dev after hard sessions",
                value = paste0(ifelse(rhr_hard > 0, "+", ""), rhr_hard, " bpm"),
                p("Sessions above 75th pct SessionLoad"),
                showcase = bs_icon("lightning-charge"),
                theme = ifelse(!is.na(rhr_hard) && rhr_hard > 1, "warning", "success"))
    )
  })

  # ── Comparison table ───────────────────────────────────────────────────────
  output$compare_tbl <- renderDT({
    pid_a <- input$indiv_pid
    pid_b <- input$indiv_pid_b
    d_a   <- df_filt() %>% filter(User_id == pid_a)
    d_b   <- df_filt() %>% filter(User_id == pid_b)

    vars  <- c("TSD","SE","SF","RHR_dev_7d","SessionLoad",
               "RelativeIntensity","VO2max_imputed_final")
    lbls  <- c("Sleep Duration (hrs)","Sleep Efficiency","Sleep Fragmentation",
               "RHR Deviation 7d (bpm)","Session Load",
               "Relative Intensity","VO2max (ml/kg/min)")

    tbl <- mapply(function(v, l) {
      xa <- d_a[[v]]; xb <- d_b[[v]]
      pooled_sd <- sd(c(xa, xb), na.rm = TRUE)
      diff_sds  <- abs(mean(xa, na.rm=TRUE) - mean(xb, na.rm=TRUE)) /
                   ifelse(is.na(pooled_sd) | pooled_sd == 0, 1, pooled_sd)
      data.frame(
        Variable   = l,
        `A Mean`   = round(mean(xa, na.rm=TRUE),2),
        `A Median` = round(median(xa, na.rm=TRUE),2),
        `A SD`     = round(sd(xa, na.rm=TRUE),2),
        `B Mean`   = round(mean(xb, na.rm=TRUE),2),
        `B Median` = round(median(xb, na.rm=TRUE),2),
        `B SD`     = round(sd(xb, na.rm=TRUE),2),
        `|Diff| SDs` = round(diff_sds, 2),
        check.names = FALSE
      )
    }, vars, lbls, SIMPLIFY = FALSE) %>% bind_rows()

    colnames(tbl)[c(2,3,4)] <- paste0(pid_a, c(" Mean"," Median"," SD"))
    colnames(tbl)[c(5,6,7)] <- paste0(pid_b, c(" Mean"," Median"," SD"))

    datatable(tbl, rownames = FALSE, options = list(dom = "t", pageLength = 10)) %>%
      formatStyle("|Diff| SDs",
        backgroundColor = styleInterval(c(0.5, 1),
                                        c("#F0FDF4","#FEF9C3","#FEE2E2")),
        fontWeight = "bold")
  })

  # ── Methods: variable dictionary ──────────────────────────────────────────
  output$dict_tbl <- renderDT({
    datatable(
      data_dict %>%
        mutate(`Missing %` = Missing_pct) %>%
        select(Column, Group, `Missing %`, Description),
      rownames = FALSE, filter = "top",
      options = list(pageLength = 15, dom = "ftp")
    ) %>%
      formatStyle("Missing %",
        background = styleInterval(c(5, 20),
                                   c("#DCFCE7","#FEF9C3","#FEE2E2")),
        fontWeight = "bold")
  })

  # ── Methods: cohort table ──────────────────────────────────────────────────
  output$cohort_tbl <- renderDT({
    tbl <- df %>%
      group_by(User_id) %>%
      summarise(
        Age          = first(Age),
        Sex          = first(Sex),
        `Total days` = n(),
        Cohort       = first(analysis_cohort),
        `Staging mode` = first(staging_mode),
        `TSD missing %` = round(100*mean(is.na(TSD)), 1),
        `BB missing %`  = round(100*mean(is.na(DailyBodyBattery.chargedValue)), 1),
        .groups = "drop"
      )
    datatable(tbl, rownames = FALSE, options = list(dom = "t")) %>%
      formatStyle("Cohort",
        backgroundColor = styleEqual(c("full","no_sleep_stages"),
                                     c("#DCFCE7","#FEF9C3")))
  })

  # ── Methods: session info ──────────────────────────────────────────────────
  output$session_info <- renderPrint({ sessionInfo() })
}

# =============================================================================
shinyApp(ui, server)