# modules/mod_recovery.R  — RHR_dev_7d primary (all 14p), BB supplementary

mod_recovery_ui <- function(id) {
  ns <- NS(id)
  tagList(
    fluidRow(
      column(3,
        box(width = 12, title = "Controls", status = "primary", solidHeader = TRUE,
          h5(icon("heart"), " RHR Deviation"),
          sliderInput(ns("roll_w"), "Rolling window (days)", 7, 42, 14),
          sliderInput(ns("tsd_lag"), "Lag TSD by (days)", min = 0, max = 3, value = 1, step = 1),
          hr(),
          h5(icon("battery-half"), " Body Battery (supplementary)"),
          tags$p(style = "color:#64748B;font-size:0.8rem;",
            "BB available for 11/14 participants.",
            tags$br(), "P008 = 0% coverage (old device).",
            tags$br(), "P009 = 17.7%, P010 = 24.0%."),
          hr(),
          h5(icon("lungs"), " Respiration")
        )
      ),
      column(9,
        tabBox(width = 12,
          tabPanel("RHR dev 7d (primary)",
            fluidRow(
              valueBoxOutput(ns("vb_rhr_mean"), width = 4),
              valueBoxOutput(ns("vb_rhr_sd"),   width = 4),
              valueBoxOutput(ns("vb_rhr_miss"), width = 4)
            ),
            plotOutput(ns("rhr_dev_ts"),      height = "280px"),
            plotOutput(ns("rhr_dev_heatmap"), height = "300px")
          ),
          tabPanel("RHR x Sleep",
            plotOutput(ns("rhr_sleep_lag"),  height = "340px"),
            plotOutput(ns("rhr_clean_dist"), height = "240px")
          ),
          tabPanel("RHR x Training",
            plotOutput(ns("rhr_load_scatter"), height = "300px"),
            plotOutput(ns("rhr_vo2_scatter"),  height = "260px")
          ),
          tabPanel("Body Battery (supplementary)",
            tags$div(class = "alert alert-warning",
              style = "margin:8px 0;font-size:0.85rem;",
              icon("exclamation-triangle"),
              " BB shown for reference only. P008 excluded (0% coverage). P009/P010 have 18-24% missing."),
            plotOutput(ns("bb_net_ts"),        height = "260px"),
            plotOutput(ns("bb_sleep_scatter"), height = "240px")
          ),
          tabPanel("Respiration",
            plotOutput(ns("resp_ts"),   height = "300px"),
            plotOutput(ns("resp_dist"), height = "240px")
          )
        )
      )
    )
  )
}

mod_recovery_server <- function(id, r_data) {
  moduleServer(id, function(input, output, session) {

    output$vb_rhr_mean <- renderValueBox({
      v <- round(median(r_data()$RHR_dev_7d, na.rm = TRUE), 2)
      valueBox(paste0(v, " bpm"), "Median RHR dev 7d", icon = icon("heart"), color = "blue")
    })
    output$vb_rhr_sd <- renderValueBox({
      v <- round(sd(r_data()$RHR_dev_7d, na.rm = TRUE), 2)
      valueBox(paste0(v, " bpm"), "SD RHR Dev", icon = icon("arrows-alt-h"), color = "teal")
    })
    output$vb_rhr_miss <- renderValueBox({
      miss <- sum(is.na(r_data()$RHR_dev_7d))
      pct  <- round(100 * miss / nrow(r_data()), 1)
      valueBox(paste0(pct, "%"), "Missing", icon = icon("check-circle"),
               color = if (pct <= 10) "green" else "orange")
    })

    output$rhr_dev_ts <- renderPlot({
      df <- r_data() %>%
        filter(!is.na(RHR_dev_7d)) %>%
        group_by(User_id) %>% arrange(calendar_date) %>%
        mutate(rhr_roll = roll_mean(RHR_dev_7d, k = input$roll_w)) %>% ungroup()
      ggplot(df, aes(x = calendar_date, colour = User_id)) +
        geom_point(aes(y = RHR_dev_7d), alpha = 0.15, size = 0.7) +
        geom_line(aes(y = rhr_roll), linewidth = 0.9) +
        geom_hline(yintercept = 0, linetype = "dashed", colour = "#94A3B8") +
        scale_colour_participants() +
        scale_x_date(date_breaks = "6 months", date_labels = "%b\n%Y") +
        facet_wrap(~User_id, ncol = 4) +
        theme_garmin() +
        labs(x = NULL, y = "RHR deviation from 7d mean (bpm)", colour = NULL,
             title = paste0(input$roll_w, "-day rolling RHR Deviation — all 14 participants"),
             subtitle = paste0("Positive = above recent baseline (fatigue)  |  Negative = below (well-recovered)  |  ", COHORT_NOTE))
    })

    output$rhr_dev_heatmap <- renderPlot({
      df <- r_data() %>% filter(!is.na(RHR_dev_7d)) %>%
        mutate(yr = year(calendar_date), wk = isoweek(calendar_date),
               dow = wday(calendar_date, label = TRUE, abbr = TRUE, week_start = 1))
      ggplot(df, aes(x = wk, y = dow, fill = RHR_dev_7d)) +
        geom_tile(colour = "white", linewidth = 0.2) +
        scale_fill_gradient2(low = "#2563EB", mid = "white", high = "#DC2626",
                             midpoint = 0, name = "RHR dev\n(bpm)") +
        facet_grid(User_id ~ yr) +
        theme_garmin() +
        theme(axis.text.x = element_blank(), panel.grid = element_blank(),
              strip.text.y = element_text(size = 7)) +
        labs(x = "Week of year", y = NULL,
             title = "RHR Deviation calendar heatmap — all 14 participants")
    })

    output$rhr_sleep_lag <- renderPlot({
      lag_n <- as.integer(input$tsd_lag)
      df <- r_data() %>%
        filter(!is.na(RHR_dev_7d), !is.na(TSD), analysis_cohort == "full") %>%
        group_by(User_id) %>% arrange(calendar_date) %>%
        mutate(TSD_lagged = dplyr::lag(TSD, lag_n)) %>% ungroup() %>%
        filter(!is.na(TSD_lagged))
      ggplot(df, aes(x = TSD_lagged, y = RHR_dev_7d, colour = User_id)) +
        geom_point(alpha = 0.2, size = 0.9) +
        geom_smooth(method = "lm", se = TRUE, linewidth = 0.9, alpha = 0.15) +
        geom_hline(yintercept = 0, linetype = "dashed", colour = "#94A3B8") +
        scale_colour_participants() +
        facet_wrap(~User_id, ncol = 4) +
        theme_garmin() +
        labs(x = paste0("TSD lagged ", lag_n, "d (h)"), y = "RHR Deviation (bpm)", colour = NULL,
             title = paste0("RHR Deviation vs TSD (lag ", lag_n, "d) — full cohort only"))
    })

    output$rhr_clean_dist <- renderPlot({
      df <- r_data() %>% filter(!is.na(DailyRestingHeartRate_clean))
      ggplot(df, aes(x = DailyRestingHeartRate_clean, fill = User_id, colour = User_id)) +
        geom_density(alpha = 0.25, linewidth = 0.7) +
        scale_fill_participants() + scale_colour_participants() +
        theme_garmin() +
        labs(x = "RHR clean (bpm)", y = "Density", fill = NULL, colour = NULL,
             title = "RHR distribution by participant")
    })

    output$rhr_load_scatter <- renderPlot({
      df <- r_data() %>% filter(!is.na(RHR_dev_7d), !is.na(SessionLoad))
      ggplot(df, aes(x = SessionLoad, y = RHR_dev_7d, colour = User_id)) +
        geom_point(alpha = 0.2, size = 0.9) +
        geom_smooth(method = "lm", se = FALSE, linewidth = 0.9) +
        geom_hline(yintercept = 0, linetype = "dashed", colour = "#94A3B8") +
        scale_colour_participants() +
        theme_garmin() +
        labs(x = "Session Load (HR x min)", y = "RHR Deviation (bpm)", colour = NULL,
             title = "Session Load vs RHR Deviation (same day)")
    })

    output$rhr_vo2_scatter <- renderPlot({
      df <- r_data() %>% filter(!is.na(DailyRestingHeartRate_clean), !is.na(VO2max_imputed_final))
      ggplot(df, aes(x = VO2max_imputed_final, y = DailyRestingHeartRate_clean, colour = User_id)) +
        geom_point(alpha = 0.2, size = 0.9) +
        geom_smooth(method = "lm", se = FALSE, linewidth = 0.9) +
        scale_colour_participants() +
        theme_garmin() +
        labs(x = "VO2max (ml/kg/min)", y = "RHR clean (bpm)", colour = NULL,
             title = "VO2max vs RHR — higher fitness, lower resting HR")
    })

    output$bb_net_ts <- renderPlot({
      df <- r_data() %>%
        filter(!is.na(BB_net), User_id != "P008") %>%
        group_by(User_id) %>% arrange(calendar_date) %>%
        mutate(bb_roll = roll_mean(BB_net, k = input$roll_w)) %>% ungroup()
      ggplot(df, aes(x = calendar_date, colour = User_id)) +
        geom_point(aes(y = BB_net), alpha = 0.15, size = 0.7) +
        geom_line(aes(y = bb_roll), linewidth = 0.8) +
        geom_hline(yintercept = 0, linetype = "dashed", colour = "#94A3B8") +
        scale_colour_participants() +
        scale_x_date(date_breaks = "6 months", date_labels = "%b\n%Y") +
        facet_wrap(~User_id, ncol = 3) +
        theme_garmin() +
        labs(x = NULL, y = "BB Charged - Drained", colour = NULL,
             title = paste0(input$roll_w, "-day rolling Body Battery Net (excl. P008)"),
             caption = "P008: 0% BB coverage. P009/P010: 18-24% missing.")
    })

    output$bb_sleep_scatter <- renderPlot({
      df <- r_data() %>%
        filter(!is.na(BB_net), !is.na(TSD), analysis_cohort == "full", User_id != "P008")
      ggplot(df, aes(x = TSD, y = BB_net, colour = User_id)) +
        geom_point(alpha = 0.3, size = 0.9) +
        geom_smooth(method = "lm", se = FALSE, linewidth = 0.9) +
        geom_hline(yintercept = 0, linetype = "dashed", colour = "#94A3B8") +
        scale_colour_participants() +
        theme_garmin() +
        labs(x = "TSD (h)", y = "Body Battery Net", colour = NULL,
             title = "Body Battery Net vs TSD (excl. P008)")
    })

    output$resp_ts <- renderPlot({
      df <- r_data() %>%
        filter(!is.na(`DailyRespiration.avgWakingRespirationValue`)) %>%
        group_by(User_id) %>% arrange(calendar_date) %>%
        mutate(resp_roll = roll_mean(`DailyRespiration.avgWakingRespirationValue`, k = input$roll_w)) %>%
        ungroup()
      ggplot(df, aes(x = calendar_date, colour = User_id)) +
        geom_point(aes(y = `DailyRespiration.avgWakingRespirationValue`), alpha = 0.15, size = 0.7) +
        geom_line(aes(y = resp_roll), linewidth = 0.9) +
        scale_colour_participants() +
        scale_x_date(date_breaks = "6 months", date_labels = "%b\n%Y") +
        theme_garmin() +
        labs(x = NULL, y = "Waking Respiration (brpm)", colour = NULL,
             title = paste0(input$roll_w, "-day rolling Waking Respiration Rate"))
    })

    output$resp_dist <- renderPlot({
      df <- r_data() %>% filter(!is.na(`DailyRespiration.avgWakingRespirationValue`))
      ggplot(df, aes(x = `DailyRespiration.avgWakingRespirationValue`,
                     fill = User_id, colour = User_id)) +
        geom_density(alpha = 0.25, linewidth = 0.7) +
        scale_fill_participants() + scale_colour_participants() +
        theme_garmin() +
        labs(x = "Waking Respiration (brpm)", y = "Density", fill = NULL, colour = NULL)
    })
  })
}
