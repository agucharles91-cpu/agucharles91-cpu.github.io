# modules/mod_overview.R

mod_overview_ui <- function(id) {
  ns <- NS(id)
  tagList(
    fluidRow(
      valueBoxOutput(ns("vb_participants"), width = 2),
      valueBoxOutput(ns("vb_rows"),         width = 2),
      valueBoxOutput(ns("vb_tsd_med"),      width = 2),
      valueBoxOutput(ns("vb_se_med"),       width = 2),
      valueBoxOutput(ns("vb_vo2_mean"),     width = 2),
      valueBoxOutput(ns("vb_rhr_mean"),     width = 2)
    ),
    fluidRow(
      box(title = "Metric Coverage per Participant", width = 8, status = "primary",
          plotOutput(ns("completeness_heatmap"), height = "360px")),
      box(title = "Cohort split", width = 4, status = "info",
          plotOutput(ns("cohort_pie"),  height = "160px"),
          hr(),
          tableOutput(ns("cohort_table")))
    ),
    fluidRow(
      box(title = "Observation timeline — each tick is one day of data",
          width = 12, status = "primary",
          plotOutput(ns("timeline"), height = "200px"))
    )
  )
}

mod_overview_server <- function(id, r_data) {
  moduleServer(id, function(input, output, session) {

    output$vb_participants <- renderValueBox({
      valueBox(n_distinct(r_data()$User_id), "Participants",
               icon = icon("users"), color = "blue")
    })
    output$vb_rows <- renderValueBox({
      valueBox(format(nrow(r_data()), big.mark = ","), "Rows",
               icon = icon("table"), color = "navy")
    })
    output$vb_tsd_med <- renderValueBox({
      v <- round(median(r_data()$TSD, na.rm = TRUE), 2)
      valueBox(paste0(v, "h"), "Median TSD", icon = icon("moon"), color = "purple")
    })
    output$vb_se_med <- renderValueBox({
      v <- round(median(r_data()$SE, na.rm = TRUE), 3)
      valueBox(v, "Median SE", icon = icon("percent"), color = "teal")
    })
    output$vb_vo2_mean <- renderValueBox({
      v <- round(mean(r_data()$VO2max_imputed_final, na.rm = TRUE), 1)
      valueBox(paste0(v, " ml/kg/min"), "Mean VO\u2082max", icon = icon("heartbeat"), color = "green")
    })
    output$vb_rhr_mean <- renderValueBox({
      v <- round(mean(r_data()$DailyRestingHeartRate_clean, na.rm = TRUE), 1)
      valueBox(paste0(v, " bpm"), "Mean RHR", icon = icon("heart"), color = "red")
    })

    output$completeness_heatmap <- renderPlot({
      req(nrow(r_data()) > 0)
      metrics <- c("TSD","SE","REMSleepProp_final","SleepMidpoint",
                   "VO2max_imputed_final","DailyRestingHeartRate_clean",
                   "RHR_dev_7d","SessionLoad","RelativeIntensity",
                   "DailyRespiration.avgWakingRespirationValue")
      df    <- r_data()
      users <- unique(df$User_id)

      heat <- do.call(rbind, lapply(users, function(u) {
        u_vals <- df[df$User_id == u, , drop = FALSE]
        data.frame(
          User_id     = u,
          metric      = metrics,
          pct_present = sapply(metrics, function(m) {
            col <- u_vals[[m]]
            if (is.null(col)) return(0)
            100 * mean(!is.na(col))
          }),
          stringsAsFactors = FALSE
        )
      }))

      ggplot(heat, aes(x = metric, y = User_id, fill = pct_present)) +
        geom_tile(colour = "white", linewidth = 0.5) +
        geom_text(aes(label = paste0(round(pct_present), "%")), size = 3,
                  colour = ifelse(heat$pct_present > 50, "white", "#1E293B")) +
        scale_fill_gradient2(low = "#DC2626", mid = "#FCD34D", high = "#16A34A",
                             midpoint = 50, limits = c(0,100), name = "% Present") +
        scale_x_discrete(labels = function(x) gsub("\\.", "\n", gsub("_", " ", x))) +
        theme_garmin() +
        theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 9)) +
        labs(x = NULL, y = NULL,
             title   = "Metric coverage",
             subtitle = COHORT_NOTE)
    })

    output$cohort_pie <- renderPlot({
      req(nrow(r_data()) > 0)
      df <- r_data() %>% count(analysis_cohort)
      ggplot(df, aes(x = "", y = n, fill = analysis_cohort)) +
        geom_col(width = 1, colour = "white") +
        coord_polar(theta = "y") +
        scale_fill_manual(values = COHORT_COLS, drop = FALSE) +
        theme_void() +
        theme(legend.position = "right", legend.text = element_text(size = 9)) +
        labs(fill = NULL)
    })

    output$cohort_table <- renderTable({
      r_data() %>%
        count(analysis_cohort, name = "n") %>%
        mutate(pct = paste0(round(100*n/sum(n), 1), "%")) %>%
        rename(Cohort = analysis_cohort, Rows = n, `%` = pct)
    }, striped = TRUE, hover = TRUE, bordered = TRUE, spacing = "s")

    output$timeline <- renderPlot({
      req(nrow(r_data()) > 0)
      ggplot(r_data(), aes(x = calendar_date, y = User_id, colour = analysis_cohort)) +
        geom_point(shape = "|", size = 2, alpha = 0.4) +
        scale_colour_manual(values = COHORT_COLS, drop = FALSE) +
        scale_x_date(date_breaks = "6 months", date_labels = "%b %Y") +
        theme_garmin() +
        theme(axis.text.x = element_text(angle = 30, hjust = 1)) +
        labs(x = NULL, y = NULL, colour = "Cohort",
             subtitle = COHORT_NOTE)
    })
  })
}
