# modules/mod_sleep.R

mod_sleep_ui <- function(id) {
  ns <- NS(id)
  tagList(
    fluidRow(
      column(3,
        box(width = 12, title = "Controls", status = "primary", solidHeader = TRUE,
          selectInput(ns("metric"), "Sleep metric",
            choices = c(
              "Total Sleep Duration (h)"    = "TSD",
              "Sleep Efficiency"            = "SE",
              "Sleep Fragmentation"         = "SF",
              "Time In Bed (h)"             = "TIB_h",
              "Sleep Midpoint"              = "SleepMidpoint",
              "Sleep Midpoint SD [28d]"     = "SleepMidpoint_sd",
              "Sleep Score (composite)"     = "SleepScore"
            ), selected = "TSD"),
          hr(),
          checkboxInput(ns("smooth"), "LOESS smoother", TRUE),
          sliderInput(ns("span"), "Smoother span", 0.1, 0.9, 0.25, step = 0.05),
          hr(),
          checkboxInput(ns("facet"), "Facet by participant", FALSE),
          hr(),
          sliderInput(ns("roll_window"), "Rolling mean window (days)", 7, 42, 14),
          hr(),
          numericInput(ns("tsd_target"), "TSD target (h, for debt plot)", 8, 4, 12, 0.5)
        )
      ),
      column(9,
        fluidRow(
          valueBoxOutput(ns("vb_mean"),   width = 3),
          valueBoxOutput(ns("vb_median"), width = 3),
          valueBoxOutput(ns("vb_sd"),     width = 3),
          valueBoxOutput(ns("vb_pct_target"), width = 3)
        ),
        tabBox(width = 12,
          tabPanel("Time Series",  plotOutput(ns("ts_plot"),      height = "340px")),
          tabPanel("Distribution", plotOutput(ns("dist_plot"),    height = "340px")),
          tabPanel("By Weekday",   plotOutput(ns("weekday_plot"), height = "340px")),
          tabPanel("Sleep Debt",   plotOutput(ns("debt_plot"),    height = "340px")),
          tabPanel("Heatmap Calendar", plotOutput(ns("cal_heatmap"), height = "380px"))
        )
      )
    ),
    fluidRow(
      box(width = 12, title = "Weekly Summary Table", status = "info", collapsible = TRUE,
          DTOutput(ns("weekly_table")))
    )
  )
}

mod_sleep_server <- function(id, r_data) {
  moduleServer(id, function(input, output, session) {

    r_sleep <- reactive({
      df <- r_data() %>% filter(analysis_cohort == "full")
      if (input$metric %in% c("SE","SF","SleepScore","DeepSleepProp",
                              "LightSleepProp","REMSleepProp_final","AwakeSleepProp")) {
        df <- df %>% filter(!is.na(.data[[input$metric]]))
      } else {
        df <- df %>% filter(!is.na(.data[[input$metric]]))
      }
      df
    })

    # KPI boxes
    output$vb_mean <- renderValueBox({
      v <- round(median(r_sleep()[[input$metric]], na.rm = TRUE), 3)
      valueBox(v, paste("Median", input$metric), icon = icon("chart-line"), color = "blue")
    })
    output$vb_median <- renderValueBox({
      v <- round(median(r_sleep()[[input$metric]], na.rm = TRUE), 3)
      valueBox(v, paste("Median", input$metric), icon = icon("equals"), color = "teal")
    })
    output$vb_sd <- renderValueBox({
      v <- round(sd(r_sleep()[[input$metric]], na.rm = TRUE), 3)
      valueBox(v, "Std Dev", icon = icon("arrows-alt-h"), color = "purple")
    })
    output$vb_pct_target <- renderValueBox({
      if (input$metric == "TSD") {
        pct <- round(100 * mean(r_sleep()$TSD >= input$tsd_target, na.rm = TRUE), 1)
        valueBox(paste0(pct, "%"), paste0("Nights ≥ ", input$tsd_target, "h"),
                 icon = icon("check-circle"), color = if (pct >= 70) "green" else "orange")
      } else {
        valueBox("—", "N/A for this metric", icon = icon("minus"), color = "gray")
      }
    })

    # Time series
    output$ts_plot <- renderPlot({
      df <- r_sleep()
      p <- ggplot(df, aes(x = calendar_date, y = .data[[input$metric]],
                          colour = User_id, group = User_id)) +
        geom_point(alpha = 0.25, size = 0.8)

      if (input$smooth) {
        p <- p + geom_smooth(method = "loess", span = input$span,
                             se = FALSE, linewidth = 1)
      }

      if (input$facet) p <- p + facet_wrap(~User_id, ncol = 4)

      if (input$metric == "TSD" && !input$facet) {
        p <- p + geom_hline(yintercept = input$tsd_target, linetype = "dashed",
                            colour = "#DC2626", linewidth = 0.8)
      }

      p + scale_colour_participants() +
        scale_x_date(date_breaks = "6 months", date_labels = "%b\n%Y") +
        theme_garmin() +
        labs(x = NULL, y = metric_labels[input$metric], colour = NULL,
             title = paste(metric_labels[input$metric], "— longitudinal"),
             subtitle = paste0(nrow(df), " obs  |  ", COHORT_NOTE))
    })

    # Distribution
    output$dist_plot <- renderPlot({
      df <- r_sleep()
      ggplot(df, aes(x = .data[[input$metric]], fill = User_id, colour = User_id)) +
        geom_density(alpha = 0.25, linewidth = 0.7) +
        geom_vline(aes(xintercept = median(.data[[input$metric]], na.rm = TRUE)),
                   colour = "#1B3A6B", linetype = "dashed", linewidth = 1) +
        scale_fill_participants() +
        scale_colour_participants() +
        theme_garmin() +
        labs(x = metric_labels[input$metric], y = "Density",
             title = paste("Distribution of", metric_labels[input$metric]),
             fill = NULL, colour = NULL)
    })

    # Weekday boxplot
    output$weekday_plot <- renderPlot({
      df <- r_sleep()
      ggplot(df, aes(x = weekday, y = .data[[input$metric]], fill = weekday)) +
        geom_boxplot(outlier.alpha = 0.3, outlier.size = 0.8, notch = FALSE, alpha = 0.8) +
        scale_fill_manual(values = c(
          Monday="#BFDBFE", Tuesday="#BAE6FD", Wednesday="#A7F3D0",
          Thursday="#DDD6FE", Friday="#FDE68A", Saturday="#FCA5A5", Sunday="#FBCFE8"
        )) +
        stat_summary(fun = mean, geom = "point", shape = 18, size = 3, colour = "#1B3A6B") +
        facet_wrap(~User_id, ncol = 4, scales = "free_y") +
        theme_garmin() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7)) +
        labs(x = NULL, y = metric_labels[input$metric],
             title = "Distribution by day of week", fill = NULL)
    })

    # Sleep Debt
    output$debt_plot <- renderPlot({
      df <- r_sleep() %>%
        filter(input$metric == "TSD" | TRUE) %>%
        filter(!is.na(TSD)) %>%
        group_by(User_id) %>%
        arrange(calendar_date) %>%
        mutate(
          rolling_tsd = roll_mean(TSD, k = input$roll_window),
          debt        = input$tsd_target - rolling_tsd
        ) %>% ungroup()

      ggplot(df, aes(x = calendar_date, y = debt, colour = User_id, fill = User_id)) +
        geom_ribbon(aes(ymin = pmin(debt, 0), ymax = 0), alpha = 0.15) +
        geom_ribbon(aes(ymin = 0, ymax = pmax(debt, 0)), alpha = 0.15) +
        geom_line(linewidth = 0.7) +
        geom_hline(yintercept = 0, colour = "#1B3A6B", linewidth = 0.6) +
        scale_colour_participants() +
        scale_fill_participants() +
        facet_wrap(~User_id, ncol = 4) +
        scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
        theme_garmin() +
        labs(x = NULL, y = paste0("Sleep debt vs ", input$tsd_target, "h target (h)"),
             title = paste0(input$roll_window, "-day rolling mean TSD vs target"),
             colour = NULL, fill = NULL)
    })

    # Calendar heatmap
    output$cal_heatmap <- renderPlot({
      df <- r_sleep() %>%
        filter(!is.na(.data[[input$metric]])) %>%
        mutate(
          yr  = lubridate::year(calendar_date),
          wk  = lubridate::isoweek(calendar_date),
          dow = lubridate::wday(calendar_date, label = TRUE, abbr = TRUE, week_start = 1)
        )

      ggplot(df, aes(x = wk, y = dow, fill = .data[[input$metric]])) +
        geom_tile(colour = "white", linewidth = 0.3) +
        scale_fill_gradient2(low = "#DC2626", mid = "#FCD34D", high = "#16A34A",
                             midpoint = median(df[[input$metric]], na.rm = TRUE),
                             name = input$metric) +
        facet_grid(User_id ~ yr) +
        theme_garmin() +
        theme(axis.text.x = element_blank(), panel.grid = element_blank()) +
        labs(x = "Week of year", y = NULL,
             title = paste("Calendar heatmap:", metric_labels[input$metric]))
    })

    # Weekly summary table
    output$weekly_table <- renderDT({
      r_sleep() %>%
        mutate(week_start = lubridate::floor_date(calendar_date, "week")) %>%
        group_by(User_id, week_start) %>%
        summarise(
          n        = n(),
          mean_val = round(mean(.data[[input$metric]], na.rm = TRUE), 3),
          sd_val   = round(sd(.data[[input$metric]], na.rm = TRUE), 3),
          min_val  = round(min(.data[[input$metric]], na.rm = TRUE), 3),
          max_val  = round(max(.data[[input$metric]], na.rm = TRUE), 3),
          .groups  = "drop"
        ) %>%
        rename(Participant = User_id, `Week start` = week_start,
               N = n, Mean = mean_val, SD = sd_val, Min = min_val, Max = max_val) %>%
        arrange(desc(`Week start`))
    }, options = list(pageLength = 12, scrollX = TRUE), rownames = FALSE)
  })
}
