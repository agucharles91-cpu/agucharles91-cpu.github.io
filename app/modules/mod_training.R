# modules/mod_training.R

RUNNING_TYPES <- c("running","treadmill_running","track_running","trail_running")

mod_training_ui <- function(id) {
  ns <- NS(id)
  tagList(
    fluidRow(
      column(3,
        box(width=12, title="Controls", status="primary", solidHeader=TRUE,
          checkboxGroupInput(ns("act_types"), "Activity types",
            choices=c("running","treadmill_running","walking","strength_training",
                      "cycling","indoor_cycling","hiit","yoga","other"),
            selected=c("running","treadmill_running")),
          hr(),
          sliderInput(ns("load_window"), "Rolling window (days)", 7, 42, 14),
          hr(),
          selectInput(ns("intensity_metric"), "Intensity metric",
            c("Relative Intensity"="RelativeIntensity",
              "Session Load (HR×min)"="SessionLoad",
              "Efficiency Index"="EfficiencyIndex",
              "Speed (km/h)"="ActivitiesAvgSpeed_kmh"), "RelativeIntensity"),
          hr(),
          checkboxInput(ns("show_pmc"), "Show ATL/CTL/TSB (PMC)", FALSE),
          selectInput(ns("pmc_uid"), "PMC participant", choices=NULL)
        )
      ),
      column(9,
        tabBox(width=12,
          tabPanel("Training Timeline",
            plotOutput(ns("load_ts"), height="320px"),
            plotOutput(ns("speed_dist"), height="240px")
          ),
          tabPanel("Performance Management (PMC)",
            plotOutput(ns("pmc_plot"), height="400px")
          ),
          tabPanel("Activity Heatmap",
            plotOutput(ns("act_heatmap"), height="420px")
          ),
          tabPanel("Intensity Distribution",
            plotOutput(ns("intensity_violin"), height="300px"),
            plotOutput(ns("intensity_scatter"), height="260px")
          ),
          tabPanel("Training × Sleep",
            fluidRow(
              column(6, plotOutput(ns("load_sleep_scatter"), height="300px")),
              column(6, plotOutput(ns("load_rhr_scatter"),   height="300px"))
            ),
            plotOutput(ns("load_lag_plot"), height="260px")
          )
        )
      )
    )
  )
}

mod_training_server <- function(id, r_data) {
  moduleServer(id, function(input, output, session) {

    observe({
      uids <- sort(unique(r_data()$User_id))
      updateSelectInput(session, "pmc_uid", choices=uids, selected=uids[1])
    })

    r_act <- reactive({
      r_data() %>%
        filter(activity_day, ActivitiesType %in% input$act_types) %>%
        filter(!is.na(SessionLoad))
    })

    # Training load time series
    output$load_ts <- renderPlot({
      df <- r_act() %>%
        group_by(User_id) %>% arrange(calendar_date) %>%
        mutate(rolling_load = roll_mean(SessionLoad, k=input$load_window, fill=NA)) %>%
        ungroup()

      ggplot(df, aes(x=calendar_date, colour=User_id)) +
        geom_point(aes(y=SessionLoad), alpha=0.2, size=0.8) +
        geom_line(aes(y=rolling_load), linewidth=0.9) +
        scale_colour_participants() +
        scale_x_date(date_breaks="6 months", date_labels="%b\n%Y") +
        theme_garmin() +
        labs(x=NULL, y="Session Load (HR × min)", colour=NULL,
             title=paste0(input$load_window, "-day rolling Session Load"),
             subtitle=COHORT_NOTE)
    })

    output$speed_dist <- renderPlot({
      df <- r_act() %>% filter(!is.na(ActivitiesAvgSpeed_kmh))
      ggplot(df, aes(x=ActivitiesAvgSpeed_kmh, fill=ActivitiesType)) +
        geom_histogram(bins=40, alpha=0.8, position="identity") +
        scale_fill_manual(values=c(running="#2563EB",treadmill_running="#3B82F6",
                                   walking="#16A34A",other="#94A3B8",
                                   track_running="#0891B2",trail_running="#065F46")) +
        theme_garmin() +
        facet_wrap(~User_id, ncol=4, scales="free_y") +
        labs(x="Speed (km/h)", y="Count", fill="Activity",
             title="Speed distribution by activity type and participant")
    })

    # PMC (ATL/CTL/TSB)
    output$pmc_plot <- renderPlot({
      req(input$pmc_uid)
      df <- r_data() %>%
        filter(User_id == input$pmc_uid) %>%
        arrange(calendar_date) %>%
        mutate(load_val = ifelse(is.na(SessionLoad), 0, SessionLoad)) %>%
        mutate(
          ATL = as.numeric(stats::filter(load_val, filter=1/7,  method="recursive", sides=1)),
          CTL = as.numeric(stats::filter(load_val, filter=1/42, method="recursive", sides=1)),
          TSB = CTL - ATL
        )

      p1 <- ggplot(df, aes(x=calendar_date)) +
        geom_area(aes(y=ATL), fill="#2563EB", alpha=0.4) +
        geom_area(aes(y=CTL), fill="#DC2626", alpha=0.3) +
        geom_line(aes(y=ATL, colour="ATL (Fatigue)"), linewidth=0.9) +
        geom_line(aes(y=CTL, colour="CTL (Fitness)"), linewidth=0.9) +
        scale_colour_manual(values=c("ATL (Fatigue)"="#2563EB","CTL (Fitness)"="#DC2626")) +
        theme_garmin() + labs(x=NULL, y="Load units", colour=NULL,
          title=paste("PMC —", input$pmc_uid), subtitle="ATL=7d EWM, CTL=42d EWM")

      p2 <- ggplot(df, aes(x=calendar_date, y=TSB)) +
        geom_ribbon(aes(ymin=pmin(TSB,0),ymax=0), fill="#DC2626",alpha=0.3) +
        geom_ribbon(aes(ymin=0,ymax=pmax(TSB,0)), fill="#16A34A",alpha=0.3) +
        geom_line(linewidth=0.7, colour="#1B3A6B") +
        geom_hline(yintercept=c(-30,0), linetype=c("dashed","solid"), colour=c("#DC2626","#475569")) +
        theme_garmin() + labs(x=NULL, y="TSB (Form = CTL − ATL)")

      cowplot::plot_grid(p1, p2, ncol=1, rel_heights=c(0.6,0.4))
    })

    # Activity heatmap
    output$act_heatmap <- renderPlot({
      df <- r_data() %>%
        filter(!is.na(ActivitiesType)) %>%
        count(User_id, weekday, ActivitiesType) %>%
        filter(ActivitiesType %in% input$act_types)

      ggplot(df, aes(x=weekday, y=ActivitiesType, fill=n)) +
        geom_tile(colour="white", linewidth=0.4) +
        geom_text(aes(label=n), size=3) +
        scale_fill_gradient(low="#DBEAFE", high="#1B3A6B", name="Count") +
        facet_wrap(~User_id, ncol=4) +
        theme_garmin() +
        theme(axis.text.x=element_text(angle=45,hjust=1,size=8)) +
        labs(x=NULL, y=NULL, title="Activity frequency: day of week × activity type")
    })

    # Intensity distribution violin
    output$intensity_violin <- renderPlot({
      df <- r_act() %>% filter(!is.na(.data[[input$intensity_metric]]))
      ggplot(df, aes(x=User_id, y=.data[[input$intensity_metric]], fill=User_id)) +
        geom_violin(alpha=0.7, trim=FALSE) +
        geom_boxplot(width=0.15, fill="white", outlier.size=0.5) +
        scale_fill_participants() +
        theme_garmin() +
        theme(axis.text.x=element_text(angle=45,hjust=1)) +
        labs(x=NULL, y=metric_labels[input$intensity_metric], fill=NULL,
             title=paste("Distribution of", input$intensity_metric, "by participant"))
    })

    output$intensity_scatter <- renderPlot({
      df <- r_act() %>%
        filter(!is.na(.data[[input$intensity_metric]]), !is.na(ActivitiesAvgSpeed_kmh))
      ggplot(df, aes(x=ActivitiesAvgSpeed_kmh, y=.data[[input$intensity_metric]],
                     colour=User_id)) +
        geom_point(alpha=0.35, size=1) +
        geom_smooth(method="lm", se=FALSE, linewidth=0.8) +
        scale_colour_participants() +
        theme_garmin() +
        labs(x="Speed (km/h)", y=metric_labels[input$intensity_metric], colour=NULL,
             title=paste(input$intensity_metric, "vs Speed"))
    })

    # Training × Sleep
    output$load_sleep_scatter <- renderPlot({
      df <- r_data() %>%
        filter(!is.na(SessionLoad), !is.na(TSD),
               analysis_cohort %in% c("full","legacy_staging"))
      ggplot(df, aes(x=SessionLoad, y=TSD, colour=User_id)) +
        geom_point(alpha=0.3, size=1) +
        geom_smooth(method="lm", se=FALSE, linewidth=0.9) +
        scale_colour_participants() +
        theme_garmin() +
        labs(x="Session Load", y="TSD (h)", colour=NULL,
             title="Session Load vs Same-night TSD")
    })

    output$load_rhr_scatter <- renderPlot({
      df <- r_data() %>%
        filter(!is.na(SessionLoad), !is.na(DailyRestingHeartRate_clean))
      ggplot(df, aes(x=SessionLoad, y=DailyRestingHeartRate_clean, colour=User_id)) +
        geom_point(alpha=0.3, size=1) +
        geom_smooth(method="lm", se=FALSE, linewidth=0.9) +
        scale_colour_participants() +
        theme_garmin() +
        labs(x="Session Load", y="RHR (bpm)", colour=NULL,
             title="Session Load vs RHR (same day)")
    })

    output$load_lag_plot <- renderPlot({
      df <- r_data() %>%
        filter(!is.na(TSD)) %>%
        group_by(User_id) %>%
        arrange(calendar_date) %>%
        mutate(
          load_lag1 = dplyr::lag(SessionLoad, 1),
          load_lag2 = dplyr::lag(SessionLoad, 2)
        ) %>% ungroup() %>%
        tidyr::pivot_longer(c(load_lag1,load_lag2),
                            names_to="lag", values_to="prev_load") %>%
        filter(!is.na(prev_load))

      ggplot(df, aes(x=prev_load, y=TSD, colour=lag)) +
        geom_point(alpha=0.15, size=0.8) +
        geom_smooth(method="lm", se=FALSE, linewidth=1) +
        facet_wrap(~User_id, ncol=4) +
        scale_colour_manual(values=c(load_lag1="#2563EB",load_lag2="#7C3AED"),
                            labels=c("Lag 1d","Lag 2d")) +
        theme_garmin() +
        labs(x="Previous session load", y="TSD (h)", colour=NULL,
             title="Prior-day training load vs next-night sleep (lag 1d & 2d)")
    })
  })
}
