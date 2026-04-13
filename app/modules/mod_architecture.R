# modules/mod_architecture.R  — Stage proportions + Circadian rhythm

mod_arch_ui <- function(id) {
  ns <- NS(id)
  tagList(
    fluidRow(
      column(3,
        box(width = 12, title = "Controls", status = "primary", solidHeader = TRUE,
          h5(icon("moon"), " Stage Proportions"),
          selectInput(ns("arch_group"), "Group by",
            c("Participant"="User_id","Year"="year","Weekday"="weekday",
              "Weekend"="is_weekend","Activity day"="activity_day"), "User_id"),
          radioButtons(ns("arch_view"), "Chart type",
            c("Stacked bar"="stack","Radar"="radar","Scatter REM%"="scatter"), "stack"),
          hr(),
          h5(icon("clock"), " Circadian / Midpoint"),
          selectInput(ns("circ_metric"), "Circadian metric",
            c("Sleep Midpoint"="SleepMidpoint",
              "Midpoint SD [28d]"="SleepMidpoint_sd",
              "Sleep Start (h)"="sleep_start_h",
              "Sleep End (h)"="sleep_end_h"), "SleepMidpoint"),
          checkboxInput(ns("circ_imputed"), "Include imputed midpoints (P006)", FALSE),
          hr(),
          sliderInput(ns("sri_window"), "SRI rolling window (days)", 14, 90, 30)
        )
      ),
      column(9,
        tabBox(width = 12,
          tabPanel("Stage Proportions",
            plotOutput(ns("stage_plot"), height = "380px"),
            hr(),
            tableOutput(ns("stage_table"))
          ),
          tabPanel("Sleep Midpoint",
            plotOutput(ns("midpoint_ts"),  height = "280px"),
            plotOutput(ns("midpoint_dist"), height = "220px")
          ),
          tabPanel("Circadian Regularity (SRI)",
            plotOutput(ns("sri_plot"), height = "340px"),
            fluidRow(
              column(6, plotOutput(ns("midpoint_violin"), height = "240px")),
              column(6, tableOutput(ns("sri_table")))
            )
          ),
          tabPanel("Midpoint vs TSD",
            plotOutput(ns("midpoint_tsd"), height = "380px")
          )
        )
      )
    )
  )
}

mod_arch_server <- function(id, r_data) {
  moduleServer(id, function(input, output, session) {

    r_stage <- reactive({
      r_data() %>%
        filter(stage_comparable | analysis_cohort == "full") %>%
        filter(!is.na(DeepSleepProp), !is.na(REMSleepProp_final))
    })

    r_circ <- reactive({
      df <- r_data()
      if (!input$circ_imputed) {
        df <- df %>% filter(sm_source %in% c("calculated","original",NA) |
                              is.na(sm_source))
        df <- df %>% filter(!is_imputed_tsd)
      }
      df %>% filter(!is.na(SleepMidpoint))
    })

    # Stage plot
    output$stage_plot <- renderPlot({
      df <- r_stage()

      if (input$arch_view == "stack") {
        # Melt to long for stacked bar
        long <- df %>%
          group_by(across(all_of(input$arch_group))) %>%
          summarise(
            Deep  = mean(DeepSleepProp,  na.rm = TRUE),
            Light = mean(LightSleepProp, na.rm = TRUE),
            REM   = mean(REMSleepProp_final, na.rm = TRUE),
            Awake = mean(AwakeSleepProp, na.rm = TRUE),
            .groups = "drop"
          ) %>%
          tidyr::pivot_longer(c(Deep,Light,REM,Awake), names_to="Stage", values_to="prop") %>%
          mutate(Stage = factor(Stage, levels = c("Awake","REM","Light","Deep")))

        ggplot(long, aes(x = .data[[input$arch_group]], y = prop, fill = Stage)) +
          geom_col(position = "stack", alpha = 0.9) +
          scale_fill_manual(values = STAGE_COLS) +
          scale_y_continuous(labels = scales::percent) +
          coord_flip() +
          theme_garmin() +
          labs(x = NULL, y = "Proportion of TSD", fill = "Stage",
               title = "Mean sleep stage proportions",
               subtitle = COHORT_NOTE,
             caption = "Only 4-stage REM nights used (sleep_prop_source = 4stage_rem_comparable)")

      } else if (input$arch_view == "scatter") {
        ggplot(df, aes(x = TSD, y = REMSleepProp_final, colour = User_id)) +
          geom_point(alpha = 0.4, size = 1.5) +
          geom_smooth(method = "lm", se = FALSE, linewidth = 0.9) +
          scale_colour_participants() +
          scale_y_continuous(labels = scales::percent) +
          theme_garmin() +
          labs(x = "TSD (h)", y = "REM proportion",
               title = "REM% vs Total Sleep Duration", colour = NULL)

      } else {
        # Radar (using simple coord_polar)
        radar_df <- df %>%
          group_by(User_id) %>%
          summarise(Deep=mean(DeepSleepProp,na.rm=TRUE),
                    Light=mean(LightSleepProp,na.rm=TRUE),
                    REM=mean(REMSleepProp_final,na.rm=TRUE),
                    Awake=mean(AwakeSleepProp,na.rm=TRUE),.groups="drop") %>%
          tidyr::pivot_longer(-User_id, names_to="Stage", values_to="prop")

        ggplot(radar_df, aes(x = Stage, y = prop, colour = User_id, group = User_id)) +
          geom_line() + geom_point(size=2) +
          coord_polar() +
          scale_colour_participants() +
          scale_y_continuous(labels = scales::percent) +
          theme_garmin() +
          labs(title = "Sleep stage radar — per participant", colour = NULL, x=NULL, y=NULL)
      }
    })

    output$stage_table <- renderTable({
      r_stage() %>%
        group_by(User_id) %>%
        summarise(
          N     = n(),
          Deep  = paste0(round(100*mean(DeepSleepProp,na.rm=TRUE),1),"%"),
          Light = paste0(round(100*mean(LightSleepProp,na.rm=TRUE),1),"%"),
          REM   = paste0(round(100*mean(REMSleepProp_final,na.rm=TRUE),1),"%"),
          Awake = paste0(round(100*mean(AwakeSleepProp,na.rm=TRUE),1),"%"),
          .groups="drop"
        ) %>% rename(Participant=User_id)
    }, striped=TRUE, hover=TRUE, bordered=TRUE, spacing="s")

    # Midpoint time series
    output$midpoint_ts <- renderPlot({
      df <- r_circ()
      ggplot(df, aes(x=calendar_date, y=SleepMidpoint, colour=User_id)) +
        geom_point(alpha=0.3, size=0.8) +
        geom_smooth(method="loess", span=0.3, se=FALSE, linewidth=0.9) +
        geom_hline(yintercept=3, linetype="dashed", colour="#94A3B8") +
        scale_colour_participants() +
        scale_x_date(date_breaks="6 months", date_labels="%b\n%Y") +
        theme_garmin() +
        labs(x=NULL, y="Midpoint (h from midnight)", colour=NULL,
             title="Sleep Midpoint over time  (dashed = 03:00)")
    })

    output$midpoint_dist <- renderPlot({
      df <- r_circ()
      ggplot(df, aes(x=SleepMidpoint, fill=User_id)) +
        geom_histogram(bins=48, alpha=0.7, position="identity") +
        scale_fill_participants() +
        theme_garmin() +
        facet_wrap(~User_id, ncol=4, scales="free_y") +
        labs(x="Sleep Midpoint (h from midnight)", y="Count", fill=NULL)
    })

    # SRI plot
    output$sri_plot <- renderPlot({
      df <- r_circ() %>%
        group_by(User_id) %>% arrange(calendar_date) %>%
        mutate(
          sm_diff  = abs(c(NA, diff(SleepMidpoint))),
          sm_diff  = ifelse(sm_diff > 12, 24 - sm_diff, sm_diff),
          sri_roll = 100*(1 - roll_mean(sm_diff, k=input$sri_window)/12)
        ) %>% ungroup()

      ggplot(df, aes(x=calendar_date, y=sri_roll, colour=User_id)) +
        geom_line(alpha=0.8, linewidth=0.7) +
        geom_hline(yintercept=c(0,100), linetype="dashed", colour="#CBD5E1") +
        geom_hline(yintercept=50, linetype="dashed", colour="#F59E0B") +
        scale_colour_participants() +
        scale_x_date(date_breaks="6 months", date_labels="%b\n%Y") +
        facet_wrap(~User_id, ncol=4) +
        theme_garmin() +
        labs(x=NULL, y="SRI (rolling)", colour=NULL,
             title=paste0(input$sri_window, "-day rolling Sleep Regularity Index"),
             subtitle="Higher = more regular sleep timing  |  Yellow dashed = 50")
    })

    output$sri_table <- renderTable({
      r_circ() %>%
        group_by(User_id) %>%
        summarise(
          `Mean midpoint` = paste0(round(mean(SleepMidpoint,na.rm=TRUE),2),"h"),
          `SD midpoint`   = paste0(round(sd(SleepMidpoint,na.rm=TRUE),2),"h"),
          `SRI (approx)`  = round(compute_sri(SleepMidpoint[!is.na(SleepMidpoint)]),1),
          .groups="drop"
        ) %>% rename(Participant=User_id)
    }, striped=TRUE, hover=TRUE, bordered=TRUE)

    output$midpoint_violin <- renderPlot({
      df <- r_circ()
      ggplot(df, aes(x=User_id, y=SleepMidpoint, fill=User_id)) +
        geom_violin(alpha=0.7, trim=FALSE) +
        geom_boxplot(width=0.15, fill="white", outlier.size=0.5) +
        scale_fill_participants() +
        theme_garmin() +
        theme(axis.text.x=element_text(angle=45,hjust=1)) +
        labs(x=NULL, y="Sleep Midpoint (h)", fill=NULL,
             title="Midpoint distribution by participant")
    })

    output$midpoint_tsd <- renderPlot({
      df <- r_circ() %>% filter(!is.na(TSD))
      ggplot(df, aes(x=SleepMidpoint, y=TSD, colour=User_id)) +
        geom_point(alpha=0.3, size=1) +
        geom_smooth(method="lm", se=FALSE, linewidth=0.9) +
        scale_colour_participants() +
        facet_wrap(~User_id, ncol=4) +
        theme_garmin() +
        labs(x="Sleep Midpoint (h)", y="TSD (h)", colour=NULL,
             title="TSD vs Sleep Midpoint — per participant regression")
    })
  })
}
