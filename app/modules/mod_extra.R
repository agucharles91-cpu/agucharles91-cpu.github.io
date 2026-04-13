# modules/mod_vo2max.R ────────────────────────────────────────────────────────

mod_vo2_ui <- function(id) {
  ns <- NS(id)
  tagList(
    fluidRow(
      column(3,
        box(width=12, title="Controls", status="primary", solidHeader=TRUE,
          radioButtons(ns("vo2_src"), "Source filter",
            c("Observed + LOCF"="all","Observed only"="observed","LOCF only"="locf"),
            "all"),
          checkboxInput(ns("smooth"), "Add trend line", TRUE),
          sliderInput(ns("smooth_span"), "Span", 0.1, 0.9, 0.4),
          checkboxInput(ns("changepoint"), "Mark change-points (±2 SD)", FALSE)
        )
      ),
      column(9,
        tabBox(width=12,
          tabPanel("Trajectories",
            plotOutput(ns("vo2_lines"), height="380px")
          ),
          tabPanel("Distribution",
            plotOutput(ns("vo2_box"), height="300px"),
            plotOutput(ns("vo2_density"), height="240px")
          ),
          tabPanel("Annual Change",
            plotOutput(ns("vo2_annual"), height="300px"),
            tableOutput(ns("delta_table"))
          ),
          tabPanel("VO2max vs Sleep",
            plotOutput(ns("vo2_sleep"), height="360px")
          )
        )
      )
    )
  )
}

mod_vo2_server <- function(id, r_data) {
  moduleServer(id, function(input, output, session) {

    r_vo2 <- reactive({
      df <- r_data() %>% filter(!is.na(VO2max_imputed_final))
      if (input$vo2_src == "observed") df <- df %>% filter(vo2_source == "observed")
      if (input$vo2_src == "locf")     df <- df %>% filter(grepl("LOCF", vo2_source))
      df
    })

    output$vo2_lines <- renderPlot({
      df <- r_vo2()
      p <- ggplot(df, aes(x=calendar_date, y=VO2max_imputed_final, colour=User_id))
      if (input$changepoint) {
        anom <- df %>% group_by(User_id) %>% arrange(calendar_date) %>%
          mutate(flag=flag_anomalies(VO2max_imputed_final)) %>% ungroup() %>% filter(flag)
        p <- p + geom_point(data=anom, shape=4, size=3, colour="#DC2626", stroke=1.5)
      }
      p + geom_line(aes(alpha=ifelse(vo2_source=="observed",1,0.4)),linewidth=0.7) +
        { if(input$smooth) geom_smooth(method="loess",span=input$smooth_span,se=FALSE,linewidth=1.2) } +
        scale_colour_participants() +
        scale_alpha_identity() +
        scale_x_date(date_breaks="6 months", date_labels="%b\n%Y") +
        facet_wrap(~User_id, ncol=4, scales="free_y") +
        theme_garmin() +
        labs(x=NULL, y="VO₂max (ml/kg/min)", colour=NULL,
             title="VO₂max trajectories (faded = LOCF-imputed)",
             caption="×  = anomaly (>2 SD from rolling mean)")
    })

    output$vo2_box <- renderPlot({
      df <- r_vo2() %>% filter(vo2_source == "observed")
      ggplot(df, aes(x=User_id, y=VO2max_imputed_final, fill=User_id)) +
        geom_boxplot(alpha=0.7, outlier.alpha=0.4, outlier.size=0.8) +
        scale_fill_participants() +
        theme_garmin() +
        theme(axis.text.x=element_text(angle=45,hjust=1)) +
        labs(x=NULL, y="VO₂max (ml/kg/min)", fill=NULL,
             title="VO₂max distribution (observed values only)")
    })

    output$vo2_density <- renderPlot({
      df <- r_vo2() %>% filter(vo2_source == "observed")
      ggplot(df, aes(x=VO2max_imputed_final, fill=User_id, colour=User_id)) +
        geom_density(alpha=0.3, linewidth=0.7) +
        scale_fill_participants() + scale_colour_participants() +
        theme_garmin() +
        labs(x="VO₂max (ml/kg/min)", y="Density", fill=NULL, colour=NULL)
    })

    output$vo2_annual <- renderPlot({
      df <- r_vo2() %>% filter(vo2_source == "observed") %>%
        group_by(User_id, year) %>%
        summarise(mean_vo2=mean(VO2max_imputed_final,na.rm=TRUE), .groups="drop")
      ggplot(df, aes(x=year, y=mean_vo2, colour=User_id, group=User_id)) +
        geom_line(linewidth=1) + geom_point(size=3) +
        scale_colour_participants() +
        theme_garmin() +
        labs(x="Year", y="Mean VO₂max (ml/kg/min)", colour=NULL,
             title="Annual mean VO₂max per participant")
    })

    output$delta_table <- renderTable({
      r_vo2() %>% filter(vo2_source == "observed") %>%
        group_by(User_id, year) %>%
        summarise(mean=round(mean(VO2max_imputed_final,na.rm=TRUE),1), .groups="drop") %>%
        tidyr::pivot_wider(names_from=year, values_from=mean, names_prefix="Y") %>%
        rename(Participant=User_id)
    }, striped=TRUE, hover=TRUE, bordered=TRUE)

    output$vo2_sleep <- renderPlot({
      df <- r_data() %>%
        filter(!is.na(VO2max_imputed_final), !is.na(TSD),
               analysis_cohort %in% c("full","legacy_staging"))
      ggplot(df, aes(x=TSD, y=VO2max_imputed_final, colour=User_id)) +
        geom_point(alpha=0.3, size=0.9) +
        geom_smooth(method="lm", se=FALSE, linewidth=0.9) +
        scale_colour_participants() + facet_wrap(~User_id,ncol=4) +
        theme_garmin() +
        labs(x="TSD (h)", y="VO₂max (ml/kg/min)", colour=NULL,
             title="VO₂max vs Total Sleep Duration")
    })
  })
}


# modules/mod_individual.R ────────────────────────────────────────────────────

mod_individual_ui <- function(id) {
  ns <- NS(id)
  tagList(
    fluidRow(
      column(3,
        box(width=12, title="Controls", status="primary", solidHeader=TRUE,
          selectInput(ns("uid"), "Participant", choices=NULL),
          selectInput(ns("primary"), "Primary metric",
            choices=c("TSD"="TSD","SE"="SE","VO2max"="VO2max_imputed_final",
                      "RHR"="DailyRestingHeartRate_clean","Session Load"="SessionLoad",
                      "Sleep Midpoint"="SleepMidpoint"), "TSD"),
          selectInput(ns("secondary"), "Secondary metric",
            choices=c("RHR"="DailyRestingHeartRate_clean","VO2max"="VO2max_imputed_final",
                      "Session Load"="SessionLoad","BB net"="BB_net",
                      "TSD"="TSD"), "DailyRestingHeartRate_clean"),
          sliderInput(ns("roll"), "Rolling window (days)", 7, 90, 30),
          checkboxInput(ns("anomaly"), "Highlight anomalies (±2 SD)", TRUE),
          hr(),
          valueBoxOutput(ns("kpi_tsd"),  width=12),
          valueBoxOutput(ns("kpi_se"),   width=12),
          valueBoxOutput(ns("kpi_vo2"),  width=12),
          valueBoxOutput(ns("kpi_rhr"),  width=12)
        )
      ),
      column(9,
        tabBox(width=12,
          tabPanel("Dual-axis Timeline",
            plotOutput(ns("dual_ts"), height="360px")
          ),
          tabPanel("Stage Composition",
            plotOutput(ns("stage_over_time"), height="320px"),
            plotOutput(ns("stage_pie"), height="240px")
          ),
          tabPanel("Activity Profile",
            plotOutput(ns("act_bar"), height="280px"),
            plotOutput(ns("act_weekly"), height="240px")
          ),
          tabPanel("Anomalies",
            plotOutput(ns("anomaly_plot"), height="360px"),
            DTOutput(ns("anomaly_table"))
          )
        )
      )
    )
  )
}

mod_individual_server <- function(id, r_data) {
  moduleServer(id, function(input, output, session) {

    observe({
      uids <- sort(unique(r_data()$User_id))
      updateSelectInput(session, "uid", choices=uids, selected=uids[1])
    })

    r_p <- reactive({
      r_data() %>% filter(User_id == input$uid) %>% arrange(calendar_date)
    })

    # KPI value boxes
    output$kpi_tsd <- renderValueBox({
      v <- round(mean(r_p()$TSD,na.rm=TRUE),2)
      valueBox(paste0(v,"h"), "Mean TSD", icon=icon("moon"), color="blue", width=NULL)
    })
    output$kpi_se <- renderValueBox({
      v <- round(mean(r_p()$SE,na.rm=TRUE),3)
      valueBox(v, "Mean SE", icon=icon("percent"), color="teal", width=NULL)
    })
    output$kpi_vo2 <- renderValueBox({
      v <- round(mean(r_p()$VO2max_imputed_final,na.rm=TRUE),1)
      valueBox(paste0(v," ml/kg/min"), "Mean VO₂max", icon=icon("heartbeat"), color="green", width=NULL)
    })
    output$kpi_rhr <- renderValueBox({
      v <- round(mean(r_p()$DailyRestingHeartRate_clean,na.rm=TRUE),1)
      valueBox(paste0(v," bpm"), "Mean RHR", icon=icon("heart"), color="red", width=NULL)
    })

    output$dual_ts <- renderPlot({
      df <- r_p() %>%
        filter(!is.na(.data[[input$primary]]) | !is.na(.data[[input$secondary]])) %>%
        mutate(
          p_roll = roll_mean(.data[[input$primary]],   k=input$roll),
          s_roll = roll_mean(.data[[input$secondary]], k=input$roll)
        )

      # Normalise secondary to primary scale for dual axis
      p_range <- range(df[[input$primary]], na.rm=TRUE)
      s_range <- range(df[[input$secondary]], na.rm=TRUE)
      scale_to <- function(x) {
        (x - s_range[1]) / diff(s_range) * diff(p_range) + p_range[1]
      }

      p <- ggplot(df, aes(x=calendar_date)) +
        geom_point(aes(y=.data[[input$primary]]), colour="#2563EB", alpha=0.2, size=0.8) +
        geom_line(aes(y=p_roll), colour="#2563EB", linewidth=1.1) +
        geom_line(aes(y=scale_to(s_roll)), colour="#DC2626", linewidth=1.1, linetype="dashed")

      if (input$anomaly) {
        anom <- df %>% mutate(flag=flag_anomalies(.data[[input$primary]],window=input$roll)) %>% filter(flag, !is.na(flag))
        if (nrow(anom) > 0) p <- p + geom_point(data=anom, aes(y=.data[[input$primary]]), shape=4,size=3,colour="#DC2626",stroke=1.5)
      }

      p + scale_x_date(date_breaks="6 months",date_labels="%b\n%Y") +
        scale_y_continuous(
          name = metric_labels[input$primary],
          sec.axis = sec_axis(~ (. - p_range[1])/diff(p_range)*diff(s_range)+s_range[1],
                              name = metric_labels[input$secondary])
        ) +
        theme_garmin() +
        theme(axis.title.y.left  = element_text(colour="#2563EB"),
              axis.title.y.right = element_text(colour="#DC2626")) +
        labs(x=NULL, title=paste(input$uid, "—", input$primary, "(blue) vs", input$secondary, "(red)"),
             subtitle=paste0(input$roll, "-day rolling mean  |  ✕ = anomaly >2 SD"))
    })

    output$stage_over_time <- renderPlot({
      df <- r_p() %>%
        filter(analysis_cohort == "full", !is.na(DeepSleepProp)) %>%
        select(calendar_date, Deep=DeepSleepProp, Light=LightSleepProp,
               REM=REMSleepProp_final, Awake=AwakeSleepProp) %>%
        tidyr::pivot_longer(-calendar_date, names_to="Stage", values_to="prop") %>%
        mutate(Stage=factor(Stage,levels=c("Awake","REM","Light","Deep")))

      ggplot(df, aes(x=calendar_date, y=prop, fill=Stage)) +
        geom_area(position="stack", alpha=0.85) +
        scale_fill_manual(values=STAGE_COLS) +
        scale_y_continuous(labels=scales::percent) +
        scale_x_date(date_breaks="6 months",date_labels="%b\n%Y") +
        theme_garmin() +
        labs(x=NULL,y="Proportion",fill="Stage",
             title=paste(input$uid,"— Sleep stage composition over time (4-stage nights only)"))
    })

    output$stage_pie <- renderPlot({
      df <- r_p() %>% filter(analysis_cohort=="full") %>%
        summarise(Deep=mean(DeepSleepProp,na.rm=TRUE),Light=mean(LightSleepProp,na.rm=TRUE),
                  REM=mean(REMSleepProp_final,na.rm=TRUE),Awake=mean(AwakeSleepProp,na.rm=TRUE)) %>%
        tidyr::pivot_longer(everything(),names_to="Stage",values_to="prop")
      ggplot(df, aes(x="",y=prop,fill=Stage)) +
        geom_col(width=1,colour="white") + coord_polar(theta="y") +
        scale_fill_manual(values=STAGE_COLS) +
        geom_text(aes(label=paste0(Stage,"\n",scales::percent(prop,accuracy=0.1))),
                  position=position_stack(vjust=0.5),colour="white",size=4,fontface="bold") +
        theme_void() + labs(fill=NULL, title=paste(input$uid,"— Mean stage proportions"))
    })

    output$act_bar <- renderPlot({
      df <- r_p() %>% filter(activity_day, !is.na(ActivitiesType)) %>%
        count(ActivitiesType) %>% arrange(desc(n))
      ggplot(df, aes(x=reorder(ActivitiesType,n), y=n, fill=ActivitiesType)) +
        geom_col(alpha=0.85) +
        scale_fill_manual(values=ACTIVITY_COLS, na.value="#94A3B8") +
        coord_flip() + theme_garmin() + theme(legend.position="none") +
        labs(x=NULL,y="Sessions",title=paste(input$uid,"— Activity type frequency"))
    })

    output$act_weekly <- renderPlot({
      df <- r_p() %>% filter(activity_day, !is.na(ActivitiesType)) %>%
        count(weekday, ActivitiesType)
      ggplot(df, aes(x=weekday, y=n, fill=ActivitiesType)) +
        geom_col(position="stack", alpha=0.85) +
        scale_fill_manual(values=ACTIVITY_COLS, na.value="#94A3B8") +
        theme_garmin() + theme(axis.text.x=element_text(angle=30,hjust=1)) +
        labs(x=NULL, y="Sessions", fill="Activity", title="Weekly activity pattern")
    })

    output$anomaly_plot <- renderPlot({
      df <- r_p() %>%
        filter(!is.na(.data[[input$primary]])) %>%
        mutate(flag=flag_anomalies(.data[[input$primary]], window=input$roll),
               roll_m=roll_mean(.data[[input$primary]], k=input$roll),
               roll_s=roll_sd(.data[[input$primary]], k=input$roll))

      ggplot(df, aes(x=calendar_date)) +
        geom_ribbon(aes(ymin=roll_m-2*roll_s, ymax=roll_m+2*roll_s), fill="#DBEAFE",alpha=0.5) +
        geom_line(aes(y=roll_m), colour="#2563EB", linewidth=0.8) +
        geom_point(aes(y=.data[[input$primary]], colour=flag), size=1.5, alpha=0.7) +
        scale_colour_manual(values=c("FALSE"="#94A3B8","TRUE"="#DC2626"),
                            labels=c("Normal","Anomaly"), na.value="#94A3B8") +
        scale_x_date(date_breaks="6 months",date_labels="%b\n%Y") +
        theme_garmin() +
        labs(x=NULL, y=metric_labels[input$primary], colour=NULL,
             title=paste(input$uid,"—", input$primary, "anomaly detection"),
             subtitle=paste0("Blue band = ±2 SD from ",input$roll,"-day rolling mean"))
    })

    output$anomaly_table <- renderDT({
      r_p() %>%
        filter(!is.na(.data[[input$primary]])) %>%
        mutate(flag=flag_anomalies(.data[[input$primary]], window=input$roll)) %>%
        filter(flag) %>%
        select(calendar_date, User_id, value=all_of(input$primary),
               weekday, activity_day, staging_mode) %>%
        rename(Date=calendar_date, Participant=User_id,
               Value=value, Weekday=weekday,
               Activity=activity_day, Staging=staging_mode) %>%
        arrange(desc(Date))
    }, options=list(pageLength=8, scrollX=TRUE), rownames=FALSE)
  })
}


# modules/mod_explorer.R ──────────────────────────────────────────────────────

mod_explorer_ui <- function(id) {
  ns <- NS(id)
  tagList(
    fluidRow(
      column(3,
        box(width=12, title="Column filter", status="primary", solidHeader=TRUE,
          checkboxGroupInput(ns("col_groups"), "Show column groups",
            choices=c("Identity"="id","Sleep"="sleep","Activity"="activity",
                      "Recovery"="recovery","Provenance"="prov"),
            selected=c("id","sleep","activity","recovery")),
          hr(),
          downloadButton(ns("dl_csv"), "Download filtered CSV", class="btn-success btn-sm"),
          br(),br(),
          downloadButton(ns("dl_rds"), "Download as .rds", class="btn-info btn-sm")
        )
      ),
      column(9,
        box(width=12, title="Data Explorer", status="primary",
          fluidRow(
            column(4, selectInput(ns("filter_cohort"), "Cohort filter",
              c("All"="","full","legacy_staging","tsd_imputed","no_sleep_stages"), "")),
            column(4, selectInput(ns("filter_uid"), "Participant", c("All"=""), "All")),
            column(4, selectInput(ns("filter_staging"), "Staging mode", c("All"=""), "All"))
          ),
          DTOutput(ns("table"))
        ),
        fluidRow(
          box(width=6, title="Missingness summary", status="info",
              plotOutput(ns("miss_plot"), height="260px")),
          box(width=6, title="Column data types", status="info",
              tableOutput(ns("dtype_table")))
        )
      )
    )
  )
}

mod_explorer_server <- function(id, r_data) {
  moduleServer(id, function(input, output, session) {

    # Dynamic filter choices
    observe({
      uids <- c("All"="", sort(unique(r_data()$User_id)))
      stagings <- c("All"="", sort(unique(as.character(r_data()$staging_mode))))
      updateSelectInput(session, "filter_uid", choices=uids)
      updateSelectInput(session, "filter_staging", choices=stagings)
    })

    COL_GROUPS <- list(
      id = c("User_id","Age","Sex","Height_cm","Weight_kg","calendar_date",
             "weekday","is_weekend","year","month","analysis_cohort","staging_mode"),
      sleep = c("TSD","SE","SF","TIB_h","SleepMidpoint","SleepMidpoint_sd","SleepScore",
                "DeepSleepProp","LightSleepProp","REMSleepProp_final","AwakeSleepProp",
                "SleepTotalSleepSeconds","SleepDeepSleepSeconds",
                "SleepLightSleepSeconds","SleepRemSleepSeconds","SleepAwakeSleepSeconds"),
      activity = c("activity_day","ActivitiesType","ActivitiesDurationMinutes",
                   "ActivitiesDistance_km","ActivitiesAvgHr","ActivitiesMaxHr",
                   "SessionLoad","RelativeIntensity","EfficiencyIndex",
                   "ActivitiesAvgSpeed_kmh","ActivitiesVO2MaxValue"),
      recovery = c("DailyRestingHeartRate_clean","RHR_dev","VO2max_imputed_final",
                   "BB_net","DailyBodyBattery.chargedValue","DailyBodyBattery.drainedValue",
                   "DailyRespiration.avgWakingRespirationValue",
                   "DailyTotalSteps","DailyTotalKilocalories"),
      prov = c("rem_source","tib_source","ss_source","se_source","sf_source",
               "sleep_prop_source","sm_source","vo2_source","BB_analysis_flag","resp_analysis_flag")
    )

    r_filtered <- reactive({
      df <- r_data()
      if (nzchar(input$filter_cohort)) df <- df %>% filter(analysis_cohort == input$filter_cohort)
      if (nzchar(input$filter_uid))    df <- df %>% filter(User_id == input$filter_uid)
      if (nzchar(input$filter_staging))df <- df %>% filter(staging_mode == input$filter_staging)
      sel_cols <- unlist(COL_GROUPS[input$col_groups])
      sel_cols <- sel_cols[sel_cols %in% names(df)]
      df %>% select(all_of(sel_cols))
    })

    output$table <- renderDT({
      r_filtered()
    }, options=list(pageLength=15,scrollX=TRUE,dom="lfrtip"), rownames=FALSE, filter="top")

    output$dl_csv <- downloadHandler(
      filename=function() paste0("garmin_filtered_",Sys.Date(),".csv"),
      content=function(file) readr::write_csv(r_filtered(), file)
    )
    output$dl_rds <- downloadHandler(
      filename=function() paste0("garmin_filtered_",Sys.Date(),".rds"),
      content=function(file) saveRDS(r_filtered(), file)
    )

    output$miss_plot <- renderPlot({
      df <- r_filtered()
      miss <- sapply(df, function(x) 100*mean(is.na(x)))
      miss_df <- data.frame(col=names(miss), pct=miss) %>%
        filter(pct > 0) %>% arrange(desc(pct)) %>% head(20)
      if (nrow(miss_df) == 0) {
        ggplot() + annotate("text",x=0.5,y=0.5,label="No missing values in selection",size=5) +
          theme_void()
      } else {
        ggplot(miss_df, aes(x=reorder(col,pct), y=pct, fill=pct)) +
          geom_col() +
          scale_fill_gradient(low="#FCD34D", high="#DC2626") +
          coord_flip() + theme_garmin() +
          theme(legend.position="none") +
          labs(x=NULL, y="% missing", title="Missingness (top 20 cols)")
      }
    })

    output$dtype_table <- renderTable({
      df <- r_filtered()
      data.frame(
        Column=names(df),
        Type=sapply(df, function(x) class(x)[1]),
        `Non-NA`=sapply(df, function(x) sum(!is.na(x))),
        `Missing%`=paste0(round(100*sapply(df,function(x) mean(is.na(x))),1),"%")
      ) %>% head(20)
    }, striped=TRUE, spacing="s", bordered=TRUE)
  })
}
