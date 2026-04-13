# helpers/plot_themes.R  — shared ggplot2 theme + palette
library(ggplot2)

# ── Palette ─────────────────────────────────────────────────────────────────
PALETTE_14 <- c(
  P001="#2563EB", P002="#DC2626", P003="#16A34A", P004="#9333EA",
  P005="#EA580C", P006="#0D9488", P008="#DB2777", P009="#CA8A04",
  P010="#7C3AED", P011="#0891B2", P012="#65A30D", P013="#BE123C",
  P014="#1D4ED8", P015="#B45309"
)

STAGE_COLS <- c(
  Deep  = "#1B3A6B", Light = "#60A5FA", REM = "#7C3AED", Awake = "#FCA5A5"
)

COHORT_COLS <- c(
  full            = "#16A34A",
  no_sleep_stages = "#94A3B8"
)

ACTIVITY_COLS <- c(
  running          = "#2563EB", treadmill_running = "#3B82F6",
  walking          = "#16A34A", strength_training = "#EA580C",
  cycling          = "#9333EA", indoor_cycling    = "#A855F7",
  hiit             = "#DC2626", yoga              = "#0D9488",
  other            = "#94A3B8"
)

# ── Base theme ───────────────────────────────────────────────────────────────
theme_garmin <- function(base_size = 13) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.background  = element_rect(fill = "#F8FAFC", colour = NA),
      panel.background = element_rect(fill = "#FFFFFF",  colour = NA),
      panel.grid.major = element_line(colour = "#E2E8F0", linewidth = 0.4),
      panel.grid.minor = element_blank(),
      panel.border     = element_rect(colour = "#CBD5E1", fill = NA, linewidth = 0.5),
      axis.text        = element_text(colour = "#475569", size = base_size * 0.8),
      axis.title       = element_text(colour = "#1E293B", size = base_size * 0.9, face = "bold"),
      plot.title       = element_text(colour = "#1B3A6B", size = base_size * 1.2, face = "bold", hjust = 0),
      plot.subtitle    = element_text(colour = "#475569", size = base_size * 0.85, hjust = 0),
      legend.background= element_rect(fill = "#F8FAFC", colour = "#E2E8F0"),
      legend.key       = element_rect(fill = NA),
      strip.background = element_rect(fill = "#1B3A6B", colour = NA),
      strip.text       = element_text(colour = "white", face = "bold", size = base_size * 0.85),
      plot.caption     = element_text(colour = "#94A3B8", size = base_size * 0.7)
    )
}

# Convenience: participant colour scale
scale_colour_participants <- function(...) {
  scale_colour_manual(values = PALETTE_14, ...)
}
scale_fill_participants <- function(...) {
  scale_fill_manual(values = PALETTE_14, ...)
}
