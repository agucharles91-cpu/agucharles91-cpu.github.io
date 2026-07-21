# renv_setup.R
# Run this script ONCE in a fresh R session to initialise the reproducible
# package environment. After this, use renv::restore() on any other machine.
#
# Step 1 — Install renv itself (skip if already installed)
if (!requireNamespace("renv", quietly = TRUE)) {
  install.packages("renv")
}

# Step 2 — Initialise renv for this project (creates renv/ and renv.lock)
# This will prompt you to confirm; type y and press Enter.
renv::init(bioconductor = FALSE)

# Step 3 — Install all required packages
pkgs <- c(
  "shiny",
  "bslib",
  "tidyverse",     # includes lubridate, dplyr, ggplot2, readr, etc.
  "plotly",
  "DT",
  "shinycssloaders",
  "zoo",
  "scales",
  "mgcv",
  "ggcorrplot",
  "lme4",
  "lmerTest",
  "testthat"
)
install.packages(pkgs)

# Step 4 — Snapshot the environment to renv.lock
# This records every package + version so others can restore exactly.
renv::snapshot()

# Step 5 — Verify
cat("\n--- Environment snapshot complete ---\n")
cat("Commit renv.lock to version control.\n")
cat("Others restore with: renv::restore()\n\n")

# Step 6 — Record sessionInfo for README
si <- capture.output(sessionInfo())
writeLines(si, "session_info.txt")
cat("sessionInfo() written to session_info.txt\n")
