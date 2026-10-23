#!/usr/bin/env Rscript
# Mixed models of PLAN.md §7, reported as a descriptive model-based summary (PREREGISTRATION.md §6; decisions 18,
# 20 and 21). They do not enter the attribution grade, which uses a channel-level test in attribution.py.
# Called by csdiar/analysis/attribution.py, which writes the input CSV and reads the JSON.
# Usage: Rscript glmm.R <data.csv> <out.json> <binomial|gaussian>
# Input columns: y, cond (1 = switch point, 0 = control), speaker, video, clip, set, pause_bin, pos_bin and the
# covariates d_f0, d_energy, d_rate, d_tilt, d_snr (absolute left-right changes). Rows with a missing y or
# covariate are dropped (complete cases; counts by condition before and after are reported). Covariates are
# z-scored over the rows used; pause_bin and pos_bin enter as factors when they have more than one level.
# Random-effects structures are tried in the pre-registered order; each drop is cumulative:
#   full      (1 + cond | speaker) + (1 | video/clip) + (1 | set)
#   no_slope  (1 | speaker) + (1 | video/clip) + (1 | set)
#   no_video  (1 | speaker) + (1 | clip) + (1 | set)        ("drop video": (1 | video/clip) becomes (1 | clip))
#   no_set    (1 | speaker) + (1 | clip)
# Each structure is fitted with the bobyqa optimizer and, if that fit errors or reports a convergence warning,
# again with Nelder_Mead. A structure is accepted when a fit has no error, no convergence warning and is not
# singular (lme4::isSingular, tol 1e-4); the first accepted structure is used. If none is accepted, the last
# fit that ran without error is reported and flagged (accepted = false); if no fit ran, failed = true.
# The chosen structure is then refitted without the covariates, so the change in the condition coefficient can be
# read off. Inference on the condition coefficient is a Wald z test (computed by the caller); binomial models use
# the Laplace approximation (glmer default).
suppressPackageStartupMessages({ library(lme4); library(jsonlite) })
a <- commandArgs(trailingOnly = TRUE)
stopifnot(length(a) == 3, a[3] %in% c("binomial", "gaussian"))
d <- read.csv(a[1], stringsAsFactors = FALSE)
family <- a[3]
covs <- c("d_f0", "d_energy", "d_rate", "d_tilt", "d_snr")
n_input <- as.list(table(factor(d$cond, levels = c(0, 1))))
d <- d[complete.cases(d[, c("y", "cond", covs)]), ]
n_complete <- as.list(table(factor(d$cond, levels = c(0, 1))))
for (v in covs) d[[v]] <- as.numeric(scale(d[[v]]))
for (v in c("speaker", "video", "clip", "set", "pause_bin", "pos_bin")) d[[v]] <- factor(d[[v]])
structures <- c(full = "(1 + cond | speaker) + (1 | video/clip) + (1 | set)",
                no_slope = "(1 | speaker) + (1 | video/clip) + (1 | set)",
                no_video = "(1 | speaker) + (1 | clip) + (1 | set)",
                no_set = "(1 | speaker) + (1 | clip)")
factors <- paste0(if (nlevels(d$pause_bin) > 1) " + pause_bin" else "", if (nlevels(d$pos_bin) > 1) " + pos_bin" else "")
fixed_cov <- paste0("y ~ cond + ", paste(covs, collapse = " + "), factors)
fixed_nocov <- paste0("y ~ cond", factors)

fit_one <- function(f, optimizer) {
  w <- character(0)
  m <- tryCatch(withCallingHandlers(
    if (family == "binomial") {
      glmer(as.formula(f), data = d, family = binomial,
            control = glmerControl(optimizer = optimizer, optCtrl = list(maxfun = 2e5)))
    } else {
      lmer(as.formula(f), data = d, REML = TRUE, control = lmerControl(optimizer = optimizer))
    },
    warning = function(x) { w <<- c(w, conditionMessage(x)); invokeRestart("muffleWarning") }),
    error = function(e) { w <<- c(w, paste("ERROR:", conditionMessage(e))); NULL })
  list(model = m, warnings = w, optimizer = optimizer,
       error = is.null(m), unconverged = any(grepl("converge", w, ignore.case = TRUE)),
       singular = if (is.null(m)) NA else isSingular(m, tol = 1e-4))
}
fit_structure <- function(f) {
  r <- fit_one(f, "bobyqa")
  if (r$error || r$unconverged) {
    r2 <- fit_one(f, "Nelder_Mead")
    if (!r2$error && (r$error || !r2$unconverged)) r <- r2
  }
  r
}
coef_row <- function(m, name = "cond") {
  cf <- tryCatch(summary(m)$coefficients, error = function(e) NULL)
  if (is.null(cf) || !(name %in% rownames(cf))) return(NULL)
  list(estimate = cf[name, 1], se = cf[name, 2], z = cf[name, 1] / cf[name, 2])
}
summ <- function(r) list(optimizer = r$optimizer, error = r$error, unconverged = r$unconverged,
                         singular = r$singular, warnings = r$warnings)

tried <- list(); chosen <- NULL; fit <- NULL; last_ran <- NULL
for (s in names(structures)) {
  r <- fit_structure(paste(fixed_cov, "+", structures[[s]]))
  tried[[s]] <- summ(r)
  if (!r$error) last_ran <- list(name = s, fit = r)
  if (!r$error && !r$unconverged && isFALSE(r$singular)) { chosen <- s; fit <- r; break }
}
out <- list(family = family, n_input = n_input, n_complete = n_complete, n_switch = sum(d$cond == 1),
            fixed_formula = fixed_cov, tried = tried, R = R.version.string, lme4 = format(packageVersion("lme4")))
if (is.null(chosen) && is.null(last_ran)) {
  out$failed <- TRUE
} else {
  out$failed <- FALSE
  out$accepted <- !is.null(chosen)
  if (is.null(chosen)) { chosen <- last_ran$name; fit <- last_ran$fit }
  nocov <- fit_one(paste(fixed_nocov, "+", structures[[chosen]]), fit$optimizer)
  out$structure <- chosen
  out$structure_formula <- structures[[chosen]]
  out$cond <- coef_row(fit$model)
  out$cond_without_covariates <- if (nocov$error) NULL else coef_row(nocov$model)
  out$without_covariates_fit <- summ(nocov)
  out$covariates <- lapply(setNames(covs, covs), function(v) coef_row(fit$model, v))
  out$n_levels <- lapply(d[, c("speaker", "video", "clip", "set")], nlevels)
}
write_json(out, a[2], auto_unbox = TRUE, digits = NA, pretty = TRUE, null = "null")
