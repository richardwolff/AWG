#!/usr/bin/env Rscript
# Per-species ZERO-INFLATED beta GLMM via glmmTMB.
#
# Zero-inflated counterpart of sweep_pairtype_per_species_glmmTMB.R. That script
# squeezes exact-zero Jaccards to eps = 1e-3; here the zeros keep their own
# Bernoulli component:
#
#   P(y = 0)             = pi        logit(pi) = intercept only
#   y | y > 0 ~ Beta(mu, phi)        logit(mu) = beta * pair_type + gamma * FST
#                                                + (1|pop_i) + (1|pop_j)
#
# The zero component is INTERCEPT-ONLY on purpose. Per-species sample sizes
# (25-280 observations) will not support pair-type covariates in both components,
# so the pair-type effect is estimated on the conditional mean -- i.e. "among
# pairs that share any sweeps at all, do same-lifestyle pairs share more?" --
# with a constant per-species rate of structural zeros divided out. That is a
# narrower question than the pooled model answers, and is stated as such.
#
# Species with no exact zeros are fit without a zi component (it would be
# unidentifiable); the `zi_used` column records which species got one.
#
# Population set matches the pooled model (20 curated populations) so the two
# analyses in the section describe the same data.
#
# Tests, as in the non-zi script:
#   - omnibus pair_type LRT (drop both indicators; df = 2, or 1 if only one present)
#   - per-contrast LRT for Within-ind vs Cross and Within-non-ind vs Cross
#   - BH-FDR across species, separately per test.
#
# INPUTS. The model is unchanged; only where the data comes from has moved.
# Instead of a pre-built sweep_pairtype_data.csv this assembles the pair table
# itself, from three sources:
#
#   analysis/jaccard_species/<species>.csv   one square population x population
#                                            Jaccard matrix per species, as
#                                            written by jaccard_sweeps.py
#   fst_pairwise_goodstrain.csv              species, pop1, pop2, fst
#   metadata/all_strains_metadata.txt        lifestyle of each population, from
#                                            which pair_type is derived
#
# The assembled table is written out alongside the results so the model input
# can be inspected, and so other scripts can consume it the way they consumed
# sweep_pairtype_data.csv.

suppressPackageStartupMessages({
  library(glmmTMB)
  library(readr)
  library(dplyr)
})

MIN_OBS         <- 25
MIN_PAIR_TYPES  <- 2
MIN_PER_TYPE    <- 3
UPPER_EPS       <- 1e-4

JACCARD_DIR <- 'analysis/jaccard_species'
FST_F       <- 'fst_pairwise_goodstrain.csv'
META_F      <- 'metadata/all_strains_metadata.txt'
POPS_F      <- 'metadata/good_populations.txt'
PAIRS_OUT   <- 'analysis/sweep_pairtype_data_goodstrain.csv'

# --plain refits the SAME species/population subset as a plain beta with the old
# eps = 1e-3 squeeze, so the zero-inflation effect can be separated from the
# effect of the population filter when comparing against the published numbers.
PLAIN <- '--plain' %in% commandArgs(trailingOnly = TRUE)
OUT_F <- if (PLAIN)
  'analysis/sweep_pairtype_per_species_plain20_glmmTMB.csv' else
  'analysis/sweep_pairtype_per_species_zibeta_glmmTMB.csv'

# Populations to analyse. Taken from good_populations.txt when it is there, so
# this tracks the rest of the pipeline; otherwise the 20 curated populations the
# pooled model used, which is what the published numbers were fit on.
keep_pops <- if (file.exists(POPS_F)) {
  trimws(readLines(POPS_F))
} else {
  c('AG','Austria','BF','China','DM','Denmark','Fiji','Finland',
    'France','GH','Germany','Israel','KY','Kazakhstan','Netherlands',
    'SW','Spain','Sweden','UK','US')
}
keep_pops <- keep_pops[nzchar(keep_pops)]
cat(sprintf("Populations considered: %d (%s)\n", length(keep_pops),
            if (file.exists(POPS_F)) POPS_F else "built-in list"))

# ── Jaccard: one square matrix per species ────────────────────────────────────
# Written by Python, so missing entries are the string "nan" rather than R's
# NaN; they are the diagonal and any pair whose sweep union was empty, and are
# dropped below.
read_jaccard <- function(path) {
  m <- as.matrix(read.csv(path, row.names = 1, check.names = FALSE,
                          na.strings = c('NA', 'nan', 'NaN', '')))
  pops <- rownames(m)
  if (length(pops) < 2) return(NULL)
  idx <- which(upper.tri(m), arr.ind = TRUE)          # each unordered pair once
  data.frame(species = tools::file_path_sans_ext(basename(path)),
             pop_i   = pops[idx[, 1]],
             pop_j   = pops[idx[, 2]],
             jaccard = as.numeric(m[idx]),
             stringsAsFactors = FALSE)
}

jac_files <- list.files(JACCARD_DIR, pattern = '\\.csv$', full.names = TRUE)
if (!length(jac_files)) stop(sprintf("No Jaccard matrices in %s", JACCARD_DIR))

df <- bind_rows(lapply(jac_files, read_jaccard))
cat(sprintf("Jaccard: %d pairs from %d species in %s\n",
            nrow(df), length(unique(df$species)), JACCARD_DIR))

n_nan <- sum(is.na(df$jaccard))
df <- df[!is.na(df$jaccard), ]
if (n_nan) cat(sprintf("  dropped %d pairs with an undefined Jaccard\n", n_nan))

# ── FST, joined on the unordered pair ─────────────────────────────────────────
pair_key <- function(species, a, b) paste(species, pmin(a, b), pmax(a, b), sep = '|')

fst <- read_csv(FST_F, show_col_types = FALSE)
fst$key <- pair_key(fst$species, fst$pop1, fst$pop2)
fst <- fst[!duplicated(fst$key), c('key', 'fst')]

df$key <- pair_key(df$species, df$pop_i, df$pop_j)
df <- left_join(df, fst, by = 'key')

n_pre <- nrow(df)
df <- df[!is.na(df$fst), ]
cat(sprintf("FST: matched %d of %d pairs from %s\n", nrow(df), n_pre, FST_F))

# ── pair_type from the lifestyle of each population ───────────────────────────
life <- read_csv(META_F, show_col_types = FALSE) %>%
  select(Population, lifestyle_population) %>%
  filter(!is.na(lifestyle_population)) %>%
  distinct()
lifemap <- setNames(life$lifestyle_population, life$Population)

li <- lifemap[df$pop_i]
lj <- lifemap[df$pop_j]
df$pair_type <- ifelse(is.na(li) | is.na(lj), NA_character_,
                ifelse(li != lj, 'Cross-lifestyle',
                ifelse(li == 'ind', 'Within-ind', 'Within-non-ind')))

n_pre <- nrow(df)
df <- df[!is.na(df$pair_type), ]
if (nrow(df) < n_pre)
  cat(sprintf("Lifestyle: dropped %d pairs with an unclassified population\n",
              n_pre - nrow(df)))

n_all <- nrow(df)
df <- df[df$pop_i %in% keep_pops & df$pop_j %in% keep_pops, ]
cat(sprintf("Population filter: kept %d of %d rows\n", nrow(df), n_all))

dir.create(dirname(PAIRS_OUT), showWarnings = FALSE, recursive = TRUE)
write_csv(df[, c('species', 'pop_i', 'pop_j', 'jaccard', 'fst', 'pair_type')],
          PAIRS_OUT)
cat(sprintf("Model input written to %s\n", PAIRS_OUT))

# zeros stay exact (zi models them); only the upper boundary is squeezed.
# Under --plain both tails are squeezed to 1e-3, reproducing the old response.
df$y <- if (PLAIN) pmin(pmax(df$jaccard, 1e-3), 1 - 1e-3) else
                   pmin(df$jaccard, 1 - UPPER_EPS)
cat(sprintf("Mode: %s\n", if (PLAIN) "PLAIN beta (eps = 1e-3 squeeze)" else
                            "ZERO-INFLATED beta"))
cat(sprintf("Exact zeros kept for the zi component: %d (%.1f%%); exact ones squeezed: %d\n",
            sum(df$jaccard == 0), 100 * mean(df$jaccard == 0), sum(df$jaccard == 1)))

df$pair_type <- factor(df$pair_type,
                       levels = c('Cross-lifestyle', 'Within-ind', 'Within-non-ind'))
df$within_ind     <- as.integer(df$pair_type == 'Within-ind')
df$within_non_ind <- as.integer(df$pair_type == 'Within-non-ind')

species_list <- sort(unique(df$species))
results <- list()

# Fit with the RE fallback cascade of the non-zi script, carrying ziformula.
fit_one <- function(formula_full, formula_null, sub, ziform) {
  two_re <- function() {
    list(full = glmmTMB(formula_full, data = sub, ziformula = ziform,
                        family = beta_family(link = 'logit'),
                        map = list(theta = factor(c(1, 1)))),
         null = glmmTMB(formula_null, data = sub, ziformula = ziform,
                        family = beta_family(link = 'logit'),
                        map = list(theta = factor(c(1, 1)))),
         method = 'eqvar_2RE')
  }
  one_re <- function() {
    ff <- update(formula_full, . ~ . - (1 | pop_j))
    fn <- update(formula_null, . ~ . - (1 | pop_j))
    list(full = glmmTMB(ff, data = sub, ziformula = ziform,
                        family = beta_family(link = 'logit')),
         null = glmmTMB(fn, data = sub, ziformula = ziform,
                        family = beta_family(link = 'logit')),
         method = 'pop_i_only')
  }
  no_re <- function() {
    ff <- update(formula_full, . ~ . - (1 | pop_j) - (1 | pop_i))
    fn <- update(formula_null, . ~ . - (1 | pop_j) - (1 | pop_i))
    list(full = glmmTMB(ff, data = sub, ziformula = ziform,
                        family = beta_family(link = 'logit')),
         null = glmmTMB(fn, data = sub, ziformula = ziform,
                        family = beta_family(link = 'logit')),
         method = 'no_RE')
  }
  ok <- function(o) {
    if (is.null(o)) return(FALSE)
    conv <- identical(o$full$fit$convergence, 0L) &&
            identical(o$null$fit$convergence, 0L)
    finite <- is.finite(as.numeric(logLik(o$full))) &&
              is.finite(as.numeric(logLik(o$null)))
    conv && finite
  }
  for (f in list(two_re, one_re, no_re)) {
    out <- tryCatch(f(), error = function(e) NULL, warning = function(w) NULL)
    if (ok(out)) return(out)
  }
  NULL
}

cat(sprintf("Total species in dataset: %d\n", length(species_list)))

for (sp in species_list) {
  sub <- df[df$species == sp, ]
  if (nrow(sub) < MIN_OBS) next
  pt_counts <- table(sub$pair_type)
  if (sum(pt_counts >= MIN_PER_TYPE) < MIN_PAIR_TYPES) next
  has_WI <- pt_counts['Within-ind']      >= MIN_PER_TYPE
  has_WN <- pt_counts['Within-non-ind']  >= MIN_PER_TYPE
  has_CR <- pt_counts['Cross-lifestyle'] >= MIN_PER_TYPE
  if (!has_CR) next

  n_zero  <- sum(sub$jaccard == 0)
  ziform  <- if (!PLAIN && n_zero > 0) ~1 else ~0   # ~0 disables zero inflation

  rhs <- "fst + (1 | pop_i) + (1 | pop_j)"
  if (has_WI) rhs <- paste0("within_ind + ", rhs)
  if (has_WN) rhs <- paste0("within_non_ind + ", rhs)
  formula_full <- as.formula(paste0("y ~ ", rhs))
  formula_null <- as.formula("y ~ fst + (1 | pop_i) + (1 | pop_j)")

  fits <- fit_one(formula_full, formula_null, sub, ziform)
  if (is.null(fits)) { cat(sprintf("  skipped (no converging fit): %s\n", sp)); next }

  ll_full <- as.numeric(logLik(fits$full))
  ll_null <- as.numeric(logLik(fits$null))
  df_omnibus <- as.numeric(has_WI) + as.numeric(has_WN)
  lr_omn <- 2 * (ll_full - ll_null)
  p_omn  <- pchisq(lr_omn, df = df_omnibus, lower.tail = FALSE)

  co <- fixef(fits$full)$cond
  se_full <- sqrt(diag(vcov(fits$full)$cond))
  b_WI <- if (has_WI) co["within_ind"]         else NA_real_
  b_WN <- if (has_WN) co["within_non_ind"]     else NA_real_
  s_WI <- if (has_WI) se_full["within_ind"]    else NA_real_
  s_WN <- if (has_WN) se_full["within_non_ind"] else NA_real_

  drop_lrt <- function(term) {
    tryCatch({
      m <- update(fits$full, as.formula(paste0(". ~ . - ", term)))
      if (!identical(m$fit$convergence, 0L)) return(NA_real_)
      pchisq(2 * (ll_full - as.numeric(logLik(m))), df = 1, lower.tail = FALSE)
    }, error = function(e) NA_real_)
  }
  p_lr_WI <- if (has_WI && has_WN) drop_lrt("within_ind")     else if (has_WI) p_omn else NA_real_
  p_lr_WN <- if (has_WN && has_WI) drop_lrt("within_non_ind") else if (has_WN) p_omn else NA_real_

  results[[sp]] <- data.frame(
    species = sp, method = fits$method, zi_used = (!PLAIN && n_zero > 0),
    n_zero = n_zero,
    n_obs = nrow(sub),
    n_cross = pt_counts['Cross-lifestyle'],
    n_within_ind = pt_counts['Within-ind'],
    n_within_non_ind = pt_counts['Within-non-ind'],
    beta_within_ind = b_WI, se_within_ind = s_WI, p_lrt_within_ind = p_lr_WI,
    beta_within_non_ind = b_WN, se_within_non_ind = s_WN,
    p_lrt_within_non_ind = p_lr_WN,
    omnibus_chi2 = lr_omn, omnibus_df = df_omnibus, omnibus_p = p_omn,
    stringsAsFactors = FALSE)
}

res <- bind_rows(results)
cat(sprintf("\nFitted: %d species  (%d with a zi component)\n",
            nrow(res), sum(res$zi_used)))

res$q_within_ind     <- p.adjust(res$p_lrt_within_ind, method = 'BH')
res$q_within_non_ind <- p.adjust(res$p_lrt_within_non_ind, method = 'BH')
res$q_omnibus        <- p.adjust(res$omnibus_p, method = 'BH')
res$OR_within_ind     <- exp(res$beta_within_ind)
res$OR_within_non_ind <- exp(res$beta_within_non_ind)

write_csv(res, OUT_F)

cat("\n────────── Summary ──────────\n")
cat(sprintf("Species fitted:  %d\n", nrow(res)))
sig_wi <- res %>% filter(!is.na(q_within_ind), q_within_ind < 0.05)
sig_wn <- res %>% filter(!is.na(q_within_non_ind), q_within_non_ind < 0.05)
cat(sprintf("Within-ind significant at FDR 5%%:     %d\n", nrow(sig_wi)))
if (nrow(sig_wi)) print(sig_wi %>% select(species, OR_within_ind, p_lrt_within_ind,
                                          q_within_ind, n_obs, n_zero),
                        row.names = FALSE, digits = 4)
cat(sprintf("Within-non-ind significant at FDR 5%%: %d\n", nrow(sig_wn)))
if (nrow(sig_wn)) print(sig_wn %>% select(species, OR_within_non_ind,
                                          p_lrt_within_non_ind, q_within_non_ind,
                                          n_obs, n_zero),
                        row.names = FALSE, digits = 4)
n_wi <- sum(!is.na(res$OR_within_ind))
cat(sprintf("Species with OR_within_ind > 1:      %d of %d (%.0f%%)\n",
            sum(res$OR_within_ind > 1, na.rm = TRUE), n_wi,
            100 * mean(res$OR_within_ind > 1, na.rm = TRUE)))
n_wn <- sum(!is.na(res$OR_within_non_ind))
cat(sprintf("Species with OR_within_non_ind > 1:  %d of %d (%.0f%%)\n",
            sum(res$OR_within_non_ind > 1, na.rm = TRUE), n_wn,
            100 * mean(res$OR_within_non_ind > 1, na.rm = TRUE)))
cat(sprintf("glmmTMB version: %s\n", as.character(packageVersion('glmmTMB'))))
cat(sprintf("\nSaved %s\n", OUT_F))
