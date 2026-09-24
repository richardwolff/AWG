"""
Run the whole iLDS workflow for every population of a species.

For each population of `--species` that is in the curated population list and
has at least `--min_genomes` good strains:

  0. reduce the population to one representative per clonal cluster, so that
     near-identical strains do not inflate LD
  1. read the alignment (`config.haplotype_dir % species` unless `--haplotypes`
     is given), subset to those genomes
  2. build a synonymous-site LD table

An equal sample of pairs is then drawn from every population's table and
pooled, and one LD decay curve is fit to that pool. Its decay distance is the
species-level tract length: every population is scanned against it, so window
sizes and iLDS scores stay comparable between populations. No population gets
a decay fit of its own.

Outputs, under `--output` (`config.scan_directory` by default), so that each
population lands in `config.scan_directory/<species>/<population>`:

    <species>/populations_summary.txt      genome counts and per-stage filters
    <species>/results.txt                  one row per population
    <species>/pooled_params.txt            species-level decay parameters
    <species>/pooled_df4_mean.txt          pooled LD decay curve
    <species>/pooled_decay.png             pooled curve and fit
    <species>/<population>/full_scan.txt   iLDS scan
    <species>/<population>/peaks.pkl       clustered sweeps
    <species>/<population>/samples.txt     genomes used
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import argparse
import os
import sys
import pickle
import time
import traceback

import config
import haplotype_utils
import population_utils
import make_ld_table
import fit_ld_curves
import scan_utils
import peak_utils
import plot_species_scans


def haplotype_candidates(species):
    """Where an alignment for `species` may live, in the order they are tried."""
    candidates = []

    template = getattr(config, "haplotype_dir", None)
    if template:
        candidates.append(template % species if "%s" in template
                          else os.path.join(template, species))

    # local checkout / test data
    candidates += [os.path.join("test_data", species, "haplotypes"),
                   os.path.join("test_data", species)]

    return candidates


def find_haplotypes(species, haplotype_dir=None):
    """
    Locate the alignment for a species.

    An explicit `--haplotypes` wins; otherwise `config.haplotype_dir % species`
    is used, falling back to `test_data/<species>` so a local checkout works
    without editing config.
    """
    if haplotype_dir is not None:
        if not os.path.exists(haplotype_dir):
            raise ValueError(f"{haplotype_dir} not found")
        return haplotype_dir

    candidates = haplotype_candidates(species)

    for path in candidates:
        if os.path.exists(path):
            return path

    raise ValueError(f"Could not find haplotypes for {species}; pass --haplotypes. "
                     f"Looked in: {', '.join(candidates)}")


def load_core_genes(species, core_genes_dir=None):
    """
    Core gene ids for a species, or None when there is no such list.

    Tries an explicit directory, then `core_genes/` beside the scripts, then
    `config.core_genes_dir` -- which may point somewhere that only exists on
    the cluster.
    """
    candidates = [core_genes_dir, "core_genes", getattr(config, "core_genes_dir", None)]

    for directory in candidates:
        if not directory:
            continue
        path = os.path.join(directory, f"{species}.txt")
        if os.path.exists(path):
            genes = pd.read_csv(path, index_col=0).index
            return set(genes.astype(str))

    return None


def restrict_to_core(dfH, core_genes):
    """Drop sites outside the core genome."""
    if core_genes is None:
        return dfH

    in_core = pd.Index(dfH.index.get_level_values("gene_id")).isin(core_genes)

    return dfH.loc[np.asarray(in_core)]


def prepare_population(genomes, haplotypes, args, core_genes):
    """Read the alignment for one population and restrict it to the core genome."""
    dfH = haplotype_utils.read_haplotypes(haplotypes, args.na_values, args.compression,
                                          genomes, prefilter=True)

    if core_genes is not None:
        before = dfH.shape[0]
        dfH = restrict_to_core(dfH, core_genes)
        sys.stderr.write(f"core genes: kept {dfH.shape[0]} of {before} variants\n")

    if dfH.shape[0] == 0:
        raise ValueError("no common variants left for this population "
                         "(check --maf, --core_genes and that the alignment covers these genomes)")

    site_type = dfH.index.get_level_values("site_type")
    sys.stderr.write(f"{dfH.shape[0]} common variants "
                     f"({int((site_type == 'nonsyn').sum())} nonsyn, "
                     f"{int((site_type == 'syn').sum())} syn) "
                     f"across {len(genomes)} genomes\n")

    return dfH


def fit_decay(df_r2, out_dir, species, label, prefix=""):
    """
    Fit an LD decay curve and write its parameters, binned curve and figure.

    Returns:
        (pd.Series, pd.DataFrame, np.ndarray): fitted params, binned curve, popt.
    """
    import matplotlib.pyplot as plt

    df4_mean, popt4, params = fit_ld_curves.return_ld_fits(df_r2)

    params.to_csv(f"{out_dir}/{prefix}params.txt", sep="\t")
    df4_mean.to_csv(f"{out_dir}/{prefix}df4_mean.txt")

    fit_ld_curves.plot_fit(df4_mean, popt4, params, species, label, save=False)
    plt.gcf().savefig(f"{out_dir}/{prefix}decay", bbox_inches="tight")
    plt.close("all")

    return params, df4_mean, popt4


def population_ld_table(dfH, out_dir, args, write=False):
    """Synonymous-site LD table for one population."""
    df_r2 = make_ld_table.build_ld_table(dfH, site_type="syn", maf=args.maf,
                                         max_dist=args.max_dist, max_pairs=args.max_pairs,
                                         seed=args.seed)

    sys.stderr.write(f"LD table: {df_r2.shape[0]} pairs\n")

    if write:
        df_r2.to_parquet(f"{out_dir}/df_r2.pq", index=False)

    return df_r2


def subsample_pairs(df_r2, n_target, rng):
    """
    A roughly `n_target`-row sample of an LD table, drawn independently of
    distance and r^2 so pooled bin means stay unbiased.
    """
    if df_r2.shape[0] <= n_target:
        return df_r2

    keep = rng.random(df_r2.shape[0]) < (n_target / df_r2.shape[0])

    return df_r2.loc[keep]


def scan_population(dfH, out_dir, dec_dist, window_size=None, num_bins=None):
    """Run the iLDS scan and peak clustering, writing both outputs."""
    window_size, dec_dist, num_bins = scan_utils.resolve_window(
        dfH, window_size=window_size, dec_dist=dec_dist, num_bins=num_bins)

    df_rnrs = scan_utils.perform_scan(dfH, window_size, num_bins)

    clusters = peak_utils.perform_peak_clustering(df_rnrs, dfH, dec_dist=dec_dist)

    df_rnrs.to_csv(f"{out_dir}/full_scan.txt")

    with open(f"{out_dir}/peaks.pkl", "wb") as handle:
        pickle.dump(clusters, handle)

    return df_rnrs, clusters, window_size, num_bins


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--species', required=True, type=str,
                        help="Species name, as it appears in the metadata")

    parser.add_argument('--haplotypes', default=None, type=str,
                        help="Alignment file or folder. Defaults to config.haplotype_dir %% species.")

    parser.add_argument('--metadata', default=None, type=str,
                        help="Strain metadata table (default config.metadata_file)")

    parser.add_argument('--good_populations', default=None, type=str,
                        help="File listing populations to analyze (default config.good_populations_file)")

    parser.add_argument('--populations', default=None, type=str,
                        help="Comma-separated subset of populations to run")

    parser.add_argument('--min_genomes', default=config.min_genomes, type=int,
                        help="Skip populations with fewer good strains than this")

    parser.add_argument('--good_strains', dest="good_strains", action="store_true",
                        default=config.require_good_strain,
                        help="Require the good_strain flag")
    parser.add_argument('--all_strains', dest="good_strains", action="store_false",
                        help="Do not require good_strain")

    parser.add_argument('--core_genes', dest="core_genes", action="store_true",
                        default=config.core_genes_only,
                        help="Restrict to the species' core genes")
    parser.add_argument('--all_genes', dest="core_genes", action="store_false",
                        help="Use every gene, not just the core genome")

    parser.add_argument('--trim_clonal', dest="trim_clonal", action="store_true",
                        default=config.trim_clonal,
                        help="Keep one representative per clonal cluster")
    parser.add_argument('--keep_clonal', dest="trim_clonal", action="store_false",
                        help="Use every strain, including clonal duplicates")

    parser.add_argument('--divergences', default=None, type=str,
                        help="Pairwise divergence matrix. Defaults to config.genetic_distances_df %% species.")

    parser.add_argument('--close_pair_thresh', default=config.close_pair_thresh, type=float,
                        help="Divergence below which two strains count as clonal")

    parser.add_argument('--core_genes_dir', default=None, type=str)

    parser.add_argument('--compression', default=None, type=str)

    parser.add_argument('--na_values', default=config.na_values,
                        help="Value denoting a missing call (the parquet alignments use -1)")

    parser.add_argument('--maf', default=config.common_variant_maf, type=float)

    parser.add_argument('--max_dist', default=config.max_dist, type=float,
                        help="Maximum separation of pairs entering the LD table")

    parser.add_argument('--max_pairs', default=config.ld_max_pairs, type=int,
                        help="Cap on pairs in the LD table, sampled uniformly. 0 keeps every pair.")

    parser.add_argument('--write_ld_table', action="store_true",
                        help="Also write each population's LD table to disk")

    parser.add_argument('--dec_dist', default=None, type=float,
                        help="Use this decay distance for every population, skipping the pooled fit")

    parser.add_argument('--pool_pairs', default=config.pool_pairs, type=int,
                        help="Total pairs in the pooled LD table, drawn equally from each population")

    parser.add_argument('--cache_budget_gb', default=config.cache_budget_gb, type=float,
                        help="Keep filtered haplotypes in memory between the two passes up to this "
                             "much; populations beyond it are re-read before scanning.")

    parser.add_argument('--window_size', default=None, type=int,
                        help="Force a fixed window size instead of deriving it from the decay distance")

    parser.add_argument('--num_bins', default=None, type=int)

    parser.add_argument('--seed', default=config.seed, type=int)

    parser.add_argument('--plot', dest="plot", action="store_true",
                        default=config.plot_scans,
                        help="Draw the combined figure of every population's scan when done")
    parser.add_argument('--no_plot', dest="plot", action="store_false",
                        help="Skip that figure")

    parser.add_argument('-o', '--output',
                        default=getattr(config, "scan_directory", "test_output"), type=str,
                        help="Results are written to <output>/<species>/<population>. "
                             "Defaults to config.scan_directory.")

    args = parser.parse_args()

    haplotypes = find_haplotypes(args.species, args.haplotypes)
    sys.stderr.write(f"Alignment: {haplotypes}\n")

    available = population_utils.alignment_samples(haplotypes)
    metadata = population_utils.load_metadata(args.metadata)
    good_populations = population_utils.load_good_populations(args.good_populations)

    ## every candidate population, before the size threshold: clonal trimming
    ## comes first, so that the threshold counts independent strains
    populations = population_utils.species_populations(
        args.species, metadata=metadata, good_populations=good_populations,
        min_genomes=1, available=available,
        require_good_strain=args.good_strains)

    genomes_before_trim = {p: len(g) for p, g in populations.items()}

    if args.trim_clonal:
        everyone = sorted({g for genomes in populations.values() for g in genomes})
        sys.stderr.write(f"\nTrimming clonal clusters (divergence < {args.close_pair_thresh})\n")

        divergence = population_utils.load_divergences(
            args.species, strains=everyone, path=args.divergences)

        populations = population_utils.trim_populations(
            populations, divergence=divergence, threshold=args.close_pair_thresh)

        dropped = sum(genomes_before_trim[p] - len(g) for p, g in populations.items())
        sys.stderr.write(f"{len(everyone)} genomes -> "
                         f"{sum(len(g) for g in populations.values())} representatives "
                         f"({dropped} clonal duplicates removed)\n")

        del divergence

    summary = population_utils.describe_populations(
        args.species, metadata=metadata, good_populations=good_populations,
        min_genomes=args.min_genomes, available=available,
        require_good_strain=args.good_strains,
        representatives=populations if args.trim_clonal else None)

    ## now apply the size threshold to what is left
    populations = {p: g for p, g in populations.items() if len(g) >= args.min_genomes}

    if args.populations:
        wanted = [p for p in args.populations.split(",") if p]
        populations = {p: g for p, g in populations.items() if p in wanted}

    species_dir = os.path.join(args.output, args.species)
    os.makedirs(species_dir, exist_ok=True)
    summary.to_csv(f"{species_dir}/populations_summary.txt", sep="\t")

    sys.stderr.write(f"\n{len(populations)} populations with >= {args.min_genomes} genomes: "
                     f"{', '.join(populations)}\n\n")

    core_genes = load_core_genes(args.species, args.core_genes_dir) if args.core_genes else None
    if args.core_genes and core_genes is None:
        parser.error(f"--core_genes given but no core gene list found for {args.species} "
                     f"(looked for {args.species}.txt in core_genes/ and config.core_genes_dir)")

    rng = np.random.default_rng(args.seed)

    ## ------------------------------------------------------------------
    ## Pass 1: LD tables, sampled into the species-level pool
    ## ------------------------------------------------------------------
    share = max(args.pool_pairs // max(len(populations), 1), 1)
    budget_bytes = args.cache_budget_gb * 1e9

    stats, pool_chunks, cache, cached_bytes = {}, [], {}, 0
    failed = set()

    for i, (population, genomes) in enumerate(populations.items(), 1):

        sys.stderr.write(f"\n{'='*60}\n[LD {i}/{len(populations)}] {args.species} / {population} "
                         f"({len(genomes)} genomes)\n{'='*60}\n")

        pop_dir = os.path.join(species_dir, population)
        os.makedirs(pop_dir, exist_ok=True)

        with open(f"{pop_dir}/samples.txt", "w") as handle:
            handle.write("\n".join(genomes) + "\n")

        started = time.time()

        try:
            dfH = prepare_population(genomes, haplotypes, args, core_genes)

            df_r2 = population_ld_table(dfH, pop_dir, args, write=args.write_ld_table)

            pool_chunks.append(subsample_pairs(df_r2, share, rng))
            del df_r2

            site_type = dfH.index.get_level_values("site_type")
            stats[population] = {"population": population,
                                 "genomes": len(genomes),
                                 "genomes_before_trim": genomes_before_trim[population],
                                 "clonal_dropped": genomes_before_trim[population] - len(genomes),
                                 "variants": dfH.shape[0],
                                 "nonsyn": int((site_type == "nonsyn").sum()),
                                 "syn": int((site_type == "syn").sum()),
                                 "seconds_ld": round(time.time() - started, 1)}

            if cached_bytes + dfH.values.nbytes <= budget_bytes:
                cache[population] = dfH
                cached_bytes += dfH.values.nbytes
            else:
                del dfH

        except Exception:
            sys.stderr.write(f"\nFAILED (LD stage): {population}\n")
            traceback.print_exc()
            failed.add(population)
            stats[population] = {"population": population, "genomes": len(genomes),
                                 "failed_stage": "ld"}

    sys.stderr.write(f"\nheld {len(cache)}/{len(populations)} populations in memory "
                     f"({cached_bytes/1e9:.2f} GB)\n")

    ## ------------------------------------------------------------------
    ## One tract length for the species, from the pooled fit
    ## ------------------------------------------------------------------
    tract_length = np.nan

    if args.dec_dist is not None:
        dec_dist, source = args.dec_dist, "--dec_dist"
        del pool_chunks

    elif not pool_chunks:
        pd.DataFrame(list(stats.values())).to_csv(f"{species_dir}/results.txt",
                                                  sep="\t", index=False)
        sys.exit(f"\nno population of {args.species} produced an LD table, "
                 f"so there is no pooled decay curve to fit\n")

    else:
        sys.stderr.write(f"\n{'='*60}\nPooled decay fit across {len(pool_chunks)} populations"
                         f"\n{'='*60}\n")
        pooled = pd.concat(pool_chunks, ignore_index=True)
        sys.stderr.write(f"pooled LD table: {pooled.shape[0]} pairs\n")

        pooled_params, _, _ = fit_decay(pooled, species_dir, args.species, "pooled",
                                        prefix="pooled_")
        del pooled, pool_chunks

        dec_dist = float(pooled_params.loc["decay_points"])
        tract_length = float(pooled_params.loc["tract_length"])
        source = "pooled fit"

    sys.stderr.write(f"\nspecies-level decay distance ({source}): {dec_dist}\n")

    ## ------------------------------------------------------------------
    ## Pass 2: scan every population against that decay distance
    ## ------------------------------------------------------------------
    results = []
    for i, (population, genomes) in enumerate(populations.items(), 1):

        if population in failed:
            results.append(stats[population])
            continue

        sys.stderr.write(f"\n{'='*60}\n[scan {i}/{len(populations)}] {args.species} / {population}"
                         f"\n{'='*60}\n")

        pop_dir = os.path.join(species_dir, population)
        started = time.time()

        try:
            dfH = cache.pop(population, None)
            if dfH is None:
                dfH = prepare_population(genomes, haplotypes, args, core_genes)

            df_rnrs, clusters, window_size, num_bins = scan_population(
                dfH, pop_dir, dec_dist=dec_dist,
                window_size=args.window_size, num_bins=args.num_bins)

            row = dict(stats[population])
            row.update({"tract_length_pooled": tract_length,
                        "l_DD_used": dec_dist,
                        "window_size": int(window_size),
                        "num_bins": int(num_bins),
                        "windows_scanned": int(df_rnrs.shape[0]),
                        "significant_windows": int(df_rnrs["significance"].sum()),
                        "sweeps": len(clusters),
                        "seconds_scan": round(time.time() - started, 1)})
            results.append(row)

            del dfH

        except Exception:
            sys.stderr.write(f"\nFAILED (scan stage): {population}\n")
            traceback.print_exc()
            row = dict(stats[population])
            row["failed_stage"] = "scan"
            results.append(row)

        pd.DataFrame(results).to_csv(f"{species_dir}/results.txt", sep="\t", index=False)

    sys.stderr.write(f"\n\n{'='*60}\nAll populations complete\n{'='*60}\n")
    print(pd.DataFrame(results).to_string(index=False))

    ## draw every population's scan on one figure. Populations that failed have
    ## no full_scan.txt and are simply left out, and a failure here must not
    ## lose the scans that just took hours to compute.
    if args.plot:
        sys.stderr.write(f"\n{'='*60}\nPlotting\n{'='*60}\n")
        try:
            plot_species_scans.plot_from_directory(args.species, species_dir)
        except Exception:
            sys.stderr.write("\nFAILED: combined figure\n")
            traceback.print_exc()
