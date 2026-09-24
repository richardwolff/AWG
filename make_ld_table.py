"""
Build a pairwise LD table (genomic distance, r^2) from an annotated alignment.

This is the input `fit_ld_curves.py` expects: a table with columns `D` and
`r2`, one row per pair of variants, from which the LD decay curve and hence
the decay distance `l_DD` (the `--dec_dist` argument of `run_iLDS.py`) is fit.

By default the table is built from intermediate frequency synonymous variants,
which is what the decay fit in Wolff & Garud (2025) is based on.
"""

import numpy as np
import pandas as pd
import argparse
import sys
import os

import config
import haplotype_utils
import population_utils
import r2_utils


## Rows of the r^2 matrix computed per pass. Peak memory is roughly
## BLOCK_ROWS * (sites within max_dist) * 8 bytes * 4.
BLOCK_ROWS = config.ld_block_rows


def block_pair_r2(A, mask, lo, hi, col_lo, col_hi):
    """
    r^2 between variants [lo, hi) and variants [col_lo, col_hi).

    Same estimator as `r2_utils.pairwise_r2` -- allele frequencies are taken
    over the samples covered at both sites -- but accumulated with four matrix
    products so the whole block goes through BLAS in one call:

        n    = mask_i . mask_j        (samples covered at both sites)
        s_a  = A_i    . mask_j        (allele count at i over those samples)
        s_b  = mask_i . A_j
        s_ab = A_i    . A_j

    `A` is the genotype matrix with missing calls zeroed, so those products are
    already restricted to jointly covered samples. Genotypes are 0/1, so every
    sum is an exact integer in float64 and the result agrees bit-for-bit with
    the per-pair loop in `r2_utils`.

    Args:
        A (np.ndarray): (N, M) genotypes, NaN replaced by 0.
        mask (np.ndarray): (N, M) 1.0 where a call is present, else 0.0.
        lo, hi (int): row range of the block.
        col_lo, col_hi (int): column range to pair the block against.

    Returns:
        np.ndarray: (hi - lo, col_hi - col_lo) of r^2, NaN where undefined.
    """
    A_i, mask_i = A[lo:hi], mask[lo:hi]
    A_j, mask_j = A[col_lo:col_hi].T, mask[col_lo:col_hi].T

    n = mask_i @ mask_j

    with np.errstate(invalid="ignore", divide="ignore"):
        f_a = (A_i @ mask_j) / n
        f_b = (mask_i @ A_j) / n
        f_ab = (A_i @ A_j) / n

        f_a_f_b = f_a * f_b
        r2 = ((f_ab - f_a_f_b) ** 2) / (f_a_f_b * (1.0 - f_a) * (1.0 - f_b))

    # no shared coverage, or a site that is monomorphic among the shared
    # samples, carries no LD information
    r2[(n == 0) | (f_a == 0) | (f_a == 1) | (f_b == 0) | (f_b == 1)] = np.nan

    return r2


def count_pairs(site_pos, max_dist=None):
    """Number of pairs on a contig separated by <= max_dist, from positions alone."""
    N = site_pos.shape[0]

    if max_dist is None:
        return N * (N - 1) // 2

    j_end = np.searchsorted(site_pos, site_pos + max_dist, side="right")

    return int(np.sum(j_end - np.arange(N) - 1))


def contig_ld_table(values, site_pos, max_dist=None, block_rows=BLOCK_ROWS,
                    keep_frac=None, rng=None):
    """
    (D, r2) for every pair of variants on one contig separated by <= max_dist.

    Pairs are enumerated in blocks of rows rather than as one N x N matrix, so
    memory stays bounded by `block_rows` instead of growing with the square of
    the number of variants.

    Args:
        values (np.ndarray): (N, M) genotype matrix, sorted by site_pos.
        site_pos (np.ndarray): (N,) site positions, ascending.
        max_dist (float, optional): only keep pairs closer than this.
        block_rows (int): rows per pass.
        keep_frac (float, optional): retain this fraction of pairs, drawn
            independently of distance and r^2 so binned means stay unbiased.
        rng (np.random.Generator, optional): source for that draw.

    Returns:
        (np.ndarray, np.ndarray): distances and r^2, NaN pairs already dropped.
    """
    N = values.shape[0]

    if keep_frac is not None and rng is None:
        rng = np.random.default_rng()

    mask = np.isfinite(values).astype(np.float64)
    A = np.where(mask > 0, values, 0.0)

    # furthest partner of each variant: contiguous, because site_pos is sorted
    if max_dist is None:
        j_end = np.full(N, N)
    else:
        j_end = np.searchsorted(site_pos, site_pos + max_dist, side="right")

    d_out, r2_out = [], []

    for lo in range(0, N, block_rows):

        hi = min(lo + block_rows, N)

        col_lo, col_hi = lo, int(j_end[lo:hi].max())
        if col_hi <= col_lo + 1:
            continue

        r2 = block_pair_r2(A, mask, lo, hi, col_lo, col_hi)

        rows = site_pos[lo:hi, None]
        cols = site_pos[None, col_lo:col_hi]
        d = cols - rows

        # upper triangle only (each unordered pair once), within max_dist,
        # and defined
        keep = (np.arange(col_lo, col_hi)[None, :] > np.arange(lo, hi)[:, None])
        if max_dist is not None:
            keep &= d <= max_dist
        keep &= np.isfinite(r2)

        d_block, r2_block = d[keep], r2[keep]

        if keep_frac is not None and keep_frac < 1.0:
            drawn = rng.random(d_block.shape[0]) < keep_frac
            d_block, r2_block = d_block[drawn], r2_block[drawn]

        d_out.append(d_block)
        r2_out.append(r2_block)

    if not d_out:
        return np.empty(0), np.empty(0)

    return np.concatenate(d_out), np.concatenate(r2_out)


def build_ld_table(dfH, site_type=config.ld_site_type, maf=config.common_variant_maf,
                   max_dist=None, max_pairs=None, seed=config.seed):
    """
    Pairwise LD table across all contigs.

    Args:
        dfH (pd.DataFrame): haplotypes, multiindex (contig, gene_id, site_pos, site_type).
        site_type (str): "syn", "nonsyn", or "all".
        maf (float): keep variants with maf <= f < 1 - maf.
        max_dist (float, optional): only keep pairs closer than this.
        max_pairs (int, optional): cap the table at roughly this many pairs by
            sampling uniformly. The decay curve is a mean over a few hundred
            distance bins, so a bounded sample estimates it just as well as
            every pair, and diverse species can have billions of them.
        seed (int, optional): seed for that sampling.

    Returns:
        pd.DataFrame: columns ["D", "r2"].
    """
    index = dfH.index

    if site_type != "all":
        selected = np.asarray(index.get_level_values("site_type") == site_type)
    else:
        selected = np.ones(dfH.shape[0], dtype=bool)

    values = dfH.to_numpy(dtype=np.float64)[selected]
    contig = np.asarray(index.get_level_values("contig"))[selected]
    site_pos = np.asarray(index.get_level_values("site_pos"), dtype=np.int64)[selected]

    # intermediate frequency variants. Haplotypes are polarized to the
    # consensus, so f <= 0.5 and the upper bound never binds; it is applied
    # anyway so the function is correct on unpolarized input too.
    called = np.count_nonzero(np.isfinite(values), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.nansum(values, axis=1) / called

    common = (f >= maf) & (f < 1 - maf)

    values, contig, site_pos = values[common], contig[common], site_pos[common]

    sys.stderr.write(f"{int(common.sum())} {site_type} variants with {maf} <= f < {1 - maf}\n")

    contigs = np.unique(contig)

    positions = {}
    for c in contigs:
        pos_c = site_pos[contig == c]
        positions[c] = np.sort(pos_c, kind="stable")

    keep_frac = None
    if max_pairs is not None:
        total = sum(count_pairs(positions[c], max_dist) for c in contigs)
        if total > max_pairs:
            keep_frac = max_pairs / total
            sys.stderr.write(f"sampling {max_pairs} of {total} pairs "
                             f"(fraction {keep_frac:.4g})\n")

    rng = np.random.default_rng(seed)

    d_all, r2_all = [], []

    for c in contigs:

        on_contig = contig == c

        pos_c = site_pos[on_contig]
        vals_c = values[on_contig]

        # pairs are enumerated as a band around the diagonal, which requires
        # positions in ascending order
        order = np.argsort(pos_c, kind="stable")
        pos_c, vals_c = pos_c[order], np.ascontiguousarray(vals_c[order])

        d, r2 = contig_ld_table(vals_c, pos_c, max_dist=max_dist,
                                keep_frac=keep_frac, rng=rng)

        sys.stderr.write(f"\t{c}: {pos_c.shape[0]} variants, {d.shape[0]} pairs\n")

        d_all.append(d)
        r2_all.append(r2)

    return pd.DataFrame({"D": np.concatenate(d_all).astype(np.int32),
                         "r2": np.concatenate(r2_all)})


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input',
                        help="File or folder where haplotypes are stored",
                        required=True,
                        type=str)

    parser.add_argument('--compression',
                        help="If haplotypes are saved in compressed format, compression type (gzip, bz2, zip)",
                        default=None,
                        type=str)

    parser.add_argument('--allowed_samples',
                        help="If subsetting to some set of samples, store as a text file and pass the path here",
                        default=None,
                        type=str)

    parser.add_argument('--na_values',
                        help="If missing data is denoted with a specific string/value specify here",
                        default=None,
                        type=str)


    parser.add_argument('--species',
                        help="Species name, used to locate the divergence matrix for clonal trimming",
                        default=None,
                        type=str)

    parser.add_argument('--divergences',
                        help="Pairwise divergence matrix. Giving this, or --species, trims to one representative per clonal cluster.",
                        default=None,
                        type=str)

    parser.add_argument('--close_pair_thresh',
                        help="Divergence below which two strains count as clonal",
                        default=config.close_pair_thresh,
                        type=float)

    parser.add_argument('--site_type',
                        help="Site class to compute LD between",
                        default=config.ld_site_type,
                        choices=["syn", "nonsyn", "all"])

    parser.add_argument('--maf',
                        help="Minor allele frequency threshold; keeps maf <= f < 1 - maf",
                        default=config.common_variant_maf,
                        type=float)

    parser.add_argument('--max_dist',
                        help="Only record pairs separated by less than this many bp. This is also the range the decay curve is fit over (config.max_dist). Pass 0 for no limit.",
                        default=config.max_dist,
                        type=float)

    parser.add_argument('--max_pairs',
                        help="Cap the table at roughly this many pairs, sampled uniformly. Pass 0 to keep every pair.",
                        default=config.ld_max_pairs,
                        type=int)

    parser.add_argument('--seed',
                        help="Seed for pair sampling",
                        default=config.seed,
                        type=int)

    parser.add_argument('-o', '--output',
                        help="Path to write the LD table to (.pq for parquet, otherwise csv)",
                        required=True,
                        type=str)

    args = parser.parse_args()

    max_dist = None if args.max_dist <= 0 else args.max_dist

    allowed_samples = args.allowed_samples
    if args.species or args.divergences:
        allowed_samples = population_utils.trim_allowed_samples(
            allowed_samples, alignment=args.input, species=args.species,
            divergences=args.divergences, threshold=args.close_pair_thresh)

    ## `prefilter` drops rare and poorly covered sites while the alignment is
    ## being parsed. It filters at config.common_variant_maf, so it can only be
    ## used when that is no stricter than the threshold requested here.
    prefilter = args.maf >= config.common_variant_maf

    dfH = haplotype_utils.read_haplotypes(args.input, args.na_values, args.compression,
                                          allowed_samples, prefilter=prefilter)

    if not prefilter:
        dfH = haplotype_utils.return_common_filtered(dfH)

    sys.stderr.write("Haplotypes successfully read in\n\n")

    df_r2 = build_ld_table(dfH, site_type=args.site_type, maf=args.maf, max_dist=max_dist,
                           max_pairs=(args.max_pairs or None), seed=args.seed)

    sys.stderr.write(f"\nLD table: {df_r2.shape[0]} pairs\n")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if args.output.endswith(".pq") or args.output.endswith(".parquet"):
        df_r2.to_parquet(args.output, index=False)
    else:
        df_r2.to_csv(args.output, index=False)

    sys.stderr.write(f"Written to {args.output}\n")
