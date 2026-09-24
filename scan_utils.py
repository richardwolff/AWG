import numpy as np
import pandas as pd
import r2_utils   # custom module: likely implements LD calculations
from scipy.stats import norm, chi2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import sys
from collections import defaultdict
import config

## --------------------------------------------------
## 1. Utility to compute site distances
## --------------------------------------------------

def return_site_diffs(dfH):
    """
    Compute distances between successive common nonsynonymous variants
    across contigs.

    Parameters:
        dfH (pd.DataFrame): Variants table with multiindex including
                            ("contig", "site_pos", "site_type").

    Returns:
        list: List of distances (bp) between consecutive nonsyn sites.
    """
    # Only the index is needed, so avoid `xs`/`groupby` on the (wide) genotype
    # matrix -- those copy every sample column just to read site positions.
    idx = dfH.index
    is_ns = idx.get_level_values("site_type") == "nonsyn"

    contig = np.asarray(idx.get_level_values("contig"))[is_ns]
    site_pos = np.asarray(idx.get_level_values("site_pos"))[is_ns]

    # Stable sort by contig reproduces `groupby("contig")`: groups in sorted
    # contig order, original row order preserved inside each group.
    order = np.argsort(contig, kind="stable")
    contig, site_pos = contig[order], site_pos[order]

    starts = np.flatnonzero(np.r_[True, contig[1:] != contig[:-1], True])

    site_diffs = []
    for lo, hi in zip(starts[:-1], starts[1:]):
        # Take successive differences along genome
        site_diffs.extend(np.diff(site_pos[lo:hi]))
    return site_diffs


## --------------------------------------------------
## 2. Window sizing and binning
## --------------------------------------------------

def return_window_size(site_diffs, dec_dist, num_bins=None, num_vars=config.num_vars):
    """
    Determine window size (in number of variants) and number of bins
    used in LD scan.

    Window size = median number of variants spanning `dec_dist` bp.

    Parameters:
        site_diffs (list): distances between consecutive nonsyn sites
        dec_dist (int): decay distance (bp) threshold
        num_bins (int): optional, number of bins
        num_vars (int): target number of LD measurements per bin

    Returns:
        (int, int): window size, number of bins
    """
    # For each starting variant, count how many downstream sites span dec_dist.
    # cumsum(site_diffs[i:])[k] == C[i+k] - C[i-1], so a single global prefix
    # sum plus a binary search replaces the O(n^2) per-start cumsum.
    site_diffs = np.asarray(site_diffs, dtype=np.int64)

    # a negative gap means the alignment was not in position order, which would
    # make every window and distance downstream meaningless
    if site_diffs.size and site_diffs.min() < 0:
        raise ValueError("site_diffs contains negative gaps: haplotypes are not "
                         "sorted by position within each contig")

    C = np.cumsum(site_diffs)
    C0 = np.concatenate(([0], C[:-1]))          # C0[i] == sum(site_diffs[:i])

    # first index m with C[m] > dec_dist + C0[i]
    first = np.searchsorted(C, C0 + dec_dist, side="right")
    found = first < C.shape[0]
    dists = (first - np.arange(C.shape[0]))[found]

    # Median window length (number of variants)
    window_size = int(np.median(dists) + 1)

    # If not provided, calculate bins based on target num_vars
    if num_bins is None:
        num_bins = bins_from_num_vars(window_size, num_vars)

    sys.stderr.write(f"(window size, num bins): ({window_size},{num_bins}) \n")

    return window_size, num_bins


def return_span_dists(site_diffs, window_size):
    """
    Total span (bp) of every consecutive run of `window_size + 1` variants.

    Used to infer a decay distance when the user supplied a window size
    instead. Computed from a prefix sum rather than re-summing each slice.
    """
    C0 = np.concatenate(([0], np.cumsum(np.asarray(site_diffs, dtype=np.int64))))

    n = len(site_diffs) - window_size - 1
    if n <= 0:
        return np.empty(0, dtype=np.int64)

    return C0[window_size + 1:window_size + 1 + n] - C0[:n]


def bins_from_num_vars(window_size, num_vars=config.num_vars,
                       min_bins=config.min_bins, max_bins=config.max_bins):
    """
    Estimate number of bins so each bin contains ~num_vars LD measurements.

    Formula is based on number of unique pairwise comparisons in a window.

    Parameters:
        window_size (int): number of variants in a window
        num_vars (int): desired comparisons per bin
        min_bins, max_bins (int): bounds on number of bins

    Returns:
        int: number of bins
    """
    # Total pairwise comparisons in a window = n choose 2
    nb = int(((window_size**2 - window_size) / 2) / num_vars)

    if min_bins is not None:
        nb = max(nb, min_bins)
    if max_bins is not None:
        nb = min(nb, max_bins)

    return nb


def resolve_window(dfH, window_size=None, dec_dist=None, num_bins=None):
    """
    Settle the window size, decay distance and bin count for a scan.

    Exactly the resolution `run_iLDS.py` performs: whichever of window size and
    decay distance is missing is inferred from the other via the spacing of
    common nonsynonymous variants, and the bin count follows from the window
    unless it was given.

    Returns:
        (int, float, int): window_size, dec_dist, num_bins
    """
    ## if decay distance has been specified but window size hasn't
    if window_size is None and dec_dist is not None:
        site_diffs = return_site_diffs(dfH)
        window_size, num_bins = return_window_size(site_diffs, dec_dist)

    ## if window size has been specified but decay distance hasn't
    elif window_size is not None and dec_dist is None:
        site_diffs = return_site_diffs(dfH)

        dists = return_span_dists(site_diffs, window_size)

        dec_dist = np.median(dists)
        sys.stderr.write(f"Inferred decay distance for peak clustering: {dec_dist}\n\n")

    ## raise an error if neither window size nor decay distance has been specified
    elif window_size is None and dec_dist is None:
        raise ValueError("Must specify one of either window size or decay distance")

    ## if the number of bins hasn't been specified, get it
    if num_bins is None:
        num_bins = bins_from_num_vars(window_size)
        sys.stderr.write(f"Inferred num_bins = {num_bins}\n\n")

    return window_size, dec_dist, num_bins


## --------------------------------------------------
## 3. Core iLDS scan
## --------------------------------------------------

def _syn_window_bounds(site_pos_syn, lo, hi):
    """
    Indices of the syn sites strictly inside (lo, hi), as (first, last - 1).

    Reproduces `np.argwhere((pos > lo) * (pos < hi))` -> (first, last - 1) but
    with a binary search when positions are sorted, which they are for any
    input the rest of the scan makes sense on.

    Returns (l_syn, u_syn) or None when the interval contains no syn site.
    """
    first = np.searchsorted(site_pos_syn, lo, side="right")
    last = np.searchsorted(site_pos_syn, hi, side="left") - 1

    if first > last:
        return None

    return first, last - 1


def _bin_stats(labels_idx, values, n_labels):
    """
    Per-bin count, mean and standard error for one set of LD measurements.

    `labels_idx` gives, for each measurement, its position in the shared array
    of bin labels. Equivalent to `groupby("D_bins")` -> mean()/std()/size(),
    with pandas' ddof=1 convention (std is NaN for singleton bins).
    """
    count = np.bincount(labels_idx, minlength=n_labels)
    total = np.bincount(labels_idx, weights=values, minlength=n_labels)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count

    # second pass about the group mean: numerically stable, and matches the
    # magnitude of error pandas' Welford accumulation produces
    dev = values - mean[labels_idx]
    ss = np.bincount(labels_idx, weights=dev * dev, minlength=n_labels)

    sem = np.full(n_labels, np.nan)
    multi = count > 1
    sem[multi] = np.sqrt(ss[multi] / (count[multi] - 1)) / np.sqrt(count[multi])

    return count, mean, sem


def perform_scan(dfH, window_size, num_bins,
                 alpha=config.alpha, sample_num=config.sample_num, min_syn=config.min_syn):
    """
    Perform iLDS scan across genome.

    iLDS (integrated LD score) compares decay of LD at nonsyn vs syn sites,
    and local LD to global LD, and computes significance.

    Parameters:
        dfH (pd.DataFrame): haplotypes matrix, multiindex = (contig, site_pos, site_type)
        window_size (int): number of variants per window
        num_bins (int): number of distance bins per window
        alpha (float): significance threshold
        sample_num (int): number of genomewide LD pairs to sample
        min_syn (int): minimum syn sites required in a window

    Returns:
        pd.DataFrame: results with r2 integrals, test statistics, iLDS values, p-values
    """

    ## Split dataframe into nonsyn and syn variants.
    ## The genotype matrix is pulled out once as a contiguous float64 array and
    ## sliced positionally from here on; `xs`/`get_group` would copy it again
    ## for every contig.
    index = dfH.index
    contigs = index.get_level_values("contig").unique()

    genotypes = np.ascontiguousarray(dfH.to_numpy(), dtype=np.float64)
    contig_lvl = np.asarray(index.get_level_values("contig"))
    pos_lvl = np.asarray(index.get_level_values("site_pos"), dtype=np.int64)
    is_ns = np.asarray(index.get_level_values("site_type") == "nonsyn")

    ns_rows = np.flatnonzero(is_ns)
    syn_rows = np.flatnonzero(~is_ns)

    contig_ns, contig_syn = contig_lvl[ns_rows], contig_lvl[syn_rows]

    # Store site positions and indices for each contig
    all_contigs = np.unique(contig_lvl)
    site_pos_ns = {c: pos_lvl[ns_rows[contig_ns == c]] for c in all_contigs}
    site_pos_syn = {c: pos_lvl[syn_rows[contig_syn == c]] for c in all_contigs}

    rows_ns = {c: ns_rows[contig_ns == c] for c in all_contigs}
    rows_syn = {c: syn_rows[contig_syn == c] for c in all_contigs}

    site_idx_ns = {c: index.droplevel("site_type")[rows_ns[c]].tolist() for c in all_contigs}

    # contigs carrying no nonsyn variant at all are simply absent, as they were
    # from the `groupby(...).size()` these dicts replace
    N_ns = {c: site_pos_ns[c].shape[0] for c in all_contigs if site_pos_ns[c].shape[0] > 0}
    N_syn = {c: site_pos_syn[c].shape[0] for c in all_contigs}

    ## Filter out contigs with < window_size number of non_syn variants
    contig_ns_sizes = pd.Series(N_ns)
    good_contigs = contig_ns_sizes.loc[contig_ns_sizes >= window_size].index
    bad_contigs = contig_ns_sizes.loc[contig_ns_sizes < window_size].index

    num_good_contigs = len(good_contigs)

    if num_good_contigs < len(contigs):
        sys.stderr.write(f"Excluded contigs: {bad_contigs}")

    if window_size < 2:
        raise ValueError(f"window_size must be at least 2, got {window_size}")

    window_bounds = defaultdict(dict)

    wn_half = window_size//2

    ## ------------------------------------------
    ## Build windows of nonsyn + corresponding syn
    ## ------------------------------------------
    ## Instead of an N x N boolean mask per contig, record for each site the
    ## furthest partner it is ever paired with (`j_end`). The windows are
    ## sliding blocks, so the pairs they cover form a band around the diagonal
    ## and `j_end` describes it exactly.
    j_end_ns, j_end_syn = {}, {}

    for contig in good_contigs:

        n_ns, n_syn = N_ns[contig], N_syn[contig]
        pos_ns, pos_syn = site_pos_ns[contig], site_pos_syn[contig]

        je_ns = np.arange(1, n_ns + 1)
        je_syn = np.arange(1, n_syn + 1)

        sorted_syn = n_syn < 2 or bool(np.all(np.diff(pos_syn) >= 0))

        for i in range(wn_half, n_ns - wn_half):

            # Define nonsyn window bounds
            l_ns, u_ns = i - wn_half, i + wn_half
            je_ns[l_ns:u_ns] = u_ns          # u_ns increases with i, so this is a max
            bounds = (pos_ns[l_ns], pos_ns[u_ns])

            # Define syn window bounds (subset syn sites in same region)
            if sorted_syn:
                syn_bounds = _syn_window_bounds(pos_syn, bounds[0], bounds[1])
            else:
                syn_where = np.argwhere((pos_syn > bounds[0]) * (pos_syn < bounds[1]))
                syn_bounds = None if len(syn_where) == 0 else (syn_where[0][0], syn_where[-1][0] - 1)

            if syn_bounds is not None:
                l_syn, u_syn = syn_bounds
                if u_syn > l_syn:
                    je_syn[l_syn:u_syn] = np.maximum(je_syn[l_syn:u_syn], u_syn)
                window_bounds[contig][site_idx_ns[contig][i]] = {
                    "nonsyn": (l_ns, u_ns), "syn": (l_syn, u_syn)
                }

        j_end_ns[contig] = je_ns
        j_end_syn[contig] = je_syn

    ## ------------------------------------------
    ## Compute distance matrices and r^2 matrices
    ## ------------------------------------------
    ## Both are stored banded: entry [i, k] refers to the pair (i, i + k + 1).
    W_ns, W_syn, d_ns, d_syn, r2_ns, r2_syn = {}, {}, {}, {}, {}, {}

    for c in good_contigs:

        W_ns[c] = max(int((j_end_ns[c] - np.arange(N_ns[c])).max()) - 1, 1)
        W_syn[c] = max(int((j_end_syn[c] - np.arange(N_syn[c])).max()) - 1, 1)

        d_ns[c] = r2_utils.distance_band(site_pos_ns[c], W_ns[c])
        d_syn[c] = r2_utils.distance_band(site_pos_syn[c], W_syn[c])

        r2_ns[c] = r2_utils.banded_r2(genotypes[rows_ns[c]], j_end_ns[c], W_ns[c])
        r2_syn[c] = r2_utils.banded_r2(genotypes[rows_syn[c]], j_end_syn[c], W_syn[c])

    # Flatten to long format (already NaN-free, as `dropna` used to guarantee)
    long_ns = [r2_utils.band_to_long(r2_ns[c], d_ns[c], j_end_ns[c]) for c in good_contigs]
    long_syn = [r2_utils.band_to_long(r2_syn[c], d_syn[c], j_end_syn[c]) for c in good_contigs]

    all_r2 = np.concatenate([p[0] for p in long_ns] + [p[0] for p in long_syn])
    all_D = np.concatenate([p[1] for p in long_ns] + [p[1] for p in long_syn])

    del long_ns, long_syn

    ## ------------------------------------------
    ## Loop over windows, calculate integrals
    ## ------------------------------------------
    sites, r2N_l, r2S_l, r2L_l, rL_std_l, pval_l = [], [], [], [], [], []
    d_bins_dic = {}

    sys.stderr.write("\nCalculating AUC(r^2_N), AUC(r^2_S), & AUC(r^2_local) within each window\n")

    bb = np.linspace(0, 100, num_bins + 1)
    min_syn_pairs = min_syn*(min_syn - 1)/2

    insuff = 0
    for contig in good_contigs:

        r2_band_ns, d_band_ns = r2_ns[contig], d_ns[contig]
        r2_band_syn, d_band_syn = r2_syn[contig], d_syn[contig]

        for site,item in window_bounds[contig].items():

            # Extract pairwise LD + distance for nonsyn and syn windows
            l,u = item["nonsyn"]
            r2_ns_window = r2_utils.block_triu(r2_band_ns, l, u)
            d_ns_window = r2_utils.block_triu(d_band_ns, l, u)

            l,u = item["syn"]
            r2_syn_window = r2_utils.block_triu(r2_band_syn, l, u)
            d_syn_window = r2_utils.block_triu(d_band_syn, l, u)

            # Drop pairs with undefined LD (the old `.dropna()`)
            keep = ~np.isnan(r2_syn_window)
            r2_syn_window, d_syn_window = r2_syn_window[keep], d_syn_window[keep]

            # Skip windows with too few syn sites
            if r2_syn_window.shape[0] < min_syn_pairs:
                if insuff == 0:
                    sys.stderr.write(f"\nWindows containing insufficient syn variants:\n")
                sys.stderr.write(f"\t\t{site}\n")
                insuff+=1
                continue

            keep = ~np.isnan(r2_ns_window)
            r2_ns_window, d_ns_window = r2_ns_window[keep], d_ns_window[keep]

            # Distance binning (binning performed based on percentiles based of nonsyn distances)
            d_bins = np.percentile(d_ns_window, bb)
            d_bins[-1] = d_bins[-1] + .5

            # Assign bins. Note the (deliberate) wrap-around of `- 1`: syn pairs
            # falling outside the nonsyn distance range land in the last bin.
            lab_ns = d_bins[np.digitize(d_ns_window, d_bins) - 1]
            lab_syn = d_bins[np.digitize(d_syn_window, d_bins) - 1]

            # Shared, sorted set of bin labels -- the groupby keys
            labels = np.union1d(lab_ns, lab_syn)
            n_labels = labels.shape[0]

            idx_ns = np.searchsorted(labels, lab_ns)
            idx_syn = np.searchsorted(labels, lab_syn)

            # take means to get LD decay curves in window (r2N, r2S, r2L),
            # plus standard errors for r2N and r2L
            cnt_ns, mean_ns, sem_ns = _bin_stats(idx_ns, r2_ns_window, n_labels)
            cnt_syn, mean_syn, _ = _bin_stats(idx_syn, r2_syn_window, n_labels)

            # Local LD is the concatenation of the nonsyn and syn measurements
            cnt_L, mean_L, sem_L = _bin_stats(
                np.concatenate([idx_ns, idx_syn]),
                np.concatenate([r2_ns_window, r2_syn_window]),
                n_labels)

            # subset to shared indices
            good = (cnt_ns > 0) & (cnt_syn > 0)
            x = labels[good]

            # intergrate to get auc(r2N) and auc(r2S) and standard error of auc(r2N)
            r2N = r2_utils.auc(mean_ns[good], x)
            r2S = r2_utils.auc(mean_syn[good], x)

            r2N_std = r2_utils.auc(sem_ns[good], x)

            # obtain p-value of r2N > r2S
            pval_window = norm.cdf((r2S - r2N)/r2N_std)

            # integrate to auc(r2L) and standard error of auc(r2L)
            r2L = r2_utils.auc(mean_L[good], x)
            rL_std = r2_utils.auc(sem_L[good], x)

            # store integrals, bins, and p-values for window
            d_bins_dic[site] = d_bins

            sites.append(site)
            r2N_l.append(r2N)
            r2S_l.append(r2S)
            r2L_l.append(r2L)
            rL_std_l.append(rL_std)
            pval_l.append(pval_window)

    del r2_ns, r2_syn, d_ns, d_syn

    ## ------------------------------------------
    ## Global LD decay (baseline)
    ## ------------------------------------------
    sys.stderr.write("\nAUC(r^2_N), AUC(r^2_S), & AUC(r^2_local) calculated.\n\nNow calculating AUC(r^2_genomewide) in each window.")

    df_rnrs = pd.DataFrame(
        {"r2N": r2N_l, "r2S": r2S_l, "r2L": r2L_l, "rL_std": rL_std_l, "rNrS_pval": pval_l},
        index=pd.MultiIndex.from_tuples(sites, names=dfH.index.names[:-1]))

    # identical draw to `all_LD.sample(sample_num, replace=True)`, which is
    # `random_state.choice(len(all_LD), size=sample_num, replace=True)`
    samp = np.random.choice(all_r2.shape[0], size=sample_num, replace=True)
    samp_r2, samp_D = all_r2[samp], all_D[samp]

    del all_r2, all_D

    ## Every window re-bins the *same* bootstrap sample, only the bin edges
    ## change. Collapsing the sample onto its distinct distances once, and
    ## keeping running totals, turns each window's work from a full pass over
    ## `sample_num` rows into a handful of binary searches.
    uniq_D, inv = np.unique(samp_D, return_inverse=True)
    cum_n = np.concatenate(([0], np.cumsum(np.bincount(inv, minlength=uniq_D.shape[0]))))
    cum_r2 = np.concatenate(([0.0], np.cumsum(np.bincount(inv, weights=samp_r2,
                                                          minlength=uniq_D.shape[0]))))

    del samp, samp_r2, samp_D, inv

    all_window_means, all_window_pval = {}, {}
    jc = np.arange(10, 110, 10)
    num_nonsyn_percentiles = np.percentile(range(sum(pd.Series(N_ns))), jc)

    r2L_vals = df_rnrs["r2L"].to_numpy()
    rL_std_vals = df_rnrs["rL_std"].to_numpy()

    # For each window, calculate genomewide expected LD
    j = 0
    for idxx, d_bins in d_bins_dic.items():

        # edges[k] = first sampled distance >= d_bins[k]; measurements in
        # [edges[k], edges[k+1]) are exactly those digitize sends to bin k
        edges = np.searchsorted(uniq_D, d_bins, side="left")
        # measurements exactly equal to the (open) upper limit form their own bin
        edges = np.append(edges, np.searchsorted(uniq_D, d_bins[-1], side="right"))

        n_bin = cum_n[edges[1:]] - cum_n[edges[:-1]]
        occupied = n_bin > 0

        x = d_bins[occupied]
        y = ((cum_r2[edges[1:]] - cum_r2[edges[:-1]])[occupied]) / n_bin[occupied]

        r_global = r2_utils.auc(y, x)
        all_window_means[idxx] = r_global
        all_window_pval[idxx] = (r_global - r2L_vals[j]) / rL_std_vals[j]

        j += 1
        if j > num_nonsyn_percentiles[0] + 1:
            sys.stderr.write(f"\n\t{jc[0]}% complete")
            num_nonsyn_percentiles, jc = num_nonsyn_percentiles[1:], jc[1:]

    # Finalize genomewide baseline
    all_window_means = pd.Series(all_window_means)
    all_window_pval = norm.cdf(pd.Series(all_window_pval))

    df_rnrs["r2G"] = all_window_means
    df_rnrs["r2G_pval"] = all_window_pval

    ## ------------------------------------------
    ## Test statistics and iLDS
    ## ------------------------------------------
    # Difference nonsyn - syn (r2N - r2S)
    df_rnrs["rNrS"] = df_rnrs["r2N"] - df_rnrs["r2S"]
    # transformed to mean 0, std 1
    df_rnrs["rNrS_trans"] = (df_rnrs["rNrS"] - df_rnrs["rNrS"].mean()) / df_rnrs["rNrS"].std()

    # Difference local - genomewide (r2L - r2G)
    df_rnrs["r_rg"] = df_rnrs["r2L"] - df_rnrs["r2G"]
    # transformed to mean 0, std 1
    df_rnrs["r_rg_transform"] = (df_rnrs["r_rg"] - df_rnrs["r_rg"].mean()) / df_rnrs["r_rg"].std()

    # iLDS statistic = squared sum of both normalized contrasts
    df_rnrs["iLDS"] = df_rnrs["r_rg_transform"]**2 + df_rnrs["rNrS_trans"]**2
    # chi2 test w/ 2 df as sum of squares of ~ standard normals under neutrality
    df_rnrs["iLDS_pval"] = chi2.sf(df_rnrs["iLDS"], df=2)

    # Call significance only if all tests pass at threshold alpha
    df_rnrs["significance"] = ((df_rnrs["iLDS_pval"] < alpha) &
                               (df_rnrs["rNrS_pval"] < alpha) &
                               (df_rnrs["r2G_pval"] < alpha))

    sys.stderr.write("\nAll components of iLDS calculated, and significance assessed. Scan complete.\n")

    return df_rnrs


## --------------------------------------------------
## 4. Plotting utility
## --------------------------------------------------

def contig_offsets(df_rnrs):
    """Cumulative offset per contig, laying the contigs end to end."""
    off_dic, off = {}, 0
    for contig, df_group in df_rnrs.groupby("contig"):
        off_dic[contig] = off
        off += df_group.index.get_level_values("site_pos").max()

    return off_dic


def plot_scan(df_rnrs, cluster_endpoints, clus_contig,
              output_scan_dir=None, offset=5*1e4,
              fig=None, ax=None, ax_scat=None, legend=False,
              off_dic=None, xmax=None, tight=True):
    """
    Plot genomewide iLDS scan with significant windows highlighted.

    Parameters:
        df_rnrs (pd.DataFrame): scan results
        cluster_endpoints (dict): start/end coords of peaks
        clus_contig (dict): mapping of cluster -> contig
        output_scan_dir (str): if given, save figure here
        offset (float): padding for plot
        fig, ax: optional matplotlib objects
        ax_scat: optional axis for the peak strip under the scan. Omitted when
            drawing into a caller-supplied `fig` without one.
        legend (bool): whether to draw legend
        off_dic (dict, optional): contig -> offset. Supply the same mapping for
            several scans to put them on one genome coordinate system.
        xmax (float, optional): right hand limit, shared the same way.
        tight (bool): run tight_layout. Turn off when the caller lays out a
            figure of several scans and will do it once at the end.

    Returns:
        fig, ax: matplotlib objects
    """
    if fig is None:
        fig = plt.figure(figsize=(16,6))
        gs = gridspec.GridSpec(12, 10, hspace=0.2, wspace=0)
        ax = fig.add_subplot(gs[:11, :])
        ax_scat = fig.add_subplot(gs[11:, :])
        ax_scat.set_facecolor("snow")

    # Compute linearized genome positions for plotting
    if off_dic is None:
        off_dic = contig_offsets(df_rnrs)

    df_rnrs = df_rnrs.sort_index(level=["contig","site_pos"])
    # Offset each contig onto a single linear axis (one aligned add, rather
    # than a chained `.loc[contig][col] +=`, which is a no-op under
    # copy-on-write pandas)
    df_rnrs["all_site_pos"] = (df_rnrs.index.get_level_values("site_pos").values
                               + df_rnrs.index.get_level_values("contig").map(off_dic).values)
    df_rnrs.set_index('all_site_pos', append=True, inplace=True)

    # Significant calls
    df_rnrs_pass = df_rnrs.query("significance == True")
    site_pos = df_rnrs.index.get_level_values("all_site_pos")

    # Axis limits
    ax.set_xlim(site_pos.min() - offset,site_pos.max() + offset)

    ax.set_xticks([])

    if ax_scat is not None:
        ax_scat.set_xlim(site_pos.min() - offset,site_pos.max() + offset)
        ax_scat.set_ylim([0,1])
        ax_scat.set_yticks([])

    ax.axhline(0,color="k",alpha=1,lw=1.5)

    ax.scatter(site_pos,df_rnrs["iLDS"].values,color="darkcyan",zorder=9,edgecolor="grey",lw=.2,s=50,label="Not significant")

    ax.scatter(df_rnrs_pass.index.get_level_values("all_site_pos"),df_rnrs_pass["iLDS"].values,color="tomato",zorder=9,edgecolor="grey",lw=.2,s=75,label="Significant")

    ax.set_ylabel("iLDS",size=15)

    if ax_scat is not None:
        ax_scat.set_xlabel("Genome position (bp)",size=15)

        off = 1e3
        for key in cluster_endpoints.keys():

            item = list(cluster_endpoints[key])

            item[0] = item[0] + off_dic[clus_contig[key]]
            item[1] = item[1] + off_dic[clus_contig[key]]

            ax_scat.add_patch(Rectangle((item[0] - off, 0), (item[1] - item[0] + 2*off), 1,facecolor="tomato",edgecolor="tomato",lw=2.5))

    if xmax is None:
        xmax = site_pos.max()

    ax.set_xlim([0,xmax*1.05])
    if ax_scat is not None:
        ax_scat.set_xlim([0,xmax*1.05])

    if legend:
        fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.865),prop={"size":15})
    if tight:
        fig.tight_layout()
    
    if output_scan_dir is not None:
        fig.savefig(f"{output_scan_dir}/scan_figure")

    return(fig,ax)
