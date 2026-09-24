import numpy as np
import pandas as pd
from numba import njit, prange
import matplotlib.pyplot as plt
from scipy import integrate
from statistics import NormalDist

## `scipy.integrate.trapz` was renamed `trapezoid` (and removed in SciPy >= 1.14)
_trapezoid = getattr(integrate, "trapezoid", None) or integrate.trapz


######### Pairwise r^2 kernels #########
##
## All kernels below share the same arithmetic:
##
##     for a pair of sites (i,j), restrict to the samples covered at *both*
##     sites, then
##         f_A  = mean(x_i),  f_B = mean(x_j),  f_AB = mean(x_i * x_j)
##         r^2  = (f_AB - f_A f_B)^2 / (f_A f_B (1 - f_A)(1 - f_B))
##
## Genotypes are exactly 0.0/1.0, so the running sums below are exact integers
## in float64 and therefore agree bit-for-bit with the equivalent
## `np.mean(a[mask])` formulation, while avoiding the temporary mask/gather
## arrays that dominated the original inner loop.


@njit(cache=True, inline="always")
def _pair_r2(a, b, M):
    """r^2 between two genotype vectors, restricted to jointly covered samples."""
    n = 0
    s_a = 0.0
    s_b = 0.0
    s_ab = 0.0

    for k in range(M):
        av = a[k]
        bv = b[k]
        # `av == av` is a NaN test that numba compiles to a single instruction
        if av == av and bv == bv:
            n += 1
            s_a += av
            s_b += bv
            s_ab += av * bv

    if n == 0:
        return np.nan

    f_a = s_a / n
    f_b = s_b / n
    f_ab = s_ab / n

    f_a_f_b = f_a * f_b
    if f_a_f_b == 0.0:
        return np.nan

    return ((f_ab - f_a_f_b) ** 2) / (f_a_f_b * (1.0 - f_a) * (1.0 - f_b))


@njit(cache=True, parallel=True)
def quick_r2(dfH_ns_v, S, N):
    """
    Upper-triangular r^2 matrix, computed only for the pairs flagged in `S`.

    Retained for backwards compatibility; `banded_r2` is preferred inside the
    scan because it does not allocate an N x N matrix.
    """
    M = dfH_ns_v.shape[1]
    R2 = np.full((N, N), np.nan)

    for i in prange(N - 1):
        a = dfH_ns_v[i]
        for j in range(i + 1, N):
            if S[i, j]:
                R2[i, j] = _pair_r2(a, dfH_ns_v[j], M)

    return R2


@njit(cache=True, parallel=True)
def banded_r2(X, j_end, W):
    """
    r^2 for every pair (i,j) with i < j < j_end[i], stored in banded form.

    The windowed masks used by the scan only ever link sites that are close
    together along the contig, so the full N x N matrix is almost entirely
    empty. Storing band `k = j - i - 1` instead keeps memory at O(N * W)
    rather than O(N^2).

    Args:
        X (np.ndarray): (N, M) float64 genotype matrix for one contig.
        j_end (np.ndarray): (N,) exclusive upper bound on j for each i.
        W (int): band width, i.e. max(j_end - arange(N)) - 1.

    Returns:
        np.ndarray: (N, W) array of r^2 values, NaN outside the band.
    """
    N = X.shape[0]
    M = X.shape[1]
    R2 = np.full((N, W), np.nan)

    for i in prange(N):
        a = X[i]
        for j in range(i + 1, j_end[i]):
            R2[i, j - i - 1] = _pair_r2(a, X[j], M)

    return R2


@njit(cache=True, parallel=True)
def pairwise_r2(X):
    """
    Full symmetric r^2 matrix for all pairs of sites (diagonal included).

    Equivalent to `calc_r2(*return_F_FAB(X, N), N)` after the F == 0 / F == 1
    entries have been NaN-ed out, but in a single pass and without the two
    intermediate N x N frequency matrices.
    """
    N = X.shape[0]
    M = X.shape[1]
    R2 = np.empty((N, N))

    for i in prange(N):
        a = X[i]
        for j in range(i, N):
            n = 0
            s_a = 0.0
            s_b = 0.0
            s_ab = 0.0

            b = X[j]
            for k in range(M):
                av = a[k]
                bv = b[k]
                if av == av and bv == bv:
                    n += 1
                    s_a += av
                    s_b += bv
                    s_ab += av * bv

            if n == 0:
                r2 = np.nan
            else:
                f_a = s_a / n
                f_b = s_b / n
                # frequencies of 0 or 1 carry no LD information
                if f_a == 0.0 or f_a == 1.0 or f_b == 0.0 or f_b == 1.0:
                    r2 = np.nan
                else:
                    f_ab = s_ab / n
                    f_a_f_b = f_a * f_b
                    r2 = ((f_ab - f_a_f_b) ** 2) / (f_a_f_b * (1.0 - f_a) * (1.0 - f_b))

            R2[i, j] = r2
            R2[j, i] = r2

    return R2


@njit(cache=True, parallel=True)
def calc_r2(F, F_AB, N):

    df_r2 = np.zeros((N, N))

    for i in prange(N):
        for j in range(i, N):

            p_A = F[i, j]
            p_B = F[j, i]
            p_AB = F_AB[i, j]

            if p_A == p_A and p_B == p_B and p_AB == p_AB:

                p_Ap_B = p_A * p_B
                r2 = ((p_AB - p_Ap_B) ** 2) / (p_Ap_B * (1 - p_A) * (1 - p_B))

            else:

                r2 = np.nan

            df_r2[i, j] = r2
            df_r2[j, i] = r2

    return df_r2


### takes allele frequencies (f_A,f_B,f_AB) w/r/t samples where both sites are covered
### necessary in r2 calculations to account for missing data
@njit(cache=True, parallel=True)
def return_F_FAB(df, N):

    M = df.shape[1]

    F = np.zeros((N, N))
    F_AB = np.zeros((N, N))

    for i in prange(N):
        a = df[i]
        for j in range(i, N):

            b = df[j]

            n = 0
            s_a = 0.0
            s_b = 0.0
            s_ab = 0.0

            for k in range(M):
                av = a[k]
                bv = b[k]
                if av == av and bv == bv:
                    n += 1
                    s_a += av
                    s_b += bv
                    s_ab += av * bv

            if n > 0:

                F[i][j] = s_a / n
                F[j][i] = s_b / n

                shared_locs = s_ab / n

                F_AB[i][j] = shared_locs
                F_AB[j][i] = shared_locs

            else:

                F[i][j] = np.nan
                F[j][i] = np.nan
                F_AB[i][j] = np.nan
                F_AB[j][i] = np.nan

    return (F, F_AB)


@njit(cache=True)
def block_triu(band, l, u):
    """
    Upper triangle (k=1) of the sub-block [l,u) x [l,u) of a banded matrix.

    Values come out in the same order as `take_triu` on the corresponding dense
    sub-block: row-major, i.e. (l,l+1), (l,l+2), ..., (u-2,u-1).
    """
    n = u - l
    if n < 2:
        # empty block: matches `take_triu` on a degenerate/inverted slice
        return np.empty(0, band.dtype)

    out = np.empty((n * (n - 1)) // 2, band.dtype)

    p = 0
    for a in range(l, u):
        for k in range(u - a - 1):
            out[p] = band[a, k]
            p += 1

    return out


@njit(cache=True)
def band_to_long(r2_band, d_band, j_end):
    """
    Flatten a banded r^2 matrix (and its companion distance band) to the
    non-NaN (r2, D) pairs, in dense-upper-triangle order.

    Equivalent to `take_triu(dense_r2)` / `take_triu(dense_D)` followed by
    `dropna()`, without ever materialising the N x N matrices.
    """
    N = r2_band.shape[0]

    n_keep = 0
    for i in range(N):
        for j in range(i + 1, j_end[i]):
            if r2_band[i, j - i - 1] == r2_band[i, j - i - 1]:
                n_keep += 1

    r2_out = np.empty(n_keep)
    d_out = np.empty(n_keep)

    p = 0
    for i in range(N):
        for j in range(i + 1, j_end[i]):
            v = r2_band[i, j - i - 1]
            if v == v:
                r2_out[p] = v
                d_out[p] = d_band[i, j - i - 1]
                p += 1

    return r2_out, d_out


def distance_band(site_pos, W):
    """Banded companion to `banded_r2`: D[i, k] = |site_pos[i+k+1] - site_pos[i]|."""
    site_pos = np.asarray(site_pos, dtype=np.float64)
    N = site_pos.shape[0]

    d_band = np.zeros((N, W))
    for k in range(W):
        n = N - k - 1
        if n <= 0:
            break
        np.abs(site_pos[k + 1:] - site_pos[:n], out=d_band[:n, k])

    return d_band


## returns a pandas dataframe with columns (r2,D)
def return_r2(df):

    df_r2 = pairwise_r2(np.ascontiguousarray(df.values, dtype=np.float64))
    df_r2 = pd.DataFrame(df_r2, index=df.index, columns=df.index)

    return (df_r2)


## Function for calculating AUC
def ld_integrate(df_r2, int_key="r2"):

    return auc(df_r2[int_key].values, df_r2.D_bins.values)


## Bare-array form of `ld_integrate`, used on the hot paths of the scan
def auc(y, x):

    span = x.max() - x.min()

    return _trapezoid(y, x) / span


## General purpose utility for easily taking the upper triangle of an array
def take_triu(df):

    N = df.shape[0]
    p = np.triu_indices(N, k=1)

    return (df[p])

def take_ld_mean(LD_gb,key="r2"):
    return(LD_gb.mean()[key])

def take_ld_CI(LD_gb,CI_thresh,key="r2"):
    
    alpha = NormalDist().inv_cdf((1 + CI_thresh) / 2.)
    
    LD_std = LD_gb.std()[key]
    LD_n = LD_gb.size()
    
    return((alpha*LD_std/np.sqrt(LD_n)))

def bound_ld_CI(LD_gb,CI_thresh=0.999,key="r2"):
    
    LD_ld = take_ld_mean(LD_gb,key)
    
    LD_CI = take_ld_CI(LD_gb,CI_thresh=CI_thresh,key=key)
    
    LD_ld_out = pd.DataFrame(LD_ld,columns=[key])
    
    LD_ld_out[f"{key}_min"] = LD_ld - LD_CI
    LD_ld_out[f"{key}_max"] = LD_ld + LD_CI
    
    return(LD_ld_out.reset_index())

def plot_ld_CI(df_ld_dic,key,fig=None,ax=None):
    
    if fig is None and ax is None:
        fig,ax = plt.subplots(figsize=(12,8))
    
    df_1_ld = df_ld_dic["1"]
    df_4_ld = df_ld_dic["4"]
    
    if df_1_ld.shape[0] > 0:
        
        ax.plot(df_1_ld.D_bins,df_1_ld[key],color="red",zorder=20,lw=7)
        ax.plot(df_1_ld.D_bins,df_1_ld[key],color="k",zorder=11,lw=7.5)
        ax.fill_between(df_1_ld.D_bins,df_1_ld[f"{key}_min"],df_1_ld[f"{key}_max"],color="tomato",alpha=.1)

    ax.plot(df_4_ld.D_bins,df_4_ld[key],color="deepskyblue",zorder=10,lw=7)
    ax.plot(df_4_ld.D_bins,df_4_ld[key],color="k",zorder=1,lw=7.5)
