"""
Jaccard index between every pair of populations, per species.

Reads the table `shared_sweeps.py` writes and asks, for each species and each
pair of populations, what fraction of the sweeps either of them carries are
carried by both:

    J(i, j) = |sweeps in i and j| / |sweeps in i or j|

One matrix per species is written to `analysis/jaccard_species/<species>.csv`,
populations on both axes and NaN down the diagonal, matching the layout the
rest of the analysis expects.

A population the species was never scanned in cannot be compared: its row and
column are left out of that species' matrix entirely, rather than counted as a
population that simply has no sweeps.
"""

import numpy as np
import pandas as pd
import argparse
import os
import sys

import config
import shared_sweeps

JACCARD_SUBDIR = "jaccard_species"


def default_analysis_dir():
    """
    Where to put `jaccard_species/`.

    `config.analysis_dir` when it can be created -- its parent exists -- and a
    local `analysis/` otherwise, so a checkout works without editing config.
    """
    configured = getattr(config, "analysis_dir", None)

    if configured and os.path.isdir(os.path.dirname(configured.rstrip("/"))):
        return configured

    return "analysis"


def jaccard(present_i, present_j, empty_union=np.nan):
    """
    Jaccard index of two presence vectors.

    Args:
        present_i, present_j (np.ndarray): boolean, one entry per sweep.
        empty_union: what to return when neither population carries any sweep,
            so the index is 0/0. NaN by default -- two populations with nothing
            to compare are not evidence of similarity.

    Returns:
        float
    """
    either = int(np.count_nonzero(present_i | present_j))

    if either == 0:
        return empty_union

    return int(np.count_nonzero(present_i & present_j)) / either


def species_jaccard(sweeps, populations=None, empty_union=np.nan):
    """
    Pairwise Jaccard matrix for one species' sweeps.

    Args:
        sweeps (pd.DataFrame): the rows of the shared-sweeps table for one
            species, with one column per population holding 1/0/NA.
        populations (iterable, optional): restrict to these populations.
        empty_union: value for a pair whose union is empty.

    Returns:
        pd.DataFrame: square, symmetric, NaN on the diagonal. Only populations
        the species was actually scanned in appear.
    """
    candidates = [c for c in sweeps.columns if c not in shared_sweeps.SWEEP_COLUMNS]

    if populations is not None:
        candidates = [c for c in candidates if c in set(populations)]

    # a population with any NA here was never scanned for this species
    scanned = sorted(c for c in candidates if sweeps[c].notna().all())

    if not scanned:
        return pd.DataFrame()

    present = {p: sweeps[p].to_numpy(dtype=float) > 0 for p in scanned}

    matrix = pd.DataFrame(np.nan, index=scanned, columns=scanned, dtype=float)

    for a, pop_i in enumerate(scanned):
        for pop_j in scanned[a + 1:]:
            value = jaccard(present[pop_i], present[pop_j], empty_union)
            matrix.loc[pop_i, pop_j] = value
            matrix.loc[pop_j, pop_i] = value

    # the diagonal is left NaN, as in the rest of the analysis: a population
    # compared with itself is 1 by construction and carries no information
    return matrix


def write_species_matrices(table, out_dir, populations=None, empty_union=np.nan):
    """
    Write one Jaccard matrix per species.

    Returns:
        dict: species -> path written.
    """
    os.makedirs(out_dir, exist_ok=True)

    written = {}

    for species, sweeps in table.groupby("species", sort=True):

        matrix = species_jaccard(sweeps, populations=populations,
                                 empty_union=empty_union)

        if matrix.empty:
            sys.stderr.write(f"  {species}: no population scanned, skipped\n")
            continue

        path = os.path.join(out_dir, f"{species}.csv")
        # `nan` spelled out, as the existing jaccard_species files do
        matrix.to_csv(path, na_rep="nan")

        upper = matrix.to_numpy()[np.triu_indices(matrix.shape[0], 1)]
        finite = upper[np.isfinite(upper)]

        sys.stderr.write(f"  {species}: {matrix.shape[0]} populations, "
                         f"{sweeps.shape[0]} sweeps, "
                         f"mean J = {finite.mean():.3f}\n" if finite.size
                         else f"  {species}: {matrix.shape[0]} populations, "
                              f"{sweeps.shape[0]} sweeps, no comparable pairs\n")

        written[species] = path

    return written


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', default=None, type=str,
                        help="Shared sweeps table (default <scan_directory>/shared_sweeps.txt)")

    parser.add_argument('-o', '--analysis_dir', default=None, type=str,
                        help=f"Folder to write {JACCARD_SUBDIR}/ into. Defaults to config.analysis_dir.")

    parser.add_argument('--species', default=None, type=str,
                        help="Comma-separated subset of species")

    parser.add_argument('--populations', default=None, type=str,
                        help="Comma-separated subset of populations")

    parser.add_argument('--empty_union', default="nan", choices=["nan", "zero"],
                        help="Value for a pair of populations that between them carry no sweeps, "
                             "so the index is 0/0. 'nan' by default; 'zero' matches the older convention.")

    args = parser.parse_args()

    table_path = args.input or os.path.join(
        getattr(config, "scan_directory", "test_output"), "shared_sweeps.txt")

    if not os.path.exists(table_path):
        parser.error(f"{table_path} not found; run shared_sweeps.py first")

    table = pd.read_csv(table_path, sep="\t")

    if args.species:
        wanted = [s for s in args.species.split(",") if s]
        table = table.loc[table["species"].isin(wanted)]

    populations = None
    if args.populations:
        populations = [p for p in args.populations.split(",") if p]

    out_dir = os.path.join(args.analysis_dir or default_analysis_dir(), JACCARD_SUBDIR)

    empty_union = np.nan if args.empty_union == "nan" else 0.0

    sys.stderr.write(f"{table.shape[0]} sweeps, {table['species'].nunique()} species "
                     f"from {table_path}\n\n")

    written = write_species_matrices(table, out_dir, populations=populations,
                                     empty_union=empty_union)

    sys.stderr.write(f"\n{len(written)} matrices written to {out_dir}\n")
