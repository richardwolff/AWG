"""
Work out which sweeps are shared between which populations.

Every population of a species is scanned separately, so the same sweep is
called once per population that carries it, at slightly different coordinates.
This groups those calls together: two sweeps belong to the same group when the
gap between them -- the left endpoint of the right sweep minus the right
endpoint of the left one -- is no wider than `config.sweep_merge_dist`. Sweeps
that genuinely overlap have a negative gap and always merge.

Run over every species in `config.good_species`, it writes one file covering
all of them: one row per sweep group, identified by a hash, with a column per
population holding

    1   the population was scanned and called this sweep
    0   the population was scanned and did not call it
    NA  the species was never scanned in that population

Species with no finished scan contribute no sweeps; they are listed on stderr
so the file's coverage is never silently partial.
"""

import numpy as np
import pandas as pd
import argparse
import hashlib
import os
import sys
from collections import defaultdict

import config
import peak_utils
import population_utils

SWEEP_COLUMNS = ["species", "sweep", "contig", "start", "end", "length",
                 "n_populations", "populations"]


def sweep_hash(species, contig, start, end, length=12):
    """
    Stable identifier for a sweep group.

    Derived from the species, contig and the group's extent, so the same group
    gets the same hash on a rerun. Adding or removing populations can move the
    extent, and will then move the hash.
    """
    key = f"{species}|{contig}|{int(start)}|{int(end)}"

    return hashlib.sha1(key.encode()).hexdigest()[:length]


def scanned_populations(species_dir):
    """Populations of a species that have a finished scan."""
    if not os.path.isdir(species_dir):
        return []

    return sorted(name for name in os.listdir(species_dir)
                  if os.path.isfile(os.path.join(species_dir, name, "full_scan.txt"))
                  and os.path.isfile(os.path.join(species_dir, name, "peaks.pkl")))


def population_sweeps(species_dir, population):
    """
    Sweeps called in one population, as (contig, start, end).

    Endpoints are the first and last significant site in the peak, exactly as
    `plot_scan` draws them.
    """
    clusters = pd.read_pickle(os.path.join(species_dir, population, "peaks.pkl"))

    endpoints, contigs = peak_utils.return_cluster_endpoints(clusters)

    return [(contigs[key], int(endpoints[key][0]), int(endpoints[key][1]))
            for key in sorted(endpoints)]


def merge_sweeps(sweeps, max_gap=config.sweep_merge_dist):
    """
    Group sweeps that sit within `max_gap` of one another along a contig.

    Args:
        sweeps (iterable): (population, contig, start, end) tuples.
        max_gap (float): widest gap that still counts as the same sweep.

    Returns:
        list[dict]: one per group, with contig, start, end and the set of
        populations carrying it, ordered by contig then position.
    """
    by_contig = defaultdict(list)
    for population, contig, start, end in sweeps:
        by_contig[contig].append((start, end, population))

    groups = []

    for contig in sorted(by_contig):

        current = None

        # sorting by start lets one pass do the grouping; `end` is carried as a
        # running maximum because a later sweep can finish earlier
        for start, end, population in sorted(by_contig[contig]):

            if current is not None and start - current["end"] <= max_gap:
                current["end"] = max(current["end"], end)
                current["populations"].add(population)
                current["n_calls"] += 1
            else:
                current = {"contig": contig, "start": start, "end": end,
                           "populations": {population}, "n_calls": 1}
                groups.append(current)

    return groups


def species_sweep_table(species, species_dir, populations=None,
                        max_gap=config.sweep_merge_dist):
    """
    Sweep groups for one species, with presence per population.

    Populations that were never scanned are left out here; the caller fills
    them in as NA once it knows the full set of populations.

    Returns:
        (pd.DataFrame, list): the table, and the populations that were scanned.
    """
    scanned = scanned_populations(species_dir)

    if populations is not None:
        scanned = [p for p in scanned if p in set(populations)]

    if not scanned:
        return pd.DataFrame(columns=SWEEP_COLUMNS), []

    sweeps = [(population,) + sweep
              for population in scanned
              for sweep in population_sweeps(species_dir, population)]

    groups = merge_sweeps(sweeps, max_gap=max_gap)

    rows = []
    for group in groups:
        row = {"species": species,
               "sweep": sweep_hash(species, group["contig"], group["start"], group["end"]),
               "contig": group["contig"],
               "start": group["start"],
               "end": group["end"],
               "length": group["end"] - group["start"],
               "n_populations": len(group["populations"]),
               "populations": ",".join(sorted(group["populations"]))}

        for population in scanned:
            row[population] = int(population in group["populations"])

        rows.append(row)

    return pd.DataFrame(rows), scanned


def build_table(species_list, scan_dir, populations=None,
                max_gap=config.sweep_merge_dist):
    """
    Sweep sharing across every species, as one table.

    Species with no finished scan are reported and skipped rather than raising,
    so a run over the whole of `good_species` completes whatever subset has
    been scanned so far.

    A species/population pair that was never scanned is NA, which is what
    distinguishes "no sweep here" from "never looked".

    Returns:
        (pd.DataFrame, dict): the table, and species -> scanned populations.
    """
    tables, scanned_by_species, skipped = [], {}, []

    for species in species_list:

        species_dir = os.path.join(scan_dir, species)

        try:
            table, scanned = species_sweep_table(species, species_dir,
                                                 populations=populations,
                                                 max_gap=max_gap)
        except Exception as error:
            sys.stderr.write(f"  {species}: FAILED to read scans ({error})\n")
            skipped.append(species)
            continue

        if not scanned:
            skipped.append(species)
            continue

        sys.stderr.write(f"  {species}: {table.shape[0]} sweep groups "
                         f"across {len(scanned)} populations\n")

        tables.append(table)
        scanned_by_species[species] = set(scanned)

    if skipped:
        sys.stderr.write(f"\n{len(skipped)} species with no finished scan, "
                         f"contributing no sweeps:\n")
        for species in skipped:
            sys.stderr.write(f"  {species}\n")

    if not tables:
        sys.stderr.write("\nNo scans found for any species\n")
        return pd.DataFrame(columns=SWEEP_COLUMNS), scanned_by_species

    combined = pd.concat(tables, ignore_index=True)

    all_populations = sorted({p for scanned in scanned_by_species.values() for p in scanned})
    all_populations = population_utils.sort_populations_by_region(all_populations)

    # a population this species was never scanned in is NA, not 0
    for population in all_populations:
        if population not in combined.columns:
            combined[population] = pd.NA

        unscanned = ~combined["species"].map(
            lambda s: population in scanned_by_species.get(s, set()))
        combined.loc[unscanned, population] = pd.NA

        combined[population] = combined[population].astype("Int64")

    return combined[SWEEP_COLUMNS + all_populations], scanned_by_species


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--species', default=None, type=str,
                        help="Comma-separated species. Defaults to every species in config.good_species.")

    parser.add_argument('-o', '--output_path',
                        default=getattr(config, "scan_directory", "test_output"), type=str,
                        help="Folder holding <species>/<population>/. Defaults to config.scan_directory.")

    parser.add_argument('--populations', default=None, type=str,
                        help="Comma-separated subset of populations to consider")

    parser.add_argument('--max_gap', default=config.sweep_merge_dist, type=float,
                        help="Widest gap between two sweeps that still counts as one sweep")

    parser.add_argument('--out', default=None, type=str,
                        help="Where to write the table (default <output_path>/shared_sweeps.txt)")

    args = parser.parse_args()

    if args.species:
        species_list = [s for s in args.species.split(",") if s]
    else:
        species_list = list(config.good_species)

    populations = None
    if args.populations:
        populations = [p for p in args.populations.split(",") if p]

    sys.stderr.write(f"{len(species_list)} species, grouping sweeps within "
                     f"{args.max_gap:g} bp of one another\n\n")

    table, scanned_by_species = build_table(species_list, args.output_path,
                                            populations=populations,
                                            max_gap=args.max_gap)

    out = args.out or os.path.join(args.output_path, "shared_sweeps.txt")
    table.to_csv(out, sep="\t", index=False, na_rep="NA")

    sys.stderr.write(f"\n{table.shape[0]} sweep groups from "
                     f"{len(scanned_by_species)}/{len(species_list)} species "
                     f"written to {out}\n")
