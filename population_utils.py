"""
Resolve which strains belong to which population, for an arbitrary species.

The strain metadata is a table with one row per genome; the columns this module
needs are `Genome`, `Species`, `Population` and `good_strain`. Populations are
restricted to a curated list (``good_populations.txt``) and to those with enough
genomes to be worth scanning.
"""

import numpy as np
import pandas as pd
import os
import sys
from collections import defaultdict

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import config

DEFAULT_METADATA = config.metadata_file
DEFAULT_GOOD_POPULATIONS = config.good_populations_file
DEFAULT_POPULATION_REGIONS = config.population_regions_file

GENOME_COL = "Genome"
SPECIES_COL = "Species"
POPULATION_COL = "Population"
GOOD_STRAIN_COL = "good_strain"

REQUIRED_COLUMNS = ["contig", "gene_id", "site_pos", "site_type"]


def _first_existing(*candidates):
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def _metadata_path(filename):
    """Locate a metadata file via config, a local metadata/, or the cwd."""
    resolver = getattr(config, "metadata_path", None)
    if resolver is not None:
        found = resolver(filename)
        if os.path.exists(found):
            return found

    return _first_existing(os.path.join(getattr(config, "metadata_dir", ""), filename),
                           os.path.join("metadata", filename),
                           filename)


def load_metadata(path=None):
    """
    Read the strain metadata table.

    Looks at `path`, then the working directory, then `config.metadata_dir`.
    """
    path = _first_existing(path, _metadata_path(DEFAULT_METADATA))

    if path is None:
        raise ValueError(f"Could not find strain metadata ({DEFAULT_METADATA})")

    metadata = pd.read_csv(path)

    missing = [c for c in [GENOME_COL, SPECIES_COL, POPULATION_COL] if c not in metadata.columns]
    if missing:
        raise ValueError(f"Metadata is missing required columns: {missing}")

    return metadata


def load_good_populations(path=None):
    """Read the curated population list, or None if there isn't one."""
    path = _first_existing(path, _metadata_path(DEFAULT_GOOD_POPULATIONS))

    if path is None:
        return None

    return [line.strip() for line in open(path) if line.strip()]


def alignment_samples(haplotype_path):
    """
    Sample names present in an alignment, without reading any genotypes.

    Reads the parquet schema or the header line, so this stays cheap even for a
    multi-gigabyte alignment.
    """
    import haplotype_utils

    if os.path.isdir(haplotype_path):
        names = sorted(os.listdir(haplotype_path))
        parquet = [f for f in names if haplotype_utils._is_parquet(f)]
        names = parquet or names

        for name in names:
            candidate = os.path.join(haplotype_path, name)
            if os.path.isfile(candidate):
                haplotype_path = candidate
                break

    if haplotype_utils._is_parquet(haplotype_path):
        columns = haplotype_utils._parquet_columns(haplotype_path)
    else:
        compression = "gzip" if str(haplotype_path).endswith(".gz") else None
        header = haplotype_utils._read_header(haplotype_path, compression)
        delimiter = "," if "," in header else "\t"
        columns = header.rstrip("\r\n").split(delimiter)

    return [c for c in columns if c not in REQUIRED_COLUMNS]


def load_population_regions(path=None):
    """
    Map each population to its region.

    Returns:
        dict: population -> region, or {} when there is no such file.
    """
    path = _first_existing(path, _metadata_path(DEFAULT_POPULATION_REGIONS))

    if path is None:
        return {}

    regions = pd.read_csv(path, sep="\t")

    return dict(zip(regions["population"].astype(str).str.strip(),
                    regions["region"].astype(str).str.strip()))


def sort_populations_by_region(populations, regions=None, order=None):
    """
    Group populations by region and sort alphabetically inside each.

    Regions come out in `order` (`config.region_order` by default); any region
    not named there follows, alphabetically, and populations with no region at
    all come last.

    Args:
        populations (iterable): population names.
        regions (dict, optional): population -> region.
        order (list, optional): region order.

    Returns:
        list: populations, grouped and sorted.
    """
    if regions is None:
        regions = load_population_regions()

    if order is None:
        order = list(getattr(config, "region_order", []))

    rank = {region: i for i, region in enumerate(order)}

    # regions present but unranked keep a stable, alphabetical place after them
    unranked = sorted({regions.get(p) for p in populations
                       if regions.get(p) is not None and regions.get(p) not in rank})
    for i, region in enumerate(unranked):
        rank[region] = len(order) + i

    missing = sorted(p for p in populations if p not in regions)
    if missing:
        sys.stderr.write(f"No region for {', '.join(missing)}; placing them last\n")

    def key(population):
        region = regions.get(population)
        return (len(rank) if region is None else rank[region], population)

    return sorted(populations, key=key)


######### Clonal clusters #########
##
## Closely related strains (ds < config.close_pair_thresh) are near-identical
## copies of one another. Leaving them all in inflates LD -- every clonal pair
## contributes r^2 ~ 1 at every distance -- so a scan should see one
## representative per clonal cluster.


def divergence_path(species, path=None):
    """
    Locate the pairwise synonymous divergence matrix for a species.

    Tries an explicit path, then `config.genetic_distances_df` (with and
    without the `_ds` suffix the files sometimes carry), then a local
    `config.genetic_distances_dir`.
    """
    candidates = [path]

    template = getattr(config, "genetic_distances_df", None)
    if template and "%s" in template:
        candidates += [template % species, template % f"{species}_ds"]

    local = getattr(config, "genetic_distances_dir", "genomewide_divergence")
    if local:
        candidates += [os.path.join(local, f"{species}_ds.txt"),
                       os.path.join(local, f"{species}.txt")]

    found = _first_existing(*candidates)

    if found is None:
        raise ValueError(f"No divergence matrix for {species}; looked in: "
                         + ", ".join(c for c in candidates if c))

    return found


def load_divergences(species=None, strains=None, path=None):
    """
    Square matrix of pairwise divergence, restricted to `strains`.

    These matrices are one row and column per genome in the whole cohort and
    run to gigabytes, so only the requested columns are decoded and only the
    requested rows kept.

    Args:
        species (str, optional): used to locate the file.
        strains (iterable, optional): subset to keep. None reads everything.
        path (str, optional): explicit path to the matrix.

    Returns:
        pd.DataFrame: symmetric divergence matrix indexed by genome.
    """
    path = path or divergence_path(species)

    # take the header as written: the leading index column is usually unnamed,
    # and pandas renames it while pyarrow does not
    with open(path) as handle:
        header = handle.readline().rstrip("\r\n")

    delimiter = "," if "," in header else "\t"
    columns = header.split(delimiter)

    if strains is None:
        divergence = pd.read_csv(path, sep=delimiter, index_col=0)
    else:
        positions = {name: i for i, name in enumerate(columns)}
        # strains with no column here are handled by the caller as singletons
        wanted = [s for s in dict.fromkeys(strains) if s in positions]

        if not wanted:
            return pd.DataFrame(index=pd.Index([], name="genome"))

        try:
            import pyarrow.csv as pv

            table = pv.read_csv(path, convert_options=pv.ConvertOptions(
                include_columns=[columns[0]] + wanted))
            divergence = table.to_pandas().set_index(columns[0])
        except ImportError:
            divergence = pd.read_csv(path, sep=delimiter, index_col=0,
                                     usecols=[0] + [positions[s] for s in wanted])

        divergence = divergence.loc[divergence.index.isin(set(wanted))]

    divergence.index.name = "genome"

    return divergence


def clonal_clusters(divergence, threshold=config.close_pair_thresh):
    """
    Groups of strains joined by divergence below `threshold`.

    Strains are nodes; an edge is drawn between any two whose divergence is
    below the threshold, and each connected component is one clonal cluster.
    Relatedness is transitive here by construction: A-B and B-C close enough
    puts A and C in the same cluster even if they are not themselves close.

    A missing divergence is not an edge, so strains with no data come out as
    singletons rather than being silently merged.

    Args:
        divergence (pd.DataFrame): square symmetric divergence matrix.
        threshold (float): divergence below which two strains are clonal.

    Returns:
        list[list[str]]: clusters, each sorted, ordered by first member.
    """
    strains = list(divergence.index)

    values = divergence.to_numpy()
    with np.errstate(invalid="ignore"):
        adjacency = values < threshold      # NaN compares False: no edge

    _, labels = connected_components(csr_matrix(adjacency), directed=False)

    grouped = defaultdict(list)
    for strain, label in zip(strains, labels):
        grouped[label].append(strain)

    return sorted((sorted(members) for members in grouped.values()),
                  key=lambda members: members[0])


def trim_clonal_clusters(strains, divergence=None, species=None,
                         threshold=config.close_pair_thresh, path=None):
    """
    Reduce a set of strains to one representative per clonal cluster.

    The representative is the first member of the cluster by name, so the
    choice is deterministic and does not depend on input ordering. Strains
    absent from the divergence matrix are kept as singletons.

    Args:
        strains (iterable): genome names to trim.
        divergence (pd.DataFrame, optional): a matrix already in memory; it may
            cover more strains than `strains` and is subset here.
        species (str, optional): used to load the matrix when not supplied.
        threshold (float): divergence below which two strains are clonal.
        path (str, optional): explicit path to the matrix.

    Returns:
        list[str]: representatives, sorted.
    """
    strains = sorted(dict.fromkeys(strains))

    if divergence is None:
        divergence = load_divergences(species=species, strains=strains, path=path)

    present = [s for s in strains if s in divergence.index]
    absent = [s for s in strains if s not in divergence.index]

    if not present:
        return strains

    sub = divergence.loc[present, present]

    representatives = [cluster[0] for cluster in clonal_clusters(sub, threshold)]

    return sorted(representatives + absent)


def trim_populations(populations, species=None, divergence=None,
                     threshold=config.close_pair_thresh, path=None):
    """
    Apply `trim_clonal_clusters` to every population.

    The divergence matrix is loaded once for the union of all populations and
    reused, rather than re-read per population.

    Returns:
        dict: population -> representative genomes.
    """
    if divergence is None:
        everyone = sorted({g for genomes in populations.values() for g in genomes})
        divergence = load_divergences(species=species, strains=everyone, path=path)

    return {population: trim_clonal_clusters(genomes, divergence=divergence,
                                             threshold=threshold)
            for population, genomes in populations.items()}


def trim_allowed_samples(allowed_samples=None, alignment=None, species=None,
                         divergences=None, threshold=config.close_pair_thresh):
    """
    Resolve a sample selection and reduce it to clonal representatives.

    Convenience wrapper for the standalone entry points, which take
    `--allowed_samples` as a list or a text file and may leave it unset to mean
    "every sample in the alignment".

    Args:
        allowed_samples (list | str, optional): samples, or a file of them.
        alignment (str, optional): alignment to take the sample list from when
            `allowed_samples` is None.
        species (str, optional): used to locate the divergence matrix.
        divergences (str, optional): explicit path to the divergence matrix.
        threshold (float): divergence below which two strains are clonal.

    Returns:
        list[str]: representative samples.
    """
    import haplotype_utils

    if allowed_samples is None:
        if alignment is None:
            raise ValueError("need allowed_samples or an alignment to trim")
        samples = alignment_samples(alignment)
    else:
        samples = haplotype_utils._resolve_allowed_samples(allowed_samples)

    divergence = load_divergences(species=species, strains=samples, path=divergences)

    representatives = trim_clonal_clusters(samples, divergence=divergence,
                                           threshold=threshold)

    sys.stderr.write(f"clonal trimming: {len(samples)} samples -> "
                     f"{len(representatives)} representatives\n")

    return representatives


######### Population selection #########


def species_populations(species, metadata=None, good_populations=None,
                        min_genomes=config.min_genomes, available=None,
                        require_good_strain=config.require_good_strain):
    """
    Genomes of `species` grouped by population.

    Args:
        species (str): species name as it appears in the metadata.
        metadata (pd.DataFrame, optional): as returned by `load_metadata`.
        good_populations (list, optional): populations to keep. None keeps all.
        min_genomes (int): drop populations with fewer genomes than this.
        available (iterable, optional): sample names present in the alignment;
            genomes absent from it are dropped before the size threshold is
            applied, so a population is only kept if it really has enough data.
        require_good_strain (bool): keep only rows flagged `good_strain`.

    Returns:
        dict: population -> sorted list of genome names, largest population first.
    """
    if metadata is None:
        metadata = load_metadata()

    rows = metadata.loc[metadata[SPECIES_COL] == species]

    if rows.empty:
        raise ValueError(f"No genomes for species {species} in the metadata")

    if require_good_strain:
        if GOOD_STRAIN_COL not in rows.columns:
            raise ValueError(f"Metadata has no '{GOOD_STRAIN_COL}' column")
        flag = rows[GOOD_STRAIN_COL]
        # the column may be stored as text depending on how it was written
        if flag.dtype == object:
            flag = flag.astype(str).str.strip().str.lower() == "true"
        rows = rows.loc[flag.fillna(False).astype(bool)]

    if good_populations is not None:
        rows = rows.loc[rows[POPULATION_COL].isin(good_populations)]

    if available is not None:
        rows = rows.loc[rows[GENOME_COL].isin(set(available))]

    counts = rows[POPULATION_COL].value_counts()
    keep = counts.loc[counts >= min_genomes]

    populations = {}
    for pop in keep.index:
        populations[pop] = sorted(rows.loc[rows[POPULATION_COL] == pop, GENOME_COL])

    return populations


def describe_populations(species, metadata=None, good_populations=None,
                         min_genomes=config.min_genomes, available=None,
                         require_good_strain=config.require_good_strain,
                         representatives=None):
    """
    Per-population genome counts before and after each filter, as a DataFrame.

    Useful for reporting why a population was or was not scanned.
    """
    if metadata is None:
        metadata = load_metadata()

    rows = metadata.loc[metadata[SPECIES_COL] == species]

    stages = {"all": rows}

    if require_good_strain:
        flag = rows[GOOD_STRAIN_COL]
        if flag.dtype == object:
            flag = flag.astype(str).str.strip().str.lower() == "true"
        rows = rows.loc[flag.fillna(False).astype(bool)]
        stages["good_strain"] = rows

    if good_populations is not None:
        rows = rows.loc[rows[POPULATION_COL].isin(good_populations)]
        stages["good_population"] = rows

    if available is not None:
        rows = rows.loc[rows[GENOME_COL].isin(set(available))]
        stages["in_alignment"] = rows

    summary = pd.DataFrame({name: frame[POPULATION_COL].value_counts()
                            for name, frame in stages.items()}).fillna(0).astype(int)

    if representatives is not None:
        summary["clonal_representatives"] = pd.Series(
            {p: len(g) for p, g in representatives.items()}).reindex(summary.index).fillna(0).astype(int)

    final = summary.columns[-1]
    summary["scanned"] = summary[final] >= min_genomes

    return summary.sort_values(final, ascending=False)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="List the populations of a species")

    parser.add_argument('--species', required=True, type=str)
    parser.add_argument('--metadata', default=None, type=str)
    parser.add_argument('--good_populations', default=None, type=str)
    parser.add_argument('--haplotypes', default=None, type=str,
                        help="Alignment to check sample availability against")
    parser.add_argument('--min_genomes', default=config.min_genomes, type=int)
    parser.add_argument('--all_strains', action="store_true",
                        help="Do not require good_strain")

    args = parser.parse_args()

    available = alignment_samples(args.haplotypes) if args.haplotypes else None

    summary = describe_populations(args.species,
                                   metadata=load_metadata(args.metadata),
                                   good_populations=load_good_populations(args.good_populations),
                                   min_genomes=args.min_genomes,
                                   available=available,
                                   require_good_strain=not args.all_strains)

    print(summary.to_string())
