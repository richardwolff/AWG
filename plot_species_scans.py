"""
Plot every population's iLDS scan for one species on a single figure.

Each population gets the same pair of panels `plot_scan` draws for a single
scan -- iLDS against genome position, with a strip of called sweeps beneath --
stacked on a shared genome coordinate system so peaks line up vertically
between populations.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import argparse
import os
import sys

import config
import scan_utils
import peak_utils
import population_utils

## rows of the gridspec per population: 11 for the scan, 1 for the sweep strip
SCAN_ROWS = 11
STRIP_ROWS = 1
PANEL_ROWS = SCAN_ROWS + STRIP_ROWS


def population_dirs(species_dir):
    """Populations under a species directory that hold a finished scan."""
    if not os.path.isdir(species_dir):
        raise ValueError(f"{species_dir} not found")

    return sorted(name for name in os.listdir(species_dir)
                  if os.path.isfile(os.path.join(species_dir, name, "full_scan.txt")))


def order_populations(species_dir, populations, regions=None):
    """
    Group populations by region, alphabetically within each.

    Region order comes from `config.region_order`. With no region mapping
    available, falls back to the order `results.txt` lists them in, then to
    alphabetical.
    """
    if regions:
        return population_utils.sort_populations_by_region(populations, regions)

    results = os.path.join(species_dir, "results.txt")

    if not os.path.exists(results):
        return sorted(populations)

    listed = pd.read_csv(results, sep="\t")["population"].tolist()

    return ([p for p in listed if p in populations]
            + [p for p in populations if p not in listed])


def load_scan(species_dir, population):
    """One population's scan table and its called sweeps."""
    pop_dir = os.path.join(species_dir, population)

    df_rnrs = pd.read_csv(f"{pop_dir}/full_scan.txt", index_col=[0, 1, 2])
    clusters = pd.read_pickle(f"{pop_dir}/peaks.pkl")

    return df_rnrs, clusters


def shared_offsets(scans):
    """
    One contig -> offset mapping covering every population.

    `plot_scan` on its own derives offsets from the scan it is given, so two
    populations that happen to end at different positions on a contig would be
    laid out differently. Taking the furthest position seen on each contig,
    across all populations, keeps the panels on one coordinate system.
    """
    furthest = {}

    for df_rnrs, _ in scans.values():
        for contig, group in df_rnrs.groupby("contig"):
            end = group.index.get_level_values("site_pos").max()
            furthest[contig] = max(furthest.get(contig, 0), end)

    off_dic, off = {}, 0
    for contig in sorted(furthest):
        off_dic[contig] = off
        off += furthest[contig]

    return off_dic, off


def region_legend_entries(populations, regions, region_colors):
    """
    Legend handles mapping region to colour, in the order the regions appear.

    Regions sharing a colour -- Europe and North America do -- collapse into one
    entry rather than two identical swatches.
    """
    by_colour = {}
    for population in populations:
        region = regions.get(population)
        colour = region_colors.get(region)
        if colour is None:
            continue
        by_colour.setdefault(colour, [])
        if region not in by_colour[colour]:
            by_colour[colour].append(region)

    return [Patch(facecolor=colour, edgecolor="none", label=" / ".join(names))
            for colour, names in by_colour.items()]


def plot_species_scans(species, scans, off_dic, xmax, output=None,
                       panel_height=3.0, width=16, share_y=False, regions=None,
                       region_colors=None):
    """
    Stack one `plot_scan` panel pair per population.

    Args:
        species (str): used for the figure title.
        scans (dict): population -> (scan table, clusters), in plot order.
        off_dic (dict): shared contig offsets.
        xmax (float): shared right hand limit.
        output (str, optional): path to save to.
        panel_height (float): inches per population.
        width (float): figure width in inches.
        share_y (bool): put every population on one iLDS scale. Off by default:
            a single strong sweep in one population otherwise flattens the
            rest, and peaks still line up horizontally without it.
        regions (dict, optional): population -> region.
        region_colors (dict, optional): region -> colour. Each population's
            label is drawn in its region's colour, with a legend mapping the
            colours back to regions.

    Returns:
        (fig, list): the figure and its scan axes.
    """
    n = len(scans)

    fig = plt.figure(figsize=(width, panel_height * n))
    gs = gridspec.GridSpec(PANEL_ROWS * n, 10, hspace=0.2, wspace=0)

    regions = regions or {}
    region_colors = region_colors or {}

    axes = []
    for i, (population, (df_rnrs, clusters)) in enumerate(scans.items()):

        top = PANEL_ROWS * i
        ax = fig.add_subplot(gs[top:top + SCAN_ROWS, :])
        ax_scat = fig.add_subplot(gs[top + SCAN_ROWS:top + PANEL_ROWS, :])
        ax_scat.set_facecolor("snow")

        cluster_endpoints, clus_contig = peak_utils.return_cluster_endpoints(clusters)

        scan_utils.plot_scan(df_rnrs, cluster_endpoints, clus_contig,
                             fig=fig, ax=ax, ax_scat=ax_scat,
                             off_dic=off_dic, xmax=xmax, tight=False)

        # name the population inside the panel, so stacking costs no height,
        # and colour it by region rather than spelling the region out
        colour = region_colors.get(regions.get(population), "black")

        ax.text(0.006, 0.93, population, transform=ax.transAxes,
                size=15, va="top", ha="left", color=colour, weight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="none", alpha=0.75), zorder=20)

        # only the bottom panel carries the genome axis
        if i < n - 1:
            ax_scat.set_xlabel("")
            ax_scat.set_xticklabels([])

        axes.append(ax)

    if share_y:
        low = min(ax.get_ylim()[0] for ax in axes)
        high = max(ax.get_ylim()[1] for ax in axes)
        for ax in axes:
            ax.set_ylim(low, high)

    fig.tight_layout()

    # place the title and legend just above the panels, once the layout is
    # settled, so stacking many populations does not open a gap at the top
    top = axes[0].get_position().y1
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.9, top),
               prop={"size": 15}, ncol=2, frameon=True)

    # the region key sits on its own row above, so it cannot run into the
    # title on the left or the marker key on the right. One legend row is a
    # fixed height in inches, which is a shrinking fraction of a tall figure.
    row = 0.34 / fig.get_figheight()

    region_handles = region_legend_entries(scans.keys(), regions, region_colors)
    if region_handles:
        fig.legend(handles=region_handles, loc="lower center",
                   bbox_to_anchor=(0.5, top + row), prop={"size": 13},
                   ncol=len(region_handles), frameon=True, handlelength=1.2,
                   columnspacing=1.2)

    fig.text(0.11, top + 0.002, species.replace("_", " "),
             size=20, style="italic", ha="left", va="bottom")

    if output is not None:
        fig.savefig(output, bbox_inches="tight")
        sys.stderr.write(f"Written to {output}\n")

    return fig, axes


def plot_from_directory(species, species_dir, populations=None, figure=None,
                        panel_height=3.0, width=16, share_y=False):
    """
    Find the finished scans under a species directory and plot them together.

    Populations still missing a `full_scan.txt` -- because they failed, or have
    not run yet -- are skipped, so this is safe to call at the end of a run in
    which some populations did not complete.

    Returns:
        str: the path written, or None when there was nothing to plot.
    """
    available = population_dirs(species_dir)

    if populations is not None:
        available = [p for p in available if p in set(populations)]

    if not available:
        sys.stderr.write(f"No finished scans under {species_dir}; nothing to plot\n")
        return None

    regions = population_utils.load_population_regions()
    available = order_populations(species_dir, available, regions)

    scans = {population: load_scan(species_dir, population) for population in available}

    sys.stderr.write(f"{len(scans)} populations: {', '.join(scans)}\n")

    off_dic, genome_length = shared_offsets(scans)
    sys.stderr.write(f"{len(off_dic)} contig(s), {genome_length} bp laid end to end\n")

    figure = figure or os.path.join(species_dir, "scan_figure_all_populations.png")

    plot_species_scans(species, scans, off_dic, genome_length,
                       output=figure, panel_height=panel_height,
                       width=width, share_y=share_y, regions=regions,
                       region_colors=getattr(config, "region_color", {}))

    return figure


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--species', required=True, type=str,
                        help="Species whose scans to plot")

    parser.add_argument('-o', '--output_path',
                        default=getattr(config, "scan_directory", "test_output"), type=str,
                        help="Folder holding <species>/<population>/. Defaults to config.scan_directory.")

    parser.add_argument('--populations', default=None, type=str,
                        help="Comma-separated subset of populations to plot")

    parser.add_argument('--figure', default=None, type=str,
                        help="Where to write the figure (default <output_path>/<species>/scan_figure_all_populations.png)")

    parser.add_argument('--panel_height', default=3.0, type=float,
                        help="Inches per population")

    parser.add_argument('--width', default=16, type=float,
                        help="Figure width in inches")

    parser.add_argument('--share_y', dest="share_y", action="store_true",
                        default=False,
                        help="Put every population on one iLDS scale, so peak heights are comparable")
    parser.add_argument('--free_y', dest="share_y", action="store_false",
                        help="Let each population set its own iLDS scale (default)")

    args = parser.parse_args()

    species_dir = os.path.join(args.output_path, args.species)

    populations = None
    if args.populations:
        populations = [p for p in args.populations.split(",") if p]

    if plot_from_directory(args.species, species_dir, populations=populations,
                           figure=args.figure, panel_height=args.panel_height,
                           width=args.width, share_y=args.share_y) is None:
        raise ValueError(f"No finished scans under {species_dir}")
