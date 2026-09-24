import pandas as pd
import os
import numpy as np
import os.path 
from math import log10
import pandas as pd
import sys

def print_var(var_name):
    print(globals()[var_name])    
    

### coefficients & inputs

common_variant_maf = 0.2
maf = common_variant_maf
clus_dist = 2.5*1e4
close_pair_thresh = 5*1e-4
count_thresh = 0.5
num_vars=50
min_bins=8
max_dist = 5*1e5      ## max separation of variant pairs: LD tables and the decay fit

common_variant_maf = 0.2
count_thresh = 0.5

clus_dist = 2.5*1e4
linkage_thresh = 0.5

min_bins = 3
max_bins = None

num_vars = 50
min_syn = 6

alpha = 0.05

sample_num = int(1e6)

## location of scripts directory
scripts_dir = "/u/home/r/rwolff/AWG/" 
figures_dir = f"{scripts_dir}/figures" 
analysis_dir = f"{scripts_dir}/analysis" 
scan_directory = f"{analysis_dir}/scans"
core_genes_dir = f"{scripts_dir}/core_genes"


## metadata
metadata_dir = f"{scripts_dir}/metadata"

## resolve a metadata file: metadata_dir first, then a local metadata/ or the
## working directory, so the same config works on the cluster and in a checkout
def metadata_path(filename):
    for directory in (metadata_dir, "metadata", "."):
        candidate = os.path.join(directory, filename)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(metadata_dir, filename)

## list of species we will be analyzing. The file is a plain one-per-line list
## with no header, so read it as one -- `index_col=0` swallows the first
## species as a header row.
good_species = pd.Index(pd.read_csv(metadata_path("good_species.txt"),
                                    header=None).iloc[:, 0].astype(str).str.strip())

base_dir="/u/project/ngarud/Garud_lab/awi_gen/"

LD_dir=f"{base_dir}/LD"

## base directory to store downloaded files
raw_dir = "/u/project/ngarud/Garud_lab/awi_gen/data"

species_base_dir = raw_dir + "/%s/snps/"

ref_files_dir = species_base_dir + "ref_files/"

## base directory to write processed (i.e. annotated) haplotypes
haplotype_dir = raw_dir + "/%s/snps/ref_files/haplotypes" 

## base directory for summaries of poly sites and annotation of state (syn/non)

## base directory for clade control - both genetic distance matrices (mem-intensive)
##  and plots
genetic_distances_df = f"{base_dir}/ds_dir/%s.txt"

## local fallback, checked when the path above does not exist
genetic_distances_dir = "genomewide_divergence"


########################################################################
### iLDS scan parameters
###
### Settings that are invariant across scans. Every one of these is the
### default for the corresponding command line flag, so a run can still
### override any of them without editing this file.
########################################################################

## --- which genomes go into a scan ---------------------------------------

## strain metadata table, and the curated list of populations to analyze.
## Looked for as given, then in the working directory, then in metadata_dir.
metadata_file = "all_strains_metadata.txt"
good_populations_file = "good_populations.txt"

## population -> region, used to group populations in figures
population_regions_file = "population_regions.txt"

## order regions are presented in. Regions absent from this list follow, in
## alphabetical order; populations are alphabetical within their region.
region_order = ["West Africa", "South Africa", "East Africa", "Oceania",
                "Europe", "North America", "Asia"]

## colour each region is drawn in, taken from the figure 1 palette
## (ds_circular_trees.REGION_COLOR). Europe and North America deliberately
## share one colour, as they do there; regions sharing a colour are collapsed
## into a single legend entry.
region_color = {"West Africa":   "#ff7f0e",   # orange
                "South Africa":  "#2ca02c",   # green
                "East Africa":   "#9467bd",   # purple
                "Oceania":       "#8c564b",   # brown, 'Fiji' in figure 1
                "Europe":        "#1f77b4",   # blue
                "North America": "#1f77b4",   # blue, shared with Europe
                "Asia":          "#d62728"}   # red

## skip populations with fewer than this many genomes
min_genomes = 20

## restrict to strains flagged good_strain in the metadata
require_good_strain = True

## keep only one representative per clonal cluster -- strains connected by
## pairwise divergence below close_pair_thresh. Clonal duplicates otherwise
## inflate LD, contributing r^2 ~ 1 at every distance.
trim_clonal = True

## --- which sites go into a scan -----------------------------------------

## restrict to the species' core genome (core_genes_dir/<species>.txt).
## Accessory genes carry presence/absence structure that distorts LD.
core_genes_only = True

## value denoting a missing call. The parquet alignments write -1, which an
## integer column cannot express as NaN. None means "only empty fields".
na_values = -1

## --- LD table -----------------------------------------------------------

## site class the LD decay curve is fit to
ld_site_type = "syn"

## only record pairs closer than this -- `max_dist`, set above, which is also
## the upper end of the range the decay curve is fit over.

## cap on the pairs in one population's LD table, sampled uniformly
ld_max_pairs = int(2e7)

## rows of the r^2 matrix computed per pass when building an LD table
ld_block_rows = 512

## --- decay curve fit ----------------------------------------------------

## lower end of the distance range the binned decay curve is fit over.
## The upper end is `max_dist`, the same bound the LD tables use.
fit_min_dist = 2499.5

## --- species level tract length -----------------------------------------

## the decay distance used for scanning comes from one fit to LD pairs pooled
## equally across populations; populations are not fit individually.

## total pairs in the pooled LD table, drawn equally from each population
pool_pairs = int(2e7)

## --- execution ----------------------------------------------------------

## rows parsed per pass when streaming an alignment
read_chunk_rows = 50_000

## filtered haplotypes are held in memory between the two passes up to this
## much; populations beyond it are re-read before scanning
cache_budget_gb = 8.0

## seed for pair sampling, so a scan is reproducible
seed = 0

## draw the combined figure of every population's scan once a species finishes
plot_scans = True

## --- sharing sweeps between populations ---------------------------------

## two sweeps are the same sweep when the gap between them is no wider than
## this: the left endpoint of the right sweep, minus the right endpoint of the
## left one. Sweeps that genuinely overlap have a negative gap and so always
## merge. Merging is single linkage, so a chain of sweeps each within this
## distance of the next forms one group.
sweep_merge_dist = 3*1e3
