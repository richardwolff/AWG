import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import r2_utils
import os,sys
import config



def _connected_component(adjacency, node):
    """
    Boolean mask of the nodes reachable from `node` in a dense boolean
    adjacency matrix.

    Replaces building a networkx graph from the adjacency matrix and scanning
    its components, which costs an object per node/edge on every peak.
    """
    n_comp, labels = connected_components(csr_matrix(adjacency), directed=False)

    return labels == labels[node]


def perform_peak_clustering(df_rnrs,dfH,dec_dist = config.clus_dist,clus_dist=config.clus_dist,linkage_thresh=config.linkage_thresh):
    
    dfH = dfH.droplevel("site_type")
    
    df_rnrs_pass = df_rnrs.loc[df_rnrs["significance"]]

    df_rnrs_pass_cp = df_rnrs_pass.copy()

    ## LD and distances between significant sites are the same no matter which
    ## peak is being carved out, so compute them once per contig and take
    ## sub-matrices below rather than recomputing on every pass of the loop.
    contig_cache = {}
    for contig, df_group in df_rnrs_pass.groupby("contig"):

        idxs = df_group.index
        site_pos = idxs.get_level_values("site_pos")

        contig_cache[contig] = (
            {site: i for i, site in enumerate(idxs)},
            np.abs(np.subtract.outer(site_pos, site_pos)).astype(float),
            r2_utils.pairwise_r2(np.ascontiguousarray(dfH.loc[idxs].to_numpy(), dtype=np.float64)),
        )

    clusters = {}
    j = 1

    while df_rnrs_pass_cp.shape[0] > 0:

        idxmax = df_rnrs_pass_cp["iLDS"].idxmax()

        idxmax_contig = idxmax[0]
        idxmax_contig_idxs = df_rnrs_pass_cp.groupby("contig").get_group(idxmax_contig).index

        positions, D_all, LD_all = contig_cache[idxmax_contig]

        # positions of the still-unassigned sites within the cached matrices
        take = np.fromiter((positions[site] for site in idxmax_contig_idxs),
                           dtype=np.intp, count=len(idxmax_contig_idxs))

        D = D_all[np.ix_(take, take)]
        LD_locus = LD_all[np.ix_(take, take)]

        with np.errstate(invalid="ignore"):
            within_peak = (D <= dec_dist) & (LD_locus >= linkage_thresh)

        foc_peak = idxmax_contig_idxs[
            _connected_component(within_peak, idxmax_contig_idxs.get_loc(idxmax))]

        if len(foc_peak) > 1:

            clusters[j] = foc_peak
            j+=1

            # first and last site of the peak along the contig
            peak_site_pos = foc_peak.get_level_values("site_pos")
            peak_min = foc_peak[int(np.argmin(peak_site_pos))]
            peak_max = foc_peak[int(np.argmax(peak_site_pos))]

            peak_min_iloc = idxmax_contig_idxs.get_loc(peak_min)
            peak_max_iloc = idxmax_contig_idxs.get_loc(peak_max)

            drop_peak = ((D[peak_min_iloc] <= clus_dist) |
                         (D[peak_max_iloc] <= clus_dist))

            # peak body (positional span) plus the shoulders either side
            drop_peak[peak_min_iloc:peak_max_iloc] = True

            df_rnrs_pass_cp.drop(idxmax_contig_idxs[drop_peak],inplace=True)

        else:

            df_rnrs_pass_cp.drop(idxmax,inplace=True)

    sys.stderr.write(f"\n\nNumber of sweeps detected: {j - 1}\n\n")



    first_elem_pos = []
    for key,item in clusters.items():
        first_elem_pos.append(item[0][-1])

    clus_order = {k+1:np.argsort(first_elem_pos)[k] for k in range(len(first_elem_pos))}

    clus_order = pd.Series(clus_order)

    clus_order = clus_order + 1

    clusters_cp = {}
    cluster_key_list = list(clusters.keys())

    for key in pd.Series(clus_order).index:

        clusters_cp[key] = clusters[clus_order.loc[key]]

    clusters = clusters_cp 
        
    return(clusters)


def return_cluster_endpoints(clusters):
    
    cluster_endpoints = {}
    cluster_numpoints = {}
    clus_contig = {}
    
    for key in np.sort(list(clusters.keys())):

        item = clusters[key]

        gene_list = list(set(item.get_level_values("gene_id")))
        
        clus_contig[key] = item.get_level_values("contig")[0]
        
        clus_site_pos = item.get_level_values("site_pos")
        
        cluster_endpoints[key] = (clus_site_pos.min(),clus_site_pos.max())
         
    return(cluster_endpoints,clus_contig)


def annotate_clusters(clusters,genes_filename):
    
    gff = pd.read_csv(genes_filename,index_col=0)
    
    clus_sites = {}
    gene_desc = {}
    for key,clus in clusters.items():

        for site in clus:

            gene_id = site[1]

            # one lookup per gene rather than one per site
            if gene_id not in gene_desc:
                gene_desc[gene_id] = gff.loc[gene_id]

            clus_sites[site] = gene_desc[gene_id]
            
    clus_sites = pd.DataFrame(clus_sites).T
    
    clus_sites.index.names = ["contig","gene_id","site_pos"]
            
    return(clus_sites)
