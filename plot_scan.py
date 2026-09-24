import scan_utils
import pandas as pd
import peak_utils

if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-o','--output_path',
                        help="filepath to output",
                        type=str)
    
    args = parser.parse_args()
    
    output_path = args.output_path
    
    clusters = pd.read_pickle(f"{output_path}/peaks.pkl")    
    df_rnrs = pd.read_csv(f"{output_path}/full_scan.txt",index_col=[0,1,2,3])
    
    ## identify endpoints of clusters (for bottom panel of iLDS scan figures-see e.g. Fig 3ABC)
    cluster_endpoints,clus_contig = peak_utils.return_cluster_endpoints(clusters)    
    
    ## plot scan
    scan_utils.plot_scan(df_rnrs,cluster_endpoints,clus_contig,output_path)
    