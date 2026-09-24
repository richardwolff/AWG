import peak_utils
import pandas as pd

if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-g','--genes_file',
                        help="filepath to gene annotations",
                        type=str)
 
    parser.add_argument('-o','--output_path',
                        help="filepath to output",
                        type=str)
    
    args = parser.parse_args()
    
    genes_file = args.genes_file
    output_path = args.output_path
    
    clusters = pd.read_pickle(f"{output_path}/peaks.pkl")
    
    ## identify annotations of genes for each significant variant lying within a peak
    ## e.g. cog category, ec number, etc
    clus_sites = peak_utils.annotate_clusters(clusters,genes_file)
    
    clus_sites.to_csv(f"{output_path}/peak_annotations.txt")