import numpy as np
import pandas as pd
import gzip
import bz2
import zipfile
import os
import warnings
import config

REQUIRED_COLUMNS = ["contig", "gene_id", "site_pos", "site_type"]
VALID_SITE_TYPES = ["syn", "nonsyn"]

## Rows parsed per pass when streaming an alignment. Peak memory during the
## read is roughly CHUNK_ROWS * n_samples * 8 bytes, independent of file size.
CHUNK_ROWS = config.read_chunk_rows

######### Basic reading utilities #########

def _read_header(filepath, compression):
    """Return the first line of a (possibly compressed) alignment file."""
    if compression == 'gzip':
        with gzip.open(filepath, 'rt') as f:
            return f.readline()
    elif compression == 'bz2':
        with bz2.open(filepath, 'rt') as f:
            return f.readline()
    elif compression == 'zip':
        with zipfile.ZipFile(filepath) as z:
            with z.open(z.namelist()[0]) as f:
                return f.readline().decode()
    elif compression is None:
        with open(filepath, 'r') as f:
            return f.readline()
    else:
        raise ValueError("Unsupported compression type.")


PARQUET_SUFFIXES = (".pq", ".parquet")


def _is_parquet(filepath):
    return str(filepath).endswith(PARQUET_SUFFIXES)


def _parquet_columns(filepath):
    """Column names of a parquet alignment, index levels included."""
    import pyarrow.parquet as pq

    return list(pq.ParquetFile(filepath).schema_arrow.names)


def _read_parquet_chunks(filepath, usecols, chunksize):
    """
    Yield row-group batches of a parquet alignment as DataFrames.

    Only the requested sample columns are decoded, so subsetting to one
    population reads a slice of the file proportional to that population rather
    than all of it. The site columns are the frame's index when pandas wrote
    the file, so they are asked for explicitly and pushed back into columns to
    match the CSV path.
    """
    import pyarrow.parquet as pq

    handle = pq.ParquetFile(filepath)
    stored = set(handle.schema_arrow.names)

    if usecols is None:
        columns = None
    else:
        columns = [c for c in usecols if c in stored]
        for c in REQUIRED_COLUMNS:
            if c not in columns and c in stored:
                columns.append(c)

    for batch in handle.iter_batches(batch_size=chunksize, columns=columns):
        df = batch.to_pandas()

        missing_as_index = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_as_index:
            df = df.reset_index()

        yield df


def _resolve_allowed_samples(allowed_samples):
    """Accept a list of sample names or a path to a one-per-line text file."""

    ## if passed a list, simply proceed
    if isinstance(allowed_samples, (list, tuple)):
        return list(allowed_samples)

    ## can provide allowed samples as a text file
    elif isinstance(allowed_samples, str) and os.path.exists(allowed_samples):
        with open(allowed_samples, 'r') as file:
            return [line.strip() for line in file.readlines()]

    ## if the allowed_samples file can't be found
    else:
        raise ValueError("allowed_samples file not found")


def _numeric_na_values(na_values):
    """The numeric entries of `na_values`, as floats; [] if there are none."""
    if na_values is None:
        return []

    if isinstance(na_values, (str, int, float)):
        na_values = [na_values]

    out = []
    for v in na_values:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            # non-numeric sentinels are handled by the csv parser
            continue

    return out


def _check_allele_values(values):
    """True if every entry is 0, 1 or missing."""

    if values.dtype.kind in "fc":
        return bool(np.all(np.isnan(values) | (values == 0) | (values == 1)))
    if values.dtype.kind in "iub":
        return bool(np.all((values == 0) | (values == 1)))

    # object/string columns: fall back to an elementwise check
    flat = values.ravel()
    return all(pd.isna(x) or x in [0, 1, 0.0, 1.0] for x in flat)


def _process_chunk(df, seen_sites, prefilter, na_values=None):
    """
    Apply the per-site checks and filters to one chunk of an alignment.

    Every step here is row-local (or, for duplicate detection, tracked across
    chunks via `seen_sites`), so streaming gives exactly the same result as
    processing the whole file at once.

    Returns (frame, n_dropped) where `frame` is None if nothing survived.
    """
    # Filter rows based on site_type
    df = df[df["site_type"].isin(VALID_SITE_TYPES)]
    if df.empty:
        return None, 0

    # Ensure site_pos is of type int
    try:
        site_pos = df["site_pos"].astype(int)
    except ValueError:
        raise ValueError("Column 'site_pos' contains non-integer values and cannot be converted to int.")

    # Set required columns as the index
    index = pd.MultiIndex.from_arrays(
        [df["contig"], df["gene_id"], site_pos, df["site_type"]], names=REQUIRED_COLUMNS)

    sample_cols = df.columns.drop(REQUIRED_COLUMNS)
    values = df[sample_cols].to_numpy()

    # Missing calls may be written as a sentinel rather than left empty (the
    # parquet alignments use -1, which an integer column cannot express as NaN)
    na_numeric = _numeric_na_values(na_values)
    if na_numeric:
        values = values.astype(float)
        values[np.isin(values, na_numeric)] = np.nan

    # Check that all values are either 0, 1, or NaN
    if not _check_allele_values(values):
        raise ValueError("All values in the DataFrame must be either 0 (reference), 1 (alternate), or NaN (missing).")

    # Convert all values to float. Should be redundant after checking all values are 0, 1, or nan above, but an additional check.
    try:
        values = values.astype(float)
    except ValueError as e:
        raise ValueError(f"Error converting DataFrame values to float (0, 1, nan): {e}")

    ## check if there are any duplicated sites; keep only the first occurrence
    keep = np.ones(values.shape[0], dtype=bool)
    for i, site in enumerate(index):
        if site in seen_sites:
            keep[i] = False
        else:
            seen_sites.add(site)

    n_dropped = int(values.shape[0] - keep.sum())
    if n_dropped:
        values, index = values[keep], index[keep]

    ## polarize haplotypes to consensus (0: major, 1: minor)
    _polarize_values(values)

    if prefilter:
        ## keep only the common, well-covered variants the scan will use
        keep = _common_mask(values) & _count_mask(values)
        if not keep.all():
            values, index = values[keep], index[keep]

    if values.shape[0] == 0:
        return None, n_dropped

    return pd.DataFrame(values, index=index, columns=sample_cols), n_dropped


def process_site_data(filepath, na_values=None, compression=None, allowed_samples=None,
                      prefilter=False, chunksize=CHUNK_ROWS):
    """
    Reads an input file containing an alignment as a DataFrame. Checks to see if the DataFrame is in a format which can be recognized by iLDS. 

    The file is parsed in chunks and each chunk is checked, de-duplicated and
    polarized before the next one is read, so peak memory tracks `chunksize`
    rather than the size of the alignment.

    Args:
        filepath (str): Path to the file containing the haplotypes (can be comma or tab-separated).
        na_values (list, optional): Values to consider as NaN—i.e. missing alleles/sites for a given haplotype. Default is None.
        compression (str, optional): Compression type for the file. Default is None.
        allowed_samples (list, optional): List of samples to include as columns in the DataFrame. By default, use all samples.
        prefilter (bool, optional): Drop rare variants and poorly covered sites while reading, rather than after the whole alignment is in memory. The surviving sites are exactly those `return_common_filtered` would keep.
        chunksize (int, optional): Rows parsed per pass.

    Returns:
        pd.DataFrame: Filtered and processed pandas DataFrame of haplotypes. Each site is indexed by its contig, the gene it lies in, the position of the site along the contig, and the annotation of the site/polymorphism as either synonymous or non-synonymous. 

    Raises:
        ValueError: If required columns are missing, `site_type` values are invalid, if `site_pos` cannot be converted to int, 
                    if any value in the DataFrame is not 0, 1, or NaN, or if any sample in allowed_samples is missing.
    """
    parquet = _is_parquet(filepath)

    if parquet:
        try:
            columns = _parquet_columns(filepath)
        except Exception as e:
            raise ValueError(f"Error reading parquet schema: {e}")
        delimiter = None
    else:
        # Determine delimiter by inspecting the file
        try:
            header = _read_header(filepath, compression)

            if ',' in header:
                delimiter = ','
            elif '\t' in header:
                delimiter = '\t'
            else:
                raise ValueError("Unable to determine file delimiter. Ensure the file is comma or tab-separated.")
        except Exception as e:
            raise ValueError(f"Error reading file header: {e}")

        columns = list(pd.Index(header.rstrip("\r\n").split(delimiter)))

    # Check for required columns
    if not all(col in columns for col in REQUIRED_COLUMNS):
        raise ValueError(f"Missing required levels for site index. Required index levels: {REQUIRED_COLUMNS}")

    # If allowed_samples is specified, ensure those columns are present
    usecols = None
    if allowed_samples is not None:

        allowed_samples = _resolve_allowed_samples(allowed_samples)

        missing_samples = [sample for sample in allowed_samples if sample not in columns]
        if missing_samples:
            raise ValueError(f"Samples in allowed_samples missing from data: {missing_samples}")

        # Read only required columns and allowed_samples
        usecols = REQUIRED_COLUMNS + allowed_samples

    # Read the file
    try:
        if parquet:
            reader = _read_parquet_chunks(filepath, usecols, chunksize)
        else:
            reader = pd.read_csv(filepath, delimiter=delimiter, na_values=na_values,
                                 compression=compression, usecols=usecols,
                                 chunksize=chunksize)

        seen_sites = set()
        n_dropped = 0
        frames = []

        for chunk in reader:
            if usecols is not None:
                # `usecols` does not preserve the requested column order
                chunk = chunk[usecols]
            # drop duplicated sample columns, keeping the first occurrence
            chunk = chunk.loc[:, ~chunk.columns.duplicated(keep='first')]

            frame, dropped = _process_chunk(chunk, seen_sites, prefilter, na_values)
            n_dropped += dropped
            if frame is not None:
                frames.append(frame)

    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")

    if not seen_sites:
        raise ValueError(f"No rows with 'site_type' values in {VALID_SITE_TYPES}.")

    if n_dropped:
        warnings.warn("Detected and removed duplicated sites or samples")

    if not frames:
        sample_cols = [c for c in (usecols if usecols is not None else columns)
                       if c not in REQUIRED_COLUMNS]
        df_haps = pd.DataFrame(columns=sample_cols, dtype=float,
                               index=pd.MultiIndex.from_arrays([[]] * 4, names=REQUIRED_COLUMNS))
    elif len(frames) == 1:
        df_haps = frames[0]
    else:
        df_haps = pd.concat(frames)

    # Everything downstream -- window construction, the banded r^2 kernels, the
    # distance between neighbouring variants -- assumes sites ascend along each
    # contig. Alignments are not always written that way: an alignment ordered
    # by gene_id walks backwards at every gene boundary. Sorting is stable, so
    # it is a no-op on a file that was already in position order.
    df_haps = sort_sites(df_haps)

    print(df_haps.head())

    return df_haps

def process_folder(folder_path, na_values=None, compression=None, allowed_samples=None,
                   prefilter=False):
    """
    Processes all files in a folder using `process_site_data`, concatenates the results, and sorts the sites by contig, then by position along the contig.
    For instance, if we have different files containing alignments for each contig, this is useful.
    """
    dataframes = []

    filenames = sorted(os.listdir(folder_path))

    # a folder holding the same alignment as both parquet and text should be
    # read once, from the parquet
    parquet_files = [f for f in filenames if _is_parquet(f)]
    if parquet_files:
        filenames = parquet_files

    for filename in filenames:
        filepath = os.path.join(folder_path, filename)

        if not os.path.isfile(filepath):
            continue

        try:
            df = process_site_data(filepath, na_values=na_values, compression=compression,
                                   allowed_samples=allowed_samples, prefilter=prefilter)
            dataframes.append(df)
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    if not dataframes:
        raise ValueError("No valid files were processed.")

    ## check that all alignments have the same samples
    missing_columns_info = check_columns_consistency(dataframes)   
    if missing_columns_info is not None:
        raise ValueError(missing_columns_info)
        
    # Concatenate all DataFrames
    combined_df = pd.concat(dataframes) if len(dataframes) > 1 else dataframes[0]

    # Sort index by ['contig', 'site_pos']
    try:
        combined_df = sort_sites(combined_df)
    except KeyError:
        raise ValueError("Index levels ['contig', 'site_pos'] not found in the DataFrame.")

    return combined_df


def read_haplotypes(input_path, na_values=None, compression=None, allowed_samples=None,
                    prefilter=False):

    # Check if it's a file
    if os.path.isfile(input_path):
        df_haplotypes = process_site_data(input_path, na_values, compression, allowed_samples,
                                          prefilter=prefilter)

    # Check if it's a directory
    elif os.path.isdir(input_path):
        df_haplotypes = process_folder(input_path, na_values, compression, allowed_samples,
                                       prefilter=prefilter)

    else:
        raise ValueError(f"{input_path} is neither a file nor a directory.")    

    return(df_haplotypes)

def sort_sites(df):
    """Order sites by contig, then by position along the contig."""
    if df.shape[0] == 0:
        return df

    return df.sort_index(level=["contig", "site_pos"], sort_remaining=False)


def check_columns_consistency(dfs):
    # Get the columns of the first dataframe as the reference
    reference_columns = set(dfs[0].columns)
    
    missing_columns_info = []
    
    # Iterate over each dataframe in the list
    for idx, df in enumerate(dfs):
        # Get the columns of the current dataframe
        current_columns = set(df.columns)
        
        # Compare the current dataframe columns with the reference columns
        missing_columns = reference_columns - current_columns
        extra_columns = current_columns - reference_columns
        
        # If there are missing or extra columns, store the info
        if missing_columns:
            missing_columns_info.append(f"DataFrame {idx} is missing samples: {', '.join(missing_columns)}")
        if extra_columns:
            missing_columns_info.append(f"DataFrame {idx} has extra samples: {', '.join(extra_columns)}")
    
    # Return the results
    return missing_columns_info if missing_columns_info else None


def remove_duplicates(df):
    
    # Check if columns are duplicated and keep only the first occurrence
    
    df_cleaned_cols = df.loc[:, ~df.columns.duplicated(keep='first')]
    
    df_cleaned = df_cleaned_cols[~df_cleaned_cols.index.duplicated(keep='first')]

    return df_cleaned

 

########################################

######### Processing utilities #########

def _nan_row_mean(values):
    """Per-row allele frequency, ignoring missing calls (all-NaN rows -> NaN)."""
    counts = np.count_nonzero(~np.isnan(values), axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nansum(values, axis=1) / counts


def _polarize_values(values):
    """In-place consensus polarization of a (sites x samples) float array."""
    flip = _nan_row_mean(values) > 0.5
    values[flip] = 1.0 - values[flip]

    return values


def _common_mask(values, common_thresh=config.common_variant_maf):
    """Rows whose minor allele frequency lies inside [thresh, 1 - thresh]."""
    f = _nan_row_mean(values)

    with np.errstate(invalid="ignore"):
        return (f >= common_thresh) & (f <= 1 - common_thresh)


def _count_mask(values, count_thresh=config.count_thresh):
    """Rows with more than `count_thresh` of samples called."""
    return np.count_nonzero(~np.isnan(values), axis=1) > (values.shape[1] * count_thresh)


## Sets major allele to state 0, minor allele to 1    
def polarize_by_consensus(dfx):
    """
    Given a genotype dataframe (rows = variants, columns = samples),
    flips allele states so that the major allele is always represented as 0
    and the minor allele as 1.

    Parameters:
        dfx (pd.DataFrame): Binary allele calls (0/1) per sample per variant.

    Returns:
        pd.DataFrame: Polarized genotype matrix with consistent allele coding.
    """

    # For each variant (row), compute mean allele frequency across samples.
    # If frequency > 0.5, then "1" is the major allele -> flip values (1 -> 0, 0 -> 1).
    # Done on the underlying array: the equivalent `.loc[mask] = 1 - .loc[mask]`
    # realigns and re-blocks the whole frame on every call.
    values = dfx.to_numpy(dtype=float, copy=False)
    flip = _nan_row_mean(values) > 0.5

    if flip.any():
        dfx.iloc[flip] = 1.0 - values[flip]

    return dfx


## Returns variants with minor allele frequency greater than threshold (20% by default)
## Can be run without first polarizing by consensus
def return_common(df, common_thresh = config.common_variant_maf):
    """
    Filters variants by minor allele frequency (MAF).

    Parameters:
        df (pd.DataFrame): Binary allele calls (0/1) per sample per variant.
        common_thresh (float): MAF threshold (default from config).

    Returns:
        pd.DataFrame: Subset of variants with MAF between `common_thresh` 
                      and (1 - common_thresh).
    """

    # Keep only variants where MAF >= threshold and <= 1 - threshold
    # i.e., filter out rare and fixed alleles.
    intermediate_frequency_variants = _common_mask(df.to_numpy(dtype=float, copy=False),
                                                   common_thresh)

    # Subset the dataframe to keep only common variants
    dfH = df.loc[intermediate_frequency_variants]

    return dfH


## Filters out sites with high missingness
def filter_counts(dfH, count_thresh = config.count_thresh):
    """
    Removes variants (rows) with too much missing data.

    Parameters:
        dfH (pd.DataFrame): Genotype matrix (rows = variants, cols = samples).
        count_thresh (float): Minimum fraction of non-missing data required.

    Returns:
        pd.DataFrame: Filtered dataframe with only variants passing missingness threshold.
    """

    # For each variant, count the number of non-missing genotype calls.
    # A variant passes if this count exceeds (number_of_samples * threshold).
    good_idxs = _count_mask(dfH.to_numpy(dtype=float, copy=False), count_thresh)

    # Keep only variants that pass missingness filter
    dfH = dfH.loc[good_idxs]

    return dfH


## Returns common variants with low missingness 
def return_common_filtered(df):
    """
    Wrapper function: filters variants for both allele frequency 
    and missingness.

    Both filters are row-local and idempotent, so this is a no-op when the
    haplotypes were already read with `prefilter=True`.

    Parameters:
        df (pd.DataFrame): Binary allele calls (0/1).

    Returns:
        pd.DataFrame: Subset of common variants with low missingness.
    """
    
    dfp = polarize_by_consensus(df)

    values = dfp.to_numpy(dtype=float, copy=False)

    # First filter by allele frequency, then filter by missingness
    dfH = dfp.loc[_common_mask(values) & _count_mask(values)]

    return dfH
