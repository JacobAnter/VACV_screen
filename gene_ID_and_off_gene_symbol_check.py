import pandas as pd

from CSV_file_utils import update_gene_ID_and_off_gene_symbol

# Two columns of interest for the following endeavour are "ID" as well
# as "ID_manufacturer"
# Upon closer scrutiny, it became apparent that contrary to their names,
# both features basically harbour the same information, namely the NCBI
# Gene ID of the targeted gene
# However, they differ in that while the feature "ID" is populated
# depending on whether the respective experiment was successful or not,
# "ID_manufacturer" is continuously populated, irrespective of the
# experiment's outcome
# Hence, the entries of the "ID_manufacturer" feature are employed in
# order to query NCBI's gene database
# Note that the features "ID" and "ID_manufacturer" only harbour NCBI
# Gene IDs in the case of siRNA, pooled siRNA and esiRNA

# Load the screen data
# Bear in mind that for certain columns, the data type has to be
# manually specified
dtype_dict = {
    "Ensembl_ID_OnTarget_Ensembl_GRCh38_release_87": str,
    "Ensembl_ID_OnTarget_NCBI_HeLa_phs000643_v3_p1_c1_HMB": str,
    "Gene_Description": str,
    "ID": str,
    "ID_OnTarget_Ensembl_GRCh38_release_87": str,
    "ID_OnTarget_Merge": str,
    "ID_OnTarget_NCBI_HeLa_phs000643_v3_p1_c1_HMB": str,
    "ID_OnTarget_RefSeq_20170215": str,
    "ID_manufacturer": str,
    "Name_alternatives": str,
    "PLATE_QUALITY_DESCRIPTION": str,
    "RefSeq_ID_OnTarget_RefSeq_20170215": str,
    "Seed_sequence_common": str,
    "WELL_QUALITY_DESCRIPTION": str,
    "siRNA_error": str,
    "siRNA_number": str,
    "Precursor_Name": str
}

# Dask DataFrames exhibit a peculiarity regarding the index labels: By
# default, the index labels are integers, just as with Pandas
# DataFrames. However, unlike Pandas DataFrames, the index labels do not
# monotonically increase from 0, but restart at 0 for each partition,
# thereby resulting in duplicated index labels (Dask subdivides a Dask
# DataFram into multiple so-called partitions as the whole idea behind
# Dask is to handle large data sets in a memory-efficient way, https://
# docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.reset_
# index.html)
# Hence, performing operations with Dask DataFrames might potentially
# raise the `ValueError: cannot reindex on an axis with duplicate
# labels` error
# In this case, loading the entire data set into a Pandas DataFrame is
# feasible, which is why this is preferred to loading it into a Dask
# DataFrame (strangely enough, this has not been possible in the very
# beginning, which is why Dask was used in the first place)
csv_df = pd.read_csv(
    "VacciniaReport_20170223-0958_ZScored_conc_and_NaN_adjusted.csv",
    sep="\t",
    dtype=dtype_dict
)

# In order to keep track of records withdrawn by NCBI, a column bearing
# the name "Withdrawn_by_NCBI" is inserted into the DataFrame after the
# "ID_manufacturer" column
# Conversion to a list maintains the ordering
column_names = csv_df.columns.to_list()
insertion_index = column_names.index("ID_manufacturer") + 1
csv_df.insert(insertion_index, "Withdrawn_by_NCBI", "No")

updated_csv_df = update_gene_ID_and_off_gene_symbol(csv_df)

# As a last step, save the new Pandas DataFrame to a new CSV file
updated_csv_df.to_csv(
    "Vaccinia_Report_NCBI_Gene_IDs_and_official_gene_symbols_updated.csv",
    # Bear in mind that if a different separator than the default one
    # (comma) is desired, specifying the desired separator is also
    # necessary when saving DataFrames to a CSV file! Apparently, once a
    # CSV file has been read into a Pandas DataFrame, all information
    # regarding the separator is discarded
    sep="\t",
    index=False
)