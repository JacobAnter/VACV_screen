import time

import numpy as np
import pandas as pd
from biotite.database import entrez

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
    "Vaccinia_Report_errors_fixed.csv",
    sep="\t",
    dtype=dtype_dict
)

ID_change_str = "This record was replaced with GeneID:"

# Retrieve the gene IDs to be iterated over
# They are confined to the single/pooled siRNA and esiRNA subset as the
# data set also comprises miRNA data
subset_gene_IDs = np.unique(csv_df.loc[
    # Bear in mind that due to operator precedence, i.e. the logical OR
    # operator being evaluated before the equality check, the equality
    # check has to be surrounded by parentheses
    (csv_df["WellType"] == "SIRNA")
    |
    (csv_df["WellType"] == "POOLED_SIRNA")
    |
    (csv_df["WellType"] == "ESIRNA"),
    "ID_manufacturer"
])

for gene_ID in subset_gene_IDs:
    # Query NCBI's gene database with the Gene ID currently dealt with
    # Code execution is suspended for one second in order to avoid
    # server-side errors
    # There are 23,413 unique gene IDs in total in the subset; for each
    # of those gene IDs, the NCBI database is queried at least once,
    # entailing the suspension of code execution for one second; in
    # total, this amounts to a "waiting time" of 23,413 seconds, which
    # corresponds to roughly six-and-a-half hours
    # To be on the safe side, the maximum wall clock time in the shell
    # script is going to be set to 18 hours
    time.sleep(1)
    # As simply suspending code execution for a couple of seconds
    # unfortunately does not prevent the occurrence of errors
    # altogether, a try/except statement is incorporated retrying the
    # dabase query for three times in total
    for _ in range(3):
        try:
            NCBI_entry = entrez.fetch_single_file(
                uids=[gene_ID],
                file_name=None,
                db_name="gene",
                ret_type="",
                ret_mode="text"
            )
            break
        except:
            time.sleep(1)
    else:
        print(
            f"Database query wasn't successful for gene ID {gene_ID}."
        )
    
    # As the file_name is specified to be None, Biotite's
    # fetch_single_file() function returns a StringIO object
    # It's content can be accessed via the getvalue() method
    # Note that the getvalue() method is preferred to the read() method
    # as the latter moves the cursor to the last index so that
    # repeatedly using the read() method returns an empty string
    NCBI_entry_str = NCBI_entry.getvalue()

    # Bear in mind that only records having been replaced altogether are
    # affected by the mistake
    # Hence, modifications to the DataFrame are only done in the case of
    # such records
    if ID_change_str in NCBI_entry_str:
        # The respective record has been replaced altogether with a new
        # ID
        # Hence, the new ID is retrieved and used to query NCBI's gene
        # database
        NCBI_entry_str_list = NCBI_entry_str.split("\n")
        # For some strange reason, the string retrieved from the NCBI
        # entry contains blank lines; they are removed
        while "" in NCBI_entry_str_list:
            NCBI_entry_str_list.remove("")
        
        # The new ID is comprised in the penultimate list element and
        # conveniently enough separated from the preceding string by a
        # space character
        new_gene_ID = NCBI_entry_str_list[-2].split()[-1]
        csv_df.loc[
            csv_df["ID_manufacturer"] == gene_ID,
            ["ID", "ID_manufacturer"]
        ] = new_gene_ID

        # Again, in a bid to prevent the occurrence of server-side
        # errors, code execution is suspended for one second
        time.sleep(1)
        for _ in range(3):
            try:
                NCBI_entry = entrez.fetch_single_file(
                    uids=[new_gene_ID],
                    file_name=None,
                    db_name="gene",
                    ret_type="",
                    ret_mode="text"
                )
                break
            except:
                time.sleep(1)
        else:
            print(
                "Querying the database wasn't successful for the "
                f"updated gene ID {new_gene_ID} (formerly gene ID "
                f"{gene_ID})."
            )
        
        NCBI_entry_str = NCBI_entry.getvalue()
        NCBI_entry_str_list = NCBI_entry_str.split("\n")
        while "" in NCBI_entry_str_list:
            NCBI_entry_str_list.remove("")
        
        # The official gene symbol is comprised in the first list
        # element, but is preceded by the string "1. ", which
        # encompasses three characters
        official_gene_symbol = NCBI_entry_str_list[0][3:]

        csv_df.loc[
            csv_df["ID_manufacturer"] == new_gene_ID,
            "Name"
        ] = official_gene_symbol

# As a last step, save the new Pandas DataFrame to a new CSV file
csv_df.to_csv(
    "Vaccinia_Report_correct_indexing.csv",
    sep="\t",
    index=False
)