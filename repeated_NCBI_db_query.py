"""
Unfortunately, the NCBI database query failed 1553 rows during the first
attempt. Therefore, the queries are repeated for those 1553 rows. A more
sophisticated approach would consist of leveraging the fact that many
gene IDs occur multiple times in the data set and thereby the
possibility that the correct name has been stored in another location.
However, it is conceivable that beyond the name, an error also occurred
during the retrieval of the gene ID. Thus, to be on the safe side, the
queries are repeated altogether.
"""

import time

import pandas as pd
from biotite.database import entrez

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

csv_df = pd.read_csv(
    "Vaccinia_Report_intermediate_postprocessing.csv",
    sep="\t",
    dtype=dtype_dict
)

# Determine the indices of the rows the "Name" value of which has been
# assigned to an error message
error_indices = csv_df.index[
    csv_df["Name"].str.contains(
        "External+viewer+error",
        regex=False
    )
    |
    csv_df["Name"].str.contains(
        "OCTYPE html PUBLIC",
        regex=False
    )
].to_list()

ID_change_str = "This record was replaced with GeneID:"

for idx in error_indices:
    NCBI_Gene_ID = csv_df.iloc[idx]["ID_manufacturer"]
    
    # Query NCBI's gene database with the Gene ID currently dealt with
    # Code execution is suspended for one second in order to avoid
    # server-side errors
    time.sleep(1)
    # As simply suspending code execution for a couple of seconds
    # unfortunately does not prevent the occurrence of errors
    # altogether, a try/except statement is incorporated retrying the
    # dabase query for three times in total
    for _ in range(3):
        try:
            NCBI_entry = entrez.fetch_single_file(
                uids=[NCBI_Gene_ID],
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
            "Database query wasn't successful for the row with index "
            f"{idx}."
        )
    
    NCBI_entry_str = NCBI_entry.getvalue()

    # Different approaches are necessary depending on whether merely
    # the official gene symbol was altered, whereas the NCBI Gene ID
    # remained unchanged, or the record has been replaced altogether
    # with an entirely new ID
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
        csv_df.at[idx, "ID"] = new_gene_ID
        csv_df.at[idx, "ID_manufacturer"] = new_gene_ID

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
                f"updated gene ID {new_gene_ID} (row {idx})."
            )
        
        NCBI_entry_str = NCBI_entry.getvalue()
        NCBI_entry_str_list = NCBI_entry_str.split("\n")
        while "" in NCBI_entry_str_list:
            NCBI_entry_str_list.remove("")
        
        # The official gene symbol is comprised in the first list
        # element, but is preceded by the string "1. ", which
        # encompasses three characters
        official_gene_symbol = NCBI_entry_str_list[0][3:]

        csv_df.at[idx, "Name"] = official_gene_symbol
    else:
        # Remove blank lines from the string retrieved from the NCBI
        # entry
        NCBI_entry_str_list = NCBI_entry_str.split("\n")
        while "" in NCBI_entry_str_list:
            NCBI_entry_str_list.remove("")
        
        # Following the removal of empty strings, the official gene
        # symbol is represented by the first list element, but it is
        # preceded by the string "1. ", which encompasses three
        # characters
        # Hence, prior to comparing the gene names provided by the VACV
        # screen data set to the official gene symbols, the first list
        # element has to be sliced accordingly
        official_gene_symbol = NCBI_entry_str_list[0][3:]
        csv_df.at[idx, "Name"] = official_gene_symbol

        # In addition, the corresponding cell of the "ID" feature is
        # populated with the NCBI Gene ID harboured by the
        # "ID_manufacturer" feature (remember that cells of the "ID"
        # feature are not continuously populated)
        csv_df.at[idx, "ID"] = NCBI_Gene_ID

# As a last step, save the new Pandas DataFrame to a new CSV file
csv_df.to_csv(
    "Vaccinia_Report_errors_fixed.csv",
    sep="\t",
    index=False
)