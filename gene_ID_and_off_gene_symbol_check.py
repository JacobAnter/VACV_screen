import time

import pandas as pd
from biotite.database import entrez

# For the sake of modularity and thereby also robustness of the code,
# functions are defined
def update_gene_ID_and_off_gene_symbol(csv_df):
    """
    Perform NCBI Gene database queries for selected gene IDs in a Pandas
    DataFrame and update both the gene ID and the official gene symbol
    in case of alterations.

    The gene ID and official gene symbol update is accomplished in two
    steps. The first step consists of exclusively updating single gene
    IDs. In the second step, rows harbouring multiple gene IDs and
    accordingly gene names are addressed.

    The function takes the issues/mistakes made previously into account
    via the following corrections:
    1.) `continue` statements have been introduced after failed database
    queries.
    2.) In the case of records having been replaced altogether with a
    new one, indexing has been corrected such that following the
    splitting of the entry string, the penultimate, i.e. second to last
    list element is retrieved.
    3.) In an effort to reduce the execution time, it is not iterated
    over each and every line of the DataFrame. Instead, the fact that
    many gene IDs occur multiple times is leveraged by iterating over
    the gene IDs and thereby modifying multiple rows at once.

    Parameters
    ----------
    csv_df: Pandas DataFrame
        A Pandas DataFrame harbouring gene IDs in conjunction with the
        corresponding gene names. It is assumed that the gene IDs are
        stored in the "ID_manufacturer" column, whereas the gene names
        are contained in the "Name" column. The DataFrame is also
        expected to comprise a column named "Withdrawn_by_NCBI", the
        purpose of which is to keep track of withdrawn NCBI records.

    Returns
    -------
    csv_df: Pandas DataFrame
        The updated Pandas DataFrame, i.e. it contains the updated gene
        ID and/or gene name in case of corresponding alterations.
    """

    # Retrieve single gene IDs
    # They are confined to the single/pooled siRNA and esiRNA subset as
    # the data set also comprises miRNA data
    single_gene_IDs = csv_df.loc[
        (
            (csv_df["WellType"] == "SIRNA")
            |
            (csv_df["WellType"] == "POOLED_SIRNA")
            |
            (csv_df["WellType"] == "ESIRNA")
        )
        &
        (~csv_df["ID_manufacturer"].str.contains(";")),
        "ID_manufacturer"
    ].unique()

    # Also retrieve multiple gene IDs
    # The mutiple gene IDs are also confined to the single/pooled siRNA
    # and esiRNA subset
    multiple_gene_IDs = csv_df.loc[
        (
            (csv_df["WellType"] == "SIRNA")
            |
            (csv_df["WellType"] == "POOLED_SIRNA")
            |
            (csv_df["WellType"] == "ESIRNA")
        )
        &
        csv_df["ID_manufacturer"].str.contains(";"),
        "ID_manufacturer"
    ].unique()

    discontinuation_str = "This record was discontinued."
    ID_change_str = "This record was replaced with GeneID:"


    for single_gene_ID in single_gene_IDs:
        # Query NCBI's gene database with the Gene ID currently dealt
        # with
        # Code execution is suspended for one second in order to avoid
        # server-side errors
        # There are 23,341 unique single gene IDs in total in the
        # subset; for each of those gene IDs, the NCBI database is 
        # queried at least once, entailing the suspension of code
        # execution for one second; in total, this amounts to a "waiting
        # time" of 23,341 seconds, which corresponds to roughly
        # six-and-a-half hours
        time.sleep(1)
        # As simply suspending code execution for a couple of seconds
        # unfortunately does not prevent the occurrence of errors
        # altogether, a try/except statement is incorporated retrying
        # the dabase query for three times in total
        for _ in range(3):
            try:
                NCBI_entry = entrez.fetch_single_file(
                    uids=[single_gene_ID],
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
                "Database query wasn't successful for the single gene "
                f"ID {single_gene_ID}."
            )
            continue
        
        # As the file_name is specified to be None, Biotite's
        # fetch_single_file() function returns a StringIO object
        # It's content can be accessed via the getvalue() method
        # Note that the getvalue() method is preferred to the read()
        # method as the latter moves the cursor to the last index so
        # that repeatedly using the read() method returns an empty
        # string
        NCBI_entry_str = NCBI_entry.getvalue()

        # Different approaches are necessary depending on whether merely
        # the official gene symbol was altered, whereas the NCBI Gene ID
        # remained unchanged, or the record has been replaced altogether
        # with an entirely new ID
        if ID_change_str in NCBI_entry_str:
            # The respective record has been replaced altogether with a
            # new ID
            # Hence, the new ID is retrieved and used to query NCBI's
            # gene database
            NCBI_entry_str_list = NCBI_entry_str.split("\n")

            # For some strange reason, the string retrieved from the
            # NCBI entry contains blank lines; they are removed
            while "" in NCBI_entry_str_list:
                NCBI_entry_str_list.remove("")
            
            # The new ID is comprised in the penultimate list element
            # and conveniently enough separated from the preceding
            # string by a space character
            new_gene_ID = NCBI_entry_str_list[-2].split()[-1]
            # As the gene ID currently dealt with may well occur outside
            # the subset, the gene ID update is conducted for the entire
            # data set; the same applies to the other updates later for
            # the "Withdrawn_by_NCBI" and "Name" columns
            csv_df.loc[
                csv_df["ID_manufacturer"] == single_gene_ID,
                "ID_manufacturer"
            ] = new_gene_ID
            # Bear in mind that the values in "ID_manufacturer" have
            # just been changed; hence, the old value must not be used
            # in the equality check!
            csv_df.loc[
                csv_df["ID_manufacturer"] == new_gene_ID,
                "ID"
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
                    f"{single_gene_ID})."
                )
                continue

            NCBI_entry_str = NCBI_entry.getvalue()
            if discontinuation_str in NCBI_entry_str:
                csv_df.loc[
                    csv_df["ID_manufacturer"] == new_gene_ID,
                    "Withdrawn_by_NCBI"
                ] = "Yes"

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

        else:
            # The gene ID remained unchanged, while the official gene
            # name may well have been changed
            # As the gene ID currently dealt with may well occur outside
            # the subset, the discontinuation state update is conducted
            # for the entire data set; the same applies to the other
            # updates later for the "Name" and "ID" columns
            if discontinuation_str in NCBI_entry_str:
                csv_df.loc[
                    csv_df["ID_manufacturer"] == single_gene_ID,
                    "Withdrawn_by_NCBI"
                ] = "Yes"
            
            # Remove blank lines from the string retrieved from the NCBI
            # entry
            NCBI_entry_str_list = NCBI_entry_str.split("\n")
            while "" in NCBI_entry_str_list:
                NCBI_entry_str_list.remove("")
            
            # Following the removal of empty strings, the official gene
            # symbol is represented by the first list element, but it is
            # preceded by the string "1. ", which encompasses three
            # characters
            # Hence, the first list element has to be sliced accordingly
            official_gene_symbol = NCBI_entry_str_list[0][3:]
            csv_df.loc[
                csv_df["ID_manufacturer"] == single_gene_ID,
                "Name"
            ] = official_gene_symbol

            # Also populate the corresponding cells of the "ID" feature
            # with the NCBI Gene ID (remember that cells of the "ID"
            # feature are not continuously populated)
            csv_df.loc[
                csv_df["ID_manufacturer"] == single_gene_ID,
                "ID"
            ] = single_gene_ID
    

    for multi_gene_ID in multiple_gene_IDs:
        # The gene IDs are concatenated into one string with semicola as
        # separator
        # In order to pass them all at once into the database query,
        # the string is converted into a comma-separated list
        multi_gene_ID_list = multi_gene_ID.split(";")

        # Query NCBI's gene database with the Gene ID currently dealt
        # with
        # Code execution is suspended for one second in order to avoid
        # server-side errors
        # There are 73 unique multi-gene IDs in total in the subset;
        # for each of those gene IDs, the NCBI database is queried at
        # least once, entailing the suspension of code execution for one
        # second; in total, this amounts to a "waiting time" of 73
        # seconds, which corresponds to slightly more than a minute
        # To be on the safe side, the maximum wall clock time in the
        # shell script is going to be set to 36 hours
        time.sleep(1)
        # As simply suspending code execution for a couple of seconds
        # unfortunately does not prevent the occurrence of errors
        # altogether, a try/except statement is incorporated retrying
        # the dabase query for three times in total
        for _ in range(3):
            try:
                NCBI_entry = entrez.fetch_single_file(
                    uids=multi_gene_ID_list,
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
                "Database query wasn't successful for the multi-gene "
                f"ID {multi_gene_ID}."
            )
            continue
        
        NCBI_entry_str = NCBI_entry.getvalue()
        # Extract the entries for the individual gene IDs
        # To this end, the fact that individual entries are separated by
        # two consecutive newline characters ("\n\n") is harnessed
        individual_entries_list = NCBI_entry_str.split("\n\n")

        # Remove any blank lines represented by empty strings
        while "" in individual_entries_list:
            individual_entries_list.remove("")

        updated_multi_gene_ID_list = []
        updated_off_gene_symbol_list = []
        updated_withdrawn_list = []

        for gene_ID, multi_gene_ID_entry in zip(
            multi_gene_ID_list, individual_entries_list
        ):
            if ID_change_str in multi_gene_ID_entry:
                # The respective record has been replaced altogether
                # with a new ID
                # Hence, the new ID is retrieved and used to query
                # NCBI's gene database
                lines_list = multi_gene_ID_entry.split("\n")
                # Again, remove blank lines represented by empty strings
                while "" in lines_list:
                    lines_list.remove("")
                
                # The new ID is comprised in the penultimate list
                # element and conveniently enough separated from the
                # preceding string by a space character
                new_gene_ID = lines_list[-2].split()[-1]
                updated_multi_gene_ID_list.append(new_gene_ID)

                # Again, in a bid to prevent the occurrence of
                # server-side errors, code execution is suspended for
                # one second
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
                        "Querying the database wasn't successful for "
                        f"the gene ID {gene_ID} (part of multi-gene ID "
                        f"{multi_gene_ID})."
                    )
                    updated_off_gene_symbol_list.append("query_failed")
                    updated_withdrawn_list.append("query_failed")
                    continue
                
                NCBI_entry_str = NCBI_entry.getvalue()
                if discontinuation_str in NCBI_entry_str:
                    updated_withdrawn_list.append("Yes")
                else:
                    updated_withdrawn_list.append("No")
                
                NCBI_entry_str_list = NCBI_entry_str.split("\n")
                while "" in NCBI_entry_str_list:
                    NCBI_entry_str_list.remove("")
                
                # The official gene symbol is comprised in the first
                # list element, but is preceded by the string "1. ",
                # which encompasses three characters
                official_gene_symbol = NCBI_entry_str_list[0][3:]
                updated_off_gene_symbol_list.append(
                    official_gene_symbol
                )

            else:
                # The gene ID remained unchanged, while the official
                # gene name may well have been changed
                if discontinuation_str in multi_gene_ID_entry:
                    updated_withdrawn_list.append("Yes")
                else:
                    updated_withdrawn_list.append("No")

                updated_multi_gene_ID_list.append(gene_ID)
                
                lines_list = multi_gene_ID_entry.split("\n")
                # Again, remove blank lines represented by empty strings
                while "" in lines_list:
                    lines_list.remove("")
                
                # Following the removal of empty strings, the official
                # gene symbol is represented by the first list element,
                # but it is preceded by the string "x. ", with x
                # representing the position index of the retrieved entry
                # The string encompasses three characters
                # Hence, the first list element has to be sliced
                # accordingly
                official_gene_symbol = lines_list[0][3:]
                updated_off_gene_symbol_list.append(
                    official_gene_symbol
                )

        # Now that all the data has been gathered for all gene IDs, the
        # Pandas DataFrame is modified
        csv_df.loc[
            # Thus far, the gene IDs have merely been gathered in the
            # corresponding list
            # Thus, row selection can still be performed via the
            # previous, unaltered multi-gene ID
            csv_df["ID_manufacturer"] == multi_gene_ID,
            "Withdrawn_by_NCBI"
        ] = ";".join(updated_withdrawn_list)

        csv_df.loc[
            csv_df["ID_manufacturer"] == multi_gene_ID,
            "Name"
        ] = ";".join(updated_off_gene_symbol_list)

        csv_df.loc[
            csv_df["ID_manufacturer"] == multi_gene_ID,
            ["ID", "ID_manufacturer"]
        ] = ";".join(updated_multi_gene_ID_list)
    
    return csv_df


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