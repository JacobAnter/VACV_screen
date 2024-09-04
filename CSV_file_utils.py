import time

import numpy as np
import pandas as pd
from biotite.database import entrez

# For the sake of modularity and thereby also robustness of the code,
# functions are defined
def update_gene_ID_and_off_gene_symbol(csv_df):
    """
    Performs NCBI Gene database queries for selected gene IDs in a
    Pandas DataFrame and updates both the gene ID and the official gene
    symbol in case of alterations. The gene IDs to perform the query
    with are confined to the single/pooled siRNA and esiRNA subset (i.e.
    rows having as "WellType" value "SIRNA", "POOLED_SIRNA" or
    "ESIRNA").

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
    4.) Some rows contain multiple gene IDs separated by semicola. A
    separate procedure has been implemented for those multi-gene IDs.
    5.) It is kept track of whether NCBI records are still valid or have
    been discontinued via the "Withdrawn_by_NCBI" column.
    6.) The values of the "Name" column are standardised by iterating
    over gene IDs present in the single/pooled siRNA and esiRNA subset,
    but modifying all rows in the CSV file having the respective gene
    ID.

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


def _query_NCBI_Gene_database_for_single_gene_IDs(
        csv_df, single_gene_IDs
):
    """
    Performs NCBI Gene Database queries for single gene IDs and updates
    both the gene ID and the official gene symbol in case of
    alterations.

    Parameters
    ----------
    csv_df: Pandas DataFrame
        A Pandas DataFrame harbouring gene IDs in conjunction with the
        corresponding gene names. It is assumed that the gene IDs are
        stored in the "ID_manufacturer" column, whereas the gene names
        are contained in the "Name" column. The DataFrame is also
        expected to comprise a column named "Withdrawn_by_NCBI", the
        purpose of which is to keep track of withdrawn NCBI records.
    
    single_gene_IDs: iterable
        An iterable harbouring the single gene IDs to perform database
        queries with.

    Returns
    -------
    csv_df: Pandas DataFrame
        The updated Pandas DataFrame, i.e. it contains the updated gene
        ID and/or gene name in case of corresponding alterations.
    """

    ID_change_str = "This record was replaced with GeneID:"
    discontinuation_str = "This record was discontinued."

    for single_gene_ID in single_gene_IDs:
        # Query NCBI's gene database with the Gene ID currently dealt
        # with
        # Code execution is suspended for one second in order to avoid
        # server-side errors
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
                    f"{single_gene_ID})."
                )
                continue
            
            NCBI_entry_str = NCBI_entry.getvalue()
            if discontinuation_str in NCBI_entry_str:
                csv_df.loc[
                    # Bear in mind that the values in "ID_manufacturer"
                    # have just been changed; hence, the old value must
                    # not be used in the equality check!
                    csv_df["ID_manufactuer"] == new_gene_ID,
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
            pass

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

    return csv_df


def mend_error_messages_single_IDs(csv_df):
    """
    Performs NCBI Gene database queries for single gene IDs an error
    message has been assigned to as "Name" value. In total, three
    different error messages are covered, two of which contain
    "External+viewer+error" as common substring.

    Instead of iterating over each affected row individually, this
    function pursues a more intelligent approach consisting of iterating
    over the unique gene IDs of the affected rows. Again, the fact that
    many gene IDs occur multiple times throughout the CSV file is
    leveraged in order to achieve a reduction of run time.

    As one run usually does not mend all error messages, which is why
    the procedure is repeated until this indeed is the case.

    Parameters
    ----------
    csv_df: Pandas DataFrame
        A Pandas DataFrame having error messages as values of its "Name"
        column for single gene IDs. Note that only the three
        abovementioned error messages are covered.

    Returns
    -------
    csv_df: Pandas DataFrame
        The Pandas DataFrame provided as input, but with correctly
        retrieved official gene symbols where the error messages
        occurred.
    """

    n_error_message_rows = np.count_nonzero(
        csv_df["Name"].str.contains(
            "External+viewer+error",
            regex=False
        )
        |
        csv_df["Name"].str.contains(
            "OCTYPE html PUBLIC",
            regex=False
        )
    )

    assert n_error_message_rows > 0, (
        "The Pandas DataFrame provided as input does not contain any "
        "error messages as \"Name\" value or at least none of the "
        "error messages covered by this function."
    )

    # As a first step, retrieve the unique gene IDs of rows having error
    # messages as "Name" value
    unique_gene_IDs_error_message_rows = csv_df.loc[
        csv_df["Name"].str.contains(
            "External+viewer+error",
            regex=False
        )
        |
        csv_df["Name"].str.contains(
            "OCTYPE html PUBLIC",
            regex=False
        )
    ]["ID_manufacturer"].unique()

    while n_error_message_rows > 0:
        csv_df = _query_NCBI_Gene_database_for_single_gene_IDs(
            csv_df, unique_gene_IDs_error_message_rows
        )

        n_error_message_rows = np.count_nonzero(
            csv_df["Name"].str.contains(
                "External+viewer+error",
                regex=False
            )
            |
            csv_df["Name"].str.contains(
                "OCTYPE html PUBLIC",
                regex=False
            )
        )

    return csv_df


def extract_valid_and_named_targets_from_df(csv_df):
    """
    Extracts from a Pandas DataFrame those targets that are known and
    still valid. The function takes account of the fact that there are
    rows containing multiple gene IDs, i.e. multiple targets. It is
    assumed that the Pandas DataFrame contains a column named
    "Withdrawn_by_NCBI" based upon which the distinction between valid
    and invalid targets is performed. It is also assumed to have columns
    termed "ID", "ID_manufacturer" and "Name".

    Parameters
    ----------
    csv_df: Pandas DataFrame
        A Pandas DataFrame invalid and unnamed targets are supposed to
        be removed from.

    Returns
    -------
    csv_df: Pandas DataFrame
        The Pandas DataFrame provided as input, but with invalid as well
        as unnammed targets having been removed.
    """

    # As a first step, remove targets the "ID_manufacturer" value of
    # which is "Not available"
    csv_df = csv_df.loc[
        # Bear in mind that due to operator precedence, i.e. the logical
        # AND (&) being evaluated before the equality check, the
        # equality check has to be surrounded by parentheses
        csv_df["ID_manufacturer"] != "Not available"
    ]

    # As a second step, remove rows containing single invalid IDs
    # To this end, the fact that the "Withdrawn_by_NCBI" value of such
    # rows equals exactly "Yes" is leveraged
    csv_df = csv_df.loc[
        ~(csv_df["Withdrawn_by_NCBI"] == "Yes")
    ]

    # As a third step, remove rows containing multiple exclusively
    # invalid IDs
    # The rationale behind the selection below is as follows: First, the
    # rows to get rid of are chosen by selecting rows containing
    # multiple IDs and rows not comprising "No" in their
    # "Withdrawn_by_NCBI" value; ultimately, the resulting Boolean
    # Series represents all rows containing multiple exclusively invalid
    # IDs
    # Finally, this Boolean Series is inverted by the tilde operator so
    # as to discard those rows while retaining all others
    csv_df = csv_df.loc[
        ~(
            csv_df["ID_manufacturer"].str.contains(";")
            &
            (~csv_df["Withdrawn_by_NCBI"].str.contains("No"))
        )
    ]

    # Finally, address the rows containing multiple gene IDs and
    # therefore multiple targets
    # Again, multiple rows are modified at once by leveraging the fact
    # that also many combinations of gene IDs occur multiple times
    # throughout the CSV file
    unique_multi_target_ids = csv_df.loc[
        # Of the rows containing multiple targets, only those are
        # addressed containing at least one invalid target
        csv_df["ID_manufacturer"].str.contains(";")
        &
        # Choosing rows the "Withdrawn_by_NCBI" value of which contains
        # "Yes" is still necessary as there is the possibility that all
        # IDs in one row are valid (i.e. have not been withdrawn and are
        # therefore represented by "No")
        csv_df["Withdrawn_by_NCBI"].str.contains("Yes"),
        "ID_manufacturer"
    ].unique()

    for ids in unique_multi_target_ids:
        # For one specific ID, the values in one column are always the
        # same
        # Hence, for each column, simply the first entry is retrieved
        # In order to access elements of a Pandas series based on their
        # position, i.e. in the same way one would index a list, `.iloc`
        # is required
        withdrawn_string = csv_df.loc[
            csv_df["ID_manufacturer"] == ids, "Withdrawn_by_NCBI"
        ].iloc[0]
        
        ID_string = csv_df.loc[
            csv_df["ID_manufacturer"] == ids, "ID"
        ].iloc[0]

        name_string = csv_df.loc[
            csv_df["ID_manufacturer"] == ids, "Name"
        ].iloc[0]

        withdrawn_list = withdrawn_string.split(";")
        ID_list = ID_string.split(";")
        ID_manufacturer_list = ids.split(";")
        name_list = name_string.split(";")

        while "Yes" in withdrawn_list:
            invalid_idx = withdrawn_list.index("Yes")

            withdrawn_list.pop(invalid_idx)
            ID_list.pop(invalid_idx)
            ID_manufacturer_list.pop(invalid_idx)
            name_list.pop(invalid_idx)
        
        # Following the removal of invalid targets, the list entries are
        # concatenated again and the respective fields in the DataFrame
        # are updated
        withdrawn_string = ";".join(withdrawn_list)
        ID_string = ";".join(ID_list)
        ID_manufacturer_string = ";".join(ID_manufacturer_list)
        name_string = ";".join(name_list)

        # Now, modify multiple rows at once by leveraging the
        # "ID_manufacturer" value
        # The "ID_manufacturer" entries are modified last so as to be
        # able to continuously use the original value for row
        # specification
        csv_df.loc[
            csv_df["ID_manufacturer"] == ids, "Withdrawn_by_NCBI"
        ] = withdrawn_string
        csv_df.loc[
            csv_df["ID_manufacturer"] == ids, "ID"
        ] = ID_string
        csv_df.loc[
            csv_df["ID_manufacturer"] == ids, "Name"
        ] = name_string
        csv_df.loc[
            csv_df["ID_manufacturer"] == ids, "ID_manufacturer"
        ] = ID_manufacturer_string

    return csv_df