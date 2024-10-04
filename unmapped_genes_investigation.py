"""
The purpose of this Python script is to determine for each of the
1,659 unmapped genes the reason, i.e. why no UniProt entry exists for
those genes. Reasons known thus far are that the gene merely gives rise
to ncRNA, or that it is a pseudogene for which a UniProt entry simply
does not exist. Gene names to which these two reasons do not apply are
gathered and later investigated.
"""

import time

import pandas as pd
from biotite.database import entrez
from biotite.database import uniprot

def _uniprot_database_search(gene_name, query):
    """
    ...

    Parameters
    ----------
    gene_name: str
        ...
    query: ...
        ...

    Returns
    -------
    uniprot_ids: ...
        ...
    """

    time.sleep(1)

    # As simply suspending code execution for a couple of seconds
    # unfortunately does not prevent the occurrence of errors
    # altogether, a try/except statement is incorporated retrying the
    # dabase query for three times in total
    for _ in range(3):
        try:
            uniprot_ids = uniprot.search(query)
            break
        except:
            time.sleep(1)
    else:
        print(
            "UniProt database query was not successful for gene "
            f"{gene_name}."
        )
        return "Query failed"

    return uniprot_ids

def investigate_unmapped_gene_names(pandas_df):
    """
    Investigates for the gene names comprised in the Pandas DataFrame
    provided as input the reason for them not having any UniProt
    entries. Reasons known thus far are either that a gene gives rise to
    a ncRNA, which, logically enough, is not translated into a protein,
    or that a gene is a pseudogene without any UniProt entries.

    Gene names to which these two reasons do not apply are gathered and
    investigated later.

    Parameters
    ----------
    pandas_df: Pandas DataFrame
        A Pandas DataFrame harbouring the gene names for which to
        identify the reason for not having any UniProt entries. The
        DataFrame is assumed to comprise four columns. The first bears
        the name "Gene_name" and, as its name already suggests, harbours
        the gene names to investigate. The second carries the name
        "Gene_ID" and stores the NCBI gene ID for the genes. The third
        bears the name "Gene_type" and the fourth is named
        "UniProt_entry_available".

    Returns
    -------
    pandas_df: Pandas DataFrame
        The Pandas DataFrame provided as input, but including for each
        of the gene names the reason for not having any UniProt entries.
    """

    gene_IDs_with_cases_not_covered = []

    for i, gene_name in enumerate(pandas_df["Gene_name"]):
        # First, it is investigated whether the gene currently dealt
        # with encodes ncRNA
        # To this end, the NCBI database is queried, which requires the
        # gene ID
        gene_ID = str(pandas_df["Gene_ID"].iloc[i])
        time.sleep(1)

        # As simply suspending code execution for a couple of seconds
        # unfortunately does not prevent the occurrence of errors
        # altogether, a try/except statement is incorporated retrying
        # the dabase query for three times in total
        for _ in range(3):
            try:
                NCBI_entry = entrez.fetch_single_file(
                    uids=[gene_ID],
                    file_name=None,
                    db_name="gene",
                    ret_type="",
                    ret_mode="xml"
                )

                # As the file_name is specified to be None, Biotite's
                # fetch_single_file() function returns a StringIO object
                # It's content can be accessed via the getvalue() method
                # Note that the getvalue() method is preferred to the
                # read() method as the latter moves the cursor to the
                # last index so that repeatedly using the read() method
                # returns an empty string
                NCBI_entry_str = NCBI_entry.getvalue()

                # It sometimes happens that the queried information is
                # incompletely transmitted
                # In such a case, the query has to be repeated
                assert "<Entrezgene_type value=" in NCBI_entry_str

                break
            except:
                time.sleep(1)
        else:
            print(
                "NCBI database query wasn't successful for "
                f"{gene_name}."
            )
            continue
        
        NCBI_entry_str_list = NCBI_entry_str.split("\n")
        
        # Determine the index of the line containing the desired
        # information, i.e. whether the gene currently dealt with
        # encodes ncRNA or not
        gene_type_line_index = [
            i for i, line in enumerate(NCBI_entry_str_list)
            if "<Entrezgene_type value=" in line
        ][0]
        gene_type_line = NCBI_entry_str_list[gene_type_line_index]
        
        if "ncRNA" in gene_type_line:
            pandas_df.at[i, "Gene_type"] = "ncRNA"
            pandas_df.at[i, "UniProt_entry_available"] = "No"
        elif "pseudo" in gene_type_line:
            pandas_df.at[i, "Gene_type"] = "Pseudogene"
            
            # Investigate whether there indeed is no UniProt entry for
            # this gene
            # To this end, the UniProt database is queried
            # The gene name and organism ID are used for the query; the
            # organism ID for Homo sapiens is 9606
            # On the one hand, Biotite's SimpleQuery() class does not
            # allow query terms to contain the following strings in
            # capital letters, amongst others: "AND", "OR", "NOT"
            # On the other hand, the UniProt database is
            # case-insensitive, i.e. the search results are invariant to
            # whether the query contains uppercase or lowercase letters
            # As some of the forbidden strings do occur in the gene
            # names to query, all letters in the gene name are converted
            # to lowercase letters prior to the query
            gene_name_for_query = gene_name.lower()

            query = (
                uniprot.SimpleQuery("gene_exact", gene_name_for_query)
                &
                uniprot.SimpleQuery("organism_id", "9606")
            )
            
            ids = _uniprot_database_search(gene_name, query)

            if ids == "Query failed":
                continue

            if len(ids) == 0:
                pandas_df.at[i, "UniProt_entry_available"] = "No"
            else:
                pandas_df.at[i, "UniProt_entry_available"] = "Yes"

        elif "protein-coding" in gene_type_line:
            pandas_df.at[i, "Gene_type"] = "Protein-coding"
            
            # Investigate whether there indeed is no UniProt entry for
            # this gene
            gene_name_for_query = gene_name.lower()

            query = (
                uniprot.SimpleQuery("gene_exact", gene_name_for_query)
                &
                uniprot.SimpleQuery("organism_id", "9606")
            )
            
            ids = _uniprot_database_search(gene_name, query)

            if ids == "Query failed":
                continue

            if len(ids) == 0:
                pandas_df.at[i, "UniProt_entry_available"] = "No"
            else:
                pandas_df.at[i, "UniProt_entry_available"] = "Yes"

        else:
            gene_IDs_with_cases_not_covered.append(gene_ID)
    
    if len(gene_IDs_with_cases_not_covered) > 0:
        print(
            "The following gene IDs have gene types other than the "
            "ones covered:\n",
            gene_IDs_with_cases_not_covered
        )

    return pandas_df

# Load the TSV file harbouring the unmapped gene names into a Pandas
# DataFrame
unmapped_genes_df = pd.read_csv(
    "unmapped_genes_info.tsv",
    sep="\t"
)

unmapped_genes_df = investigate_unmapped_gene_names(unmapped_genes_df)

# Save the updated DataFrame to a TSV file
unmapped_genes_df.to_csv(
    "unmapped_genes_info_updated.tsv",
    sep="\t",
    index=False
)