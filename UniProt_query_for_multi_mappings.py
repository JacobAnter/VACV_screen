"""
The purpose of this Python script is to determine for gene names that
have been mapped to multiple UniProt IDs which UniProt IDs to retain
(isoforms generated via alternative splicing) and which ones to discard
(proteins given rise to by similar, yet distinct genes). To this end,
the UniProt database is queried.
"""

import time

import pandas as pd
from biotite.database import uniprot

def _fetch_from_UniProt_database(gene_name, uniprot_id):
    """
    ...

    Parameters
    ----------
    gene_name: str
        ...
    uniprot_id: str
        ...

    Returns
    -------
    UniProt_entry_str: str
        ...
    """

    time.sleep(1)

    for _ in range(3):
        try:
            UniProt_entry = uniprot.fetch(
                ids=uniprot_id,
                format="fasta"
            )
            UniProt_entry_str = UniProt_entry.getvalue()

            # It sometimes happens that the queried information is
            # incompletely transmitted
            # In such a case, the query has to be repeated
            assert "GN=" in UniProt_entry_str

            break
        except:
            time.sleep(1)
    else:
        print(
            "UniProt database query was not successful for UniProt ID "
            f"{uniprot_id} (gene {gene_name})."
        )
        return "Query failed"

    return UniProt_entry_str


multi_mappings_df = pd.read_csv(
    "multi_mappings_info.tsv",
    sep="\t"
)

for i, id_str in enumerate(multi_mappings_df["UniProt_IDs_list"]):
    id_list = id_str.split(",")
    gene_name = multi_mappings_df.iloc[i]["Gene_name"]

    ids_to_omit = []
    for UniProt_ID in id_list:
        UniProt_entry_str = _fetch_from_UniProt_database(
            gene_name, UniProt_ID
        )

        if UniProt_entry_str == "Query failed":
            continue

        # The gene name of the UniProt entry currently dealt with has to
        # be extracted
        # This information is contained in the Fasta header, i.e. the
        # first line
        UniProt_entry_first_line = UniProt_entry_str.split("\n")[0]
        # Conveniently enough, the desired information is separated from
        # the adjacent text by spaces
        first_line_split = UniProt_entry_first_line.split()
        # The "split()" string method returns a list of strings;
        # determine the index of the string containing the desired
        # information
        gene_idx = [
            i for i, info in enumerate(first_line_split)
            if "GN=" in info
        ][0]
        # The string is appropriately sliced in order to extract the
        # gene name, i.e. the "GN=" part is removed
        gene_info = first_line_split[gene_idx][3:]

        if gene_info != gene_name:
            ids_to_omit.append(UniProt_ID)
    
    # Update the "UniProt_IDs_to_omit" value of the gene currently dealt
    # with
    # Prior to doing so, the list of UniProt IDs to omit is converted
    # into a string with commas between the individual IDs
    ids_to_omit_str = ",".join(ids_to_omit)
    # Empty strings are interpreted by Pandas as NaN, which is why an
    # alternative string is used indicating the lack of any IDs to
    # remove
    if ids_to_omit_str == "":
        ids_to_omit_str = "None to omit"
    multi_mappings_df.at[i, "UniProt_IDs_to_omit"] = ids_to_omit_str

# Save the updated DataFrame to a TSV file
multi_mappings_df.to_csv(
    "multi_mappings_info_updated.tsv",
    sep="\t",
    index=False
)