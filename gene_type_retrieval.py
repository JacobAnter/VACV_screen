"""
The purpose of this Python script is to retrieve the gene type of the
genes for which the mapping to either Swiss-Prot or TrEMBL was
successful. To this end, the NCBI database is queried.
"""

import time

import pandas as pd
from biotite.database import entrez

gene_type_df = pd.read_csv(
    "gene_type_query.tsv",
    sep="\t"
)

gene_IDs_with_types_not_covered = []

for i, gene_ID in enumerate(gene_type_df["Gene_ID"]):
    time.sleep(1)

    for _ in range(3):
        try:
            NCBI_entry = entrez.fetch_single_file(
                uids=[str(gene_ID)],
                file_name=None,
                db_name="gene",
                ret_type="xml"
            )

            # As the file_name is specified to be None, Biotite's
            # fetch_single_file() function returns a StringIO object
            # It's content can be accessed via the getvalue() method
            # Note that the getvalue() method is preferred to the
            # read() method as the latter moves the cursor to the last
            # index so that repeatedly using the read() method returns
            # an empty string
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
            "NCBI database query wasn't successful for gene ID "
            f"{gene_ID}."
        )
        continue

    NCBI_entry_str_list = NCBI_entry_str.split("\n")

    # Determine the index of the line containing the desired information
    gene_type_line_index = [
        i for i, line in enumerate(NCBI_entry_str_list)
        if "<Entrezgene_type value=" in line
    ][0]
    gene_type_line = NCBI_entry_str_list[gene_type_line_index]

    if "pseudo" in gene_type_line:
        gene_type_df.at[i, "Gene_type"] = "pseudo"
    elif "ncRNA" in gene_type_line:
        gene_type_df.at[i, "Gene_type"] = "ncRNA"
    elif "other" in gene_type_line:
        gene_type_df.at[i, "Gene_type"] = "other"
    elif "snoRNA" in gene_type_line:
        gene_type_df.at[i, "Gene_type"] = "snoRNA"
    elif "protein-coding" in gene_type_line:
        gene_type_df.at[i, "Gene_type"] = "protein-coding"
    elif "unknown" in gene_type_line:
        gene_type_df.at[i, "Gene_type"] = "unknown"
    else:
        gene_IDs_with_types_not_covered.append(gene_ID)

if len(gene_IDs_with_types_not_covered) > 0:
    print(
        "The following gene IDs have gene types other than the ones "
        "covered:\n",
        gene_IDs_with_types_not_covered,
        sep=""
    )

# Overwrite the original TSV file with the updated DataFrame
gene_type_df.to_csv(
    "gene_type_query.tsv",
    sep="\t",
    index=False
)