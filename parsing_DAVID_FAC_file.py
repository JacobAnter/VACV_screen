"""
The purpose of this Python script is to translate the functionality
provided by the "parseDavidFAC.m" file from Matlab into Python. The
aforementioned file is part of the source code made available publicly
in the context of the publication "RNAi Screening Reveals Proteasome-
and Cullin3-Dependent Stages in Vaccinia Virus Infection".
"""

import numpy as np
import pandas as pd

def parse_DAVID_FAC_file(file_path, mapping_option="union"):
    """
    ...

    In the DAVID FAC file, the third column of each cluster header may
    optionally contain a custom cluster annotation. If the corresponding
    field is empty, which is the default, the first annotation class
    will be the cluster name.

    Parameters
    ----------
    file_path: str
        String denoting the path to the DAVID Functional Annotation
        Clustering (FAC)file to be parsed. The file can be an Excel
        file, a txt file using tab stops as separator or a TSV file. If
        providing the path to an Excel file, make sure that the DAVID
        FAC results are located in the very first sheet.
    mapping_option: str
        String denoting the technique by which to assign genes to
        clusters. The default is "union", which means that all genes
        observed at least once are assigned to the respective cluster.
        However, as an alternative, "intersect" may be passed, which
        means that only genes present in all functional annotation
        classes of the respective cluster are included.

    Returns
    -------
    ...: ...
        ...
    ...: ...
        ...
    ...: ...
        ...
    ...: ...
        ...
    ...: ...
        ...
    """

    # Extract the name of the DAVID FAC file from the path in case it is
    # not located in the present working directory
    # Note that macOS and Linux both use the forward slash as separator
    # in paths, whereas Windows uses the backslash
    # Hence, backslashes in the file path, if present, are converted
    # into forward slashes
    file_path = file_path.replace("\\", "/")
    file_path_list = file_path.split("/")
    DAVID_FAC_file_name =file_path_list[-1]

    # Report the inputs
    if mapping_option == "union":
        print(
            "The DAVID FAC file bearing the name "
            f"\"{DAVID_FAC_file_name}\" is parsed using the "
            f"\"{mapping_option}\" method (default).\n\n"
        )
    elif mapping_option == "intersect":
        print(
            "The DAVID FAC file bearing the name "
            f"\"{DAVID_FAC_file_name}\" is parsed using the "
            f"\"{mapping_option}\" method.\n\n"
        )
    else:
        print(
            f"An unknown mapping option (\"{mapping_option}\") has "
            "been passed, which is why the DAVID FAC file bearing the "
            f"name {DAVID_FAC_file_name} is parsed using the default "
            "method \"union\".\n\n"
        )
        mapping_option = "union"
    
    # On account of the structure of the DAVID FAC file, one preliminary
    # measure needs to be taken prior to reading the file in
    # This preliminary measure consists og determining the amount of
    # colums in the DAVID file
    # This is done in order to be able to define custom column labels
    # Bear in mind that the "with" context manager is preferred in the
    # context of working with files as it automatically takes care of
    # closing files, even in case of errors/exceptions
    with open(file_path, "r") as f:
        # The amount of columns can be determined by splitting the
        # second line of the file using "\t" as separator and
        # subsequently determining the amount of elements in the
        # resulting list
        second_line = f.readlines()[1]
        n_columns = len(second_line.split("\t"))
    
    # Now that the amounf of columns has been determined, a list
    # containing the column names is created
    col_names = [
        f"Column_{i}" for i in range(1, n_columns + 1)
    ]
    
    # Read in the DAVID Functional Annotation Clustering (FAC) file
    # Bear in mind that Excel files can carry either the .xls or the
    # .xlsx extension; as the ".xls" extension also occurs as substring
    # in the other file extension, its presence is checked
    if ".xls" in DAVID_FAC_file_name:
        # to be continued ...
        pass
    else:
        DAVID_FAC_df = pd.read_csv(
            file_path,
            sep="\t",
            header=None,
            names=col_names
        )
    
    # The structure of the DAVID FAC file exhibits periodicity in that
    # multiple annotation clusters occur, which in turn also exhibit
    # periodicity with respect to sequence of their rows
    # In order to be able to leverage this periodicity, the row labels
    # are constructed in a specific manner
    # In detail, the row label for all rows belonging to one specific
    # cluster equals the number of that cluster
    # Empty rows do not have to be taken care of as the keyword argument
    # "skip_blank_lines" defaults to "True"

    # Determine the rows in which a new annotation cluster starts
    # This is accomplished by identifying rows in the very first column
    # containing the string "Annotation"
    cluster_start_boolean_series = DAVID_FAC_df["Column_1"].str.contains(
        pat="Annotation",
        regex=False
    )
    
    # Now, construct the row labels
    row_labels = []
    cluster_num = 0

    for i, entry in enumerate(cluster_start_boolean_series):
        if (entry == True) and (i != 0):
            cluster_num += 1
        row_labels.append(cluster_num)
    
    # Assign the new row labels to the DataFrame
    DAVID_FAC_df.index = row_labels

    # Extract the individual clusters via Boolean indexing
    # This requires determining the total amount of clusters
    n_clusters = DAVID_FAC_df.index.to_list()[-1]
    sub_df_list = [
        DAVID_FAC_df[DAVID_FAC_df.index == i]
        for i in range(n_clusters + 1)
    ]
    
    # Initialising the output
    enrichment_score_per_cluster = []
    label_per_cluster = []
    genes_per_cluster = []
    genes_per_class_per_cluster = []
    
    # Iterate over the individual annotation clusters
    for i, cluster_df in enumerate(sub_df_list):
        # Add a new gene list for the new cluster
        genes_per_cluster.append([])
        genes_per_class_per_cluster.append([])

        # Extract the enrichment score; it is located in the very first
        # line in the second column
        current_enrichment_score_str = cluster_df.iloc[
            0, 1
        ].split(": ")[-1]
        current_enrichment_score = float(current_enrichment_score_str)
        enrichment_score_per_cluster.append(current_enrichment_score)

        # Now, retrieve a label for the current cluster
        # As explained in the doc string, the third column of the
        # cluster header may contain a custom cluster annotation
        # Therefore, it is checked whether the corresponding cell indeed
        # is populated with a string
        # Otherwise, the first annotation class is taken as cluster
        # label; this, however, is done at a later point
        third_column_of_header = cluster_df.iloc[0, 2]
        if not np.isnan(third_column_of_header):
            label_per_cluster.append(third_column_of_header)
        
        # Iterate over the rows of the cluster DataFrame from the third
        # row onwards as it is the first row containing annotation
        # classes
        for _, annot_class_row in cluster_df.iloc[2:].iterrows():
            class_genes = annot_class_row.iloc[5].split(", ")
            print(class_genes)
            break
        break

    return


if __name__ == "__main__":
    some_path = (
        "/Users/jacobanter/Documents/Projects/Project_A_safe/"
        "VACV_screen/Related_works/RNAi_Screening_Reveals_Proteasome-"
        "and_Cullin3-Dependent_Stages_in_Vaccinia_Virus_Infection/mmc3"
        "/new_david_gene_symbols_high.txt"
    )
    parse_DAVID_FAC_file(some_path)