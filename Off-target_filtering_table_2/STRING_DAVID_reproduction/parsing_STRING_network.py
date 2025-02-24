"""
The purpose of this Python script is to translate the functionality
provided by the "parseStringNetwork.m" file from Matlab into Python. The
aforementioned file is part of the source code made publicly available
in the context of the publication "RNAi Screening Reveals Proteasome-
and Cullin3-Dependent Stages in Vaccinia Virus Infection".
"""

import numpy as np
import pandas as pd

def parse_STRING_network(string_network_file_path):
    """
    ...

    Parameters
    ----------
    string_network_file_path: str
        A string representing the path to a STRING network file, which
        is a tab-separated text file encompassing 15 columns.

    Returns
    -------
    string_network_arr: ndarray, shape=(n,n), dtype=float
        ...
    string_identifiers_arr: ndarray, shape=(n,), dtype=int
        A NumPy array containing the unique ...
    string_gene_symbols_arr: ndarray, shape=(n,), dtype=...
        ...
    ENSP_arr: ndarray, shape=(n,), dtype=...
        ...
    """

    string_output_df = pd.read_csv(
        string_network_file_path,
        sep="\t"
    )
    
    # Ensure that the file the path of which has been provided as input
    # comprises 15 columns; otherwise, an error is thrown
    assert string_output_df.shape[1] == 15, (
        "The file the path of which has been provided as input does "
        "not have the expected format. Please make sure that the file "
        "is a results file from the STRING database."
    )

    # Remove potentially occurring empty lines from the DataFrame
    string_output_df.dropna(
        how="all",
        inplace=True
    )

    # Determine unique gene IDs along with the corresponding unique
    # gene symbols and ENSP identifiers; rather than applying the
    # `unique()` function on the gene symbols and ENSP identifiers, the
    # indices of the input array that give the unique values are used to
    # extract the unique gene symbols and ENSP identifiers; this is due
    # to the fact that `np.unique()` returns the unique elements with
    # sorting, thereby potentially disrupting the joint ordering of the
    # three different identifiers
    # By default, in case an at least two-dimensional array is passed as
    # input and no axis is specified, the input array will be flattened
    # in a row-wise manner; as flattening in a column-wise manner is
    # desired, i.e. simply stacking the columns on top of each other,
    # `pd.melt()` is used prior to feeding the gene IDs to `np.unique()`
    all_string_identifiers = pd.melt(
        string_output_df.iloc[:, 2:4]
    )["value"].to_numpy()
    string_identifiers_arr, indices = np.unique(
        all_string_identifiers,
        return_index=True
    )

    all_gene_symbols = pd.melt(
        string_output_df.iloc[:, :2]
    )["value"].to_numpy()
    string_gene_symbols_arr = all_gene_symbols[indices]

    all_ENSP_identifiers = pd.melt(
        string_output_df.iloc[:, 4:6]
    )["value"].to_numpy()
    ENSP_arr = all_ENSP_identifiers[indices]

    n_unique_prots = len(string_identifiers_arr)
    string_network_arr = np.zeros(shape=(n_unique_prots,n_unique_prots))
    combined_scores = string_output_df.iloc[:, 14].to_numpy()

    for i, current_score in enumerate(combined_scores):
        # Extract the proteins of the current interaction pair and
        # determine their indices in the sorted array of unique gene IDs
        # in order to populate the network matrix at the correct
        # positions
        int_partner_1 = string_output_df.iat[i, 2]
        int_partner_2 = string_output_df.iat[i, 3]
        
        # Bear in mind that `np.nonzero()` returns a tuple of arrays
        # with the individual arrays harbouring the indices of elements
        # that are non-zero
        # Thus, the returned object must be indexed twice
        idx_1 = np.nonzero(string_identifiers_arr == int_partner_1)[0][0]
        idx_2 = np.nonzero(string_identifiers_arr == int_partner_2)[0][0]
        
        string_network_arr[idx_1, idx_2] = current_score
        string_network_arr[idx_2, idx_1] = current_score

    return (
        string_network_arr, string_identifiers_arr,
        string_gene_symbols_arr, ENSP_arr
    )