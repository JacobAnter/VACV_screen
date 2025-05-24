"""
This Python script harbours a function for the performance evaluation
of xCAPT5.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, matthews_corrcoef,\
    average_precision_score, roc_auc_score

def _compute_rearrangement_list(df_1, df_2):
    """
    ...

    Parameters
    ----------
    df_1: Pandas DataFrame, dtype=str, shape=(n, 2)
        ...
    df_2: Pandas DataFrame, dtype=str, shape=(n 2)
        ...

    Returns
    -------
    rearrangement_list: list, dtype=int, shape=(n,)
        ...
    """

    # As a first step, the  UniProt accessions in the first two columns
    # have to be merged to a single string for both DataFrames
    uniprot_acs_df_1_arr = df_1.iloc[:, :2].to_numpy()
    uniprot_acs_df_2_arr = df_2.iloc[:, :2].to_numpy()

    merged_uniprot_acs_df_1 = [
        "".join(row) for row in uniprot_acs_df_1_arr
    ]
    merged_uniprot_acs_df_2 = [
        "".join(row) for row in uniprot_acs_df_2_arr
    ]

    # Now, a dictionary is created mapping the individual elements in
    # the second array to their row number
    second_array_dict = {
        element: i for i, element in enumerate(merged_uniprot_acs_df_2)
    }
    
    # In the final step, this dictionary is used to determine the
    # rearrangement array
    rearrangement_list = [
        second_array_dict[element]
        for element in merged_uniprot_acs_df_1
    ]

    return rearrangement_list

def eval_performance(ground_truth_path, prediction_path, output_path):
    """
    ...

    Parameters
    ----------
    ground_truth_path: str
        A string denoting the path to the ground truth file. The file is
        expected to be a TSV file. Furthermore, the first two columns
        are expected to contain UniProt accessions or protein
        identifiers in general, whereas the third column is expected to
        harbour the ground truth labels, i.e. 0 or 1. The ground truth
        TSV file is also expected no to contain a header.
    prediction_path: str
        A string denoting the path to the file harbouring the predicted
        probabilities. The file is expected to be a TSV file.
        Furthermore, the firs two columns are expected to contain
        UniProt accessions or protein identifiers in general, whereas
        the third column is expected to harbour the predicted
        probabilities.
    output_path: str
        A string denoting the path to the output file, i.e. the file
        the computed performance metrics are stored to.

    Returns
    -------
    ...
    """

    ground_truth_df = pd.read_csv(
        ground_truth_path,
        sep="\t",
        header=None
    )
    predictions_df = pd.read_csv(
        prediction_path,
        sep="\t"
    )
    
    # Unfortunately, the rows of the two TSV files, i.e. the individual
    # PPI pairs, do not have the same ordering
    # Thus, the order must be adjusted
    rearrangement_list = _compute_rearrangement_list(
        ground_truth_df, predictions_df
    )

    predictions_df = predictions_df.iloc[rearrangement_list]

    # Verify that the two DataFrames now indeed have the same ordering
    uniprot_acs_df_1_arr = ground_truth_df.iloc[:, :2].to_numpy()
    uniprot_acs_df_2_arr = predictions_df.iloc[:, :2].to_numpy()

    assert (uniprot_acs_df_1_arr == uniprot_acs_df_2_arr).all(), (
        "The two DataFrames still don't have the same ordering!"
    )

    # Finally, turn to the computation of the performance metrics
    y_pred = predictions_df.iloc[:, 2].to_numpy()
    y_true = ground_truth_df.iloc[:, 2].to_numpy()

    cm = confusion_matrix(y_true, np.round(y_pred))

    acc = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])
    spec = (cm[0,0])/(cm[0,0]+cm[0,1])
    sens = (cm[1,1])/(cm[1,0]+cm[1,1])
    prec = cm[1,1]/(cm[1,1]+cm[0,1])
    rec = cm[1,1]/(cm[1,1]+cm[1,0])
    f1 = 2 * (prec * rec) / (prec + rec)
    mcc = matthews_corrcoef(y_true, np.round(y_pred))

    prc = average_precision_score(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_pred)

        metrics_result_text = (
            f"Accuracy: {acc}\n"
            f"Precision: {prec}\n"
            f"Recall: {rec}\n"
            f"Specificity; {spec}\n"
            f"F1-score: {f1}\n"
            f"MCC: {mcc}\n"
            f"AUROC: {auc}\n"
            f"AUPRC: {prc}"
        )

        print(metrics_result_text)

        # Bear in mind that in the context of working with files, the
        # `with` context mananger is preferred as it automatically takes
        # care of closing files, even in case of errors/exceptions
        with open(output_path, "w") as f:
            f.write(metrics_result_text)

    except ValueError:
        metrics_result_text = (
            f"Accuracy: {acc}\n"
            f"Precision: {prec}\n"
            f"Recall: {rec}\n"
            f"Specificity: {spec}\n"
            f"F1-score: {f1}\n"
            f"MCC: {mcc}\n"
            f"AUROC: nan\n"
            f"AUPRC: {prc}"
        )

        print(metrics_result_text)

        with open(output_path, "w") as f:
            f.write(metrics_result_text)