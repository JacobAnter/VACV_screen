"""
This Python script harbours utilities for the evaluation of the
different PPI prediction models.
"""

import math

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

def _round_half_up(number):
    """
    Rounds the number provided as input to the nearest integer using the
    rounding half up strategy, i.e. the number in the first decimal
    place being greater than or equal to five results in rounding up,
    whereas the number in the first decimal place being less than five
    results in rounding down.

    Implementing the rounding half up strategy from scratch is necessary
    as neither the built-in `round()` function nor `numpy.round()`
    implement this rounding strategy.

    Parameters
    ----------
    number: float
        The number to be rounded to the nearest integer using rounding
        half up strategy.

    Returns
    -------
    rounded_number: integer
        The input number rounded to the nearest integer using rounding
        half up strategy.
    """

    # One way of implementing the rounding half up strategy consists of
    # adding 0.5 to the number, followed by rounding down with
    # `math.floor()`
    rounded_number = math.floor(number + 0.5)

    return rounded_number


def add_labels_based_on_probs(path_tsv_files, pred_col_name, n_fold):
    """
    Determines PPI labels based on predicted probabilities with a label
    of 1 representing a positive PPI and a label of 0 representing a
    negative PPI. In detail, the labels are obtained by subjecting the
    probabilities to rounding via the rounding half up strategies.
    According to this rounding strategy, the respective place having a
    value of greater than or equal to 5 results in rounding up, whereas
    the respective place having a value of less than five results in
    rounding down. The labels are introduced into the respective TSV
    file via a new column bearing the name `label`.

    This function is designed to operate on multiple TSV files at once
    named in a systematic manner and thus lends itself to files
    generated in the context of e.g. k-fold cross-validation.

    Parameters
    ----------
    path_tsv_files: str
        A string representing the path to the TSV files to process in a
        general manner. To be more precise, it is assumed that the names
        of the files in question only differ in one position carrying an
        integer. Thus, the path has to be provided in the following
        format: /dir_1/dir_2/interaction_probs_test_set_{i}.tsv
    pred_col_name: str
        String denoting the name of the column harbouring the predicted
        PPI probabilities.
    n_fold: int
        An integer corresponding to the k in k-fold cross-validation,
        i.e. it denotes the number of chunks the original data set has
        been split into and thus the number of test sets to process.

    Returns
    -------
    None
    """

    # Iterate over the n output files
    for i in range(n_fold):
        current_tsv_path = path_tsv_files.format(i=i)
        current_output_df = pd.read_csv(
            current_tsv_path,
            sep="\t"
        )

        # It is assumed that result files from PPI prediction methods
        # encompass at most four columns, with the `label` column being
        # one of them
        # Hence, the TSV file (or strictly speaking DataFrame) at hand
        # is only processed if it does not encompass four columns
        if len(current_output_df.columns) != 4:
            current_probs = current_output_df[pred_col_name]

            current_labels = [
                _round_half_up(predicted_prob) for predicted_prob
                in current_probs
            ]

            # Finally, introduce the `label` column and overwrite the
            # original TSV file with the modified DataFrame
            current_output_df["label"] = current_labels

            current_output_df.to_csv(
                current_tsv_path,
                sep="\t",
                index=False
            )
        else:
            print(
                f"The file {current_tsv_path} already comprises a "
                "`label` column."
            )


def evaluation_k_fold_cross_val(
        ground_truth_path, splits_path, n_fold, probability_key,
        model_name, output_path
):
    """
    Evaluates a PPI prediction model's performance in the context of
    k-fold cross-validation by computing six different metrics, which
    are accuracy, precision, recall, F1-score, specificity and ROC AUC
    score. A prerequisite for employing this function is that the
    predicted labels of each test set are stored in separate files named
    in a systematic manner. See the elaborations on the `splits_path`
    parameter for more detailed information.

    To be more precise, each of the six metrics is given as mean value
    across all k test sets along with the corresponding standard
    deviation.

    Parameters
    ----------
    ground_truth_path: str
        A string representing the path to the TSV file harbouring the
        ground truth labels. In the TSV file, the column containing the
        ground truth labels must bear the name `label`.
    splits_path: str
        A string denoting the path to the test set TSV files containing
        the predicted labels in a general manner. To be more precise, it
        is assumed that the names of the files in question only differ
        in one position carrying an integer. Thus, the path has to be
        provided in the following format:
        /dir_1/dir_2/interaction_probs_test_set_{i}.tsv
        As with the ground truth TSV file denoted by
        `ground_truth_path`, the column harbouring the predicted labels
        must bear the name `label`. Apart from that, it is assumed that
        the first two columns harbour the UniProt IDs of the PPI pairs,
        and also that the PPI pairs have the same ordering as in the
        ground truth file. What is meant by that is that if a UniProt ID
        of a PPI occurs in the first column of the test set TSV file, it
        also has to occur in the first column of the ground truth TSV
        file. Conversely, a UniProt ID of a certain PPI occurring in the
        second column of the test set TSV file also has to occur in the
        second column of the ground truth TSV file.
    n_fold: int
        An integer corresponding to the k in k-fold cross-validation,
        i.e. it denotes the number of chunks the original data set has
        been split into and thus the number of test sets to process.
    probability_key: str
        A string denoting the name of the column in the results TSV
        files harbouring the predicted probabilities.
    model_name: str
        A string denoting the model to evaluate via k-fold
        cross-validation.
    output_path: str
        A string denoting the path to the output file to store the
        results in.

    Returns
    -------
    accuracy_tuple: tuple, dtype=float
        A tuple the first element of which is the mean accuracy across
        the k test sets and the second element of which is the
        corresponding standard deviation.
    precision_tuple: tuple, dtype=float
        A tuple the first element of which is the mean precision across
        the k test sets and the second element of which is the
        corresponding standard deviation.
    recall_tuple: tuple, dtype=float
        A tuple the first element of which is the mean recall across the
        k test sets and the second element of which is the corresponding
        standard deviation.
    f1_score_tuple: tuple, dtype=float
        A tuple the first element of which is the mean F1-score across
        the k test sets and the second element of which is the
        corresponding standard deviation.
    specificity_tuple: tuple, dtype=float
        A tuple the first element of which is the mean specificity
        across the k test sets and the second element of which is the
        corresponding standard deviation.
    roc_auc_score_tuple: tuple, dtype=float
        A tuple the first element of which is the mean ROC AUC score
        across the k test sets and he second element of which is the
        corresponding standard deviation.
    """

    # Load the ground truth TSV file
    ground_truth_df = pd.read_csv(ground_truth_path, sep="\t")

    # In order to be able to utilise scikit-learn's `confusion_matrix`
    # class, the ground truth labels have to be extracted for each and
    # every test set
    ground_truth_labels_per_test_set = []

    for i in range(n_fold):
        current_test_set_ground_truth_labels = []

        current_test_set_pred_df = pd.read_csv(
            splits_path.format(i=i),
            sep="\t"
        )

        # Iterate over the test set DataFrame at hand in order to
        # extract the PPIs and their ground truth labels
        for _, row in current_test_set_pred_df.iterrows():
            first_uniprot_id = row.iloc[0]
            second_uniprot_id = row.iloc[1]
            
            # Extracting a portion of a DataFrame via Boolean indexing
            # results in another DataFrame, and extracting a column (in
            # this case the `label` column) yields a Pandas Series
            # In order to obtain the actual (and in this case also only)
            # value stored in the Series, it has to be explicitly
            # accessed, for instance by integer-based indexing (`.iloc`)
            current_ground_truth_label = ground_truth_df.loc[
                (ground_truth_df.iloc[:,0] == first_uniprot_id)
                &
                (ground_truth_df.iloc[:,1] == second_uniprot_id)
            ]["label"].iloc[0]

            current_test_set_ground_truth_labels.append(
                current_ground_truth_label
            )
        
        ground_truth_labels_per_test_set.append(
            current_test_set_ground_truth_labels
        )
    
    # Now, compute the six metrics mentioned in the docstring for each
    # and every test set (accuracy, precision, recall, F1-score,
    # specificity and ROC AUC score)
    # To this end, scikit-learn's `confusion_matrix` class is utilised

    # Store the metric values for each test set in corresponding lists
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []
    specificity_list = []
    roc_auc_score_list = []

    # Iterate over the k test sets
    for i, current_ground_truths in enumerate(
        ground_truth_labels_per_test_set
    ):
        # Extract the predicted labels as well as the predicted
        # probabilities of the test set at hand
        current_test_set_pred_df = pd.read_csv(
            splits_path.format(i=i),
            sep="\t"
        )
        current_predicted_labels = current_test_set_pred_df[
            "label"
        ].to_list()
        current_predicted_probs = current_test_set_pred_df[
            probability_key
        ].to_list()

        # The predicted labels along with the ground truth labels are
        # fed into the `confusion_matrix` class
        cm = confusion_matrix(
            current_ground_truths, current_predicted_labels
        )

        # Accuracy is defined as the proportion of correct predictions
        # in all predictions made by the model and is hence computed as
        # follows:
        # (# correct predictions) / (# all predictions)
        # = (TP + TN) / (TP + TN + FP + FN)
        accuracy = (
            (cm[1,1] + cm[0,0]) / (cm[1,1] + cm[0,0] + cm[0,1] + cm[1,0])
        )
        # Precision is defined as the proportion of correct positive
        # predictions in all positive predictions and is thus computed
        # as follows:
        # (# true positives) / (# positive predictions)
        # = TP / (TP + FP)
        precision = cm[1,1] / (cm[1,1] + cm[0,1])
        # Recall, also known as sensitivity, is defined as the
        # proportion of correctly identified positive instances in all
        # positive instances and is thus computed as follows:
        # (# true positives) / (# positive instances in data set)
        # = TP / (TP + FN)
        recall = cm[1,1] / (cm[1,1] + cm[1,0])
        # F1-score is a metric incorporating both precision and recall;
        # to be more precise, the F1-score is the harmonic mean of
        # precision and recall
        # It is defined as follows:
        # 2*TP / (2*TP + FP + FN)
        f1_score = 2*cm[1,1] / (2*cm[1,1] + cm[0,1] + cm[1,0])
        # Specificity, which can be considered the opposite of recall,
        # is defined as the proportion of correctly identified negative
        # instances in all negative instances and is hence computed as
        # follows:
        # (# true negatives) / (# negative instances in data set)
        # = TN / (TN + FP)
        specificity = cm[0,0] / (cm[0,0] + cm[0,1])

        auc_score = roc_auc_score(
            current_ground_truths, current_predicted_probs
        )

        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_score_list.append(f1_score)
        specificity_list.append(specificity)
        roc_auc_score_list.append(auc_score)
    
    # Now, compute the mean as well as the standard deviation for all
    # six metrics
    accuracy_mean = np.mean(accuracy_list)
    accuracy_std = np.std(accuracy_list)
    accuracy_tuple = (accuracy_mean, accuracy_std)

    precision_mean = np.mean(precision_list)
    precision_std = np.std(precision_list)
    precision_tuple = (precision_mean, precision_std)

    recall_mean = np.mean(recall_list)
    recall_std = np.std(recall_list)
    recall_tuple = (recall_mean, recall_std)

    f1_score_mean = np.mean(f1_score_list)
    f1_score_std = np.std(f1_score_list)
    f1_score_tuple = (f1_score_mean, f1_score_std)

    specificity_mean = np.mean(specificity_list)
    specificity_std = np.std(specificity_list)
    specificity_tuple = (specificity_mean, specificity_std)

    roc_auc_score_mean = np.mean(roc_auc_score_list)
    roc_auc_score_std = np.std(roc_auc_score_list)
    roc_auc_tuple = (roc_auc_score_mean, roc_auc_score_std)

    # Regarding string padding by means of string methods such as
    # `.ljust()`, it must be noted that if string continuation by e.g.
    # parantheses is used, the entire text preceding a certain point
    # will be considered contiguous
    # Thus, in order to apply string padding to a limited string, it is
    # advisable to separate that string from the surrounding ones by
    # e.g. commas or the plus operator
    metrics_result_test = (
        f"Using {n_fold}-fold cross-validation, the metrics for "
        f"{model_name} are as follows:\n" +
        "Accuracy:".ljust(15) + f"{accuracy_mean} \xB1 {accuracy_std}\n" +
        "Precision:".ljust(15) + f"{precision_mean} \xB1 {precision_std}\n" +
        "Recall:".ljust(15) + f"{recall_mean} \xB1 {recall_std}\n" +
        "F1-score:".ljust(15) + f"{f1_score_mean} \xB1 {f1_score_std}\n" +
        "Specificity:".ljust(15) + f"{specificity_mean} \xB1 {specificity_std}\n" +
        "ROC AUC score:".ljust(15) + f"{roc_auc_score_mean} \xB1 {roc_auc_score_std}"
    )

    print(metrics_result_test)

    # Bear in mind that in the context of working with files, the `with`
    # context manager is preferred as it automatically takes care of
    # closing files, even in case of errors/exceptions
    with open(f"{output_path}", "w") as f:
        f.write(metrics_result_test)

    return accuracy_tuple, precision_tuple, recall_tuple,\
        f1_score_tuple, specificity_tuple, roc_auc_tuple