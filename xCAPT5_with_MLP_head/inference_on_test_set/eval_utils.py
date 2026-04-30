"""
The purpose of this Python script is to define utilities for performance
evaluation.

Specifically, utilities are defined for generating ROC curves, PR curves
as well as histograms visualizing the distribution of predicted
probabiltiies,
"""

import os
import math

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve,\
    auc, matthews_corrcoef, average_precision_score,\
    precision_recall_curve
from matplotlib import pyplot as plt
import matplotlib as mpl


def round_half_up(unrounded_num, decimal_place=0):
    """
    Performs half up rounding on the provided input on the specified
    decimal place, i.e. if the value in the decimal place to the right
    of the specified decimal place is greater than or equal to 5,
    rounding up is performed. Conversely, if the value is less than 5,
    rounding down is performed.

    This is accomplished in three steps. The first step consists of
    shifting the decimal point by the desired number of places. The
    second step involves adding 0.5 and the subsequent usage of
    `math.floor()`. Lastly, the third step shifts the decimal point back
    to the original position.

    Parameters
    ----------
    unrounded_num: int or float
        The input number to perform half up rounding on.
    decimal_place: int, optional
        The decimal place to perform rounding on.

    Returns
    -------
    rounded_num: float
        The input number after half up rounding on the specified decimal
        place.
    """

    # First step, i.e. shifting the decimal point by the desired number
    # of places
    unrounded_num *= 10**decimal_place

    # Second step, i.e. addition of 0.5 and subsequent usage of
    # `math.floor()`
    unrounded_num += 0.5
    shifted_and_rounded_num = math.floor(unrounded_num)

    # Third step, i.e. shifting the decimal point back to the original
    # position
    rounded_num = shifted_and_rounded_num * 10**(-decimal_place)

    return rounded_num


def generate_roc_curve(preds, ground_truth, title_info, image_path):
    """
    Generates an ROC (Receiver Operating Characteristic) curve.

    Parameters
    ----------
    preds: Pandas DataFrame
        A Pandas DataFrame storing the predicted probabilities for PPI
        pairs. It is expected to have a header, i.e. column names. It is
        expected to have a MultiIndex comprising two levels with UniProt
        accessions. One column is expected to  contain the predicted
        probabilities and to bear the name "interaction_probability",
        accordingly.
    ground_truth: Pandas DataFrame
        A Pandas DataFrame storing the ground truth labels for the PPI
        pairs. It is expected to have a header. Again, it is expected to
        have a MultiIndex comprising two levels with UniProt accessions.
        One column is expected to bear the name "label" and to contain
        the ground truth labels.
    title_info: str
        A strig providing information about the predictions the
        performance of which is assessed. It appears in the figure
        title.
    image_path: str
        A string denoting the location to store the image at.

    Returns
    -------
    roc_auc_score: float
        The ROC AUC score.
    curve_df: Pandas DataFrame
        The curve DataFrame allowing to reconstruct the ROC curve. In
        detail, the DataFrame comprises three columns, the first of
        which contains the False Positive Rate, the second of which
        stores the True Positive Rate and the third of which stores the
        threshold.
    """

    # Extract the ground truth labels
    ground_truth_arr = ground_truth["label"].to_numpy()
    
    # Reindex the DataFrame storing the predictions so that its row
    # order matches that of the ground truth DataFrame
    preds = preds.reindex(ground_truth.index)
    # Extract the predicted probabilities
    predicted_probs = preds["interaction_probability"].to_numpy()
    
    # Use the ground truth labels in conjunction with the predicted
    # probabilities so as to compute the FPR/TPR values as well as the
    # AUC score
    fpr, tpr, thresholds = roc_curve(ground_truth_arr, predicted_probs)
    roc_auc_score = auc(fpr, tpr)
    curve_df = pd.DataFrame(data={
        "FPR": fpr,
        "TPR": tpr,
        "Threshold": thresholds
    })

    # Now, turn to the generation of the ROC curve
    fig, ax = plt.subplots(1, 1)

    ax.set_title(
        f"ROC Curve {title_info}\n"
        f"ROC AUC Score: {round_half_up(roc_auc_score, 4):.4f}"
    )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    # Plot the dashed diagonal line representing a random chance model
    ax.plot([0, 1], [0, 1], lw=2, linestyle="--")

    # Plot the ROC curve
    ax.plot(fpr, tpr, lw=2)

    fig.savefig(
        image_path,
        dpi=300
    )

    return roc_auc_score, curve_df


def superimpose_roc_curves(
        preds_list, ground_truth, descriptor_list, image_dir,
        fname_info, best_in_bold=False
):
    """
    Superimposes ROC curves, i.e. generates one plot depicting multiple
    ROC curves.

    Parameters
    ----------
    preds_list: list, dtype=Pandas DataFrame, shape=(n,)
        A list storing Pandas DataFrames, which in turn store predicted
        probabilities for PPI pairs. The DataFrames are expected to have
        a header, i.e. column names. They are also expected to have a
        MultiIndex comprising two levels with UniProt accessions. One
        column is expected to  contain the predicted probabilities and
        to bear the name "interaction_probability", accordingly.
    ground_truth: Pandas DataFrame
        A Pandas DataFrame storing the ground truth labels for the PPI
        pairs. It is expected to have a header. Again, it is expected to
        have a MultiIndex comprising two levels with UniProt accessions.
        One column is expected to bear the name "label" and to contain
        the ground truth labels.
    descriptor_list: list, dtype=str, shape=(n,)
        A list comprising short descriptions of the ROC curves; their
        order has to match the order of DataFrames in `preds_list. These
        descriptions appear in the figure legend.
    image_dir: str
        A string denoting the path to the directory to store the image
        in.
    fname_info: str
        A string providing information about the ROC curves on the plot.
        It appears in the file name.
    best_in_bold: bool, optional, default=False
        A Boolean indicating whether the legend entry associated with
        the best AUC score is displayed in bold. By default, it is not.

    Returns
    -------
    None
    """

    fig, ax = plt.subplots(1, 1)

    ax.set_xlabel("False Positive Rate", fontsize=20)
    ax.set_ylabel("True Positive Rate", fontsize=20)

    ax.tick_params(labelsize=17)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    # Plot the dashed diagonal line representing a random chance model
    ax.plot([0, 1], [0, 1], lw=2, linestyle="--")

    # Load a colormap for assigning colors to the individual curves in
    # the plot
    colors = mpl.colormaps["tab10"].colors

    # Extract the ground truth labels
    ground_truth_arr = ground_truth["label"].to_numpy()

    auc_score_list = []

    # Iterate over the predictions DataFrames, extract the ROC data and
    # plot the curve
    for preds_df, description, color in zip(
        preds_list, descriptor_list, colors
    ):
        # Reindex the DataFrame storing the predictions so that its row
        # order matches that of the ground truth DataFrame
        preds_df = preds_df.reindex(ground_truth.index)
        # Extract the predicted probabilities
        predicted_probs = preds_df["interaction_probability"].to_numpy()

        # Use the ground truth labels in conjunction with the predicted
        # probabilities so as to compute the FPR/TPR values as well as
        # the AUC score
        fpr, tpr, _ = roc_curve(
            ground_truth_arr, predicted_probs
        )
        roc_auc_score = auc(fpr, tpr)
        auc_score_list.append(roc_auc_score)
        roc_auc_score = round_half_up(roc_auc_score, 4)

        # Plot the ROC curve
        ax.plot(
            fpr,
            tpr,
            color=color,
            lw=2,
            label=f"{description} (AUC={roc_auc_score:.4f})"
        )
    
    # Add a legend to the figure displaying the AUC scores of the
    # individual ROC curves
    leg = ax.legend(loc="best", fontsize=13)

    if best_in_bold:
        # Determine the indices of the best AUC score
        best_auc_idx = [
            i for i, auc in enumerate(auc_score_list)
            if auc == max(auc_score_list)
        ]

        for i, text in enumerate(leg.get_texts()):
            if i in best_auc_idx:
                text.set_fontweight("bold")
    
    fig.tight_layout()

    # The figure is saved twice, once as a PNG and once as an SVG (for
    # the publication)
    fig.savefig(
        os.path.join(
            image_dir,
            f"ROC_curves_{fname_info}.png"
        ),
        dpi=300
    )

    fig.savefig(
        os.path.join(
            image_dir,
            f"ROC_curves_{fname_info}.svg"
        )
    )


def generate_pr_curve(preds, ground_truth, title_info, image_path):
    """
    Generates a PR (Precision-Recall) curve.

    Parameters
    ----------
    preds: Pandas DataFrame
        A Pandas DataFrame storing the predicted probabilities for PPI
        pairs. It is expected to have a header, i.e. column names. It is
        expected to have a MultiIndex comprising two levels with UniProt
        accessions. One column is expected to  contain the predicted
        probabilities and to bear the name "interaction_probability",
        accordingly.
    ground_truth: Pandas DataFrame
        A Pandas DataFrame storing the ground truth labels for the PPI
        pairs. It is expected to have a header. Again, is is expected to
        have a MultiIndex comprising two levels with UniProt accessions.
        One column is expected to bear the name "label" and to contain
        the ground truth labels.
    title_info: str
        A string providing information about the predictions the
        performance of which is assessed. It appears in the figure
        title.
    image_path: str
        A string denoting the location to store the image at.

    Returns
    -------
    pr_auc_score: float
        The PR AUC score.
    curve_df: Pandas DataFrame
        The curve DataFrame allowing to reconstruct the PR curve. In
        detail, the DataFrame comprises three columns, the first of
        which contains the recall, the second of which stores the
        precision and the third of which stores the threshols.
    """

    # Extract the ground truth labels
    ground_truth_arr = ground_truth["label"].to_numpy()

    # Reindex the DataFrame storing the predictions so that its row
    # order matches that of the ground truth DataFrame
    preds = preds.reindex(ground_truth.index)
    # Extract the predicted probabilities
    predicted_probs = preds["interaction_probability"].to_numpy()

    # Use the ground truth labels in conjunction with the predicted
    # probabilities so as to compute the recall and precision values as
    # well as the AUC score
    precision, recall, thresholds = precision_recall_curve(
        ground_truth_arr, predicted_probs
    )
    thresholds = np.append(thresholds, np.inf)
    pr_auc_score = auc(recall, precision)
    curve_df = pd.DataFrame(data={
        "Recall": recall,
        "Precision": precision,
        "Threshold": thresholds
    })

    # Now, turn to the generation of the PR curve
    fig, ax = plt.subplots(1, 1)
    ax.set_title(
        f"PR Curve {title_info}\n"
        f"PR AUC score: {round_half_up(pr_auc_score, 4):.4f}"
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    # Plot the PR curve
    ax.plot(recall, precision, lw=2)

    fig.savefig(
        image_path,
        dpi=300
    )

    return pr_auc_score, curve_df


def superimpose_pr_curves(
        preds_list, ground_truth, descriptor_list, image_dir,
        fname_info, best_in_bold=False
):
    """
    Superimposes PR curves, i.e. generates one plot depicting multiple
    PR curves.

    Parameters
    ----------
    preds_list: list, dtype=Pandas DataFrame, shape=(n,)
        A list storing Pandas DataFrames, which in turn store predicted
        probabilities for PPI pairs. The DataFrames are expected to have
        a header, i.e. column names. They are also expected to have a
        MultiIndex comprising two levels with UniProt accessions. One
        column is expected to  contain the predicted probabilities and
        to bear the name "interaction_probability", accordingly.
    ground_truth: Pandas DataFrame
        A Pandas DataFrame storing the ground truth labels for the PPI
        pairs. It is expected to have a header. Again, it is expected to
        have a MultiIndex comprising two levels with UniProt accessions.
        One column is expected to bear the name "label" and to contain
        the ground truth labels.
    descriptor_list: list, dtype=str, shape=(n,)
        A list comprising short descriptions of the ROC curves; their
        order has to match the order of DataFrames in `preds_list. These
        descriptions appear in the figure legend.
    image_dir: str
        A string denoting the path to the directory to store the image
        in.
    fname_info: str
        A string providing information about the ROC curves on the plot.
        It appears in the file name.
    best_in_bold: bool, optional, default=False
        A Boolean indicating whether the legend entry associated with
        the best AUC score is displayed in bold. By default, it is not.

    Returns
    -------
    None
    """

    fig, ax = plt.subplots(1, 1)

    ax.set_xlabel("Recall", fontsize=20)
    ax.set_ylabel("Precision", fontsize=20)

    ax.tick_params(labelsize=17)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    # Load a colormap for assigning colors to the individual curves in
    # the plot
    colors = mpl.colormaps["tab10"].colors

    # Extract the ground truth labels
    ground_truth_arr = ground_truth["label"].to_numpy()

    pr_auc_score_list = []

    # Iterate over the predictions DataFrames, extract the PR curve data
    # and plot the curve
    for preds_df, description, color in zip(
        preds_list, descriptor_list, colors
    ):
        # Reindex the DataFrame storing the predictions so that its row
        # order matches that of the ground truth DataFrame
        preds_df = preds_df.reindex(ground_truth.index)
        # Extract the predicted probabilities
        predicted_probs = preds_df["interaction_probability"].to_numpy()

        # Use the ground truth labels in conjunction with the predicted
        # probabilities as as to compute the recall and precision values
        # as well as the AUC score
        precision, recall, _ = precision_recall_curve(
            ground_truth_arr, predicted_probs
        )
        pr_auc_score = auc(recall, precision)
        pr_auc_score_list.append(pr_auc_score)
        pr_auc_score = round_half_up(pr_auc_score, 4)

        # Plot the ROC curve
        ax.plot(
            recall,
            precision,
            color=color,
            lw=2,
            label=f"{description} (AUC={pr_auc_score:.4f})"
        )
    
    # Add a legend to the figure displaying the AUC scores of the
    # individual PR curves
    leg = ax.legend(loc="best", fontsize=13)

    if best_in_bold:
        # Determine the indices of the best AUC score
        best_auc_idx = [
            i for i, auc in enumerate(pr_auc_score_list)
            if auc == max(pr_auc_score_list)
        ]

        for i, text in enumerate(leg.get_texts()):
            if i in best_auc_idx:
                text.set_fontweight("bold")
    
    fig.tight_layout()

    # The figure is saved twice, once as a PNG and once as an SVG (for
    # the publication)
    fig.savefig(
        os.path.join(
            image_dir,
            f"PR_curves_{fname_info}.png"
        ),
        dpi=300
    )

    fig.savefig(
        os.path.join(
            image_dir,
            f"PR_curves_{fname_info}.svg"
        )
    )


def prob_histogram(preds, title_info, image_path):
    """
    Plots a probability histogram.

    Parameters
    ----------
    preds: Pandas DataFrame
        A Pandas DataFrame storing the predicted probabilities for PPI
        pairs. It is expected to have a header, i.e. column names. One
        column is expected to ...
    title_info: str
        ...
    image_path: str
        ...

    Returns
    -------
    None
    """

    # Extract the predicted predicted probabilities
    predicted_probs = preds["interaction_probability"].to_numpy()

    # Generate the histogram
    fig, ax = plt.subplots(1, 1)

    ax.set_title(f"Probability distribution\n{title_info}")
    ax.set_xlabel("Probability")
    ax.set_ylabel("Count")

    ax.hist(predicted_probs, bins=10)

    fig.savefig(
        image_path,
        dpi=300
    )


def evaluation_k_fold_cross_validation(
        ground_truth_path, splits_path, n_fold, probability_key,
        model_name, output_path
):
    """
    Evaluates a PPI prediction model's performance in the context of
    k-fold cross-validation by computing eight different metrics, which
    are accuracy, precision, recall, F1-score, specificity, Matthews
    correlation coefficient, ROC AUC score and AUPRC. A prerequisite for
    employing this function is that the predicted labels of each test
    set are stored in separate files named in a systematic manner. See
    the elaborations on the `splits_path` parameter for more detailed
    information.

    To be more precise, each of the eight metrics is given as mean value
    across all k test sets along with the corresponding standard
    deviation.

    Naturally, this function is not limited to k-fold cross-validation,
    but can also be employed for different random seeds.

    Parameters
    ----------
    ground_truth_path: str
        A string representing the path to the TSV file containing the
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
        `ground_truth_path`, the column containing the predicted labels
        must bear the name `label`. Apart from that, it is assumed that
        the first two columns store the UniProt IDs of the PPI pairs,
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
        Alternatively, it denotes the number of random seeds.
    probability_key: str
        A string denoting the name of the column in the results TSV
        files storing the predicted probabilities.
    model_name: str
        A string denoting the model to evaluate via k-fold
        cross-validation (or k random seeds).
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
        across the k test sets and the second element of which is the
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
    
    # Now, compute the eight metrics mentioned in the docstring for each
    # and every test set (accuracy, precision, recall, F1-score,
    # specificity, Matthews correlation coefficient, ROC AUC score and
    # AUPRC)
    # To this end, scikit-learn's `confusion_matrix` class is utilised

    # Store the metric values for each test set in corresponding lists
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []
    specificity_list = []
    mcc_list = []
    roc_auc_score_list = []
    auprc_list = []

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

        mcc = matthews_corrcoef(
            current_ground_truths, current_predicted_labels
        )

        auc_score = roc_auc_score(
            current_ground_truths, current_predicted_probs
        )

        auprc_score = average_precision_score(
            current_ground_truths, current_predicted_probs
        )

        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_score_list.append(f1_score)
        specificity_list.append(specificity)
        mcc_list.append(mcc)
        roc_auc_score_list.append(auc_score)
        auprc_list.append(auprc_score)
    
    # Now, compute the mean as well as the standard deviation for all
    # eight metrics
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

    mcc_mean = np.mean(mcc_list)
    mcc_std = np.std(mcc_list)
    mcc_tuple = (mcc_mean, mcc_std)

    roc_auc_score_mean = np.mean(roc_auc_score_list)
    roc_auc_score_std = np.std(roc_auc_score_list)
    roc_auc_tuple = (roc_auc_score_mean, roc_auc_score_std)

    auprc_mean = np.mean(auprc_list)
    auprc_std = np.std(auprc_list)
    auprc_tuple = (auprc_mean, auprc_std)

    # Regarding string padding by means of string methods such as
    # `.ljust()`, it must be noted that if string continuation by e.g.
    # parantheses is used, the entire text preceding a certain point
    # will be considered contiguous
    # Thus, in order to apply string padding to a limited string, it is
    # advisable to separate that string from the surrounding ones by
    # e.g. commas or the plus operator
    metrics_result_text = (
        f"Using {n_fold}-fold cross-validation, the metrics for "
        f"{model_name} are as follows:\n" +
        "Accuracy:".ljust(15) + f"{accuracy_mean} \xB1 {accuracy_std}\n" +
        "Precision:".ljust(15) + f"{precision_mean} \xB1 {precision_std}\n" +
        "Recall:".ljust(15) + f"{recall_mean} \xB1 {recall_std}\n" +
        "F1-score:".ljust(15) + f"{f1_score_mean} \xB1 {f1_score_std}\n" +
        "Specificity:".ljust(15) + f"{specificity_mean} \xB1 {specificity_std}\n" +
        "MCC:".ljust(15) + f"{mcc_mean} \xB1 {mcc_std}\n" +
        "ROC AUC score:".ljust(15) + f"{roc_auc_score_mean} \xB1 {roc_auc_score_std}\n" +
        "AUPRC score".ljust(15) + f"{auprc_mean} \xB1 {auprc_std}"
    )

    print(metrics_result_text)

    # Bear in mind that in the context of working with files, the `with`
    # context manager is preferred as it automatically takes care of
    # closing files, even in case of errors/exceptions
    with open(f"{output_path}", "w") as f:
        f.write(metrics_result_text)
    
    return accuracy_tuple, precision_tuple, recall_tuple,\
        f1_score_tuple, specificity_tuple, mcc_tuple, roc_auc_tuple,\
        auprc_tuple