"""
The purpose of this Python script is to provide alternatives to
conventional Z-scoring.

In detail, these alternatives encompass B-score normalization as
described by Brideau et al., median absolute deviation (MAD)-based
robust Z-Scores ...
"""

import numpy as np
import pandas as pd
from statsmodels.robust.scale import mad


def arrange_as_plate(input_df, val_col):
    """
    Arrange one-dimensional data provided in a column according to
    spatial information, i.e. row and column information.

    The column storing row information is expected to bear the name
    `WellRow`, whilst the column storing column information is expected
    to be called `WellColumn`.

    This function is not tied to a specific well plate format (e.g.
    96-well plate or 384-well plate), but automatically adapts to the
    well plate format at hand. Missing wells appear as `np.nan`.

    Parameters
    ----------
    input_df: 2D Pandas DataFrame
        The Pandas DataFrame storing the data to be re-arranged. The
        DataFrame is expected to contain the following columns:
        `WellRow`, `WellColumn` and a column bearing the name specified
        by `val_col`.
    val_col: str
        A string denoting the name of the column storing the data to be
        arranged.

    Returns
    -------
    output_df: 2D Pandas DataFrame
        The input data stored in the `val_col` column arranged according
        to the spatial information.
    row_idx: Pandas Series
        Pandas Series specifying the row location of the data.
    col_idx: Pandas Series
        Pandas Series specifying the column location of the data.
    """
    # As outlined in the docstring, the column storing row information
    # is expected to bear the name `WellRow`
    # Convert the row information to numerical indices starting from 0
    input_df["RowIdx"] = input_df["WellRow"].astype("category").cat.codes

    # Convert `WellColumn` to zero-based index
    input_df["ColIdx"] = input_df["WellColumn"] - 1

    # Get plate dimensions
    n_rows = input_df["RowIdx"].max() + 1
    n_cols = input_df["ColIdx"].max() + 1

    # Initialize an array filled with NaNs (in case not all wells are
    # present)
    output_arr = np.full((n_rows, n_cols), np.nan)

    # Populate the array
    for _, row in input_df.iterrows():
        output_arr[row["RowIdx"], row["ColIdx"]] = row[val_col]
    
    output_df = pd.DataFrame(output_arr)

    return output_df, input_df["RowIdx"], input_df["ColIdx"]


def arrange_back(input_df, row_idx, col_idx):
    """
    Convert data from well plate format back to column format according
    to the row and column information.

    This function returns a NumPy array instead of a Pandas Series. This
    is because of the fact that Pandas performs label (index and column
    names) alignment durign assignment operations, potentially resulting
    in label misalignment. When performing an assignment with a NumPy
    array, however, which lacks any labels, assignment is purely done by
    position.

    Parameters
    ----------
    input_df: 2D Pandas DataFrame
        DataFrame in the format of a well plate. It is expected to
        exclusively contain numerical data.
    row_idx: Pandas Series
        A Pandas Series storing the row indices.
    col_idx: Pandas Series
        A Pandas Series storing the column indices.

    Returns
    -------
    col_array: NumPy array
        NumPy array storing the data in column format.
    """
    n_elements = len(row_idx)
    col_array = np.full((n_elements,), 0.0)

    input_array = input_df.to_numpy()

    # Iterate jointly over the row and column indices in order to
    # populate the column array
    for i, (row_index, col_index) in enumerate(zip(row_idx, col_idx)):
        col_array[i] = input_array[row_index, col_index]

    return col_array


def _median_polish(X, max_iter=200, tol=1e-5):
    """
    Perform Tukey's median polish procedure.

    Parameters
    ----------
    X: 2D NumPy array, shape=(m, n)
        Input data to apply Tukey's median polish procedure on.
    max_iter: int, optional
        Maximum number of iterations. Default value is 200.
    tol: float, optional
        The convergence tolerance. If the absolute values of both the
        row medians and the column medians are below that tolerance, the
        iteration is stopped. Default value is 1e-5.

    Returns
    -------
    overall: float
        The overall effect.
    row_effects: 1D NumPy array
        One-dimensional NumPy array storing the row effects.
    col_effects: 1D NumPy array
        One-dimensional NumPy array storing the column effects.
    residuals: 2D NumPy array
        Two-dimensional NumPy array containing the residuals, which are
        obtained after removing the effects.
    """
    # Create a copy of the input array so as not to modify the user's
    # input
    X = X.astype(float).copy()
    n_rows, n_cols = X.shape
    row_effects = np.zeros(n_rows)
    col_effects = np.zeros(n_cols)
    overall = 0.0

    for _ in range(max_iter):
        # Row step
        row_meds = np.median(X, axis=1)
        row_effects += row_meds
        # In order to ensure broadcasting in the desired manner, the 1D
        # `row_meds` array has to be expanded to a 2D array
        # To be more precise, the medians have to be arranged as a
        # column by introducing a new axis
        X -= row_meds[:, None]

        # Column step
        col_meds = np.median(X, axis=0)
        col_effects += col_meds
        X -= col_meds

        # Update the overall effect
        row_meds_med = np.median(row_meds)
        row_effects -= row_meds_med
        col_meds_med = np.median(col_meds)
        col_effects -= col_meds_med
        overall += (row_meds_med + col_meds_med)

        # Check convergence
        if (
            np.all(np.abs(row_meds) < tol)
            and
            np.all(np.abs(col_meds) < tol)
        ):
            break
    
    residuals = X

    return overall, row_effects, col_effects, residuals


def compute_b_score(input_df, max_iter=200, tol=1e-5):
    """
    Apply B-score normalization as described by Brideau et al.

    Parameters
    ----------
    input_df: 2D Pandas DataFrame
        The input data to apply B-score normalization on.
    max_iter: int, optional
        Maximum number of iterations for Tukey's median polish
        procedure. Default value is 200.
    tol: float, optional
        The convergence tolerance for Tukey's median polish procedure.
        Default value is 1e-5.
    
    Returns
    -------
    b_scored_df: 2D Pandas DataFrame
        The input data after B-score normalization.
    """
    _, _, _, residuals = _median_polish(
        input_df.values,
        max_iter=max_iter,
        tol=tol
    )

    # Scale residuals by median absolute deviation (MAD)
    scale = mad(residuals.flatten(), c=1.4826)
    b_scored = residuals / scale
    b_scored_df = pd.DataFrame(
        b_scored,
        index=input_df.index,
        columns=input_df.columns
    )

    return b_scored_df


def compute_mad_based_Z_scores(input_df):
    """
    Computes median absolute deviation (MAD)-based robust Z-scores.

    Conventional Z-scoring is performed by subtracting the arithmetic
    mean from the raw value and dividing the resulting difference by the
    standard deviation.
    
    In MAD-based Z-scoring, however, the arithmetic mean is replaced
    with sample median and the standard deviation is replaced with the
    median absolute deviation (MAD). Multiplying MAD by 1.4826 makes it
    a Gaussian-consistent estimator of the standard deviation.

    Parameters
    ----------
    input_df: 2D Pandas DataFrame
        The input data to apply MAD-based Z-scoring on.

    Returns
    -------
    Z_scored_df: 2D Pandas DataFrame
        The input data after MAD-based Z-scoring.
    """
    input_df = input_df.astype(float).copy()

    input_vals = input_df.values

    median = np.median(input_vals)
    median_abs_dev = mad(input_vals.flatten(), c=1.4826)

    # Avoid division by zero
    if median_abs_dev == 0:
        median_abs_dev = 1

    Z_scored_df = (input_df - median) / median_abs_dev

    return Z_scored_df


def iqm_normalization(input_df):
    """
    Perform interquartile mean (IQM) normalization.

    As its name already suggests, interquartile mean (IQM) normalization
    involves the computation of the so-called interquartile mean (IQM),
    which is the mean of the middle two quartiles. In a subsequent step,
    the raw data are divided by the IQM, thus obtaining normalized
    values.

    Parameters
    ----------
    input_df: 2D Pandas DataFrame
        Pandas DataFrame to apply IQM normalization on.

    Returns
    -------
    normalized_df: 2D Pandas DataFrame
        The input data after IQM normalization.
    """
    input_array = input_df.to_numpy().flatten()

    q1, q3 = np.quantile(input_array, [0.25, 0.75])

    middle_values = input_array[
        (input_array >= q1)
        &
        (input_array <= q3)
    ]

    iqm = np.mean(middle_values)

    normalized_df = input_df / iqm

    return normalized_df