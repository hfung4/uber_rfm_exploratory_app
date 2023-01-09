"""data"""
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from itertools import repeat

"""visualization"""
import matplotlib.pyplot as plt

"""type checking"""
from typing import Union
from typing import List, Dict

num_type = Union[int, float, complex]

import streamlit as st


def remove_outliers_iqr(
    df: pd.DataFrame,
    vars: List[str],
    lwr: Dict[str, num_type] = None,
    upr: Dict[str, num_type] = None,
    plotting: bool = False,
    figsize: tuple[int, int] = (10, 5),
    prt_bounds: bool = False,
) -> pd.DataFrame:

    """
    remove_outliers_iqr:
    NOTE: only used for continuous variables!

    Given an input dataframe and a list of continuous variables,
    remove outliers that are outside of the interquantile range.
    Optionally, for each variable, generate a plot of the distribution
    before-and-after outlier removal for the variable.

    Args:
        df (pd.DataFrame): input dataframe
        vars (List[str]): a list of continuous columns
        lwr (Dict[str,num_type]): user defined lower bound of values to keep
        upr (Dict[str,num_type]): user defined upper bound of values to keep
        plotting (bool): option flag for plotting distribution of variables
        figsize (tuple[int,int]): figure size
    Returns:
        df_out (pd.DataFrame): output dataframe
    """

    # make a copy of the input dataframe
    data = df.copy()

    # Initialize dictionary with None values
    if lwr is None:
        lwr = dict(zip(vars, repeat(None)))

    if upr is None:
        upr = dict(zip(vars, repeat(None)))

    # init a list of numeric columns
    l_numeric_cols = []

    for var in vars:

        # Check to see if the column is numeric
        if not is_numeric_dtype(data[var]):
            print(
                f"Skipped: {var} is not numeric, so we cannot remove outliers based on iqr criterion."
            )
            continue

        # append var to l_numeric_cols if iteration is not skipped
        l_numeric_cols.append(var)

        # The 25th and 75th percentile (1st and 3rd quartile)
        q1 = data[var].describe()["25%"]
        q3 = data[var].describe()["75%"]

        # IQR
        iqr = q3 - q1

        # Upper and lower bound of values to keep
        if lwr[var] is None:
            lower_bound = q1 - 3 * iqr
        else:
            # use user-specified lower bound
            lower_bound = lwr[var]

        if upr[var] is None:
            upper_bound = q3 + 3 * iqr
        else:
            # use user-specified lower bound
            upper_bound = upr[var]

        # Remove outliers (only keep observations within IQR, or within user-defined bounds)
        if prt_bounds:
            print(f"{var}: keep values between {lower_bound} and {upper_bound}")
        df_out = data.loc[(data[var] < upper_bound) & (data[var] > lower_bound), :]

    # Plot distribution of variable before and after you remove the outliers
    if plotting:
        fig, axes = plt.subplots(len(l_numeric_cols), 2, figsize=figsize)

        for i, v in enumerate(l_numeric_cols):

            # Distribution of var before outliers removal
            axes[i, 0].hist(data[v], bins=30, color="k", alpha=0.5)
            axes[i, 0].set_title(f"Distribution of {v} before outliers removal")

            # Distribution of var after outliers removal
            axes[i, 1].hist(df_out[v], bins=30, color="blue", alpha=0.5)
            axes[i, 1].set_title(f"Distribution of {v} after outliers removal")

        # adjust whitespace between subplots
        plt.subplots_adjust(wspace=0.5, hspace=0.5)

    return df_out


def remove_collinear_features(
    df: pd.DataFrame,
    DV: str,
    threshold: num_type,
    drop_collinear: bool = True,
    print_collinear: bool = False,
    custom_drop: List[str] = [],
) -> pd.DataFrame:
    """
    remove_collinear_features:

    There are a number of methods for removing collinear features, such as using the
    Variance Inflation Factor. Here, We will use a simpler metric, and remove features
    that have a correlation coefficient above a certain threshold with each other.

    This function removes the collinear continuous features based on a threshold we select for the
    correlation coefficients by removing one of the two features that are compared. It also prints
    the correlations that it removes so we can see the effect of adjusting the threshold.
    We will use a threshold of 0.6 which removes one of a pair of features if the correlation
    coefficient between the features exceeds this value.

    Args:
        df (pd.DataFrame): input dataframe
        DV (str): dependent variable
        threshold (numeric): threshold of correlation above which we will drop one of the variable
        drop_collinear(bool): drop one of the collinear variables if True, otherwise return the original df
        print_collinear (bool): print collinear pairs
        custom_drop (bool): custom list of variables to drop from the dataframe in addition to the collinear ones
    Returns:
        df_out (pd.DataFrame): output dataframe
        df_collinear_pairs (pd.DataFrame): a dataframe of collinear pairs and their correlation
    """
    # make a copy of the input dataframe
    data = df.copy()

    # save the dependent variable
    y = data[DV]
    # handle features only
    df_out = data.drop(columns=[DV])

    # Calculate the correlation matrix
    corr_matrix = df_out.corr()
    iters = range(len(corr_matrix.columns) - 1)
    # Init a list of columns to be dropped
    drop_cols = []
    # Init a list of tuples that contains 1) collinear col_1,
    # 2) collinear col_2, and 3) correlation between them
    collinear_pairs = []

    # Iterate through the correlation matrix and compare correlations
    # If I have k features, corr_matrix has shape (kxk)
    for i in iters:  # iterate from 0 to k-1
        for j in range(i):  # iterate from 0 to k-1
            item = corr_matrix.iloc[j : (j + 1), (i + 1) : (i + 2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            # If correlation exceeds the threshold
            if val >= threshold:

                # Print the correlated features and the correlation value
                if print_collinear:
                    print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))

                # Add the tuple of collinear pairs and their correlation to a list
                collinear_pairs.append(
                    (col.values[0], row.values[0], round(val[0][0], 2))
                )
                # Add one of the variable to the "drop_cols" list
                drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    if drop_collinear:
        df_out = df_out.drop(columns=drops)

    # Drop additional columns that are specified by the using
    if custom_drop != []:
        df_out = df_out.drop(columns=custom_drop)

    # Add the score back in to the data
    df_out[DV] = y

    # Convert collinear_pairs to a dataframe
    df_collinear_pairs = pd.DataFrame(
        collinear_pairs, columns=["var_1", "var_2", "correlation"]
    )

    # Drop duplicated rows in terms of var_1 and var_2 (order does not matter)
    # False if row is a duplicated that needs to be removed
    mask = ~pd.DataFrame(
        np.sort(df_collinear_pairs[["var_1", "var_2"]], axis=1)
    ).duplicated()
    df_collinear_pairs = df_collinear_pairs[mask]

    return df_out, df_collinear_pairs
