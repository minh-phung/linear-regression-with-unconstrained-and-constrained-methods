import numpy as np
from sympy.functions.elementary import miscellaneous


def within_1_std_of_min (data):

    min_row = data.loc[data["test_error", "mean"].idxmin()]
    upper_bound = min_row["test_error", "mean"] + min_row["test_error", "std"]
    within_chosen = data.loc[data["test_error", "mean"] < upper_bound]
    sorted_chosen = within_chosen.sort_index()

    return sorted_chosen.iloc[0]