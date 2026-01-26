import numpy as np
from sympy.functions.elementary import miscellaneous
import pandas as pd

def within_1_std_of_min (data):

    min_row = data.loc[data["test_error", "mean"].idxmin()]
    upper_bound = min_row["test_error", "mean"] + min_row["test_error", "std"]
    within_chosen = data.loc[data["test_error", "mean"] < upper_bound]
    sorted_chosen = within_chosen.sort_index()

    out = pd.DataFrame(sorted_chosen.iloc[0])

    return out.T

