
import pandas as pd
import numpy as np

from itertools import product

# Helper function --------------------------------------------------------------
def imputation(df):
    """
    Replace missings values with the values from the same moment yesterday

    Args:
        df: Pandas DataFrame
            A pandas df filled with numeric values.

    Returns: A pandas DataFrame.
    """
    for row, col in product(range(df.shape[0]), range(df.shape[1])):
        if np.isnan(df.iloc[row, col]):
            df.iloc[row, col] = df.iloc[row - (60*24), col]
    return df


# Main function ----------------------------------------------------------------
def make_dataset(input_path, sep = ";"):
    """
    Load and prepare dataset for the prediction framework.

    Args:
        input_path: str
            The path where to get the CSV file.
        sep: str
            The separator in the pandas read_csv function.

    Returns: A pandas DataFrame.
    """
    df = pd.read_csv(input_path, sep = ";", header=0, low_memory=False,
                    infer_datetime_format=True, parse_dates={'datetime':[0,1]},
                    index_col=['datetime'])
    df.replace("?", np.nan, inplace = True)
    df = df.astype('float32')
    # Fill missing values
    df = imputation(df)
    # Reshape to daily
    daily_df = df.resample("D").sum()
    return daily_df

if __name__ == "__main__":
    df = make_dataset(input_path = "data/raw/household_power_consumption.txt")
    import pdb; pdb.set_trace()
