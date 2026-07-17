

import pandas as pd

def load_data(path):
    """
    Load CSV dataset
    """

    df = pd.read_csv(path)

    return df