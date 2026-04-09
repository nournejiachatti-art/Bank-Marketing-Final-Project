import pandas as pd

def load_bank_data(path='data/bank-full.csv'):
    return pd.read_csv(path, sep=';')