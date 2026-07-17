"""
utils.py
==========================================================
Simple shared data-loading helpers so any notebook can load
the project's data with one import, e.g.:

    from utils import load_raw_data, load_cleaned_data
    df = load_raw_data()
==========================================================
"""
import pandas as pd


def load_raw_data(path="../data/comments.csv"):
    """Load the raw comments.csv file."""
    return pd.read_csv(path)


def load_cleaned_data(path="../data/cleaned_comments.csv"):
    """Load the cleaned/preprocessed dataset produced by 02_preprocessing.ipynb."""
    return pd.read_csv(path)
