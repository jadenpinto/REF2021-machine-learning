import os
import pandas as pd

from utils.constants import DATASETS_DIR, RAW_DIR, CS_RESULTS
from utils.dataframe import log_dataframe

def get_results_dataframe():
    results_dataset_path = os.path.join(os.path.dirname(__file__), "..", DATASETS_DIR, RAW_DIR, CS_RESULTS)

    try:
        # Load the Excel file, skipping the first 6 lines -> The 7th line will be used as the header
        df = pd.read_excel(results_dataset_path, skiprows=6)
        return df

    except FileNotFoundError:
        print("File not found. Please make sure the file name and path are correct.")
    except Exception as e:
        print("An error occurred while reading the file:", str(e))

results_df = get_results_dataframe()
log_dataframe(results_df)
