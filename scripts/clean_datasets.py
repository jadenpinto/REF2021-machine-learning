import os
import pandas as pd

from utils.constants import DATASETS_DIR, REF2021_EXPORTED_DIR, CS_UOA_RESULTS

def clean_results_dataset():
    # Define the Excel file name
    results_dataset_path = os.path.join(os.path.dirname(__file__), "..", DATASETS_DIR, REF2021_EXPORTED_DIR, CS_UOA_RESULTS)

    try:
        # Load the Excel file, skipping the first 6 lines
        # The 7th line will be used as the header
        df = pd.read_excel(results_dataset_path, skiprows=6)

        # Display the first few rows to verify the result
        print("Data loaded successfully:")
        print(df.head())

        # Display the shape of the DataFrame
        print("\nNumber of rows and columns:", df.shape)

        # Columns
        print(df.columns)

    except FileNotFoundError:
        print("File not found. Please make sure the file name and path are correct.")
    except Exception as e:
        print("An error occurred while reading the file:", str(e))

clean_results_dataset()
