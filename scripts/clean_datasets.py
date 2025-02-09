import os
import pandas as pd

from utils.constants import DATASETS_DIR, REF2021_EXPORTED_DIR, CS_UOA_RESULTS, OUTPUTS_METADATA, REF2021_CLEANED_DIR, \
    CS_OUTPUTS_METADATA
from utils.dataframe import log_dataframe

def clean_results_dataset():
    # Define the Excel file name
    results_dataset_path = os.path.join(os.path.dirname(__file__), "..", DATASETS_DIR, REF2021_EXPORTED_DIR, CS_UOA_RESULTS)

    try:
        # Load the Excel file, skipping the first 6 lines -> The 7th line will be used as the header
        df = pd.read_excel(results_dataset_path, skiprows=6)
        log_dataframe(df)

    except FileNotFoundError:
        print("File not found. Please make sure the file name and path are correct.")
    except Exception as e:
        print("An error occurred while reading the file:", str(e))

def clean_outputs_dataset():
    # Define the Excel file name
    outputs_dataset_path = os.path.join(os.path.dirname(__file__), "..", DATASETS_DIR, REF2021_EXPORTED_DIR, OUTPUTS_METADATA)

    try:
        # Load the Excel file, skipping the first 4 lines -> 4th line will be used as the header
        df = pd.read_excel(outputs_dataset_path, skiprows=4)

        cs_outputs = df[df['Unit of assessment number'] == 11]
        cs_outputs = cs_outputs.drop(
            columns=['Unit of assessment number', 'Unit of assessment name']
        )

        log_dataframe(cs_outputs)

        return cs_outputs

    except FileNotFoundError:
        print("File not found. Please make sure the file name and path are correct.")
    except Exception as e:
        print("An error occurred while reading the file:", str(e))

def write_cleaned_cs_outputs(cs_outputs):
    cleaned_cs_outputs_path = os.path.join(os.path.dirname(__file__), "..", DATASETS_DIR, REF2021_CLEANED_DIR,
                                           CS_OUTPUTS_METADATA)

    cs_outputs.to_csv(cleaned_cs_outputs_path, index=False)


# for testing, make it return a df [and write to datasets], possible give argument

clean_results_dataset()
cs_outputs = clean_outputs_dataset()
write_cleaned_cs_outputs(cs_outputs)

# preprocess(), filter, remove UoA id, name, write.
