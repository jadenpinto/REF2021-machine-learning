import os
import pandas as pd

from utils.constants import DATASETS_DIR, RAW_DIR, OUTPUTS_METADATA, PROCESSED_DIR, CS_OUTPUTS_METADATA
from utils.dataframe import log_dataframe


def read_ref_outputs():
    outputs_dataset_path = os.path.join(os.path.dirname(__file__), "..", DATASETS_DIR, RAW_DIR, OUTPUTS_METADATA)

    try:
        # Load the Excel file, skipping the first 4 lines -> 4th line will be used as the header
        ref_outputs_df = pd.read_excel(outputs_dataset_path, skiprows=4)
        return ref_outputs_df

    except FileNotFoundError:
        print("File not found. Please make sure the file name and path are correct.")
    except Exception as e:
        print("An error occurred while reading the file:", str(e))

def filter_cs_outputs(ref_outputs_df):
    cs_outputs = ref_outputs_df[ref_outputs_df['Unit of assessment number'] == 11]
    cs_outputs = cs_outputs.drop(columns=['Unit of assessment number', 'Unit of assessment name'])

    return cs_outputs

def write_cs_outputs(cs_outputs):
    cs_outputs_path = os.path.join(os.path.dirname(__file__), "..", DATASETS_DIR, PROCESSED_DIR,
                                           CS_OUTPUTS_METADATA)

    cs_outputs.to_csv(cs_outputs_path, index=False)

def read_cs_outputs():
    cs_outputs_path = os.path.join(os.path.dirname(__file__), "..", DATASETS_DIR, PROCESSED_DIR,
                                   CS_OUTPUTS_METADATA)
    cs_outputs_df = pd.read_csv(cs_outputs_path)

    return cs_outputs_df

def check_inapplicable_citations(cs_outputs_df):
    # Check possible values of the 'Citations applicable' column
    citation_applicable_vals = cs_outputs_df['Citations applicable'].unique()
    print(f"List of possible values for citation applicable field: {citation_applicable_vals}")
    # ['Yes'] => Citations are applicable for all CS outputs

def count_null_citations(cs_outputs_df):
    citations_nan_count = cs_outputs_df['Citation count'].isna().sum() # 906
    print("Number of records where citation count is NaN/None:", citations_nan_count)

def count_non_journal_article_citations(cs_outputs_df):
    # Count the number of CS outputs that are not of type 'Journal Article' and have a valid citation count
    non_journal_article_cited_df = cs_outputs_df[(cs_outputs_df['Citation count'].notna()) & (cs_outputs_df['Output type'] != 'D')]
    non_journal_article_cited_count = non_journal_article_cited_df.shape[0] # 1138
    print("Number of records where Citation count is non none and Output type is not journal-article:", non_journal_article_cited_count)

    print(non_journal_article_cited_df.head().to_string())
    # Seems like most are of type E (conference outputs)
    # For conference impact factors for example, you can search for them in the CSV using ISSN & get citation counts


def process_cs_outputs():
    ref_outputs_df = read_ref_outputs()
    cs_outputs = filter_cs_outputs(ref_outputs_df)
    log_dataframe(cs_outputs)
    write_cs_outputs(cs_outputs)

process_cs_outputs()

cs_outputs_df = read_cs_outputs()
check_inapplicable_citations(cs_outputs_df)
count_null_citations(cs_outputs_df)
count_non_journal_article_citations(cs_outputs_df)
