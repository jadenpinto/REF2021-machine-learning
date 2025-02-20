import os
import pandas as pd

from utils.constants import DATASETS_DIR, RAW_DIR, PROCESSED_DIR, SCIMAGO_JOURNAL_RANK
from utils.dataframe import log_dataframe, delete_rows_by_values, log_null_values


def read_sjr_dataset():
    # Path to SCImago journal rank file
    sjr_dataset_path = os.path.join(os.path.dirname(__file__), "..", DATASETS_DIR, RAW_DIR, SCIMAGO_JOURNAL_RANK)

    try:
        sjr_df = pd.read_csv(
            sjr_dataset_path,
            delimiter=';',
            decimal=","
        )

        log_dataframe(sjr_df)

        return sjr_df

    except FileNotFoundError:
        print("File not found. Please make sure the file name and path are correct.")
    except Exception as e:
        print("An error occurred while reading the file:", str(e))

def filter_sjr_columns(sjr_df):
    sjr_columns = ['Rank', 'Title', 'Issn', 'SJR']

    sjr_df = sjr_df[sjr_columns]
    return sjr_df

def log_issn_lengths(sjr_df):
    issn_lengths = sjr_df['Issn'].astype(str).apply(len)
    unique_issn_lengths = sorted(issn_lengths.unique())

    print(f"Possible Lengths of ISSNs in SJR dataframe={unique_issn_lengths}")

def log_records_issn_len_not_eight(sjr_df):
    # Log records where the ISSN length is not eight i.e. ISSN length is 1 or 18

    # Create a temporary column to store the length of each Issn
    sjr_df['Issn_length'] = sjr_df['Issn'].astype(str).apply(len)

    # Filter DataFrame to include only rows where Issn_length is 1
    single_char_issn = sjr_df[sjr_df['Issn_length'] == 1]
    print("Logging records in DF where ISSN length is 1:")
    log_dataframe(single_char_issn)
    """
    40 rows
    Issn = "-" (in all 40 records)
    These journal do not have an Issn => Drop records where Issn = "-"
    """

    # Filter DataFrame to include only rows where Issn_length is 18
    eighteen_char_issn = sjr_df[sjr_df['Issn_length'] == 18]
    print("Logging records in DF where ISSN length is 18:")
    log_dataframe(eighteen_char_issn)
    """
    17894
    Example: Issn = "15424863, 00079235"
    ISSN length is 18, if the field contains 2 ISSN values => Normalise to 1NF
    """

    # Drop temporary column
    sjr_df.drop(
        columns=['Issn_length'], inplace = True
    )

def handle_sjr_issn(sjr_df):
    sjr_df = drop_null_issn(sjr_df)
    sjr_df = normalise_to_1nf(sjr_df)
    sjr_df = add_hyphen_issn(sjr_df)
    return sjr_df

def drop_null_issn(sjr_df):
    sjr_df = delete_rows_by_values(sjr_df, "Issn", ["-"])
    return sjr_df

def normalise_to_1nf(sjr_df):
    sjr_df = sjr_df.assign(
        Issn=sjr_df['Issn'].str.split(', ')
    ).explode('Issn')

    return sjr_df

def add_hyphen_issn(sjr_df):
    sjr_df['Issn'] = sjr_df['Issn'].apply(
        lambda issn_str: issn_str[:4] + '-' + issn_str[4:]
    )

    return sjr_df

def write_processed_sjr_csv(sjr_df):
    processed_sjr_csv_path = os.path.join(os.path.dirname(__file__), "..", DATASETS_DIR, PROCESSED_DIR,
                                           SCIMAGO_JOURNAL_RANK)

    sjr_df.to_csv(processed_sjr_csv_path, index=False)

def process_sjr_impact_factor():
    sjr_df = read_sjr_dataset()

    sjr_df = filter_sjr_columns(sjr_df)
    log_null_values(sjr_df) # Null counts: SJR = 2805, Rest = 0

    log_issn_lengths(sjr_df)                       # Possible Lengths: 1, 8, 18
    log_records_issn_len_not_eight(sjr_df)         # if 1, Issn = "-". If 18, Issn includes 2 comma separated ISSNs

    sjr_df = handle_sjr_issn(sjr_df)
    log_issn_lengths(sjr_df)                       # Possible lengths: 9. Example: 1542-4863

    # Write:
    write_processed_sjr_csv(sjr_df)

process_sjr_impact_factor()
