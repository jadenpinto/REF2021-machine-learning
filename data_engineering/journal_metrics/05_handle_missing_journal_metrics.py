import os
import pandas as pd

from utils.constants import DATASETS_DIR, PROCESSED_DIR, SNIP, CS_JOURNAL_METRICS, REFINED_DIR, SJR
from utils.dataframe import split_df_on_null_field

def main():
    """
    ETL pipeline to fill in missing journal metrics
    """
    cs_journal_metrics_df = load_cs_journal_metrics_df()
    log_null_cs_journal_metadata(cs_journal_metrics_df)
    """
    Total number of missing values: [Before handling missing values]
    ISSN            0
    Scopus_ID      96
    SNIP          129
    SJR           143
    Cite_Score    143
    """

    journal_metrics_df_handled_missing_fields = handle_missing_journal_metrics(cs_journal_metrics_df)
    log_null_cs_journal_metadata(journal_metrics_df_handled_missing_fields)
    """
    Total number of missing values: [After handling missing values]
    ISSN            0
    Scopus_ID      96
    SNIP           95
    SJR           141
    Cite_Score    143
    """

    # Explicitly map each column to its correct data-type
    ensure_uniform_data_types(journal_metrics_df_handled_missing_fields)
    # Override existing CS_JOURNAL_METRICS file
    write_journal_metrics_handled_missing_fields_df(journal_metrics_df_handled_missing_fields)

    # Algorithm: Given df -> split into 2, null not null, handle null, rename cols (if needed to match), concat and write


def load_cs_journal_metrics_df():
    """
    Load the CS CS_Journal_Metrics parquet file
    :return: Dataframe representing the CS_Journal_Metrics parquet file
    """
    cs_journal_metrics_df_path = os.path.join(os.path.dirname(__file__), "..", "..", DATASETS_DIR, REFINED_DIR,
                                              CS_JOURNAL_METRICS)
    cs_journal_metrics_df = pd.read_parquet(cs_journal_metrics_df_path, engine='fastparquet')
    return cs_journal_metrics_df

def load_processed_snip_df():
    """
    Load the SNIP parquet file
    :return: Dataframe representing the SNIP parquet file
    """
    snip_df_path = os.path.join(os.path.dirname(__file__), "..", "..", DATASETS_DIR, PROCESSED_DIR,
                                              SNIP)
    snip_df = pd.read_parquet(snip_df_path, engine='fastparquet')
    return snip_df

def load_sjr_df():
    """
    Load the SJR parquet file
    :return: Pandas dataframe of the SJR-ISSN values
    """
    processed_sjr_csv_path = os.path.join(os.path.dirname(__file__), "..", "..", DATASETS_DIR, PROCESSED_DIR,
                                          SJR)
    sjr_df = pd.read_parquet(processed_sjr_csv_path, engine='fastparquet')
    return sjr_df

def log_null_cs_journal_metadata(loaded_cs_journal_metrics_df):
    """
    Log the number of missing values in the CS journal metrics DF
    :param loaded_cs_journal_metrics_df: Pandas dataframe containing journal metrics
    """
    print("CS journal metadata fields with null counts:")
    print(loaded_cs_journal_metrics_df.isna().sum())

    null_scopus_id_cs_journal_metadata_df = loaded_cs_journal_metrics_df[
        loaded_cs_journal_metrics_df['Scopus_ID'].isna()
    ]
    print("Examples of records where Scopus ID is null:")
    print(null_scopus_id_cs_journal_metadata_df.head().to_string())
    # For these records, making an API call to the endpoint with their ISSNs resulted in a 404 RESOURCE_NOT_FOUND

    cs_journal_metadata_df_with_scopus_id_null_snip = loaded_cs_journal_metrics_df[
        ~loaded_cs_journal_metrics_df['Scopus_ID'].isna() & loaded_cs_journal_metrics_df['SNIP'].isna()
    ]
    print("Example of records where Scopus ID is valid, and SNIP is null:")
    print(cs_journal_metadata_df_with_scopus_id_null_snip.head().to_string())
    # In such records, SNIPList (and other fields like SJRList and citeScoreYearInfoList) either absent or set to null

def handle_null_sjr(journal_metrics_df):
    """
    Handle the missing SJE values - fill in using the SJR dataset
    :param journal_metrics_df: Journal metrics dataframe containing missing SJR values
    :return: Journal metrics dataframe with missing SJR handled
    """
    # Split into two dataframes, one with SJRs and another for journals missing SJRs
    null_sjr_journal_metrics_df, not_null_sjr_journal_metrics_df = split_df_on_null_field(journal_metrics_df, 'SJR')

    # Drop SJR from the null_sjr_df: [here SJR=null for all records, we try to obtain this by joining]
    null_sjr_journal_metrics_df = null_sjr_journal_metrics_df.drop(columns=['SJR'])

    # Load the processed SJR dataframe
    sjr_df = load_sjr_df()

    # Perform a merge using the null SJRs with the dataset containing the SJRs to override the null SJRs if ISSNs match
    null_sjr_with_sjr_df_joined = pd.merge(
        null_sjr_journal_metrics_df, sjr_df, left_on='ISSN', right_on='Issn', how='left'
    )
    # Drop additional fields that are not required [we're only interested in obtaining SJR]
    null_sjr_with_sjr_df_joined = null_sjr_with_sjr_df_joined.drop(columns=['Rank', 'Title', 'Issn'])

    # Ensure consistent column order
    null_sjr_with_sjr_df_joined = null_sjr_with_sjr_df_joined[['ISSN', 'Scopus_ID', 'SNIP', 'SJR', 'Cite_Score']]

    # Combine the journals who original has SJRs along with the now processed journals with missing SJRs
    journal_metrics_handled_null_sjr_df = pd.concat([null_sjr_with_sjr_df_joined, not_null_sjr_journal_metrics_df])

    return journal_metrics_handled_null_sjr_df

def handle_null_snip(journal_metrics_df):
    """
    Handle the missing SNIP values - fill in using the SNIP dataset
    :param journal_metrics_df: Journal metrics dataframe containing missing SNIP values
    :return: Journal metrics dataframe with missing SNIP handled
    """
    # Split into two dataframes, one with SNIPs and another for journals missing SNIPs
    null_snip_journal_metrics_df, not_null_snip_journal_metrics_df = split_df_on_null_field(journal_metrics_df, 'SNIP')

    # Drop SNIP column from the null SNIP DF. SNIP values will be retrieved by performing join with SNIP DF
    null_snip_journal_metrics_df = null_snip_journal_metrics_df.drop(columns=['SNIP'])

    # Load the processed dataframe containing SNIP values
    snip_df = load_processed_snip_df()

    # Perform a merge using the null SNIPs with the dataset containing the SNIPs to override the null SNIPs if ISSNs match
    null_snip_with_snip_df_joined = pd.merge(
        null_snip_journal_metrics_df, snip_df, left_on='ISSN', right_on='ISSN', how='left'
    )

    # Ensure consistent column order
    null_snip_with_snip_df_joined = null_snip_with_snip_df_joined[['ISSN', 'Scopus_ID', 'SNIP', 'SJR', 'Cite_Score']]

    # Combine the journals who original has SNIPs along with the now processed journals with missing SNIPs
    journal_metrics_handled_null_snip_df = pd.concat([null_snip_with_snip_df_joined, not_null_snip_journal_metrics_df])

    return journal_metrics_handled_null_snip_df


def handle_missing_journal_metrics(journal_metrics_df):
    """
    Handle missing values in the journal metrics
    :param journal_metrics_df: Journal metrics dataframes with missing values
    :return: Journal metrics dataframes with missing values handled
    """
    journal_metrics_df_handled_missing_fields = handle_null_sjr(journal_metrics_df)
    journal_metrics_df_handled_missing_fields = handle_null_snip(journal_metrics_df_handled_missing_fields)
    return journal_metrics_df_handled_missing_fields


def ensure_uniform_data_types(journal_metrics_df_handled_missing_fields):
    """
    fastparquet engine cannot infer the data-type of a field containing both NaN and numeric values when it is
    stored as dtype: object.
    Before writing to parquet, ensure all columns have the proper data-type

    :param journal_metrics_df_handled_missing_fields: Journal Metrics dataframe with missing values handled where
    fields as stored as dtype: object (s)
    :return: Journal Metrics dataframe with missing values handled and proper data-types for columns
    """
    numerical_cols = ["SNIP", "SJR", "Cite_Score"]
    for numerical_col in numerical_cols:
        journal_metrics_df_handled_missing_fields[numerical_col] = pd.to_numeric(
            journal_metrics_df_handled_missing_fields[numerical_col], errors="coerce"
        )

    # Keep "Scopus_ID" as a string to avoid scientific notation and preserve leading zeros (if any)
    # Example: Prevent '21100853871' from becoming 2.110085e+10
    string_cols = ["ISSN", "Scopus_ID"]
    for string_col in string_cols:
        journal_metrics_df_handled_missing_fields[string_col] = journal_metrics_df_handled_missing_fields[string_col].astype(
            "string"
        )

    return journal_metrics_df_handled_missing_fields


def write_journal_metrics_handled_missing_fields_df(journal_metrics_df_handled_missing_fields):
    """
    Write the journal metrics dataframe with missing values filled in as a parquet file
    :param journal_metrics_df_handled_missing_fields: Journal metrics dataframe with missing values filled in
    """
    cs_journal_metrics_handled_missing_data_df_path = os.path.join(
        os.path.dirname(__file__), "..", "..", DATASETS_DIR, REFINED_DIR, CS_JOURNAL_METRICS
    )

    journal_metrics_df_handled_missing_fields.to_parquet(
        cs_journal_metrics_handled_missing_data_df_path,
        engine='fastparquet'
    )

if __name__ == "__main__":
    main()
