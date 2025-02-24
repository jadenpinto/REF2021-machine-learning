import os
import pandas as pd

from utils.constants import DATASETS_DIR, PROCESSED_DIR, SNIP, CS_JOURNAL_METRICS, REFINED_DIR, SCIMAGO_JOURNAL_RANK

from utils.dataframe import split_df_on_null_field

def load_cs_journal_metrics_df():
    cs_journal_metrics_df_path = os.path.join(os.path.dirname(__file__), "..", "..", DATASETS_DIR, REFINED_DIR,
                                              CS_JOURNAL_METRICS)
    cs_journal_metrics_df = pd.read_parquet(cs_journal_metrics_df_path, engine='fastparquet')
    return cs_journal_metrics_df

def load_processed_snip_df():
    snip_df_path = os.path.join(os.path.dirname(__file__), "..", "..", DATASETS_DIR, PROCESSED_DIR,
                                              SNIP)
    snip_df = pd.read_parquet(snip_df_path, engine='fastparquet')
    return snip_df

def load_sjr_df():
    processed_sjr_csv_path = os.path.join(os.path.dirname(__file__), "..", "..", DATASETS_DIR, PROCESSED_DIR,
                                          SCIMAGO_JOURNAL_RANK)

    sjr_df = pd.read_csv(processed_sjr_csv_path)
    return sjr_df

def log_null_cs_journal_metadata(loaded_cs_journal_metrics_df):
    print("CS journal metadata fields with null counts:")
    print(loaded_cs_journal_metrics_df.isna().sum())
    """
    ISSN            0
    Scopus_ID      96
    SNIP          129
    SJR           143
    Cite_Score    143
    """

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
    null_sjr_journal_metrics_df, not_null_sjr_journal_metrics_df = split_df_on_null_field(journal_metrics_df, 'SJR')

    # Drop SJR from the null_sjr_df: [here SJR=null for all records, we try to obtain this by joining]
    null_sjr_journal_metrics_df = null_sjr_journal_metrics_df.drop(columns=['SJR'])

    sjr_df = load_sjr_df()

    null_sjr_with_sjr_df_joined = pd.merge(
        null_sjr_journal_metrics_df, sjr_df, left_on='ISSN', right_on='Issn', how='left'
    )
    # Drop additional fields that are not required [we're only interested in obtaining SJR]
    null_sjr_with_sjr_df_joined = null_sjr_with_sjr_df_joined.drop(columns=['Rank', 'Title', 'Issn'])

    # Ensure consistent column order
    null_sjr_with_sjr_df_joined = null_sjr_with_sjr_df_joined[['ISSN', 'Scopus_ID', 'SNIP', 'SJR', 'Cite_Score']]

    journal_metrics_handled_null_sjr_df = pd.concat([null_sjr_with_sjr_df_joined, not_null_sjr_journal_metrics_df])

    return journal_metrics_handled_null_sjr_df

def handle_null_snip(journal_metrics_df):
    null_snip_journal_metrics_df, not_null_snip_journal_metrics_df = split_df_on_null_field(journal_metrics_df, 'SNIP')

    # Drop SNIP column from the null SNIP DF. SNIP values will be retrieved by performing join with SNIP DF
    null_snip_journal_metrics_df = null_snip_journal_metrics_df.drop(columns=['SNIP'])

    snip_df = load_processed_snip_df() # possible rename ISSN -> Issn or ISSN_SNIP_DF

    null_snip_with_snip_df_joined = pd.merge(
        null_snip_journal_metrics_df, snip_df, left_on='ISSN', right_on='ISSN', how='left'
    )

    # Ensure consistent column order
    null_snip_with_snip_df_joined = null_snip_with_snip_df_joined[['ISSN', 'Scopus_ID', 'SNIP', 'SJR', 'Cite_Score']]

    journal_metrics_handled_null_snip_df = pd.concat([null_snip_with_snip_df_joined, not_null_snip_journal_metrics_df])

    return journal_metrics_handled_null_snip_df


def handle_missing_journal_metrics(journal_metrics_df):
    print(journal_metrics_df.isna().sum())
    journal_metrics_df_handled_missing_fields = handle_null_sjr(journal_metrics_df)
    print(journal_metrics_df_handled_missing_fields.isna().sum())
    journal_metrics_df_handled_missing_fields = handle_null_snip(journal_metrics_df_handled_missing_fields)
    print(journal_metrics_df_handled_missing_fields.isna().sum())
    return journal_metrics_df_handled_missing_fields



cs_journal_metrics_df = load_cs_journal_metrics_df()

log_null_cs_journal_metadata(cs_journal_metrics_df)

journal_metrics_df_handled_missing_fields =  handle_missing_journal_metrics(cs_journal_metrics_df)

# Write to the same location as write_cs_journal_metrics_df [overwrite existing file]


# Given df -> split into 2, null not null, handle null, rename cols (if needed to match), concat and write

"""
failed api calls here, for SJR, see if I can obtain from the SJR dataset
same for snip which (datasets in downloaded)
maybe check if cite-core has information online.
[if df just do left join with issn for null cols]

todo
in another file, create df of unique scopus ids and get metadata about the metrics john asked for [from API]
see if I can train LLMs to get score [this can be research question]
"""
