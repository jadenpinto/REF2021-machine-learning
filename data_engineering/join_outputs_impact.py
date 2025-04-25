"""
Archived Script:
The SJR metrics were obtained using the Scopus serial title API, see journal_metrics/02_scopus_serial_title_API.py
The SJR data set was used to fill in the missing values after the API calls, see journal_metrics/05_handle_missing_journal_metrics.py
"""

import os
import pandas as pd

from utils.constants import DATASETS_DIR, PROCESSED_DIR, SJR, CS_OUTPUTS_METADATA
from utils.dataframe import log_dataframe

def get_cs_outputs_metadata():
    cs_outputs_path = os.path.join(os.path.dirname(__file__), "..", DATASETS_DIR, PROCESSED_DIR,
                                           CS_OUTPUTS_METADATA)

    cs_outputs_df = pd.read_csv(cs_outputs_path)
    return cs_outputs_df

def get_journal_article_metadata(cs_outputs_df):
    journal_article_metadata = cs_outputs_df[cs_outputs_df['Output type'] == "D"]
    return journal_article_metadata

def get_sjr_impact_factor_df():
    processed_sjr_csv_path = os.path.join(os.path.dirname(__file__), "..", DATASETS_DIR, PROCESSED_DIR,
                                          SJR)
    sjr_df = pd.read_parquet(processed_sjr_csv_path, engine='fastparquet')
    return sjr_df

def join_journal_article_metadata_with_sjr():
    cs_outputs_df = get_cs_outputs_metadata()
    journal_article_metadata = get_journal_article_metadata(cs_outputs_df)
    sjr_impact_df = get_sjr_impact_factor_df()

    log_dataframe(journal_article_metadata) # Shape: 5573x40
    log_dataframe(sjr_impact_df)            # Shape: 47524x4

    # Join: journal_article_metadata [ISSN], sjr_impact_df ['Issn']
    joined_df = pd.merge(
        journal_article_metadata, sjr_impact_df, left_on='ISSN', right_on='Issn', how='left'
    )
    log_dataframe(joined_df)               # Shape: 5573x44
    return joined_df

def log_failed_joins(sjr_joined_df):
    # Failed joins: Records where Issn = None
    failed_join_count = sjr_joined_df['Issn'].isna().sum() #  303 failed joins. [successful joins = 5573-303 = 5270]
    print(f"Count of output records whose ISSN could not be found in journal metadata = {failed_join_count}")

    # Invalid SJR: Issn is Valid (successful join) but SJR is None
    joins_with_no_sjr = sjr_joined_df[
        ~sjr_joined_df['Issn'].isnull() & sjr_joined_df["SJR"].isna()
    ]
    print(f"Number of output records with successful ISSN joins but invalid SJR = {joins_with_no_sjr.shape[0]}")
    print(joins_with_no_sjr.to_string()) # 14x44 (14 such records where Issn is Valid but SJR is None)

def join_outputs_metadata_with_sjr():
    cs_outputs_df = get_cs_outputs_metadata()
    sjr_impact_df = get_sjr_impact_factor_df()

    log_dataframe(cs_outputs_df)            # Shape: 7296x40
    log_dataframe(sjr_impact_df)            # Shape: 47524x4

    joined_df = pd.merge(
        cs_outputs_df, sjr_impact_df, left_on='ISSN', right_on='Issn', how='left'
    )
    log_dataframe(joined_df)               # Shape: 7296x44

    # Failed joins: Issn = None
    print(joined_df.isna().sum()) # Issn = 1641. 1641 failed joins. [successful joins = 7296-1641 = 5655]
    # So, besides journal articles, a few entries of other types have an SJR impact [5655 - 5270 = 385]

    return joined_df

journal_metadata_with_sjr = join_journal_article_metadata_with_sjr()
outputs_with_sjr = join_outputs_metadata_with_sjr()

log_failed_joins(journal_metadata_with_sjr)
