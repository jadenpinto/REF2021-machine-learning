import os
import pandas as pd

from utils.constants import DATASETS_DIR, PROCESSED_DIR, CS_OUTPUTS_METADATA, REFINED_DIR, CS_CITATION_METRICS
from utils.dataframe import split_df_on_null_field

def get_cs_outputs_metadata():
    cs_outputs_metadata_path = os.path.join(
        os.path.dirname(__file__), "..", "..", DATASETS_DIR, PROCESSED_DIR, CS_OUTPUTS_METADATA
    )

    cs_outputs_metadata = pd.read_csv(cs_outputs_metadata_path)
    return cs_outputs_metadata

def load_cs_citation_metadata_df():
    cs_citation_metadata_df_path = os.path.join(
        os.path.dirname(__file__), "..", "..", DATASETS_DIR, REFINED_DIR, CS_CITATION_METRICS
    )
    cs_citation_metadata_df = pd.read_parquet(cs_citation_metadata_df_path, engine='fastparquet')
    return cs_citation_metadata_df

def log_missing_citations(cs_citation_metadata_df):
    outputs_missing_citations = cs_citation_metadata_df['total_citations'].isna().sum()
    print(f"Number of outputs missing citation counts: {outputs_missing_citations}")

def handle_missing_citations(cs_citation_metadata_df):
    null_citations_df, not_null_citations_df = split_df_on_null_field(
        cs_citation_metadata_df, 'total_citations'
    )

    # Drop total_citations from the null_citations_df: [here total_citations=null for all records, we try to obtain this by joining]
    null_citations_df = null_citations_df.drop(columns=['total_citations'])

    cs_outputs_metadata = get_cs_outputs_metadata()
    cs_outputs_metadata = cs_outputs_metadata[['DOI', 'Citation count']].rename(columns={'Citation count': 'total_citations'})

    null_citations_df_joined = pd.merge(
        null_citations_df,
        cs_outputs_metadata,
        left_on='DOI',
        right_on='DOI',
        how='left'
    )

    # Ensure consistent column order
    cs_citation_metadata_df_columns = cs_citation_metadata_df.columns
    null_citations_df_joined = null_citations_df_joined[cs_citation_metadata_df_columns]

    updated_citations_df = pd.concat([null_citations_df_joined, not_null_citations_df])

    return updated_citations_df

def write_handled_missing_citations_df(cs_citation_metadata_df):
    cs_citation_metadata_df_path = os.path.join(
        os.path.dirname(__file__), "..", "..", DATASETS_DIR, REFINED_DIR, CS_CITATION_METRICS
    )

    cs_citation_metadata_df.to_parquet(
        cs_citation_metadata_df_path,
        engine='fastparquet'
    )


def process_missing_citations():
    cs_citation_metadata_df = load_cs_citation_metadata_df()

    log_missing_citations(cs_citation_metadata_df) # Number of outputs missing citation counts: 174

    cs_citation_metadata_df = handle_missing_citations(cs_citation_metadata_df)

    log_missing_citations(cs_citation_metadata_df) # Number of outputs missing citation counts: 103

    write_handled_missing_citations_df(cs_citation_metadata_df)

process_missing_citations()
