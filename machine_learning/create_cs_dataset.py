import os
import pandas as pd

from utils.constants import DATASETS_DIR, RAW_DIR, OUTPUTS_METADATA, PROCESSED_DIR, CS_OUTPUTS_METADATA, REFINED_DIR, \
    CS_JOURNAL_METRICS, CS_OUTPUT_METRICS, CS_CITATION_METRICS


def get_cs_outputs_metadata():
    cs_outputs_metadata_path = os.path.join(
        os.path.dirname(__file__), "..", DATASETS_DIR, PROCESSED_DIR, CS_OUTPUTS_METADATA
    )

    cs_outputs_metadata = pd.read_csv(cs_outputs_metadata_path)
    return cs_outputs_metadata

def filter_cs_metadata_fields(cs_outputs_metadata):
    cs_outputs_metadata_fields = [
        'Institution UKPRN code', 'Institution name', 'Output type', 'Title', 'ISSN', 'DOI', 'Year',
        'Number of additional authors'
    ]
    # Possibly 'Citation count' -> but try to use this to back-fill citation API
    # Can remove later: Institution name (code is enough), Type, Title, Year
    # Institution UKPRN code: Results
    # ISSN, DOI: Join
    # Author count: Normalisation [Look into current literature, I think they use log]
    return cs_outputs_metadata[cs_outputs_metadata_fields]

def load_cs_journal_metrics_df():
    cs_journal_metrics_df_path = os.path.join(os.path.dirname(__file__), "..", DATASETS_DIR, REFINED_DIR,
                                              CS_JOURNAL_METRICS)
    cs_journal_metrics_df = pd.read_parquet(cs_journal_metrics_df_path, engine='fastparquet')
    return cs_journal_metrics_df

def load_cs_output_metrics_df():
    cs_output_metrics_df_path = os.path.join(
        os.path.dirname(__file__), "..", DATASETS_DIR, REFINED_DIR, CS_OUTPUT_METRICS
    )
    cs_output_metrics_df = pd.read_parquet(cs_output_metrics_df_path, engine='fastparquet')
    return cs_output_metrics_df

def load_cs_citation_metadata_df():
    cs_citation_metadata_df_path = os.path.join(
        os.path.dirname(__file__), "..", DATASETS_DIR, REFINED_DIR, CS_CITATION_METRICS
    )
    cs_citation_metadata_df = pd.read_parquet(cs_citation_metadata_df_path, engine='fastparquet')
    return cs_citation_metadata_df

def enrich_cs_outputs_metadata(cs_outputs_metadata):
    cs_outputs_metadata_journal_metrics = enrich_metadata_with_journal_metrics(
        cs_outputs_metadata
    )
    cs_outputs_metadata_journal_output_metrics = enrich_metadata_with_output_metrics(
        cs_outputs_metadata_journal_metrics
    )
    return cs_outputs_metadata_journal_output_metrics

def enrich_metadata_with_journal_metrics(cs_outputs_metadata):
    # Using ISSN: Retrieve SNIP, SJR, and CiteScore of journals

    cs_journal_metrics_df = load_cs_journal_metrics_df()

    cs_outputs_metadata_journal_metrics = cs_outputs_metadata.merge(
        cs_journal_metrics_df,
        on="ISSN",  # Join on ISSN column
        how="left"  # Keep all rows from cs_outputs_metadata
    )

    return cs_outputs_metadata_journal_metrics




def create_cs_dataset():
    cs_outputs_metadata = get_cs_outputs_metadata()

    cs_outputs_metadata = filter_cs_metadata_fields(cs_outputs_metadata)

    cs_outputs_enriched_metadata = enrich_cs_outputs_metadata(cs_outputs_metadata)
    # Use DOI, Use issn


#create_cs_dataset() # -> write to ML folder

"""
Dataset represent CS outputs
"""