import os
import pandas as pd

from utils.constants import DATASETS_DIR, PROCESSED_DIR, CS_OUTPUTS_METADATA, REFINED_DIR, \
    CS_JOURNAL_METRICS, CS_OUTPUT_METRICS, CS_CITATION_METRICS, MACHINE_LEARNING_DIR, CS_OUTPUTS_COMPLETE_METADATA

def main():
    """
    ETL to create dataframe of CS outputs with complete metadata (Include all available features - that are sensible)

    This file contains all possible metadata about a given output.
    It can be read to create different variations of parameters to feed the ML model =>  feature engineering file.
    """
    cs_outputs_enriched_metadata = create_cs_outputs_enriched_metadata()
    write_cs_outputs_enriched_metadata(cs_outputs_enriched_metadata)


def get_cs_outputs_metadata():
    cs_outputs_metadata_path = os.path.join(
        os.path.dirname(__file__), "..", DATASETS_DIR, PROCESSED_DIR, CS_OUTPUTS_METADATA
    )

    cs_outputs_metadata = pd.read_csv(cs_outputs_metadata_path)
    return cs_outputs_metadata

def filter_cs_metadata_fields(cs_outputs_metadata):
    # Note: Select fields that provide insight about the output. Do not have to use these fields for clustering.
    cs_outputs_metadata_fields = [
        'Institution UKPRN code', 'Institution name', 'Output type', 'Title', 'Volume title', # UKPRN => Results
        'Place', 'Publisher',
        'ISSN', 'DOI', 'Year', # ISSN, DOI => Joins
        'Number of additional authors', 'Interdisciplinary', 'Forensic science', 'Criminology',
        'Research group', 'Open access status', 'Cross-referral requested', 'Delayed by COVID19',
        'Incl sig material before 2014', 'Incl reseach process', 'Incl factual info about significance'
    ]
    # Author count => Normalisation. [no. of authors - was log scaled and used as input in 3rd paper]
    # Left out 'REF2ID' (Don't think I need this)
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

def enrich_metadata_with_output_metrics(cs_outputs_metadata):
    cs_citation_metadata_df = load_cs_citation_metadata_df()  # merge using DOI, and get scopus ID

    cs_outputs_metadata_citation_metrics_df = cs_outputs_metadata.merge(
        cs_citation_metadata_df,
        on="DOI",
        how="left"
    )

    cs_output_metrics_df = load_cs_output_metrics_df()

    cs_outputs_metadata_citation_output_metrics_df = cs_outputs_metadata_citation_metrics_df.merge( # merge using scopus id
        cs_output_metrics_df,
        on="scopus_id",
        how="left"
    )

    return cs_outputs_metadata_citation_output_metrics_df


def create_cs_outputs_enriched_metadata():
    cs_outputs_metadata = get_cs_outputs_metadata()

    cs_outputs_metadata = filter_cs_metadata_fields(cs_outputs_metadata)

    cs_outputs_enriched_metadata = enrich_cs_outputs_metadata(cs_outputs_metadata)

    return cs_outputs_enriched_metadata

def write_cs_outputs_enriched_metadata(cs_outputs_enriched_metadata):
    cs_outputs_enriched_metadata_path = os.path.join(
        os.path.dirname(__file__), "..", DATASETS_DIR, MACHINE_LEARNING_DIR, CS_OUTPUTS_COMPLETE_METADATA
    )

    cs_outputs_enriched_metadata.to_parquet(
        cs_outputs_enriched_metadata_path,
        engine='fastparquet'
    )


if __name__ == "__main__":
    main()
