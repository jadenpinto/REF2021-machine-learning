import os
import pandas as pd

from utils.constants import DATASETS_DIR, PROCESSED_DIR, CS_OUTPUTS_METADATA, REFINED_DIR, \
    CS_JOURNAL_METRICS, CS_OUTPUT_METRICS, CS_CITATION_METRICS, MACHINE_LEARNING_DIR, CS_OUTPUTS_COMPLETE_METADATA

def main():
    """
    ETL to create dataframe of CS outputs with complete metadata: (Include all available features - that are sensible)
    Metadata includes Journal metrics, citation counts, and field-normalised output performance metrics.

    This file contains all possible metadata about a given output.
    It can be read to create different variations of parameters to feed the ML model =>  feature engineering file.
    """
    cs_outputs_enriched_metadata = create_cs_outputs_enriched_metadata()
    write_cs_outputs_enriched_metadata(cs_outputs_enriched_metadata)


def get_cs_outputs_metadata():
    """
    Load a file containing metadata of outputs submitted to the CS UoA into a DataFrame
    :return: DataFraming containing metadata of outputs submitted to the CS UoA
    """
    cs_outputs_metadata_path = os.path.join(
        os.path.dirname(__file__), "..", DATASETS_DIR, PROCESSED_DIR, CS_OUTPUTS_METADATA
    )

    cs_outputs_metadata = pd.read_csv(cs_outputs_metadata_path)
    return cs_outputs_metadata

def filter_cs_metadata_fields(cs_outputs_metadata):
    """
    The CS output metadata has multiple columns, filter to keep only the necessary ones
    :param cs_outputs_metadata: DataFraming containing metadata of outputs submitted to the CS UoA
    :return: Filtered DataFraming containing metadata of outputs submitted to the CS UoA containing metadata fields
    """

    # Note: Select fields that provide insight about the output. Do not have to use these fields for clustering.
    cs_outputs_metadata_fields = [
        'Institution UKPRN code', 'Institution name', 'Output type', 'Title', 'Volume title', # UKPRN => Results
        'Place', 'Publisher',
        'ISSN', 'DOI', 'Year', # ISSN, DOI => Joins
        'Number of additional authors', 'Interdisciplinary', 'Forensic science', 'Criminology',
        'Research group', 'Open access status', 'Cross-referral requested', 'Delayed by COVID19',
        'Incl sig material before 2014', 'Incl reseach process', 'Incl factual info about significance'
    ]

    return cs_outputs_metadata[cs_outputs_metadata_fields]

def load_cs_journal_metrics_df():
    """
    Load the file containing the journal metrics of journals whose articles were submitted the CS UoA into a DataFrame
    :return: DataFrame of CS journal metrics - includes SJR, SNIP, Cite Score
    """
    cs_journal_metrics_df_path = os.path.join(os.path.dirname(__file__), "..", DATASETS_DIR, REFINED_DIR,
                                              CS_JOURNAL_METRICS)
    cs_journal_metrics_df = pd.read_parquet(cs_journal_metrics_df_path, engine='fastparquet')
    return cs_journal_metrics_df

def load_cs_output_metrics_df():
    """
    Load the file containing the field-weighted performance metrics of outputs submitted to the CS UoA into a DataFrame
    :return: DataFrame of field-weighted performance metrics of outputs submitted to the CS UoA
    """
    cs_output_metrics_df_path = os.path.join(
        os.path.dirname(__file__), "..", DATASETS_DIR, REFINED_DIR, CS_OUTPUT_METRICS
    )
    cs_output_metrics_df = pd.read_parquet(cs_output_metrics_df_path, engine='fastparquet')
    return cs_output_metrics_df

def load_cs_citation_metadata_df():
    """
    Load the file containing the citation metrics of outputs submitted to the CS UoA into a DataFrame
    :return: DataFrame of citation metrics of outputs submitted to the CS UoA
    """
    cs_citation_metadata_df_path = os.path.join(
        os.path.dirname(__file__), "..", DATASETS_DIR, REFINED_DIR, CS_CITATION_METRICS
    )
    cs_citation_metadata_df = pd.read_parquet(cs_citation_metadata_df_path, engine='fastparquet')
    return cs_citation_metadata_df

def enrich_cs_outputs_metadata(cs_outputs_metadata):
    """
    Enrich the CS outputs with journal metrics and output metrics
    :param cs_outputs_metadata: DataFrame of metadata of outputs submitted the CS UoA
    :return: Enriched DataFrame containing output metadata with journal metrics and output metrics
    """
    cs_outputs_metadata_journal_metrics = enrich_metadata_with_journal_metrics(
        cs_outputs_metadata
    )
    cs_outputs_metadata_journal_output_metrics = enrich_metadata_with_output_metrics(
        cs_outputs_metadata_journal_metrics
    )
    return cs_outputs_metadata_journal_output_metrics

def enrich_metadata_with_journal_metrics(cs_outputs_metadata):
    """
    Enrich outputs with journal metrics - SNIP, SJR, and CiteScore

    :param cs_outputs_metadata: DataFrame of CS outputs without journal metrics
    :return: DataFrame of CS outputs with journal metrics (SNIP, SJR, and CiteScore)
    """
    # Using ISSN: Retrieve SNIP, SJR, and CiteScore of journals

    cs_journal_metrics_df = load_cs_journal_metrics_df()

    cs_outputs_metadata_journal_metrics = cs_outputs_metadata.merge(
        cs_journal_metrics_df,
        on="ISSN",  # Join on ISSN column
        how="left"  # Keep all rows from cs_outputs_metadata i.e. all outputs
    )

    return cs_outputs_metadata_journal_metrics

def enrich_metadata_with_output_metrics(cs_outputs_metadata):
    """
    Enrich CS outputs by adding new columns to store output metrics - citation counts and field-normalised performance metrics
    :param cs_outputs_metadata: CS outputs DataFrame without output metrics
    :return: CS outputs DataFrame with new columns for output metrics
    """
    cs_citation_metadata_df = load_cs_citation_metadata_df()  # merge using DOI, and get scopus ID

    cs_outputs_metadata_citation_metrics_df = cs_outputs_metadata.merge(
        cs_citation_metadata_df,
        on="DOI",  # Join on DOI column
        how="left" # Keep all rows from cs_outputs_metadata i.e. all outputs
    )

    cs_output_metrics_df = load_cs_output_metrics_df()

    cs_outputs_metadata_citation_output_metrics_df = cs_outputs_metadata_citation_metrics_df.merge( # merge using scopus id
        cs_output_metrics_df,
        on="scopus_id", # Join on Scopus ID
        how="left"      # Keep all rows from cs_outputs_metadata_citation_metrics_df i.e. all outputs (with citation metrics)
    )

    return cs_outputs_metadata_citation_output_metrics_df


def create_cs_outputs_enriched_metadata():
    """
    Create the CS outputs enrich metadata parquet file by enhancing the CS outputs metadata file.
    Drop all unnecessary columns, and add new columns for journal and citation metrics
    :return: DataFrame with enriched output metadata that includes journal and output metrics
    """
    # Load all CS outputs
    cs_outputs_metadata = get_cs_outputs_metadata()

    # Drop all unnecessary columns
    cs_outputs_metadata = filter_cs_metadata_fields(cs_outputs_metadata)

    # Add new fields for journal metrics and output metrics (citation counts + field-normalised performance metrics)
    cs_outputs_enriched_metadata = enrich_cs_outputs_metadata(cs_outputs_metadata)

    return cs_outputs_enriched_metadata

def write_cs_outputs_enriched_metadata(cs_outputs_enriched_metadata):
    """
    Persist the DataFrame containing the enriched metadata of outputs submitted to the CS UoA as a parquet file
    :param cs_outputs_enriched_metadata: DataFrame containing the enriched metadata of outputs submitted to the CS UoA
    """
    cs_outputs_enriched_metadata_path = os.path.join(
        os.path.dirname(__file__), "..", DATASETS_DIR, MACHINE_LEARNING_DIR, CS_OUTPUTS_COMPLETE_METADATA
    )

    cs_outputs_enriched_metadata.to_parquet(
        cs_outputs_enriched_metadata_path,
        engine='fastparquet'
    )


if __name__ == "__main__":
    main()
