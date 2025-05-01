import os
import pandas as pd

from utils.constants import DATASETS_DIR, PROCESSED_DIR, CS_OUTPUTS_METADATA, REFINED_DIR, CS_CITATION_METRICS
from utils.dataframe import split_df_on_null_field

def main():
    """
    ETL Pipeline to fill-in the citations of outputs submitted to the CS UoA that were missing after unsuccessful API calls
    """
    process_missing_citations()

def get_cs_outputs_metadata():
    cs_outputs_metadata_path = os.path.join(
        os.path.dirname(__file__), "..", "..", DATASETS_DIR, PROCESSED_DIR, CS_OUTPUTS_METADATA
    )

    cs_outputs_metadata = pd.read_csv(cs_outputs_metadata_path)
    return cs_outputs_metadata

def load_cs_citation_metadata_df():
    """
    Load the file containing the citation metadata of outputs submitted to the CS UoA as a DataFrame
    :return: DataFrame containing the citation metadata of outputs submitted to the CS UoA
    """
    cs_citation_metadata_df_path = os.path.join(
        os.path.dirname(__file__), "..", "..", DATASETS_DIR, REFINED_DIR, CS_CITATION_METRICS
    )
    cs_citation_metadata_df = pd.read_parquet(cs_citation_metadata_df_path, engine='fastparquet')
    return cs_citation_metadata_df

def log_missing_citations(cs_citation_metadata_df):
    """
    Log the count of the outputs missing citation counts
    :param cs_citation_metadata_df: DataFrame containing the citation metadata of outputs submitted to the CS UoA
    """
    outputs_missing_citations = cs_citation_metadata_df['total_citations'].isna().sum()
    print(f"Number of outputs missing citation counts: {outputs_missing_citations}")

def fill_missing_citations(cs_citation_metadata_df, cs_outputs_metadata):
    """
    Fills missing total_citations in cs_citation_metadata_df with values from
    cs_outputs_metadata using DOI as the matching key.

    :param cs_citation_metadata_df: DataFrame containing total_citations and DOI.
    :param cs_outputs_metadata: DataFrame containing DOI and Citation count.
    :return: Updated cs_citation_metadata_df with filled total_citations.
    """

    cs_outputs_metadata = cs_outputs_metadata.drop_duplicates(subset="DOI")
    """
    Without doing above change, the number of nulls for other fields like scopus_id in merged_df increases
    If cs_outputs_metadata has duplicate DOI values, a one-to-many merge occurs.
    This creates duplicate rows in merged_df, potentially leading to more NaN values where scopus_id was previously present.
    """

    # Update the CS Citation metadata DF by Merging it with cs_outputs_metadata, to obtain the metadata 'Citation count'
    # field by joining on DOI
    merged_df = cs_citation_metadata_df.merge(
        cs_outputs_metadata[['DOI', 'Citation count']],
        on="DOI",
        how="left"
    )

    # Fill null total_citations with Citation count if available
    merged_df["total_citations"] = merged_df["total_citations"].fillna(merged_df["Citation count"])

    # Drop the extra 'Citation count' column as it's no longer needed
    merged_df.drop(columns=["Citation count"], inplace=True)

    return merged_df


def write_handled_missing_citations_df(cs_citation_metadata_df):
    """
    Write the updated cs citation metadata dataframe containing output's DOI with citation counts
    :param cs_citation_metadata_df: DataFrame of CS outputs with their DOI with citation counts
    """
    cs_citation_metadata_df_path = os.path.join(
        os.path.dirname(__file__), "..", "..", DATASETS_DIR, REFINED_DIR, CS_CITATION_METRICS
    )

    cs_citation_metadata_df.to_parquet(
        cs_citation_metadata_df_path,
        engine='fastparquet'
    )


def process_missing_citations():
    """
    ETL pipeline that reads in CS outputs citations file, and fills in missing values using the CS outputs metadata file
    """
    cs_citation_metadata_df = load_cs_citation_metadata_df()

    log_missing_citations(cs_citation_metadata_df) # Number of outputs missing citation counts: 174

    cs_outputs_metadata = get_cs_outputs_metadata()

    cs_citation_metadata_df = fill_missing_citations(cs_citation_metadata_df, cs_outputs_metadata)

    log_missing_citations(cs_citation_metadata_df) # Number of outputs missing citation counts: 101

    write_handled_missing_citations_df(cs_citation_metadata_df)

if __name__ == "__main__":
    main()
