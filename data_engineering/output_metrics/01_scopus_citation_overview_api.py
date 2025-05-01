import os
import requests
import pandas as pd
from time import sleep
from dotenv import load_dotenv

from utils.constants import DATASETS_DIR, PROCESSED_DIR, CS_OUTPUTS_METADATA, REFINED_DIR, CS_CITATION_METRICS

def main():
    """
    ETL pipeline to obtain and persist the citation counts of outputs submitted to the CS UoA
    """
    configure()

    global elsevier_api_key
    elsevier_api_key = os.getenv('elsevier_api_key')

    cs_citation_metadata_df = process_citation_metadata()
    write_cs_citation_metadata_df(cs_citation_metadata_df)


def configure():
    """
    Configure the API Key - read the environment file, and load it as an environment variable
    """
    load_dotenv()

def get_cs_outputs_metadata():
    """
    Obtain a DataFrame of containing the metadata of CS outputs - needed for creating the CS DOI dataframe
    :return: DataFrame containing outputs metadata of submissions to the CS UoA
    """
    processed_cs_outputs_metadata_path = os.path.join(os.path.dirname(__file__), "..", "..", DATASETS_DIR, PROCESSED_DIR,
                                           CS_OUTPUTS_METADATA)

    cs_outputs_metadata = pd.read_csv(processed_cs_outputs_metadata_path)
    return cs_outputs_metadata

def get_cs_doi_df():
    """
    Obtain Dataframe containing the DOIs of outputs submitted to the CS UoA
    :return: Dataframe containing the DOIs of outputs submitted to the CS UoA
    """
    cs_outputs_df = get_cs_outputs_metadata()
    cs_doi_df = cs_outputs_df[["DOI"]].drop_duplicates().dropna()
    return cs_doi_df

def get_citation_metadata(doi):
    """
    Using an outputs's DOI, make an API call to the Scopus Citation Overview API to obtain its citation metrics
    :param doi: Unique identifier for an output
    :return: JSON payload response that is returned on a successful API call to the Scopus API
    """
    citation_metadata_base_url = f"https://api.elsevier.com/content/abstract/citations"
    citation_metadata_url_params = {
        "apiKey": elsevier_api_key,
        "doi": doi,
        "httpAccept": "application/json",
        "sort": "+sort-year",
        "date": "2014-2020", # Publication duration of outputs submitted to REF2021
        "field": "scopus_id,cc,rangeCount"
    }

    try:
        response = requests.get(citation_metadata_base_url, params=citation_metadata_url_params, timeout=10)
        response.raise_for_status()

        if response.status_code == 200:
            # If the HTTP status code is 200 OK, it means the API call was successful, so retrieve the JSON response
            data = response.json()
            return data
        else:
            print(f"Request failed with status code: {response.status_code}")
            print(response.text)  # Print the response text for debugging

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")

def extract_citation_metadata(data):
    """
    Parse the Scopus API's JSON response, to obtained citation metrics
    :param data: JSON payload response that is returned after a successful API call to the Scopus Citation API
    :return: A hash-map representing the number of citations an output has received in the years 2014-2020
    """
    try:
        if not data:
            return {}

        # Extract scopus ID
        scopus_id = (
            data.get("abstract-citations-response", {})
            .get("identifier-legend", {})
            .get("identifier", [{}])[0]
            .get("scopus_id", None)
        )

        # Parse citation data:
        citation_metrics = (
            data.get("abstract-citations-response", {})
            .get("citeInfoMatrix", {})
            .get("citeInfoMatrixXML", {})
            .get("citationMatrix", {})
        )

        # Extract citation counts
        citation_data = citation_metrics.get("cc", [])

        citation_counts = []
        for citation_year_count in citation_data:
            citation_counts.append(
                int(citation_year_count.get("$", 0))
            )

        # Extract total citation count (Alternatively could sum the citation_counts array, but its available as rangeCount)
        total_citations = int(
            citation_metrics.get("rangeCount", 0)
        )

        # Return a hash-map representing the number of citations an output has received in the years 2014-2020
        return {
            "scopus_id": scopus_id,
            "citation_counts_2014": citation_counts[0],
            "citation_counts_2015": citation_counts[1],
            "citation_counts_2016": citation_counts[2],
            "citation_counts_2017": citation_counts[3],
            "citation_counts_2018": citation_counts[4],
            "citation_counts_2019": citation_counts[5],
            "citation_counts_2020": citation_counts[6],
            "total_citations": total_citations
        }

    except (IndexError, KeyError, TypeError, ValueError) as e:
        # Handle potential JSON decoding errors
        print(f"Error decoding JSON response body: {e}")
        return None


def process_citation_metadata():
    """
    Obtain the DOIs of CS outputs, and return a DataFrame with each output and its citation counts using the Citation API
    :return: DataFrame containing citation counts of outputs submitted to the CS UoA
    """
    cs_doi_df = get_cs_doi_df()

    def process_doi(doi):
        """
        For a given output identified by its DOI, return a hash-map of its citation metrics with the DOI by calling an API
        :param doi: Unique identifier for an output
        :return: Hashmap containing citation metrics and the output's DOI
        """
        citation_data = get_citation_metadata(doi)
        parsed_data = extract_citation_metadata(citation_data)
        # In the extracted citation counts hash-map, include the output's DOI
        parsed_data["DOI"] = doi
        return parsed_data

    # Instead of using an array, first filter for all DOIs, use apply to make the obtain a hash-map of parsed
    # citation metrics by making API calls, and use another apply to transform these values as a series resulting in a
    # Pandas dataframe
    cs_citation_metadata_df = cs_doi_df["DOI"].apply(process_doi).apply(pd.Series)
    return cs_citation_metadata_df

def write_cs_citation_metadata_df(cs_citation_metadata_df):
    """
    Persist the dataframe containing citation counts of outputs as a parquet file
    :param cs_citation_metadata_df: DataFrame containing citation counts of outputs submitted to the CS UoA
    """
    cs_citation_metadata_df_path = os.path.join(
        os.path.dirname(__file__), "..", "..", DATASETS_DIR, REFINED_DIR, CS_CITATION_METRICS
    )

    cs_citation_metadata_df.to_parquet(cs_citation_metadata_df_path, engine='fastparquet')


if __name__ == "__main__":
    main()
