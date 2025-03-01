import os
import requests
import pandas as pd
from time import sleep
from dotenv import load_dotenv

from utils.constants import DATASETS_DIR, PROCESSED_DIR, CS_OUTPUTS_METADATA, REFINED_DIR, CS_CITATION_METRICS


def configure():
    load_dotenv()

def get_cs_outputs_metadata():
    processed_cs_outputs_metadata_path = os.path.join(os.path.dirname(__file__), "..", "..", DATASETS_DIR, PROCESSED_DIR,
                                           CS_OUTPUTS_METADATA)

    cs_outputs_metadata = pd.read_csv(processed_cs_outputs_metadata_path)
    return cs_outputs_metadata

def get_cs_doi_df():
    cs_outputs_df = get_cs_outputs_metadata()
    cs_doi_df = cs_outputs_df[["DOI"]].drop_duplicates().dropna()
    return cs_doi_df

def get_citation_metadata(doi):
    citation_metadata_base_url = f"https://api.elsevier.com/content/abstract/citations"
    citation_metadata_url_params = {
        "apiKey": elsevier_api_key,
        "doi": doi,
        "httpAccept": "application/json",
        "sort": "+sort-year",
        "date": "2014-2020",
        "field": "scopus_id,cc,rangeCount"
    }

    try:
        response = requests.get(citation_metadata_base_url, params=citation_metadata_url_params, timeout=10)
        response.raise_for_status()

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"Request failed with status code: {response.status_code}")
            print(response.text)  # Print the response text for debugging

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")

def extract_citation_metadata(data):
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

        # Extract total citation count
        total_citations = int(
            citation_metrics.get("rangeCount", 0)
        )

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
    cs_doi_df = get_cs_doi_df()

    def process_doi(doi):
        citation_data = get_citation_metadata(doi)
        parsed_data = extract_citation_metadata(citation_data)
        parsed_data["DOI"] = doi
        return parsed_data

    cs_citation_metadata_df = cs_doi_df["DOI"].apply(process_doi).apply(pd.Series)
    return cs_citation_metadata_df

def write_cs_citation_metadata_df(cs_citation_metadata_df):
    cs_citation_metadata_df_path = os.path.join(
        os.path.dirname(__file__), "..", "..", DATASETS_DIR, REFINED_DIR, CS_CITATION_METRICS
    )

    cs_citation_metadata_df.to_parquet(cs_citation_metadata_df_path, engine='fastparquet')

configure()
elsevier_api_key = os.getenv('elsevier_api_key')

cs_citation_metadata_df = process_citation_metadata()
write_cs_citation_metadata_df(cs_citation_metadata_df)

# todo move this to different file, possible extract 3 metrics from SciVal publication API.
def load_cs_citation_metadata_df():
    cs_citation_metadata_df_path = os.path.join(
        os.path.dirname(__file__), "..", "..", DATASETS_DIR, REFINED_DIR, CS_CITATION_METRICS
    )
    cs_citation_metadata_df = pd.read_parquet(cs_citation_metadata_df_path, engine='fastparquet')
    return cs_citation_metadata_df

cs_citation_metadata_df = load_cs_citation_metadata_df()
print(cs_citation_metadata_df.tail(10).to_string())
print(cs_citation_metadata_df.shape)
