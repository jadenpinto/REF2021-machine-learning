import os
import requests
import pandas as pd
from time import sleep
from dotenv import load_dotenv

from utils.constants import DATASETS_DIR, PROCESSED_DIR, CS_OUTPUTS_METADATA, REFINED_DIR, CS_CITATION_METRICS


def configure():
    load_dotenv()

def load_cs_citation_metadata_df():
    cs_citation_metadata_df_path = os.path.join(
        os.path.dirname(__file__), "..", "..", DATASETS_DIR, REFINED_DIR, CS_CITATION_METRICS
    )
    cs_citation_metadata_df = pd.read_parquet(cs_citation_metadata_df_path, engine='fastparquet')
    return cs_citation_metadata_df

def get_cs_scopus_id_df():
    cs_citation_metadata_df = load_cs_citation_metadata_df() # (6467, 10)
    cs_scopus_id_df = cs_citation_metadata_df[["scopus_id"]].drop_duplicates().dropna() # (6259, 1)
    return cs_scopus_id_df

def get_output_metadata(scopus_id): # FieldWeightedCitationImpact & OutputsInTopCitationPercentiles
    output_metadata_base_url = "https://api.elsevier.com/analytics/scival/publication/metrics"
    output_metadata_url_params = {
        "metricTypes": "FieldWeightedCitationImpact,OutputsInTopCitationPercentiles,FieldWeightedViewsImpact",
        "showAsFieldWeighted": "true",
        "publicationIds": scopus_id,
        "yearRange": "10yrs",
        "byYear": "false",
        "apiKey": elsevier_api_key,
        "httpAccept": "application/json"
    }

    try:
        response = requests.get(output_metadata_base_url, params=output_metadata_url_params, timeout=10)
        response.raise_for_status()

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"Request failed with status code: {response.status_code}")
            print(response.text)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")



def process_citation_metrics():
    cs_scopus_id_df = get_cs_scopus_id_df()

configure()
elsevier_api_key = os.getenv('elsevier_api_key')

process_citation_metrics()




