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

def extract_output_metadata(data):
    try:
        if not data:
            return {}

        extracted_metadata = {
            "field_weighted_citation_impact": None,
            "top_citation_percentile": None, # Here, null implies 100 (in the top 100% citation percentile)
            "field_weighted_views_impact": None
        }

        json_results = data.get("results", [{}])
        json_result = json_results[0] # We only query for one document at a time
        metrics = json_result.get("metrics", [])

        for metric in metrics:

            # Extract Field Weighted Citation Impact
            if metric.get("metricType") == "FieldWeightedCitationImpact":
                extracted_metadata["field_weighted_citation_impact"] = metric.get("value")

            # Extract Top Citation Percentile
            elif metric.get("metricType") == "OutputsInTopCitationPercentiles":
                percentile_values = metric.get("values", [])

                for is_top_percentile in percentile_values:
                    if is_top_percentile.get("value") == 1:
                        extracted_metadata["top_citation_percentile"] = is_top_percentile.get("threshold")
                        break

            # Extract Field Weighted Views Impact
            elif metric.get("metricType") == "FieldWeightedViewsImpact":
                extracted_metadata["field_weighted_views_impact"] = metric.get("value")

        return extracted_metadata

    except (IndexError, KeyError, TypeError, ValueError) as e:
        # Handle potential JSON decoding errors
        print(f"Error decoding JSON response body: {e}")
        return None

def process_output_metrics():
    cs_scopus_id_df = get_cs_scopus_id_df()

    def to_do_func():
        return {}

    cs_output_metrics_df = cs_scopus_id_df["scopus_id"].apply(to_do_func).apply(pd.Series)

    # for scopus_id in cs_scopus_id_df['scopus_id'].head():
    #     print(scopus_id)
    #     #data = get_output_metadata(scopus_id)
    #     #extract = extract_output_metadata(data)
    #     #print(extract)


configure()
elsevier_api_key = os.getenv('elsevier_api_key')

process_output_metrics()



