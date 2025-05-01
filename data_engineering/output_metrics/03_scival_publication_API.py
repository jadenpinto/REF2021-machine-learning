import os
import requests
import pandas as pd
from dotenv import load_dotenv

from utils.constants import DATASETS_DIR, REFINED_DIR, CS_CITATION_METRICS, CS_OUTPUT_METRICS
from utils.API import check_api_quota


def main():
    configure()

    global elsevier_api_key
    elsevier_api_key = os.getenv('elsevier_api_key')

    cs_scopus_id_df = get_cs_scopus_id_df()
    cs_output_metrics_df = process_output_metrics(cs_scopus_id_df)
    write_cs_output_metrics_df(cs_output_metrics_df)

    retry_process_output_metrics()

    check_api_quota(
        elsevier_api_key,
        "https://api.elsevier.com/analytics/scival/publication/metrics?metricTypes=FieldWeightedCitationImpact&publicationIds=85021243138"
    )

    """
    cs_output_metrics_df = load_cs_output_metrics_df()
    print(cs_output_metrics_df.isna().sum())

    # Before handling failed API calls:

    field_weighted_citation_impact    1273
    top_citation_percentile           3212
    field_weighted_views_impact       1273
    scopus_id                            0
    dtype: int64

    # After retrying failed API calls:
    field_weighted_citation_impact       0
    top_citation_percentile           2410
    field_weighted_views_impact          0
    scopus_id                            0
    dtype: int64
    """

def configure():
    """"
    Configure the API Key - read the environment file, and load it as an environment variable
    """
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

def get_output_metadata(scopus_id): # Retrieve Field Weighted Publication metrics
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
        response = requests.get(output_metadata_base_url, params=output_metadata_url_params, timeout=100)
        response.raise_for_status()

        if response.status_code == 200:
            # If the HTTP status code is 200 OK, it means the API call was successful, so retrieve the JSON response
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

def process_output_metrics(cs_scopus_id_df):

    def process_scopus_id(scopus_id):
        output_metadata = get_output_metadata(scopus_id)
        parsed_output_data = extract_output_metadata(output_metadata)
        parsed_output_data["scopus_id"] = scopus_id
        return parsed_output_data

    cs_output_metrics_df = cs_scopus_id_df["scopus_id"].apply(process_scopus_id).apply(pd.Series)
    return cs_output_metrics_df

def write_cs_output_metrics_df(cs_output_metrics_df):
    cs_output_metrics_df_path = os.path.join(
        os.path.dirname(__file__), "..", "..", DATASETS_DIR, REFINED_DIR, CS_OUTPUT_METRICS
    )

    cs_output_metrics_df.to_parquet(cs_output_metrics_df_path, engine='fastparquet')

def load_cs_output_metrics_df():
    cs_output_metrics_df_path = os.path.join(
        os.path.dirname(__file__), "..", "..", DATASETS_DIR, REFINED_DIR, CS_OUTPUT_METRICS
    )
    cs_output_metrics_df = pd.read_parquet(cs_output_metrics_df_path, engine='fastparquet')
    return cs_output_metrics_df

def retry_process_output_metrics():
    """
    Running process_output_metrics() initially threw a rate limit exceeded error after a few successful requests
    So this method is run when the API limits are reset to process the failed records
    It makes calls to the SciVal Publication API for failed requests and rewrites to CS_Output_Metrics.parquet file
    """
    original_cs_output_metrics_df = load_cs_output_metrics_df()

    records_with_failed_api_calls = original_cs_output_metrics_df['field_weighted_citation_impact'].isna()
    succeeded_scival_api_df = original_cs_output_metrics_df[~records_with_failed_api_calls]
    failed_scival_api_df = original_cs_output_metrics_df[records_with_failed_api_calls]

    failed_cs_scopus_id_df = failed_scival_api_df[["scopus_id"]].drop_duplicates().dropna() # 1273 records failed
    retried_cs_output_metrics_df = process_output_metrics(failed_cs_scopus_id_df)

    updated_cs_output_metrics_df = pd.concat([succeeded_scival_api_df, retried_cs_output_metrics_df])
    write_cs_output_metrics_df(updated_cs_output_metrics_df)


if __name__ == "__main__":
    main()
