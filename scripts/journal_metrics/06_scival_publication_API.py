import os
import requests
import pandas as pd
from time import sleep
from dotenv import load_dotenv

from utils.constants import DATASETS_DIR, PROCESSED_DIR, CS_JOURNALS_ISSN, REFINED_DIR, CS_JOURNAL_METRICS

def configure():
    load_dotenv()


def load_cs_journal_metrics_df():
    cs_journal_metrics_df_path = os.path.join(os.path.dirname(__file__), "..", "..", DATASETS_DIR, REFINED_DIR,
                                              CS_JOURNAL_METRICS)
    cs_journal_metrics_df = pd.read_parquet(cs_journal_metrics_df_path, engine='fastparquet')
    return cs_journal_metrics_df


def get_valid_scopus_id_df(cs_journal_metrics_df):
    scopus_id_df = cs_journal_metrics_df[
        ~cs_journal_metrics_df['Scopus_ID'].isna()
    ]
    return scopus_id_df


def get_journal_citation_metadata(scopus_publication_id): # FieldWeightedCitationImpact & OutputsInTopCitationPercentiles
    citation_metadata_base_url = "https://api.elsevier.com/analytics/scival/publication/metrics"
    citation_metadata_url_params = {
        "metricTypes": "FieldWeightedCitationImpact,OutputsInTopCitationPercentiles",
        "showAsFieldWeighted": "true",
        "publicationIds": scopus_publication_id,
        "apiKey": elsevier_api_key,
        "httpAccept": "application/json"
    }

    try:
        response = requests.get(citation_metadata_base_url, params=citation_metadata_url_params, timeout=10)
        response.raise_for_status()

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"Request failed with status code: {response.status_code}")
            print(response.text)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")


def count_citation_metrics(scopus_id_df, citation_metric):
    citation_metric_count = 0

    for scopus_id in scopus_id_df['Scopus_ID']:
        journal_citation_metadata = get_journal_citation_metadata(scopus_id)

        if not journal_citation_metadata:
            continue

        if "results" not in journal_citation_metadata:
            continue
        results_list = journal_citation_metadata["results"]
        if not results_list:
            continue

        results = results_list[0]

        if "metrics" not in results:
            continue
        metrics_list = results["metrics"]
        if not metrics_list:
            continue

        field_weighted_citation_impact, outputs_in_top_citation_percentiles = metrics_list[0], metrics_list[1]

        if citation_metric == "OutputsInTopCitationPercentiles":
            if not outputs_in_top_citation_percentiles:
                continue

            outputs_in_top_citation_percentiles_values_list = outputs_in_top_citation_percentiles["values"]
            if not outputs_in_top_citation_percentiles_values_list:
                continue

            outputs_in_top_citation_percentiles_values_list_one_threshold = outputs_in_top_citation_percentiles_values_list[0]
            citation_metric_values_by_year = outputs_in_top_citation_percentiles_values_list_one_threshold["valueByYear"]

        elif citation_metric == "FieldWeightedCitationImpact":
            if not field_weighted_citation_impact:
                continue

            citation_metric_values_by_year = field_weighted_citation_impact["valueByYear"]

        if not citation_metric_values_by_year:
            continue

        for year, value in citation_metric_values_by_year.items():
            if value is not None:
                print(f"Scopus ID of the publication where {citation_metric} exists = {results["publication"]["id"]}")
                citation_metric_count += 1
                break

    print(f"The number of journals containing OutputsInTopCitationPercentiles field = {citation_metric_count} (out of {scopus_id_df.shape[0]} journals)")


def get_journal_views_metadata(scopus_publication_id): # FieldWeightedViewsImpact
    views_metadata_base_url = "https://api.elsevier.com/analytics/scival/publication/metrics"
    views_metadata_url_params = {
        "metricTypes": "FieldWeightedViewsImpact",
        "showAsFieldWeighted": "true",
        "byYear": "false",
        "publicationIds": scopus_publication_id,
        "apiKey": elsevier_api_key,
        "httpAccept": "application/json"
    }

    try:
        response = requests.get(views_metadata_base_url, params=views_metadata_url_params, timeout=10)
        response.raise_for_status()

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"Request failed with status code: {response.status_code}")
            print(response.text)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")


def count_views_metrics(scopus_id_df):
    view_metric_count = 0

    for scopus_id in scopus_id_df['Scopus_ID']:
        journal_views_metadata = get_journal_views_metadata(scopus_id)

        if not journal_views_metadata:
            continue

        if "results" not in journal_views_metadata:
            continue
        results_list = journal_views_metadata["results"]
        if not results_list:
            continue

        results = results_list[0]

        if "metrics" not in results:
            continue
        metrics_list = results["metrics"]
        if not metrics_list:
            continue

        field_weighted_views_impact = metrics_list[0]
        if not field_weighted_views_impact:
            continue

        field_weighted_views_impact_value = field_weighted_views_impact["value"]
        if field_weighted_views_impact_value is not None:
            print(f"Scopus ID of the publication where FieldWeightedViewsImpact exists = {results["publication"]["id"]}")
            view_metric_count += 1

    print(f"The number of journals containing FieldWeightedViewsImpact field = {view_metric_count} (out of {scopus_id_df.shape[0]} journals)")


configure()
elsevier_api_key = os.getenv('elsevier_api_key')

cs_journal_metrics_df = load_cs_journal_metrics_df()
scopus_id_df = get_valid_scopus_id_df(cs_journal_metrics_df)

count_citation_metrics(scopus_id_df, "OutputsInTopCitationPercentiles")
"""
Scopus ID of the publication where OutputsInTopCitationPercentiles exists = 29954
The number of journals containing OutputsInTopCitationPercentiles field = 1 (out of 1187 journals)
"""

count_citation_metrics(scopus_id_df, "FieldWeightedCitationImpact")
"""
Scopus ID of the publication where FieldWeightedCitationImpact exists = 29954
The number of journals containing OutputsInTopCitationPercentiles field = 1 (out of 1187 journals)
"""



"""
def extract_citation_data(data):
    if not data:
        return

    if "results" not in data:
        return
    results = data["results"]
    if not results:
        return

    results = data["results"][0]

    if "metrics" not in results:
        return
    metrics = results["metrics"]
    if not metrics:
        return

    FieldWeightedCitationImpact, OutputsInTopCitationPercentiles = metrics[0], metrics[1]
    if not FieldWeightedCitationImpact or not OutputsInTopCitationPercentiles:
        return

    OutputsInTopCitationPercentiles_values = OutputsInTopCitationPercentiles["values"][0]
    valueByYear = OutputsInTopCitationPercentiles_values["valueByYear"]

    for k,v in valueByYear.items():
        if v!=None:
            print(k, v, data["results"][0]["publication"]["id"])

    # return valueByYear


# print(cs_journal_metrics_df.shape)
# print(cs_journal_metrics_df.head().to_string())
# print(cs_journal_metrics_df.isna().sum())

# x = cs_journal_metrics_df[
#         ~cs_journal_metrics_df['Scopus_ID'].isna()
#     ]
# # print(x.shape)
# print(x.head(10).to_string())


#cs_journal_metrics_df = process_journal_metrics()
#write_cs_journal_metrics_df(cs_journal_metrics_df)


# get_journal_citation_metadata(27433)

# Tried all records
# Only printed: 2023 0 29954
"""




# todo -> make requests for the latest
"""
These are articles and not journals:
https://api.elsevier.com/analytics/scival/publication/metrics?metricTypes=FieldWeightedViewsImpact&publicationIds=85047328215&showAsFieldWeighted=true&apiKey=7f59af901d2d86f78a1fd60c1bf9426a&byYear=false&httpAccept=application/json
"""
