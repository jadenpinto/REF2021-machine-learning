import os
import requests
import pandas as pd
from time import sleep
from dotenv import load_dotenv

from utils.constants import DATASETS_DIR, PROCESSED_DIR, CS_JOURNALS_ISSN, REFINED_DIR, CS_JOURNAL_METRICS


def configure():
    load_dotenv()

def get_cs_journal_issns_df():
    cs_journal_ISSN_path = os.path.join(os.path.dirname(__file__), "..", "..", DATASETS_DIR, PROCESSED_DIR,
                                          CS_JOURNALS_ISSN)

    cs_journal_ISSN_df = pd.read_csv(cs_journal_ISSN_path)
    return cs_journal_ISSN_df

def get_serial_metadata(issn): # Serial Title API
    serial_title_metadata_url = f"https://api.elsevier.com/content/serial/title/issn/{issn}?apiKey={elsevier_api_key}&httpAccept=application/json&view=CITESCORE&date=2021-2021"

    try:
        response = requests.get(serial_title_metadata_url, timeout=10)
        response.raise_for_status()

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"Request failed with status code: {response.status_code}")
            print(response.text)  # Print the response text for debugging

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")

def extract_journal_metrics(data):
    try:
        if not data:
            return {}

        # Retrieve first entry
        entry = data.get("serial-metadata-response", {}).get("entry", [])[0]

        # Extract Scopus ID
        source_id = entry.get("source-id", None)

        # Extract SNIP
        snip_list = entry.get("SNIPList", {})
        if not snip_list:
            snip = None
        else:
            snip = snip_list.get("SNIP", [{}])[0].get("$", None)

        # Extract SJR
        sjr_list = entry.get("SJRList", {})
        if not sjr_list:
            sjr = None
        else:
            sjr = sjr_list.get("SJR", [{}])[0].get("$", None)

        # Extract CiteScore
        cite_score_year_info = entry.get("citeScoreYearInfoList", {}).get("citeScoreYearInfo", [{}])[0]
        cite_score = cite_score_year_info.get("citeScoreInformationList", [{}])[0].get("citeScoreInfo", [{}])[0].get("citeScore", None)

        return {"Scopus_ID": source_id, "SNIP": snip, "SJR": sjr, "Cite_Score": cite_score}

    except (IndexError, KeyError, TypeError, ValueError) as e:
        # Handle potential JSON decoding errors
        print(f"Error decoding JSON response body: {e}")
        return None

def process_journal_metrics():
    cs_journal_ISSN_df = get_cs_journal_issns_df()

    journal_metrics = []

    for issn in cs_journal_ISSN_df['ISSN']:
        try:
            journal_metrics_json_response = get_serial_metadata(issn)

            journal_metrics_metadata = extract_journal_metrics(journal_metrics_json_response)
            journal_metrics_metadata['ISSN'] = issn

            journal_metrics.append(journal_metrics_metadata)
            sleep(0.01)

        except Exception as e:
            print(f"Error processing ISSN {issn}: {str(e)}")

            journal_metrics.append({
                'ISSN': issn,
                'Scopus_ID': None,
                'SNIP': None,
                'SJR': None,
                'Cite_Score': None
            })

    # Convert results to DataFrame
    journal_metrics_df = pd.DataFrame(journal_metrics)
    # Ensure consistent column order
    journal_metrics_df = journal_metrics_df[['ISSN', 'Scopus_ID', 'SNIP', 'SJR', 'Cite_Score']]

    return journal_metrics_df

def write_cs_journal_metrics_df(cs_journal_metrics_df):
    cs_journal_metrics_df_path = os.path.join(os.path.dirname(__file__), "..", "..", DATASETS_DIR, REFINED_DIR,
                                            CS_JOURNAL_METRICS)

    cs_journal_metrics_df.to_parquet(cs_journal_metrics_df_path, engine='fastparquet')

def load_cs_journal_metrics_df():
    cs_journal_metrics_df_path = os.path.join(os.path.dirname(__file__), "..", "..", DATASETS_DIR, REFINED_DIR,
                                              CS_JOURNAL_METRICS)
    cs_journal_metrics_df = pd.read_parquet(cs_journal_metrics_df_path, engine='fastparquet')
    return cs_journal_metrics_df

def log_null_cs_journal_metadata(loaded_cs_journal_metrics_df):
    print("CS journal metadata fields with null counts:")
    print(loaded_cs_journal_metrics_df.isna().sum())
    """
    ISSN            0
    Scopus_ID      96
    SNIP          129
    SJR           143
    Cite_Score    143
    """

    null_scopus_id_cs_journal_metadata_df = loaded_cs_journal_metrics_df[
        loaded_cs_journal_metrics_df['Scopus_ID'].isna()
    ]
    print("Examples of records where Scopus ID is null:")
    print(null_scopus_id_cs_journal_metadata_df.head().to_string())
    # For these records, making an API call to the endpoint with their ISSNs resulted in a 404 RESOURCE_NOT_FOUND

    cs_journal_metadata_df_with_scopus_id_null_snip = loaded_cs_journal_metrics_df[
        ~loaded_cs_journal_metrics_df['Scopus_ID'].isna() & loaded_cs_journal_metrics_df['SNIP'].isna()
    ]
    print("Example of records where Scopus ID is valid, and SNIP is null:")
    print(cs_journal_metadata_df_with_scopus_id_null_snip.head().to_string())
    # In such records, SNIPList (and other fields like SJRList and citeScoreYearInfoList) either absent or set to null


configure()
elsevier_api_key = os.getenv('elsevier_api_key')

cs_journal_metrics_df = process_journal_metrics()
write_cs_journal_metrics_df(cs_journal_metrics_df)

loaded_cs_journal_metrics_df = load_cs_journal_metrics_df()
log_null_cs_journal_metadata(loaded_cs_journal_metrics_df)
