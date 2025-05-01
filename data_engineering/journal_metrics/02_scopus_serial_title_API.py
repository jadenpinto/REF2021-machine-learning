import os
import requests
import pandas as pd
from time import sleep
from dotenv import load_dotenv

from utils.constants import DATASETS_DIR, PROCESSED_DIR, CS_JOURNALS_ISSN, REFINED_DIR, CS_JOURNAL_METRICS

def main():
    """
    ETL Pipeline that:
        1. Gets the ISSN of all journals of the outputs submitted the CS UoA
        2. For all journals, make API calls to retrieve their Scopus ID, SNIP, SJR, and Cite Score
        3. Persist these journal metrics as a parquet file
    """
    # Securely retrieve API key:
    configure()

    global elsevier_api_key
    elsevier_api_key = os.getenv('elsevier_api_key')

    cs_journal_metrics_df = process_journal_metrics()
    write_cs_journal_metrics_df(cs_journal_metrics_df)

def configure():
    """"
    Configure the API Key - read the environment file, and load it as an environement variable
    """
    load_dotenv()

def get_cs_journal_issns_df():
    """
    Read in the file containing the ISSNs of all journals of output submissions to the CS UoA as a dataframe
    :return: Dataframe containing the ISSNs of all journals of output submissions to the CS UoA
    """
    cs_journal_ISSN_path = os.path.join(os.path.dirname(__file__), "..", "..", DATASETS_DIR, PROCESSED_DIR,
                                          CS_JOURNALS_ISSN)

    cs_journal_ISSN_df = pd.read_csv(cs_journal_ISSN_path)
    return cs_journal_ISSN_df

def get_serial_metadata(issn): # Serial Title API
    """
    Make an API call to the Scopus Serial TItle API to retrieve journal metrics
    :param issn: The ISSN (unique identifier) of a journal
    :return: JSON payload response containing journal metadata
    """
    serial_title_metadata_base_url = f"https://api.elsevier.com/content/serial/title/issn/{issn}"
    serial_title_metadata_url_params = {
        "apiKey": elsevier_api_key,
        "httpAccept": "application/json",
        "view": "CITESCORE",
        "date": "2021-2021"
    }

    try:
        response = requests.get(serial_title_metadata_base_url, params=serial_title_metadata_url_params, timeout=10)
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

def extract_journal_metrics(data):
    """
    Extract the journal metrics (Scopus ID, SNIP, SJR, and Cite Score) from the payload response after successful request
    :param data: JSON response returned by the Scopus Serial Title API
    :return: A hashmap with journal metrics: Scopus ID, SNIP, SJR, and Cite Score
    """
    try:
        if not data:
            return {}

        # Retrieve first entry - all metadata is present in this entry
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

        # Return a hashmap where the key is the Journal Metric name as a string, and its int value is the hash map's value
        return {"Scopus_ID": source_id, "SNIP": snip, "SJR": sjr, "Cite_Score": cite_score}

    except (IndexError, KeyError, TypeError, ValueError) as e:
        # Handle potential JSON decoding errors
        print(f"Error decoding JSON response body: {e}")
        return None

def process_journal_metrics():
    """
    Obtain all CS journals, using their ISSNs create new columns for their journal metrics using the Serial Title API
    :return: A DataFrame each journal (identified by its ISSN) has fields for journal metrics:
             Scopus ID, SNIP, SJR, and Cite Score
    """
    cs_journal_ISSN_df = get_cs_journal_issns_df()

    journal_metrics = [] # List of each journal and its journal metrics

    for issn in cs_journal_ISSN_df['ISSN']:
        try:
            journal_metrics_json_response = get_serial_metadata(issn)

            journal_metrics_metadata = extract_journal_metrics(journal_metrics_json_response)
            # In the extracted journal metrics hash-map, include the journal's ISSN
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

    # Convert the list containing the journal metrics to DataFrame
    journal_metrics_df = pd.DataFrame(journal_metrics)
    # Ensure order of columns is fixed
    journal_metrics_df = journal_metrics_df[['ISSN', 'Scopus_ID', 'SNIP', 'SJR', 'Cite_Score']]

    return journal_metrics_df

def write_cs_journal_metrics_df(cs_journal_metrics_df):
    """
    Write the dataframe containing all CS journals with their metrics as a parquet file
    :param cs_journal_metrics_df: DataFrame containing all CS journals with their metrics:
            ISSN, Scopus_ID, SNIP, SJR, Cite_Score
    """
    cs_journal_metrics_df_path = os.path.join(os.path.dirname(__file__), "..", "..", DATASETS_DIR, REFINED_DIR,
                                            CS_JOURNAL_METRICS)

    cs_journal_metrics_df.to_parquet(cs_journal_metrics_df_path, engine='fastparquet')


if __name__ == "__main__":
    main()
