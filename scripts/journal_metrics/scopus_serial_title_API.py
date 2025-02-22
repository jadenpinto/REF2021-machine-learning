import requests
import os
import requests
import pandas as pd

from dotenv import load_dotenv

from utils.constants import DATASETS_DIR, PROCESSED_DIR, CS_JOURNALS_ISSN


def configure():
    load_dotenv()

def get_cs_journal_issns_df():
    cs_journal_ISSN_path = os.path.join(os.path.dirname(__file__), "..", "..", DATASETS_DIR, PROCESSED_DIR,
                                          CS_JOURNALS_ISSN)

    cs_journal_ISSN_df = pd.read_csv(cs_journal_ISSN_path)
    return cs_journal_ISSN_df

def get_serial_metadata(issn, elsevier_api_key):
    serial_title_metadata_url = f"https://api.elsevier.com/content/serial/title/issn/{issn}?apiKey={elsevier_api_key}&httpAccept=application/json"

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
    except ValueError as e:
        print(f"Error decoding JSON: {e}")  # Handle potential JSON decoding errors


# serial_title_api()
configure()
elsevier_api_key = os.getenv('elsevier_api_key')

# cs_journal_ISSN_df = get_cs_journal_issns_df()
# print(cs_journal_ISSN_df.head().to_string())

data = get_serial_metadata("1064-2307", elsevier_api_key)
print(data)

"""
Scopus API Spec: https://dev.elsevier.com/sc_api_spec.html
Serial Title API: https://dev.elsevier.com/documentation/SerialTitleAPI.wadl
Serial Title API Views: https://dev.elsevier.com/sc_serial_title_views.html
Serial Title Metadata Interactive API: https://dev.elsevier.com/scopus.html#!/Serial_Title/SerialTitleMetadata

JSON response example: https://dev.elsevier.com/payloads/metadata/serialTitleResp.json
"""

# Get SNIP [standard, enhanced view]
# cites score is same [may need CITESCORE view]

# if using date argument - Date range for filtering info entries in EHANCED view, with the lowest granularity being year.
# year would be 2021




