from unittest.mock import patch, Mock, MagicMock
import pandas as pd
import importlib.util
import sys
from pathlib import Path

# Path to project root
project_root = Path(__file__).resolve().parent.parent.parent
# Path to script containing functions to test
module_path = project_root / "data_engineering" / "journal_metrics" / "02_scopus_serial_title_API.py"
# Rename module so it does not start with a number
module_name = "scopus_serial_api"

# Load module dynamically
spec = importlib.util.spec_from_file_location(module_name, module_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)

# Access functions
get_serial_metadata = module.get_serial_metadata
extract_journal_metrics = module.extract_journal_metrics
process_journal_metrics = module.process_journal_metrics

# Set up dummy API key
module.elsevier_api_key = "ABCDE1234"


def test_extract_journal_metrics():
    sample_journal_metrics_json_response_payload = {'serial-metadata-response': {'link': [{'@_fa': 'true', '@ref': 'self', '@href': 'https://api.elsevier.com/content/serial/title/issn/17416485', '@type': 'application/json'}], 'entry': [{'@_fa': 'true', 'dc:title': 'Journal of Information Science', 'dc:publisher': 'SAGE Publications Ltd', 'coverageStartYear': '1979', 'coverageEndYear': '2025', 'prism:aggregationType': 'journal', 'source-id': '12813', 'prism:issn': '0165-5515', 'prism:eIssn': '1741-6485', 'openaccess': None, 'openaccessArticle': None, 'openArchiveArticle': None, 'openaccessType': None, 'openaccessStartDate': None, 'oaAllowsAuthorPaid': None, 'subject-area': [{'@_fa': 'true', '@code': '1710', '@abbrev': 'COMP', '$': 'Information Systems'}, {'@_fa': 'true', '@code': '3309', '@abbrev': 'SOCI', '$': 'Library and Information Sciences'}], 'SNIPList': {'SNIP': [{'@_fa': 'true', '@year': '2021', '$': '1.684'}]}, 'SJRList': {'SJR': [{'@_fa': 'true', '@year': '2021', '$': '0.761'}]}, 'citeScoreYearInfoList': {'citeScoreCurrentMetric': '6.8', 'citeScoreCurrentMetricYear': '2023', 'citeScoreTracker': '7.8', 'citeScoreTrackerYear': '2024', 'citeScoreYearInfo': [{'@_fa': 'true', '@year': '2021', '@status': 'Complete', 'citeScoreInformationList': [{'@_fa': 'true', 'citeScoreInfo': [{'@_fa': 'true', 'docType': 'all', 'scholarlyOutput': '210', 'citationCount': '1441', 'citeScore': '6.9', 'percentCited': '86', 'citeScoreSubjectRank': [{'@_fa': 'true', 'subjectCode': '3309', 'rank': '17', 'percentile': '93'}, {'@_fa': 'true', 'subjectCode': '1710', 'rank': '67', 'percentile': '81'}]}]}]}]}, 'link': [{'@_fa': 'true', '@ref': 'scopus-source', '@href': 'https://www.scopus.com/source/sourceInfo.url?sourceId=12813'}, {'@_fa': 'true', '@ref': 'homepage', '@href': ''}, {'@_fa': 'true', '@ref': 'coverimage', '@href': 'https://api.elsevier.com/content/serial/title/issn/0165-5515?view=coverimage'}], 'prism:url': 'https://api.elsevier.com/content/serial/title/issn/0165-5515'}]}}
    expected_extracted_journal_metrics_metadata = {'Scopus_ID': '12813', 'SNIP': '1.684', 'SJR': '0.761', 'Cite_Score': '6.9'}

    actual_extracted_journal_metrics_metadata = extract_journal_metrics(sample_journal_metrics_json_response_payload)

    assert isinstance(actual_extracted_journal_metrics_metadata, dict)
    assert expected_extracted_journal_metrics_metadata == actual_extracted_journal_metrics_metadata

def test_extract_empty_journal_metrics():
    sample_journal_metrics_json_response_payload = {}
    expected_extracted_journal_metrics_metadata = {}

    actual_extracted_journal_metrics_metadata = extract_journal_metrics(sample_journal_metrics_json_response_payload)

    assert expected_extracted_journal_metrics_metadata == actual_extracted_journal_metrics_metadata

def test_extract_journal_metrics_with_none():
    assert extract_journal_metrics(None) == {}


def test_process_journal_metrics():
    sample_issn_df = pd.DataFrame({
        'ISSN': ['1741-6485']
    })

    sample_metadata_response = {
        'Scopus_ID': '12813', 'SNIP': '1.684', 'SJR': '0.761', 'Cite_Score': '6.9'
    }

    expected_df = pd.DataFrame([
        {
            'ISSN': '1741-6485',
            'Scopus_ID': '12813',
            'SNIP': '1.684',
            'SJR': '0.761',
            'Cite_Score': '6.9'
        }
    ])

    # Patch internal function calls on the module object
    # Patch functions get_serial_metadata and extract_journal_metrics on the module object instead of string
    # Module references the loaded module, so it should be mocked instead
    with patch.object(module, "get_cs_journal_issns_df", return_value=sample_issn_df), \
         patch.object(module, "get_serial_metadata", return_value=sample_metadata_response), \
         patch.object(module, "extract_journal_metrics", side_effect=lambda x: x):

        result_df = module.process_journal_metrics() # Actual result df

    # Assert resultant dataframe matches the expected dataframe
    pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df.reset_index(drop=True))


def test_get_serial_metadata():
    sample_issn = "1741-6485"
    sample_response_data = {'serial-metadata-response': {'link': [{'@_fa': 'true', '@ref': 'self', '@href': 'https://api.elsevier.com/content/serial/title/issn/17416485', '@type': 'application/json'}], 'entry': [{'@_fa': 'true', 'dc:title': 'Journal of Information Science', 'dc:publisher': 'SAGE Publications Ltd', 'coverageStartYear': '1979', 'coverageEndYear': '2025', 'prism:aggregationType': 'journal', 'source-id': '12813', 'prism:issn': '0165-5515', 'prism:eIssn': '1741-6485', 'openaccess': None, 'openaccessArticle': None, 'openArchiveArticle': None, 'openaccessType': None, 'openaccessStartDate': None, 'oaAllowsAuthorPaid': None, 'subject-area': [{'@_fa': 'true', '@code': '1710', '@abbrev': 'COMP', '$': 'Information Systems'}, {'@_fa': 'true', '@code': '3309', '@abbrev': 'SOCI', '$': 'Library and Information Sciences'}], 'SNIPList': {'SNIP': [{'@_fa': 'true', '@year': '2021', '$': '1.684'}]}, 'SJRList': {'SJR': [{'@_fa': 'true', '@year': '2021', '$': '0.761'}]}, 'citeScoreYearInfoList': {'citeScoreCurrentMetric': '6.8', 'citeScoreCurrentMetricYear': '2023', 'citeScoreTracker': '7.8', 'citeScoreTrackerYear': '2024', 'citeScoreYearInfo': [{'@_fa': 'true', '@year': '2021', '@status': 'Complete', 'citeScoreInformationList': [{'@_fa': 'true', 'citeScoreInfo': [{'@_fa': 'true', 'docType': 'all', 'scholarlyOutput': '210', 'citationCount': '1441', 'citeScore': '6.9', 'percentCited': '86', 'citeScoreSubjectRank': [{'@_fa': 'true', 'subjectCode': '3309', 'rank': '17', 'percentile': '93'}, {'@_fa': 'true', 'subjectCode': '1710', 'rank': '67', 'percentile': '81'}]}]}]}]}, 'link': [{'@_fa': 'true', '@ref': 'scopus-source', '@href': 'https://www.scopus.com/source/sourceInfo.url?sourceId=12813'}, {'@_fa': 'true', '@ref': 'homepage', '@href': ''}, {'@_fa': 'true', '@ref': 'coverimage', '@href': 'https://api.elsevier.com/content/serial/title/issn/0165-5515?view=coverimage'}], 'prism:url': 'https://api.elsevier.com/content/serial/title/issn/0165-5515'}]}}
    expected_base_url = f"https://api.elsevier.com/content/serial/title/issn/{sample_issn}"
    expected_params = {
        "apiKey": module.elsevier_api_key,
        "httpAccept": "application/json",
        "view": "CITESCORE",
        "date": "2021-2021"
    }

    # Mock the response:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = sample_response_data
    mock_response.raise_for_status.return_value = None

    # Test the function with mocked requests.get
    with patch("requests.get", return_value=mock_response) as mock_get:
        result = module.get_serial_metadata(sample_issn)

    # Check requests.get was called
    mock_get.assert_called_once()
    # Parse and assert the call arguments on the mock call
    args, kwargs = mock_get.call_args
    assert args[0] == expected_base_url
    assert kwargs["params"] == expected_params
    assert kwargs["timeout"] == 10

    # Assert mock response was called
    assert mock_response.raise_for_status.called
    assert mock_response.json.called

    # Check the (mocked) result returned from the (mocked) API call matches the expected result
    assert result == sample_response_data
    assert "serial-metadata-response" in result
    assert "entry" in result["serial-metadata-response"]
