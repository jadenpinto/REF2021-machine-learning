from unittest.mock import patch, MagicMock
import pandas as pd
import importlib.util
import sys
from pathlib import Path
import pytest

# Path to project root
project_root = Path(__file__).resolve().parent.parent.parent
# Path to script containing functions to test
module_path = project_root / "data_engineering" / "output_metrics" / "03_scival_publication_API.py"
# Rename module so it does not start with a number
module_name = "scival_publication_api"

# Load module dynamically
spec = importlib.util.spec_from_file_location(module_name, module_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)

# Access functions
get_output_metadata = module.get_output_metadata
extract_output_metadata = module.extract_output_metadata
process_output_metrics = module.process_output_metrics

# Set up dummy API key
module.elsevier_api_key = "ABCDE1234"

# Test set-up, create resources like dataframes and JSON hashmaps that will be used in multiple tests
@pytest.fixture
def sample_scopus_id_df():
    """
    Provides a sample DataFrame containing a single scopus ID used for multiple tests
    """
    return pd.DataFrame({
        'scopus_id': ['85021243138']
    })

@pytest.fixture
def sample_scival_output_metrics_api_payload():
    """
    Provides a sample JSON payload returned by the SciVal API for the scopus ID: 85021243138
    """
    return {'link': {'@ref': 'self', '@href': 'https://api.elsevier.com/analytics/scival/publication/metrics?metricTypes=FieldWeightedCitationImpact,OutputsInTopCitationPercentiles,FieldWeightedViewsImpact&publicationIds=85021243138&showAsFieldWeighted=true&apiKey=8b4bdd23def172d47b8e3df79f7baff7&yearRange=10yrs&httpAccept=application/json&byYear=false', '@type': 'application/json'}, 'dataSource': {'sourceName': 'Scopus', 'lastUpdated': '2025-04-16', 'metricStartYear': 2014, 'metricEndYear': 2023}, 'results': [{'metrics': [{'metricType': 'FieldWeightedCitationImpact', 'value': 1.99}, {'metricType': 'OutputsInTopCitationPercentiles', 'values': [{'threshold': 1, 'value': 0, 'percentage': 0.0}, {'threshold': 5, 'value': 0, 'percentage': 0.0}, {'threshold': 10, 'value': 0, 'percentage': 0.0}, {'threshold': 25, 'value': 1, 'percentage': 100.0}]}, {'metricType': 'FieldWeightedViewsImpact', 'value': 0.60552424}], 'publication': {'link': {'@ref': 'self', '@href': 'https://api.elsevier.com/analytics/scival/publication/85021243138?apiKey=8b4bdd23def172d47b8e3df79f7baff7&httpAccept=application/json', '@type': 'application/json'}, 'title': 'J-Logic: Logical foundations for JSON querying', 'id': 85021243138, 'doi': '10.1145/3034786.3056106', 'publicationYear': 2017, 'type': 'Conference Paper', 'sourceTitle': 'Proceedings of the ACM SIGACT-SIGMOD-SIGART Symposium on Principles of Database Systems', 'topicId': 86565, 'topicName': 'Database Systems; Data Model; Query Processing', 'topicClusterId': 1091, 'topicClusterName': 'Information System; Query Processing; Artificial Intelligence'}}]}

@pytest.fixture
def sample_scival_output_metrics_parsed_dict():
    """
    Provides a parsed and extracted JSON containing the field-normalised metrics from the
    scival API call for the scopus_id 85021243138
    """
    return {'field_weighted_citation_impact': 1.99, 'top_citation_percentile': 25, 'field_weighted_views_impact': 0.60552424}


def test_extract_output_metadata(sample_scival_output_metrics_api_payload, sample_scival_output_metrics_parsed_dict):
    expected_extracted_field_normalised_output_metrics_dict = sample_scival_output_metrics_parsed_dict

    actual_extracted_field_normalised_output_metrics_dict = extract_output_metadata(sample_scival_output_metrics_api_payload)

    assert isinstance(actual_extracted_field_normalised_output_metrics_dict, dict)
    assert expected_extracted_field_normalised_output_metrics_dict == actual_extracted_field_normalised_output_metrics_dict

def test_extract_empty_output_metadata():
    assert extract_output_metadata({}) == {}

def test_extract_output_metadata_with_none():
    assert extract_output_metadata(None) == {}

def test_process_output_metrics(
    sample_scopus_id_df,
    sample_scival_output_metrics_api_payload,
    sample_scival_output_metrics_parsed_dict
):
    # Expected final dataframe after the output metrics have been processed by making an API call on scopud_id = 85021243138
    expected_field_normalised_outputs_df = pd.DataFrame([
        {'field_weighted_citation_impact': 1.99,
         'top_citation_percentile': 25,
         'field_weighted_views_impact': 0.60552424,
         'scopus_id': '85021243138'}
    ])

    # Setup mocks
    with patch.object(
            module, "get_output_metadata", return_value=sample_scival_output_metrics_api_payload
    ) as mock_get_output_metadata:
        with patch.object(
                module, "extract_output_metadata", return_value=sample_scival_output_metrics_parsed_dict
        ) as mock_extract_output_metadata:
            # For the given scopus ID, process it to obtained field normalised outputs metrics via scival publication API:
            actual_field_normalised_outputs_df = module.process_output_metrics(sample_scopus_id_df)

    # Assertions:

    # Assert function calls:
    mock_get_output_metadata.assert_called_once()
    mock_extract_output_metadata.assert_called_once()

    # Assert expected dataframe matches the actual returned dataframe
    pd.testing.assert_frame_equal(
        actual_field_normalised_outputs_df.reset_index(drop=True),
        expected_field_normalised_outputs_df.reset_index(drop=True)
    )

def test_get_output_metadata(sample_scival_output_metrics_api_payload):
    sample_scopus_id = "85021243138"

    expected_output_metadata_base_url = "https://api.elsevier.com/analytics/scival/publication/metrics"
    expected_output_metadata_url_params = {
        "metricTypes": "FieldWeightedCitationImpact,OutputsInTopCitationPercentiles,FieldWeightedViewsImpact",
        "showAsFieldWeighted": "true",
        "publicationIds": sample_scopus_id,
        "yearRange": "10yrs",
        "byYear": "false",
        "apiKey": module.elsevier_api_key,
        "httpAccept": "application/json"
    }

    # Mock the response, such that when API is called, it returned the sample SciVal Output metrics API payload
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = sample_scival_output_metrics_api_payload
    mock_response.raise_for_status.return_value = None

    # Test the function with mocked requests.get
    with patch("requests.get", return_value=mock_response) as mock_get:
        returned_scival_api_json_payload = module.get_output_metadata(sample_scopus_id)

    # Assertions:

    # Assert API call to the Sci Val API was made - Check requests.get was called
    mock_get.assert_called_once()

    # Parse and assert the call arguments on the mock SciVal API call
    args, kwargs = mock_get.call_args
    assert args[0] == expected_output_metadata_base_url
    assert kwargs["params"] == expected_output_metadata_url_params
    assert kwargs["timeout"] == 100

    # Assert that the SciVal API responded i.e. mock response was called
    assert mock_response.raise_for_status.called
    assert mock_response.json.called

    # Assert that the SciVal API returns the mocked JSON payload as expected
    assert returned_scival_api_json_payload == sample_scival_output_metrics_api_payload
    assert "results" in returned_scival_api_json_payload
    assert "metrics" in returned_scival_api_json_payload["results"][0]


def test_retry_process_output_metrics():
    # Original dataframe with some failed API calls
    # Failed means null values for the metrics - top_citation_percentile, field_weighted_citation_impact, field_weighted_views_impact

    sample_original_df = pd.DataFrame([
        # Records for which SciVal API call was a success
        {'scopus_id': '85021243138', 'field_weighted_citation_impact': 1.99,
         'top_citation_percentile': 25, 'field_weighted_views_impact': 0.60552824},
        {'scopus_id': '85022567891', 'field_weighted_citation_impact': 0.65,
         'top_citation_percentile': 15, 'field_weighted_views_impact': 0.8104595},

        # Records for which SciVal API call failed due to timeout
        {'scopus_id': '74900319915', 'field_weighted_citation_impact': None,
         'top_citation_percentile': None, 'field_weighted_views_impact': None},
        {'scopus_id': '84981503288', 'field_weighted_citation_impact': None,
         'top_citation_percentile': None, 'field_weighted_views_impact': None}
    ])

    # Expected dataframe of records that were timed-out, to be retried:
    expected_failed_df = pd.DataFrame([
        {'scopus_id': '74900319915'},
        {'scopus_id': '84981503288'}
    ])

    # Mock retried API call results
    mock_retried_results = pd.DataFrame([
        {'scopus_id': '74900319915', 'field_weighted_citation_impact': 1.95,
         'top_citation_percentile': 25, 'field_weighted_views_impact': 0.55133456},
        {'scopus_id': '84981503288', 'field_weighted_citation_impact': 1.55,
         'top_citation_percentile': 1, 'field_weighted_views_impact': 1.23326789}
    ])

    # Expected final combined dataframe
    expected_updated_df = pd.DataFrame([
        # Original successful records
        {'scopus_id': '85021243138', 'field_weighted_citation_impact': 1.99,
         'top_citation_percentile': 25, 'field_weighted_views_impact': 0.60552424},
        {'scopus_id': '85022567891', 'field_weighted_citation_impact': 0.65,
         'top_citation_percentile': 15, 'field_weighted_views_impact': 0.8104595},

        # Newly successful records from retries; these originally were timed out
        {'scopus_id': '74900319915', 'field_weighted_citation_impact': 1.95,
         'top_citation_percentile': 25, 'field_weighted_views_impact': 0.55133456},
        {'scopus_id': '84981503288', 'field_weighted_citation_impact': 1.55,
         'top_citation_percentile': 1, 'field_weighted_views_impact': 1.23326789}
    ])

    # Setup mocks
    with patch.object(module, "load_cs_output_metrics_df", return_value=sample_original_df) as mock_load_df, \
            patch.object(module, "process_output_metrics", return_value=mock_retried_results) as mock_process_metrics, \
            patch.object(module, "write_cs_output_metrics_df") as mock_write_df:
        # Call the rety method to reprocess SciVal API calls on failed metrics
        module.retry_process_output_metrics()

    # Assertions:

    # Assert function calls:
    mock_load_df.assert_called_once()
    mock_process_metrics.assert_called_once()
    mock_write_df.assert_called_once()

    # Assert process_output_metrics was called with the correct dataframe of records that failed the original API calls
    mock_process_metrics_call_args = mock_process_metrics.call_args[0][0]
    pd.testing.assert_frame_equal(
        mock_process_metrics_call_args.reset_index(drop=True),
        expected_failed_df.reset_index(drop=True),
        check_dtype=False  # Ignore dtype differences
    )

    # Assert the write function was called updated concatenated dataframe where all SciVal API calls are successful
    mock_write_df_call_args = mock_write_df.call_args[0][0]
    pd.testing.assert_frame_equal(
        mock_write_df_call_args.reset_index(drop=True),
        expected_updated_df.reset_index(drop=True),
        check_dtype=False  # Ignore dtype differences
    )

    # Assert all records in the original dataframe were processed and persisted:
    assert len(mock_write_df_call_args) == len(sample_original_df)

    # Assert that the final processed dataframe does not have any null values, as all API calls were successful
    assert mock_write_df_call_args['field_weighted_citation_impact'].isna().sum() == 0
