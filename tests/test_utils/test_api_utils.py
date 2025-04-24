from datetime import datetime
import pytest
from unittest.mock import patch, Mock
import time

from utils.API import check_api_quota


@pytest.fixture
def mock_response():
    """
    Set-up for tests: Returns a mocked response object, with status of 200 OK
    """
    mock = Mock()
    mock.status_code = 200
    mock.headers = {}
    return mock

def test_check_api_quota_available(mock_response, capfd):
    """
    Test for checking the API quota when it is available.
    """
    mock_response.status_code = 200

    with patch('requests.get', return_value=mock_response):
        check_api_quota("test_key", "https://example_API.com/endpoint")

    # Capture standard output
    out, err = capfd.readouterr()
    assert "API call successful. Quota is available." in out


def test_check_api_quota_reset_time(mock_response, capfd):
    """
    Test for checking the API quota when it is exceeded
    """
    # API response: 429 Too Many Requests
    mock_response.status_code = 429

    # Time when the API quota resets. Example value: 9AM, April 24, 2025
    api_quota_reset_time = datetime(2025, 4, 24, 9, 0, 0)
    api_quota_reset_timestamp = int(api_quota_reset_time.timestamp())

    # Update response header to include the reset timestamp
    mock_response.headers = {"X-RateLimit-Reset": str(api_quota_reset_timestamp)}
    formatted_api_quota_reset_timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(api_quota_reset_timestamp))

    with patch('requests.get', return_value=mock_response):
        check_api_quota("test_key", "https://example_API.com/endpoint")

    # Capture standard output
    out, err = capfd.readouterr()
    assert f"API Quota exceeded. Limit resets at {formatted_api_quota_reset_timestamp} UTC." in out


def test_check_api_quota_invalid_call(mock_response, capfd):
    """
    Test for checking the API quota when the API call is invalid, but quota is available.
    """
    # Status code is not 200 OK, or 429 Too Many Requests
    mock_response.status_code = 400
    mock_response.text = "Bad Request"

    with patch('requests.get', return_value=mock_response):
        check_api_quota("test_key", "https://example_API.com/endpoint?param1=bad_param1_value")

    # Capture standard output
    out, err = capfd.readouterr()
    assert "Invalid API call. Quota is available." in out
    assert "400 - Bad Request" in out
