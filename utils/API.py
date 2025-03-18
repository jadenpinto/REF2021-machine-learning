import requests
import time

def check_api_quota(api_key, api_endpoint):
    url = api_endpoint
    params = {
        "apiKey": api_key
    }

    response = requests.get(url, params=params)

    # Too Many Requests
    if response.status_code == 429:
        reset_time = response.headers.get("X-RateLimit-Reset")
        if reset_time:
            reset_time = int(reset_time)
            formatted_reset_time = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(reset_time))
            print(f"API Quota exceeded. Limit resets at {formatted_reset_time} UTC.")
        else:
            print("API Quota exceeded.")

    elif response.status_code == 200:
        print("API call successful. Quota is available.")

    else:
        print("Invalid API call. Quota is available.")
        print(f"{response.status_code} - {response.text}")
