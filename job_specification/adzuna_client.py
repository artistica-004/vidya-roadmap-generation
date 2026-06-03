# src/adzuna_client.py
import requests
from typing import List, Dict
from .config import ADZUNA_APP_ID, ADZUNA_APP_KEY, ADZUNA_COUNTRY

BASE_URL = "https://api.adzuna.com/v1/api/jobs"


# src/adzuna_client.py
import time
import requests
from typing import List, Dict
from .config import ADZUNA_APP_ID, ADZUNA_APP_KEY, ADZUNA_COUNTRY

BASE_URL = "https://api.adzuna.com/v1/api/jobs"


def fetch_jobs_from_adzuna(
    query: str,
    page: int = 1,
    results_per_page: int = 20,
    max_retries: int = 3,
    timeout_seconds: int = 60,     # increased from 30 -> 60
) -> List[Dict]:
    """
    Fetch a page of job results from Adzuna with basic retry logic.
    Returns: list of job dicts (JSON objects).
    """
    if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
        raise ValueError("Adzuna credentials are missing in .env")

    url = f"{BASE_URL}/{ADZUNA_COUNTRY}/search/{page}"
    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_APP_KEY,
        "results_per_page": results_per_page,
        "what": query,          # keyword, e.g. 'Software Engineer'
        "content-type": "application/json",
    }

    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            print(f"[Adzuna] Requesting (attempt {attempt}): {url} with params={params}")
            resp = requests.get(url, params=params, timeout=timeout_seconds)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            print(f"[Adzuna] Got {len(results)} results for '{query}'")
            return results

        except requests.exceptions.ReadTimeout:
            print(f"[Adzuna] Read timeout on attempt {attempt} for query '{query}'")
        except requests.RequestException as e:
            # covers network errors, HTTP errors, etc.
            print(f"[Adzuna] Request failed on attempt {attempt} for query '{query}': {e}")

        # small backoff before retrying
        time.sleep(2 * attempt)

    print(f"[Adzuna] Giving up on query '{query}' after {max_retries} attempts.")
    return []
