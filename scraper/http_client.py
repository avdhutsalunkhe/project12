"""
HTTP client with retry logic, backoff, and session reuse.
"""

import logging
import time
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from scraper.config import (
    DEFAULT_HEADERS,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    RETRY_BACKOFF_FACTOR,
    RETRY_STATUS_CODES,
)

logger = logging.getLogger(__name__)


def create_session() -> requests.Session:
    """
    Build a requests.Session with automatic retry on transient failures.

    Retries on 429 / 5xx with exponential backoff.
    """
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)

    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=RETRY_BACKOFF_FACTOR,
        status_forcelist=list(RETRY_STATUS_CODES),
        allowed_methods=["GET"],
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


def fetch_page(
    session: requests.Session,
    url: str,
    timeout: int = REQUEST_TIMEOUT,
) -> Optional[str]:
    """
    GET a URL and return the response body as text.

    Returns None if the request fails after all retries.
    """
    try:
        logger.debug("GET %s", url)
        response = session.get(url, timeout=timeout)

        if response.status_code == 200:
            return response.text

        logger.warning(
            "Non-200 response: %s → %d", url, response.status_code
        )
        return None

    except requests.RequestException as exc:
        logger.error("Request failed for %s: %s", url, exc)
        return None
