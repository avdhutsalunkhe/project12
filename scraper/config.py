"""
Scraper configuration constants.
"""

# ── SHL Catalog URLs ─────────────────────────────────────────────
BASE_URL = "https://www.shl.com"
CATALOG_URL = f"{BASE_URL}/products/product-catalog/"

# type=1 → Individual Test Solutions, type=2 → Pre-packaged Job Solutions
CATALOG_TYPES = {
    "individual": 1,
    "prepackaged": 2,
}

ITEMS_PER_PAGE = 12

# ── Request Settings ─────────────────────────────────────────────
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 2  # exponential: 2s, 4s, 8s
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}

# Polite crawl delay between requests (seconds)
CRAWL_DELAY = 1.5

# ── Headers (mimic a real browser) ───────────────────────────────
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

# ── Output ───────────────────────────────────────────────────────
OUTPUT_DIR = "data"
CATALOG_FILENAME = "catalog.json"

# ── Test Type Codes (from SHL's legend) ──────────────────────────
TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}
