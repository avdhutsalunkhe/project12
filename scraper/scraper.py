"""
Main scraper orchestrator.

Coordinates:
  1. Paginating through the catalog listing
  2. Collecting all assessment URLs
  3. Fetching each detail page
  4. Parsing structured data
  5. Writing catalog.json
"""

import json
import logging
import os
import time
from typing import Any, Dict, List

from scraper.config import (
    CATALOG_TYPES,
    CATALOG_URL,
    CRAWL_DELAY,
    ITEMS_PER_PAGE,
    OUTPUT_DIR,
    CATALOG_FILENAME,
)
from scraper.http_client import create_session, fetch_page
from scraper.parsers import parse_catalog_page, parse_detail_page, parse_total_pages

logger = logging.getLogger(__name__)


def _build_catalog_url(catalog_type: int, start: int) -> str:
    """Construct a paginated catalog listing URL."""
    return f"{CATALOG_URL}?start={start}&type={catalog_type}"


def discover_assessment_urls(
    session,
    catalog_type: int,
    type_label: str,
) -> List[Dict[str, str]]:
    """
    Paginate through all listing pages for a given catalog type
    and collect every assessment URL.

    Returns: [{"name": "...", "url": "..."}]
    """
    # Fetch first page to determine total pages
    first_url = _build_catalog_url(catalog_type, 0)
    first_html = fetch_page(session, first_url)

    if not first_html:
        logger.error("Failed to fetch first catalog page for type=%s", type_label)
        return []

    total_pages = parse_total_pages(first_html)
    logger.info(
        "[%s] Detected %d pages of assessments (%d items/page)",
        type_label,
        total_pages,
        ITEMS_PER_PAGE,
    )

    all_items: List[Dict[str, str]] = []
    seen_urls: set = set()

    # Parse first page
    for item in parse_catalog_page(first_html):
        if item["url"] not in seen_urls:
            seen_urls.add(item["url"])
            all_items.append(item)

    # Fetch remaining pages
    for page_num in range(1, total_pages):
        start = page_num * ITEMS_PER_PAGE
        url = _build_catalog_url(catalog_type, start)

        logger.info(
            "[%s] Fetching listing page %d/%d (start=%d)",
            type_label, page_num + 1, total_pages, start,
        )

        time.sleep(CRAWL_DELAY)
        html = fetch_page(session, url)

        if not html:
            logger.warning("Skipping page %d -- fetch failed", page_num + 1)
            continue

        for item in parse_catalog_page(html):
            if item["url"] not in seen_urls:
                seen_urls.add(item["url"])
                all_items.append(item)

    logger.info(
        "[%s] Discovered %d unique assessments", type_label, len(all_items)
    )
    return all_items


def scrape_detail_pages(
    session,
    items: List[Dict[str, str]],
    catalog_type_label: str,
) -> List[Dict[str, Any]]:
    """
    Fetch and parse each assessment's detail page.

    Enriches the listing data with description, duration, test type, etc.
    """
    results: List[Dict[str, Any]] = []
    total = len(items)

    for idx, item in enumerate(items, 1):
        url = item["url"]
        name = item["name"]

        logger.info(
            "[%s] (%d/%d) Scraping detail: %s",
            catalog_type_label, idx, total, name,
        )

        time.sleep(CRAWL_DELAY)
        html = fetch_page(session, url)

        if not html:
            logger.warning("Skipping %s -- fetch failed", name)
            results.append({
                "name": name,
                "url": url,
                "catalog_type": catalog_type_label,
                "description": None,
                "test_type_code": None,
                "test_type": None,
                "duration_minutes": None,
                "remote_testing": None,
                "job_levels": [],
                "languages": [],
                "fact_sheet_urls": [],
                "scrape_error": True,
            })
            continue

        try:
            detail = parse_detail_page(html, url)
            detail["catalog_type"] = catalog_type_label
            detail["scrape_error"] = False

            # Prefer listing name if detail page title is missing
            if not detail.get("name"):
                detail["name"] = name

            results.append(detail)

        except Exception as exc:
            logger.exception("Parse error for %s: %s", name, exc)
            results.append({
                "name": name,
                "url": url,
                "catalog_type": catalog_type_label,
                "description": None,
                "test_type_code": None,
                "test_type": None,
                "duration_minutes": None,
                "remote_testing": None,
                "job_levels": [],
                "languages": [],
                "fact_sheet_urls": [],
                "scrape_error": True,
            })

    return results


def save_catalog(data: List[Dict[str, Any]], output_dir: str, filename: str) -> str:
    """Write the scraped catalog to a JSON file and return the path."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_assessments": len(data),
                "assessments": data,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info("Catalog saved -> %s (%d assessments)", filepath, len(data))
    return filepath


def run_scraper(
    types: List[str] | None = None,
    output_dir: str = OUTPUT_DIR,
    filename: str = CATALOG_FILENAME,
) -> str:
    """
    Full scrape pipeline:
      1. Discover all assessment URLs across catalog types
      2. Fetch each detail page
      3. Parse structured data
      4. Save to catalog.json

    Args:
        types: Which catalog types to scrape. Defaults to all.
                Options: "individual", "prepackaged"
        output_dir: Directory for the output file.
        filename: Output filename.

    Returns:
        Path to the generated catalog JSON file.
    """
    if types is None:
        types = list(CATALOG_TYPES.keys())

    session = create_session()
    all_assessments: List[Dict[str, Any]] = []

    for type_label in types:
        catalog_type = CATALOG_TYPES.get(type_label)
        if catalog_type is None:
            logger.warning("Unknown catalog type: %s -- skipping", type_label)
            continue

        logger.info("=" * 60)
        logger.info("Starting scrape: %s (type=%d)", type_label, catalog_type)
        logger.info("=" * 60)

        # Phase 1: Discover all assessment URLs
        items = discover_assessment_urls(session, catalog_type, type_label)

        if not items:
            logger.warning("No assessments found for type=%s", type_label)
            continue

        # Phase 2: Scrape detail pages
        details = scrape_detail_pages(session, items, type_label)
        all_assessments.extend(details)

    # Phase 3: Save
    filepath = save_catalog(all_assessments, output_dir, filename)

    # Summary
    errors = sum(1 for a in all_assessments if a.get("scrape_error"))
    logger.info("=" * 60)
    logger.info("SCRAPE COMPLETE")
    logger.info("  Total assessments: %d", len(all_assessments))
    logger.info("  Successful:        %d", len(all_assessments) - errors)
    logger.info("  Errors:            %d", errors)
    logger.info("  Output:            %s", filepath)
    logger.info("=" * 60)

    return filepath
