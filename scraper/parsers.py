"""
HTML parsers for SHL catalog listing pages and individual detail pages.

All parsing logic is isolated here so the scraper orchestrator stays clean.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag

from scraper.config import BASE_URL, TEST_TYPE_MAP

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Catalog Listing Parser
# ═══════════════════════════════════════════════════════════════

def parse_catalog_page(html: str) -> List[Dict[str, str]]:
    """
    Extract assessment links from a catalog listing page.

    Returns a list of dicts: [{"name": "...", "url": "..."}]
    """
    soup = BeautifulSoup(html, "html.parser")
    items: List[Dict[str, str]] = []
    seen_urls: set = set()

    # Assessment links live inside the main content area and point to
    # /products/product-catalog/view/<slug>/
    for link in soup.find_all("a", href=True):
        href: str = link["href"]

        if "/products/product-catalog/view/" not in href:
            continue

        full_url = urljoin(BASE_URL, href)

        # Deduplicate (the page renders nav elements multiple times)
        if full_url in seen_urls:
            continue
        seen_urls.add(full_url)

        name = link.get_text(strip=True)
        if name:
            items.append({"name": name, "url": full_url})

    logger.info("Parsed %d assessment links from listing page", len(items))
    return items


def parse_total_pages(html: str) -> int:
    """
    Detect the total number of pages from the pagination controls.

    Falls back to 1 if pagination is not found.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Pagination links look like: ?start=372&type=1 → page = 372/12 + 1 = 32
    max_start = 0
    for link in soup.find_all("a", href=True):
        href: str = link["href"]
        match = re.search(r"[?&]start=(\d+)", href)
        if match:
            start_val = int(match.group(1))
            if start_val > max_start:
                max_start = start_val

    if max_start == 0:
        return 1

    from scraper.config import ITEMS_PER_PAGE
    return (max_start // ITEMS_PER_PAGE) + 1


# ═══════════════════════════════════════════════════════════════
#  Detail Page Parser
# ═══════════════════════════════════════════════════════════════

def _extract_section_text(soup: BeautifulSoup, heading_text: str) -> Optional[str]:
    """
    Find an h4 (or h3/h5) containing `heading_text` and return the text
    that follows it until the next heading.
    """
    for heading in soup.find_all(re.compile(r"^h[3-5]$")):
        if heading_text.lower() in heading.get_text(strip=True).lower():
            # Collect all siblings until the next heading
            parts: List[str] = []
            for sibling in heading.next_siblings:
                if isinstance(sibling, Tag) and sibling.name and re.match(r"^h[1-5]$", sibling.name):
                    break
                text = sibling.get_text(strip=True) if isinstance(sibling, Tag) else str(sibling).strip()
                if text:
                    parts.append(text)
            return " ".join(parts).strip() or None
    return None


def _parse_test_type(raw: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract test type code and human-readable label from assessment length text.

    Example input: "Approximate Completion Time in minutes = 30 Test Type: K Remote Testing:"
    """
    code = None
    match = re.search(r"Test\s*Type:\s*([A-Z])", raw, re.IGNORECASE)
    if match:
        code = match.group(1).upper()

    label = TEST_TYPE_MAP.get(code) if code else None
    return code, label


def _parse_duration(raw: str) -> Optional[int]:
    """
    Extract duration in minutes from the assessment length section.

    Example: "Approximate Completion Time in minutes = 30"
    """
    match = re.search(r"(?:minutes|min)\s*=?\s*(\d+)", raw, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Fallback: just look for a standalone number
    match = re.search(r"(\d+)\s*(?:minutes|min)", raw, re.IGNORECASE)
    if match:
        return int(match.group(1))

    return None


def _parse_remote_testing(raw: str) -> Optional[bool]:
    """
    Determine if remote testing is supported.

    The field often appears as "Remote Testing: Yes" or just "Remote Testing:"
    """
    match = re.search(r"Remote\s*Testing:\s*(Yes|No)?", raw, re.IGNORECASE)
    if match:
        value = match.group(1)
        if value:
            return value.lower() == "yes"
    return None


def _parse_languages(raw: Optional[str]) -> List[str]:
    """Split a comma-separated language string into a clean list."""
    if not raw:
        return []
    return [lang.strip() for lang in raw.split(",") if lang.strip()]


def _parse_job_levels(raw: Optional[str]) -> List[str]:
    """Split a comma-separated job levels string into a clean list."""
    if not raw:
        return []
    return [level.strip() for level in raw.split(",") if level.strip()]


def _extract_fact_sheet_urls(soup: BeautifulSoup) -> List[str]:
    """Find all PDF fact sheet download links."""
    urls: List[str] = []
    seen: set = set()
    for link in soup.find_all("a", href=True):
        href: str = link["href"]
        if href.endswith(".pdf") and href not in seen:
            seen.add(href)
            urls.append(href)
    return urls


def parse_detail_page(html: str, assessment_url: str) -> Dict[str, Any]:
    """
    Parse a single assessment detail page into a structured dictionary.

    Returns all extractable fields; missing fields default to None / [].
    """
    soup = BeautifulSoup(html, "html.parser")

    # ── Description ──────────────────────────────────────────────
    description = _extract_section_text(soup, "Description")

    # Also check og:description as fallback
    if not description:
        og = soup.find("meta", property="og:description")
        if og and og.get("content"):
            content = og["content"]
            # OG description often has "Name: Description" format
            if ":" in content:
                description = content.split(":", 1)[1].strip()
            else:
                description = content.strip()

    # ── Job Levels ───────────────────────────────────────────────
    job_levels_raw = _extract_section_text(soup, "Job levels")
    job_levels = _parse_job_levels(job_levels_raw)

    # ── Languages ────────────────────────────────────────────────
    languages_raw = _extract_section_text(soup, "Languages")
    languages = _parse_languages(languages_raw)

    # ── Assessment Length / Test Type / Remote ────────────────────
    assessment_info = _extract_section_text(soup, "Assessment length") or ""
    duration_minutes = _parse_duration(assessment_info)
    test_type_code, test_type_label = _parse_test_type(assessment_info)
    remote_testing = _parse_remote_testing(assessment_info)

    # ── Fact Sheets ──────────────────────────────────────────────
    fact_sheets = _extract_fact_sheet_urls(soup)

    # ── Title (from og:title or page title) ──────────────────────
    title = None
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        title = og_title["content"].replace("| SHL", "").strip()
    if not title:
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True).replace("| SHL", "").strip()

    return {
        "name": title,
        "url": assessment_url,
        "description": description,
        "test_type_code": test_type_code,
        "test_type": test_type_label,
        "duration_minutes": duration_minutes,
        "remote_testing": remote_testing,
        "job_levels": job_levels,
        "languages": languages,
        "fact_sheet_urls": fact_sheets,
    }
