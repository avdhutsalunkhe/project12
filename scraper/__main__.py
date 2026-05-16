"""
CLI entry-point for the SHL catalog scraper.

Usage:
    python -m scraper                          # scrape everything
    python -m scraper --type individual        # only individual tests
    python -m scraper --type prepackaged       # only pre-packaged solutions
    python -m scraper --output data/custom.json
"""

import argparse
import logging
import sys
import time

from scraper.scraper import run_scraper


def _setup_logging(level: str = "INFO") -> None:
    """Configure root logger for CLI usage."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape the SHL product catalog into structured JSON.",
    )
    parser.add_argument(
        "--type",
        choices=["individual", "prepackaged", "all"],
        default="all",
        help="Which catalog section to scrape (default: all)",
    )
    parser.add_argument(
        "--output",
        default="data/catalog.json",
        help="Output file path (default: data/catalog.json)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )

    args = parser.parse_args()
    _setup_logging(args.log_level)

    logger = logging.getLogger(__name__)

    # Determine types to scrape
    if args.type == "all":
        types = ["individual", "prepackaged"]
    else:
        types = [args.type]

    # Split output path
    import os
    output_dir = os.path.dirname(args.output) or "data"
    filename = os.path.basename(args.output)

    start_time = time.time()

    try:
        filepath = run_scraper(
            types=types,
            output_dir=output_dir,
            filename=filename,
        )
        elapsed = time.time() - start_time
        logger.info("Done in %.1f seconds -> %s", elapsed, filepath)

    except KeyboardInterrupt:
        logger.info("Scrape interrupted by user")
        sys.exit(1)

    except Exception:
        logger.exception("Fatal error during scrape")
        sys.exit(1)


if __name__ == "__main__":
    main()
