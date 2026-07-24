#!/usr/bin/env python3
"""Download monthly New York Times Archive API responses."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path


BASE_URL = "https://api.nytimes.com/svc/archive/v1"


def parse_month(value):
    try:
        return datetime.strptime(value, "%Y-%m").date().replace(day=1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected YYYY-MM") from error


def iter_months(start_date, end_date):
    """Yield inclusive ``(year, month)`` pairs."""
    if end_date < start_date:
        raise ValueError("end date must not be earlier than start date")

    year, month = start_date.year, start_date.month
    while (year, month) <= (end_date.year, end_date.month):
        yield year, month
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1


def load_env_file(path):
    """Load simple KEY=VALUE entries without overwriting the environment."""
    path = Path(path)
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def download_data(
    start_date,
    end_date,
    output_dir,
    api_key,
    *,
    timeout=30.0,
    session=None,
):
    """Download and save one archive response per month."""
    if not api_key:
        raise ValueError("A non-empty NYT API key is required")

    if session is None:
        import requests

        session = requests.Session()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for year, month in iter_months(start_date, end_date):
        response = session.get(
            f"{BASE_URL}/{year}/{month}.json",
            params={"api-key": api_key},
            timeout=timeout,
        )
        response.raise_for_status()
        output_path = output_dir / f"{year}_{month:02d}.json"
        output_path.write_text(
            json.dumps(response.json(), ensure_ascii=False),
            encoding="utf-8",
        )
        written.append(output_path)
        print(f"Downloaded {year}-{month:02d} -> {output_path}")
    return written


def build_parser():
    parser = argparse.ArgumentParser(
        description="Download monthly responses from the NYT Archive API."
    )
    parser.add_argument("start_date", type=parse_month, help="first month (YYYY-MM)")
    parser.add_argument("end_date", type=parse_month, help="last month (YYYY-MM)")
    parser.add_argument("output_dir", type=Path, help="directory for JSON responses")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(__file__).resolve().parents[1] / ".env",
        help="optional environment file (default: repository .env)",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    load_env_file(args.env_file)
    api_key = os.environ.get("NYT_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        parser.error(
            "NYT_API_KEY is missing. Export it or copy .env.example to .env."
        )
    try:
        download_data(
            args.start_date,
            args.end_date,
            args.output_dir,
            api_key,
            timeout=args.timeout,
        )
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
