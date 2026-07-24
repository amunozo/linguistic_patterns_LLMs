#!/usr/bin/env python3
"""Extract the article records used by the generation pipeline."""

import argparse
import json
from datetime import date, datetime
from pathlib import Path


def parse_date(value):
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected YYYY-MM-DD") from error


def publication_date(article):
    value = article.get("pub_date")
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def usable_article(article, start_date=None, end_date=None):
    """Return whether an NYT record contains the fields required by the paper."""
    if not isinstance(article, dict):
        return False
    headline = article.get("headline")
    headline_text = headline.get("main") if isinstance(headline, dict) else headline
    if not isinstance(headline_text, str) or not headline_text.strip():
        return False
    paragraph = article.get("lead_paragraph")
    if not isinstance(paragraph, str) or not paragraph.strip():
        return False

    if start_date is not None or end_date is not None:
        article_date = publication_date(article)
        if article_date is None:
            return False
        if start_date is not None and article_date < start_date:
            return False
        if end_date is not None and article_date > end_date:
            return False
    return True


def input_files(path):
    path = Path(path)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob("*.json"))
    raise FileNotFoundError(f"Input path does not exist: {path}")


def records_from_payload(payload, path):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        response = payload.get("response")
        if isinstance(response, dict) and isinstance(response.get("docs"), list):
            return response["docs"]
    raise ValueError(f"Unsupported NYT JSON structure in {path}")


def prepare_articles(source, start_date=None, end_date=None):
    """Load, filter, and deduplicate NYT article records."""
    if start_date and end_date and end_date < start_date:
        raise ValueError("end date must not be earlier than start date")

    selected = []
    seen = set()
    for path in input_files(source):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for article in records_from_payload(payload, path):
            if not usable_article(article, start_date, end_date):
                continue
            identifier = article.get("_id") or article.get("web_url")
            if identifier is not None and identifier in seen:
                continue
            if identifier is not None:
                seen.add(identifier)
            selected.append(article)
    return selected


def build_parser():
    parser = argparse.ArgumentParser(
        description="Extract usable article records from NYT Archive API JSON."
    )
    parser.add_argument("input", type=Path, help="archive JSON file or directory")
    parser.add_argument("output", type=Path, help="filtered article-list JSON")
    parser.add_argument("--start", type=parse_date, help="inclusive publication date")
    parser.add_argument("--end", type=parse_date, help="inclusive publication date")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        articles = prepare_articles(args.input, args.start, args.end)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(articles, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(articles)} articles to {args.output}")


if __name__ == "__main__":
    main()
