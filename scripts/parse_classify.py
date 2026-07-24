#!/usr/bin/env python3
"""Add syntactic parses, constituency trees, and emotion labels to articles."""

import argparse
import json
from pathlib import Path


DEFAULT_EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"


def load_articles(path, limit=None):
    with Path(path).open(encoding="utf-8") as stream:
        articles = json.load(stream)
    if not isinstance(articles, list):
        raise ValueError(f"Expected a JSON list in {path}")
    if limit is not None:
        articles = articles[:limit]
    for index, article in enumerate(articles):
        if not isinstance(article, dict):
            raise ValueError(f"Article {index} is not a JSON object")
        if not isinstance(article.get("lead_paragraph"), str):
            raise ValueError(f"Article {index} has no textual lead_paragraph")
    return [dict(article) for article in articles]


def attach_parses(articles, documents):
    """Attach one Stanza document's sentence data to each article."""
    if len(articles) != len(documents):
        raise ValueError("article and document counts differ")
    for article, document in zip(articles, documents):
        article["parsed"] = document.to_dict()
        article["constituents"] = [
            str(sentence.constituency) for sentence in document.sentences
        ]


def attach_labels(articles, labels):
    if len(articles) != len(labels):
        raise ValueError("article and emotion-label counts differ")
    for article, label in zip(articles, labels):
        article["emotion"] = label


def resolve_device(requested, torch):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested


class Analyzer:
    def __init__(
        self,
        *,
        device="auto",
        emotion_model=DEFAULT_EMOTION_MODEL,
        parse=True,
        classify=True,
    ):
        import torch

        self.torch = torch
        self.device = resolve_device(device, torch)
        self.parser = None
        self.tokenizer = None
        self.classifier = None

        if parse:
            import stanza

            self.parser = stanza.Pipeline(
                lang="en",
                processors="tokenize,pos,lemma,depparse,constituency",
                use_gpu=self.device == "cuda",
                verbose=False,
            )
        if classify:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(emotion_model)
            self.classifier = AutoModelForSequenceClassification.from_pretrained(
                emotion_model
            ).to(self.device)
            self.classifier.eval()

    def parse_articles(self, articles):
        if self.parser is None:
            raise RuntimeError("Parsing resources were not initialized")
        documents = self.parser.bulk_process(
            [article["lead_paragraph"] for article in articles]
        )
        attach_parses(articles, documents)

    def classify_articles(self, articles, batch_size=32):
        from tqdm import tqdm

        if self.classifier is None or self.tokenizer is None:
            raise RuntimeError("Emotion-classification resources were not initialized")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        labels = []
        for start in tqdm(
            range(0, len(articles), batch_size),
            desc="Classifying",
        ):
            texts = [
                article["lead_paragraph"]
                for article in articles[start : start + batch_size]
            ]
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            with self.torch.inference_mode():
                logits = self.classifier(**inputs).logits
            label_ids = self.torch.argmax(logits, dim=-1).tolist()
            labels.extend(self.classifier.config.id2label[index] for index in label_ids)
        attach_labels(articles, labels)

    def process(self, articles, *, batch_size=32):
        if self.parser is not None:
            self.parse_articles(articles)
        if self.classifier is not None:
            self.classify_articles(articles, batch_size)
        return articles


def build_parser():
    parser = argparse.ArgumentParser(
        description="Annotate article JSON with Stanza parses and emotion labels."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, help="process only the first N articles")
    parser.add_argument("--skip-parsing", action="store_true")
    parser.add_argument("--skip-emotion", action="store_true")
    parser.add_argument("--emotion-model", default=DEFAULT_EMOTION_MODEL)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.skip_parsing and args.skip_emotion:
        parser.error("nothing to do: both analysis stages were disabled")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    articles = load_articles(args.input, args.limit)
    analyzer = Analyzer(
        device=args.device,
        emotion_model=args.emotion_model,
        parse=not args.skip_parsing,
        classify=not args.skip_emotion,
    )
    output = analyzer.process(articles, batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(output)} articles to {args.output}")


if __name__ == "__main__":
    main()
