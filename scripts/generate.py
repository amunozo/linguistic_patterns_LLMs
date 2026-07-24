#!/usr/bin/env python3
"""Generate news lead paragraphs with Hugging Face causal language models."""

import argparse
import gc
import json
import re
from pathlib import Path


def model_spec(value):
    """Parse ``[ALIAS=]MODEL_ID`` into a safe output alias and model ID."""
    if "=" in value:
        alias, model_id = value.split("=", 1)
    else:
        model_id = value
        alias = value.rstrip("/").split("/")[-1]
    if not model_id.strip():
        raise argparse.ArgumentTypeError("model ID cannot be empty")
    alias = re.sub(r"[^A-Za-z0-9_.-]+", "_", alias.strip())
    if not alias:
        raise argparse.ArgumentTypeError("model alias cannot be empty")
    return alias, model_id.strip()


def headline_text(article):
    headline = article.get("headline")
    if isinstance(headline, dict):
        headline = headline.get("main")
    if not isinstance(headline, str) or not headline.strip():
        raise ValueError("article is missing headline text")
    return headline.strip()


def seed_text(article, seed_words=3):
    paragraph = article.get("lead_paragraph")
    if not isinstance(paragraph, str) or not paragraph.strip():
        raise ValueError("article is missing a lead_paragraph")
    return " ".join(paragraph.split()[:seed_words])


def build_prompt(article, seed_words=3):
    """Build the paper's headline-plus-three-words generation prompt."""
    return f'"{headline_text(article)}"\n{seed_text(article, seed_words)} '


def load_articles(path, limit=None):
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        articles = json.load(stream)
    if not isinstance(articles, list):
        raise ValueError(f"Expected a JSON list in {path}")
    if limit is not None:
        articles = articles[:limit]
    for index, article in enumerate(articles):
        if not isinstance(article, dict):
            raise ValueError(f"Article {index} is not a JSON object")
        build_prompt(article)
    return articles


def attach_generations(articles, continuations, seed_words=3):
    """Return article copies whose lead paragraphs contain generated text."""
    if len(articles) != len(continuations):
        raise ValueError("article and generation counts differ")
    generated_articles = []
    for article, continuation in zip(articles, continuations):
        generated = dict(article)
        prefix = seed_text(article, seed_words)
        generated["lead_paragraph"] = f"{prefix} {continuation}".strip()
        generated_articles.append(generated)
    return generated_articles


class ArticleGenerator:
    """Thin adapter around a Hugging Face causal language model."""

    def __init__(self, model_id, *, device="auto", load_in_8bit=False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            padding_side="left",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {}
        if device == "auto":
            model_kwargs["device_map"] = "auto"
        if load_in_8bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        if device != "auto":
            self.model.to(device)
        self.model.eval()

    @property
    def device(self):
        return self.model.device

    def generate(
        self,
        articles,
        *,
        batch_size=1,
        seed_words=3,
        max_new_tokens=200,
        top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.1,
        seed=0,
    ):
        from tqdm import tqdm

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        prompts = [build_prompt(article, seed_words) for article in articles]
        self.torch.manual_seed(seed)
        continuations = []
        for start in tqdm(
            range(0, len(prompts), batch_size),
            desc="Generating",
        ):
            batch = prompts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(self.device)
            with self.torch.inference_mode():
                output_ids = self.model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            new_ids = output_ids[:, encoded["input_ids"].shape[1] :]
            continuations.extend(
                text.strip()
                for text in self.tokenizer.batch_decode(
                    new_ids,
                    skip_special_tokens=True,
                )
            )
        return attach_generations(articles, continuations, seed_words)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate lead paragraphs using the prompting setup from the paper."
    )
    parser.add_argument("--input", type=Path, required=True, help="source article JSON")
    parser.add_argument(
        "--model",
        dest="models",
        type=model_spec,
        action="append",
        required=True,
        metavar="[ALIAS=]MODEL_ID",
        help="repeat for each Hugging Face model or local model path",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed-words", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, help="process only the first N articles")
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="use bitsandbytes 8-bit quantization",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.seed_words <= 0:
        parser.error("--seed-words must be positive")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    articles = load_articles(args.input, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for alias, model_id in args.models:
        print(f"Loading {model_id} as {alias}")
        generator = ArticleGenerator(
            model_id,
            device=args.device,
            load_in_8bit=args.load_in_8bit,
        )
        output = generator.generate(
            articles,
            batch_size=args.batch_size,
            seed_words=args.seed_words,
            max_new_tokens=args.max_new_tokens,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
        )
        output_path = args.output_dir / f"{alias}.json"
        output_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote {len(output)} articles to {output_path}")
        del generator
        gc.collect()


if __name__ == "__main__":
    main()
