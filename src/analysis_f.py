"""Analysis helpers used by the paper notebook."""

from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


STAT_COLUMNS = [
    "lm",
    "% l",
    "% r",
    "avg_len",
    "avg_r_len",
    "avg_l_len",
    "std_len",
    "std_r_len",
    "std_l_len",
    "n_sentences",
]


def limit_length(sentences: Iterable[Sequence[Any]], lower: int, upper: int):
    """Return sentences whose token counts fall in the inclusive interval."""
    if lower < 0 or upper < lower:
        raise ValueError("Expected 0 <= lower <= upper")
    return [sentence for sentence in sentences if lower <= len(sentence) <= upper]


def myconverter(value: Any):
    """JSON serializer retained for compatibility with the analysis notebook."""
    if isinstance(value, datetime):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def create_length_dict(sentences_dict, max_length: int, step: int):
    """Group each model's sentences into inclusive length buckets."""
    if max_length <= 1:
        raise ValueError("max_length must be greater than 1")
    if step <= 0:
        raise ValueError("step must be positive")

    length_dict = {}
    for lower in range(1, max_length + 1, step):
        upper = min(lower + step - 1, max_length)
        key = f"{lower}_{upper}" if upper < max_length else f"{lower}_"
        length_dict[key] = {
            model: limit_length(sentences, lower, upper)
            for model, sentences in sentences_dict.items()
        }
    return length_dict


def _safe_mean(values):
    return float(np.mean(values)) if values else np.nan


def _safe_std(values):
    return float(np.std(values)) if values else np.nan


def _dependency_lengths(sentences: Iterable[Sequence[Mapping[str, Any]]]):
    absolute = []
    right = []
    left = []

    for sentence in sentences:
        for token in sentence:
            token_id = int(token["id"])
            head_id = int(token["head"])
            if head_id == 0:
                continue
            distance = head_id - token_id
            absolute.append(abs(distance))
            if distance > 0:
                left.append(distance)
            else:
                right.append(-distance)

    return absolute, right, left


def stats_df(dict_of_sentences):
    """Compute dependency-direction and dependency-length statistics by model.

    Empty subsets are represented by ``NaN`` for undefined means and standard
    deviations, rather than raising a division-by-zero exception. Column names
    are preserved for compatibility with the paper notebook.
    """
    rows = []
    for model, sentences in dict_of_sentences.items():
        absolute, right, left = _dependency_lengths(sentences)
        total = len(absolute)
        rows.append(
            {
                "lm": model,
                "% l": 100 * len(left) / total if total else np.nan,
                "% r": 100 * len(right) / total if total else np.nan,
                "avg_len": _safe_mean(absolute),
                "avg_r_len": _safe_mean(right),
                "avg_l_len": _safe_mean(left),
                "std_len": _safe_std(absolute),
                "std_r_len": _safe_std(right),
                "std_l_len": _safe_std(left),
                "n_sentences": len(sentences),
            }
        )

    return pd.DataFrame(rows, columns=STAT_COLUMNS)
