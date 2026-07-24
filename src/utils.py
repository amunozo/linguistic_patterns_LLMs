"""Utilities for reading CoNLL-U data and computing dependency statistics."""

from pathlib import Path

import numpy as np
import pandas as pd


STAT_COLUMNS = [
    "treebank",
    "% l",
    "% r",
    "avg_len",
    "avg_r_len",
    "avg_l_len",
    "std_len",
    "std_r_len",
    "std_l_len",
]


def get_files(treebank):
    """Return sorted CoNLL-U files for a file, directory, or file iterable."""
    if isinstance(treebank, (str, Path)):
        path = Path(treebank).expanduser()
        if path.is_file():
            return [path]
        if path.is_dir():
            return sorted(path.rglob("*.conllu"))
        raise FileNotFoundError(f"Treebank path does not exist: {path}")

    files = [Path(path).expanduser() for path in treebank]
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"CoNLL-U file does not exist: {missing[0]}")
    return sorted(files)


def _safe_mean(values):
    return float(np.mean(values)) if values else np.nan


def _safe_std(values):
    return float(np.std(values)) if values else np.nan


def _iter_syntactic_words(path):
    with Path(path).open(encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) != 10:
                raise ValueError(f"Expected 10 CoNLL-U columns in {path}: {line!r}")
            if not fields[0].isdigit():
                # Skip multi-word-token and empty-node rows.
                continue
            yield fields


def stats_df(treebanks):
    """Compute dependency-direction and length statistics per treebank."""
    rows = []
    for treebank in treebanks:
        absolute = []
        right = []
        left = []
        for path in get_files(treebank):
            for fields in _iter_syntactic_words(path):
                token_id = int(fields[0])
                head_id = int(fields[6])
                if head_id == 0:
                    continue
                distance = head_id - token_id
                absolute.append(abs(distance))
                if distance > 0:
                    left.append(distance)
                else:
                    right.append(-distance)

        total = len(absolute)
        rows.append(
            {
                "treebank": str(treebank),
                "% l": 100 * len(left) / total if total else np.nan,
                "% r": 100 * len(right) / total if total else np.nan,
                "avg_len": _safe_mean(absolute),
                "avg_r_len": _safe_mean(right),
                "avg_l_len": _safe_mean(left),
                "std_len": _safe_std(absolute),
                "std_r_len": _safe_std(right),
                "std_l_len": _safe_std(left),
            }
        )

    return pd.DataFrame(rows, columns=STAT_COLUMNS)


class CoNLLu:
    """Small, dependency-free reader for the fields used in the analysis."""

    def __init__(self, file):
        self.path = Path(file)
        self.text = self.path.read_text(encoding="utf-8")
        self.sentences = [
            sentence for sentence in self.text.strip().split("\n\n") if sentence.strip()
        ]
        self.lines = [line for line in self.text.splitlines() if line]
        self.constituency = []
        self.attributes = self.get_attributes()
        self.sentence_length = self.sentence_length_dist()

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

    def get_attributes(self):
        """Return column-wise attributes for syntactic word rows."""
        attributes = {
            "idx": [],
            "word": [],
            "lemma": [],
            "upos": [],
            "xpos": [],
            "feats": [],
            "head": [],
            "deprel": [],
            "deps": [],
            "misc": [],
            "arc": [],
        }
        self.constituency = []
        for line in self.lines:
            if line.startswith("# constituency = "):
                self.constituency.append(line.removeprefix("# constituency = "))
                continue
            fields = line.split("\t")
            if len(fields) != 10 or not fields[0].isdigit():
                continue
            attributes["idx"].append(fields[0])
            attributes["word"].append(fields[1])
            attributes["lemma"].append(fields[2])
            attributes["upos"].append(fields[3])
            attributes["xpos"].append(fields[4])
            attributes["feats"].append(fields[5])
            attributes["head"].append(fields[6])
            attributes["deprel"].append(fields[7])
            attributes["deps"].append(fields[8])
            attributes["misc"].append(fields[9])
            attributes["arc"].append(int(fields[6]) - int(fields[0]))
        return attributes

    @staticmethod
    def remove_outliers(data, threshold=1):
        """Remove values whose frequency is not greater than ``threshold``."""
        frequencies = {}
        for value in data:
            frequencies[value] = frequencies.get(value, 0) + 1
        return [value for value in data if frequencies[value] > threshold]

    def sentence_length_dist(self):
        """Return syntactic word counts, excluding comments and special rows."""
        return [
            sum(
                len(fields) == 10 and fields[0].isdigit()
                for fields in (line.split("\t") for line in sentence.splitlines())
            )
            for sentence in self.sentences
        ]

    def avg_sentence_length(self):
        if not self.sentence_length:
            return np.nan
        return float(np.mean(self.sentence_length))
