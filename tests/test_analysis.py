from pathlib import Path

import numpy as np
from nltk.tree import Tree

from src.analysis_f import create_length_dict, stats_df
from src.constituency import get_tree_spans
from src.utils import CoNLLu


def test_dependency_statistics_handle_direction_and_string_ids():
    sentences = {
        "model": [
            [
                {"id": "1", "head": "2"},
                {"id": "2", "head": "0"},
                {"id": "3", "head": "2"},
            ]
        ]
    }

    row = stats_df(sentences).iloc[0]

    assert row["% l"] == 50
    assert row["% r"] == 50
    assert row["avg_len"] == 1
    assert row["n_sentences"] == 1


def test_empty_dependency_bucket_returns_nan_instead_of_crashing():
    row = stats_df({"model": []}).iloc[0]

    assert np.isnan(row["avg_len"])
    assert row["n_sentences"] == 0


def test_length_buckets_are_inclusive():
    buckets = create_length_dict({"m": [[1], [1, 2], [1, 2, 3]]}, 3, 2)

    assert buckets["1_2"]["m"] == [[1], [1, 2]]
    assert buckets["3_"]["m"] == [[1, 2, 3]]


def test_get_tree_spans_propagates_ignore_non_terminal():
    tree = Tree.fromstring("(ROOT (S (NP A dog) (VP runs fast)))")

    labels = [label for _, label in get_tree_spans(tree, True, True)]

    assert labels == ["-", "-", "-"]


def test_conllu_ignores_comments_multiword_tokens_and_empty_nodes(tmp_path: Path):
    conllu = tmp_path / "sample.conllu"
    conllu.write_text(
        "# constituency = (ROOT (S I go))\n"
        "1-2\tcan't\t_\t_\t_\t_\t_\t_\t_\t_\n"
        "1\tI\tI\tPRON\t_\t_\t2\tnsubj\t_\t_\n"
        "2\tgo\tgo\tVERB\t_\t_\t0\troot\t_\t_\n"
        "2.1\tghost\tghost\tX\t_\t_\t2\tdep\t_\t_\n",
        encoding="utf-8",
    )

    parsed = CoNLLu(conllu)

    assert parsed.attributes["word"] == ["I", "go"]
    assert parsed.sentence_length == [2]
    assert parsed.constituency == ["(ROOT (S I go))"]
