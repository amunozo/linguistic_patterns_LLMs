"""Constituency-tree analysis helpers used by the paper notebook."""

from collections import Counter
from pathlib import Path


def get_tree_spans(tree, root, ignore_non_terminal=False):
    """Return ``(leaves, label)`` pairs for the non-root tree nodes."""
    spans = []
    if isinstance(tree, str):
        return spans

    if not root:
        if len(tree.leaves()) == 1 and "@" in tree.label():
            spans.append((tree.leaves(), tree.label()))
        elif len(tree.leaves()) > 1:
            label = "-" if ignore_non_terminal else tree.label()
            spans.append((tree.leaves(), label))

    for child in tree:
        if not isinstance(child, str):
            spans.extend(
                get_tree_spans(
                    child,
                    root=False,
                    ignore_non_terminal=ignore_non_terminal,
                )
            )
    return spans


def avg_non_terminal_len(path_file):
    """Return non-terminal frequencies and average span lengths."""
    from nltk.tree import Tree

    path = Path(path_file)
    non_terminals = []
    lengths_by_label = {}
    with path.open(encoding="utf-8") as stream:
        trees = [line.strip() for line in stream if line.strip()]

    for serialized_tree in trees:
        tree = Tree.fromstring(serialized_tree, remove_empty_top_bracketing=True)
        tree.collapse_unary(collapsePOS=True, collapseRoot=True, joinChar="@")
        for span_text, span_label in get_tree_spans(tree, root=True):
            uppermost_label = span_label.split("@")[0]
            non_terminals.append(uppermost_label)
            lengths_by_label.setdefault(uppermost_label, []).append(len(span_text))

    average_lengths = {
        label: float(sum(lengths) / len(lengths))
        for label, lengths in lengths_by_label.items()
    }
    return Counter(non_terminals), average_lengths
