import json
from datetime import date

import pytest

from scripts.download_articles import download_data, iter_months
from scripts.generate import attach_generations, build_prompt, model_spec
from scripts.parse_classify import attach_labels, attach_parses
from scripts.prepare_articles import prepare_articles


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeResponse({"response": {"docs": []}})


class FakeSentence:
    def __init__(self, tree):
        self.constituency = tree


class FakeDocument:
    def __init__(self):
        self.sentences = [FakeSentence("(ROOT text)")]

    def to_dict(self):
        return [[{"id": 1, "text": "text", "head": 0}]]


def article(identifier="1", publication="2024-01-24T10:00:00Z"):
    return {
        "_id": identifier,
        "headline": {"main": "A headline"},
        "lead_paragraph": "The first three words continue here.",
        "pub_date": publication,
    }


def test_month_iteration_crosses_year_boundary():
    assert list(iter_months(date(2023, 11, 1), date(2024, 2, 1))) == [
        (2023, 11),
        (2023, 12),
        (2024, 1),
        (2024, 2),
    ]


def test_download_uses_params_and_zero_padded_filenames(tmp_path):
    session = FakeSession()

    paths = download_data(
        date(2024, 1, 1),
        date(2024, 1, 1),
        tmp_path,
        "secret",
        session=session,
    )

    assert paths == [tmp_path / "2024_01.json"]
    assert session.calls[0][1]["params"] == {"api-key": "secret"}
    assert "secret" not in session.calls[0][0]


def test_prepare_articles_filters_dates_empty_text_and_duplicates(tmp_path):
    payload = {
        "response": {
            "docs": [
                article(),
                article(),
                article("old", "2023-09-30T10:00:00Z"),
                {**article("empty"), "lead_paragraph": ""},
            ]
        }
    }
    (tmp_path / "archive.json").write_text(json.dumps(payload), encoding="utf-8")

    result = prepare_articles(
        tmp_path,
        start_date=date(2023, 10, 1),
        end_date=date(2024, 1, 24),
    )

    assert [item["_id"] for item in result] == ["1"]


def test_prompt_and_generated_output_follow_paper_setup():
    source = article()

    assert build_prompt(source) == '"A headline"\nThe first three '
    assert attach_generations([source], ["generated ending"])[0][
        "lead_paragraph"
    ] == "The first three generated ending"
    assert source["lead_paragraph"] == "The first three words continue here."


def test_model_aliases_are_safe():
    assert model_spec("m7=mistralai/Mistral-7B-v0.1") == (
        "m7",
        "mistralai/Mistral-7B-v0.1",
    )
    assert model_spec("org/model") == ("model", "org/model")


def test_annotation_results_remain_aligned():
    articles = [article()]

    attach_parses(articles, [FakeDocument()])
    attach_labels(articles, ["neutral"])

    assert articles[0]["parsed"][0][0]["text"] == "text"
    assert articles[0]["constituents"] == ["(ROOT text)"]
    assert articles[0]["emotion"] == "neutral"


def test_annotation_alignment_mismatch_fails_loudly():
    with pytest.raises(ValueError, match="counts differ"):
        attach_parses([article()], [])
