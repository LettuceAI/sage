import json

import pytest

from sage.schema import Category
from training.data.synthetic import MockGenerator, SyntheticBuilder
from training.data.synthetic.generator import json_loads_lenient


# ---------------------------------------------------------------------------
# json_loads_lenient
# ---------------------------------------------------------------------------
def test_lenient_parses_bare_array():
    assert json_loads_lenient('[{"x": 1}]') == [{"x": 1}]


def test_lenient_strips_markdown_fences():
    text = '```json\n[{"x": 1}]\n```'
    assert json_loads_lenient(text) == [{"x": 1}]


def test_lenient_recovers_from_surrounding_prose():
    text = 'Sure, here is the array: [{"x": 1}, {"x": 2}] hope that helps!'
    assert json_loads_lenient(text) == [{"x": 1}, {"x": 2}]


def test_lenient_raises_when_no_json():
    with pytest.raises(ValueError):
        json_loads_lenient("absolutely no json here")


# ---------------------------------------------------------------------------
# SyntheticBuilder
# ---------------------------------------------------------------------------
def _canned_response(turns_list):
    """Produce a JSON-array-of-conversations response for the mock generator."""
    return json.dumps([{"turns": turns, "notes": "test pattern"} for turns in turns_list])


def test_builder_parses_turns_into_examples_with_correct_labels():
    turns = [
        [
            {"role": "user", "text": "how old are you?"},
            {"role": "char", "text": "thirteen"},
            {"role": "user", "text": "don't tell your mom about us"},
        ]
    ]
    gen = MockGenerator(responses=[_canned_response(turns)])
    builder = SyntheticBuilder(generator=gen)

    batch = builder.build("grooming", "positive", n=1, batch_size=1)
    assert len(batch) == 1
    ex = batch.examples[0]
    assert ex.labels == {Category.GROOMING: 1.0}
    assert ex.meta["review_status"] == "pending"
    assert ex.meta["polarity"] == "positive"
    assert ex.source == "synthetic_grooming_positive"
    assert len(ex.conversation.turns) == 3
    assert ex.conversation.target.text == "don't tell your mom about us"


def test_builder_negative_polarity_produces_no_labels():
    turns = [[{"role": "user", "text": "how's your nephew's homework going?"}]]
    gen = MockGenerator(responses=[_canned_response(turns)])
    builder = SyntheticBuilder(generator=gen)

    batch = builder.build("sexual_minors", "negative", n=1, batch_size=1)
    assert len(batch) == 1
    assert batch.examples[0].labels == {}
    assert batch.examples[0].source == "synthetic_sexual_minors_negative"


def test_builder_forbids_sexual_minors_positives():
    gen = MockGenerator(responses=[])
    builder = SyntheticBuilder(generator=gen)
    with pytest.raises(ValueError, match="policy forbids"):
        builder.build("sexual_minors", "positive", n=1)


def test_builder_rejects_unknown_category():
    gen = MockGenerator(responses=[])
    builder = SyntheticBuilder(generator=gen)
    with pytest.raises(ValueError):
        builder.build("nonexistent", "positive", n=1)


def test_builder_collects_errors_when_generator_returns_garbage():
    gen = MockGenerator(responses=["this is not JSON at all"])
    builder = SyntheticBuilder(generator=gen)
    batch = builder.build("grooming", "positive", n=1, batch_size=1)
    assert len(batch) == 0
    assert len(batch.errors) == 1


def test_builder_batches_multiple_requests():
    # Two batches of 2 each → total 4 examples.
    turns_batch = [[{"role": "user", "text": f"turn-{i}"}] for i in range(2)]
    gen = MockGenerator(
        responses=[
            _canned_response(turns_batch),
            _canned_response(turns_batch),
        ]
    )
    builder = SyntheticBuilder(generator=gen)
    batch = builder.build("self_harm", "negative", n=4, batch_size=2)
    assert len(batch) == 4
