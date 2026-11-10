from agentdock.pii import redact, redact_content


def test_redacts_each_kind():
    r = redact("ssn 123-45-6789, iban DE89 3704 0044 0532 0130 00, a@b.co, +1 415 555 0134")
    assert "123-45-6789" not in r.text and "DE89" not in r.text and "a@b.co" not in r.text
    assert "555" not in r.text
    assert r.counts.keys() >= {"SSN", "IBAN", "EMAIL", "PHONE"}


def test_leaves_normal_text_and_money_alone():  # also with NER on (PERSON only by default)
    text = "I have 12 vacation days and spent $1,450 on a flight to Berlin in 2026."
    assert redact(text).text == text


def test_content_blocks():
    out, counts = redact_content([{"type": "text", "text": "ssn 123-45-6789"}, {"type": "image"}])
    assert out[0]["text"] == "ssn [SSN REDACTED]" and out[1] == {"type": "image"}
    assert counts == {"SSN": 1}


import random

import pytest
from langchain_core.messages import AIMessage

from agentdock.pii import StreamRedactor, redact_tool_calls


@pytest.mark.parametrize(
    "text,label",
    [
        ("ship it to 221B Baker Street, London", "ADDRESS"),
        ("I live at 1600 Pennsylvania Avenue, Washington, DC 20500", "ADDRESS"),
        ("office is 42 Market St., Apt 5, San Francisco, CA 94103", "ADDRESS"),
        ("postcode SW1A 1AA please", "POSTCODE"),
        ("DOB: 04/12/1990", "DOB"),
        ("I was born on March 3, 1988", "DOB"),
        ("date of birth 1990-12-04", "DOB"),
        ("Hi, my name is Priya Raman and I need leave", "PERSON"),
        ("please ask Dr. Alan Grant", "PERSON"),
        ("Asha Rao asked about leave", "PERSON"),  # from the company directory
    ],
)
def test_names_addresses_and_dates(text, label):
    r = redact(text)
    assert label in r.counts, (text, r.text)


@pytest.mark.parametrize(
    "text",
    [
        "Decision: APPROVE for $20.00 (taxi): within policy.",
        "E101 has 3 vacation days and 8 sick days left.",
        "Up to 5 unused days carry over into the next year; the rest expire on 31 March.",
        "Employees may work remotely up to 3 days per week.",
        "Expenses over 1000 USD need manager approval from M200.",
        "Primary caregivers get 16 weeks of paid parental leave.",
    ],
)
def test_no_false_positives_on_agent_answers(text):
    assert redact(text).text == text


def test_tool_call_arguments_are_redacted_including_raw_provider_copy():
    msg = AIMessage(
        "",
        tool_calls=[{"name": "lookup", "args": {"q": "ssn 123-45-6789", "n": 3, "tags": ["a@b.co"]}, "id": "c1"}],
        additional_kwargs={"tool_calls": [{"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": '{"q": "ssn 123-45-6789"}'}}]},
    )
    out, counts = redact_tool_calls(msg)
    assert out.tool_calls[0]["args"] == {"q": "ssn [SSN REDACTED]", "n": 3, "tags": ["[EMAIL REDACTED]"]}
    assert "123-45-6789" not in str(out.additional_kwargs)
    assert counts == {"SSN": 1, "EMAIL": 1}


SECRET_TEXT = (
    "Here is the record. SSN 123-45-6789 and IBAN DE89 3704 0044 0532 0130 00. "
    "Mail a@b.co or call +1 415 555 0134. Ship to 221B Baker Street, London! "
    "Asha Rao, DOB: 04/12/1990. Thanks."
)
RAW = ["123-45-6789", "DE89 3704", "a@b.co", "555 0134", "Baker Street", "Asha Rao", "04/12/1990"]


@pytest.mark.parametrize("seed", range(40))
def test_stream_redactor_never_releases_raw_pii_under_any_chunking(seed):
    rng = random.Random(seed)
    chunks, i = [], 0
    while i < len(SECRET_TEXT):
        n = rng.randint(1, 9)
        chunks.append(SECRET_TEXT[i : i + n])
        i += n
    r = StreamRedactor()
    emitted = [r.feed(c) for c in chunks] + [r.flush()]
    joined = "".join(emitted)
    for piece in emitted:
        assert not any(raw in piece for raw in RAW)
    assert not any(raw in joined for raw in RAW)
    assert joined == redact(SECRET_TEXT).text  # same result as redacting it whole




def _has_presidio() -> bool:
    try:
        import presidio_analyzer  # noqa: F401
        import en_core_web_sm  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _has_presidio(), reason="install with `uv sync --extra ner`")
@pytest.mark.parametrize(
    "text,raw",
    [
        ("Please forward this to Jonathan Meyers in accounting.", "Jonathan Meyers"),
        ("Our client Maria Gonzalez called about the invoice.", "Maria Gonzalez"),
        ("She now lives near Koramangala.", "Koramangala"),
    ],
)
def test_presidio_engine_catches_names_and_places_patterns_cannot(monkeypatch, text, raw):
    monkeypatch.setenv("AGENTDOCK_PII_ENGINE", "presidio")
    monkeypatch.setenv("AGENTDOCK_PII_NER_ENTITIES", "PERSON,LOCATION")
    assert raw not in redact(text).text
    monkeypatch.setenv("AGENTDOCK_PII_ENGINE", "patterns")
    assert raw in redact(text).text  # proves NER did the work
