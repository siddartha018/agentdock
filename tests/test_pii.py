from agentdock.pii import redact, redact_content


def test_redacts_each_kind():
    r = redact("ssn 123-45-6789, iban DE89 3704 0044 0532 0130 00, a@b.co, +1 415 555 0134")
    assert "123-45-6789" not in r.text and "DE89" not in r.text and "a@b.co" not in r.text
    assert "555" not in r.text
    assert r.counts.keys() >= {"SSN", "IBAN", "EMAIL", "PHONE"}


def test_leaves_normal_text_and_money_alone():
    text = "I have 12 vacation days and spent $1,450 on a flight to Berlin in 2026."
    assert redact(text).text == text


def test_content_blocks():
    out, counts = redact_content([{"type": "text", "text": "ssn 123-45-6789"}, {"type": "image"}])
    assert out[0]["text"] == "ssn [SSN REDACTED]" and out[1] == {"type": "image"}
    assert counts == {"SSN": 1}
