"""PII redaction applied by the platform on the way in and on the way out.

Agents don't have to remember to scrub anything: the wrapper redacts user input
before the agent sees it, and everything the agent produces (text, tool results
and tool-call arguments) before it is stored, returned or streamed.

Two engines, chosen with ``AGENTDOCK_PII_ENGINE`` or ``pii_engine`` in
``config/platform.toml``:

- ``patterns`` (default, no dependencies): SSN, IBAN, card, email, phone,
  street address, UK postcode, date of birth, and person names found from the
  company directory or from context ("my name is ...", "Dr. ...").
- ``presidio``: everything above plus Microsoft Presidio's spaCy NER for names
  and locations the patterns can't know. Install with ``uv sync --extra ner``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

STREET = (
    r"Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl|"
    r"Terrace|Close|Crescent|Parkway|Pkwy|Highway|Hwy|Square|Sq|Strasse|Straße"
)

PATTERNS: dict[str, re.Pattern[str]] = {
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "IBAN": re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b"),
    "CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "PHONE": re.compile(r"(?<!\w)\+?\d{1,3}[ .-]?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}\b"),
    "DOB": re.compile(
        r"\b(?:DOB|D\.O\.B\.|date of birth|birth ?date|born(?: on)?)\s*:?\s*"
        r"(?:\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{4}-\d{2}-\d{2}|"
        r"(?:\d{1,2}\s+)?[A-Z][a-z]{2,8}\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}|\d{1,2}\s+[A-Z][a-z]{2,8}\s+\d{4})",
        re.IGNORECASE,
    ),
    "ADDRESS": re.compile(
        rf"\b\d{{1,6}}[A-Za-z]?\s+(?:[A-Z][a-z]+\.?\s+){{1,4}}(?:{STREET})\b\.?"
        r"(?:,?\s*(?:Apt|Apartment|Suite|Unit|Flat|#)\.?\s*\w+)?"
        r"(?:,\s*[A-Z][a-z]+(?:\s[A-Z][a-z]+){0,2})?"
        r"(?:,?\s*[A-Z]{2})?(?:\s+\d{5}(?:-\d{4})?)?"
    ),
    "POSTCODE": re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s+\d[A-Z]{2}\b"),
    "PERSON": re.compile(
        r"(?:(?<=\bmy name is )|(?<=\bname: )|(?<=\bName: )|(?<=\bMy name is ))"
        r"[A-Z][a-z]+(?:[ -][A-Z][a-z]+){0,2}"
        r"|\b(?:Mr|Mrs|Ms|Miss|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s[A-Z][a-z]+)?"
    ),
}

# Longest thing a pattern can match. A streaming redactor never releases text
# this close to the end of its buffer, so a value split across chunks is caught.
STREAM_HOLDBACK = 96


@dataclass
class Redaction:
    text: str
    counts: dict[str, int]

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def _merge(into: dict[str, int], more: dict[str, int]) -> None:
    for k, v in more.items():
        into[k] = into.get(k, 0) + v


@lru_cache(maxsize=1)
def _directory_names() -> re.Pattern[str] | None:
    """Full and first+last names of everyone in the company directory."""
    try:
        from agentdock.identity import directory
    except Exception:  # pragma: no cover - pii must work standalone
        return None
    names = sorted(
        {r["name"] for r in directory().users.values() if r.get("role") != "service" and r.get("name")},
        key=len,
        reverse=True,
    )
    if not names:
        return None
    return re.compile(r"\b(?:" + "|".join(re.escape(n) for n in names) + r")\b")


def engine_name() -> str:
    name = os.environ.get("AGENTDOCK_PII_ENGINE")
    if name:
        return name
    try:
        from agentdock.evals import platform_settings

        return str(platform_settings().get("pii_engine", "patterns"))
    except Exception:
        return "patterns"


def ner_entities() -> tuple[str, ...]:
    """Which NER entities to redact. Default PERSON only: a bare city or country
    ("a flight to Berlin") is rarely personal data, and redacting it hurts answers."""
    raw = os.environ.get("AGENTDOCK_PII_NER_ENTITIES")
    if raw:
        return tuple(e.strip().upper() for e in raw.split(",") if e.strip())
    try:
        from agentdock.evals import platform_settings

        return tuple(platform_settings().get("pii_ner_entities", ["PERSON"]))
    except Exception:
        return ("PERSON",)


@lru_cache(maxsize=4)
def _presidio(entities: tuple[str, ...]) -> Callable[[str], list[tuple[int, int, str]]]:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    nlp = NlpEngineProvider(
        nlp_configuration={"nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]}
    ).create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"])
    label = {"PERSON": "PERSON", "LOCATION": "ADDRESS"}

    def find(text: str) -> list[tuple[int, int, str]]:
        hits = analyzer.analyze(text=text, language="en", entities=[e for e in entities if e in label], score_threshold=0.6)
        return [(h.start, h.end, label[h.entity_type]) for h in hits]

    return find


def _spans(text: str) -> list[tuple[int, int, str]]:
    spans = [(m.start(), m.end(), label) for label, p in PATTERNS.items() for m in p.finditer(text)]
    names = _directory_names()
    if names is not None:
        spans += [(m.start(), m.end(), "PERSON") for m in names.finditer(text)]
    if engine_name() == "presidio":
        spans += _presidio(ner_entities())(text)
    # Resolve overlaps: earliest start wins, then the longest span.
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    chosen: list[tuple[int, int, str]] = []
    for s in spans:
        if chosen and s[0] < chosen[-1][1]:
            continue
        chosen.append(s)
    return chosen


def redact(text: str) -> Redaction:
    if not text:
        return Redaction(text, {})
    counts: dict[str, int] = {}
    out, pos = [], 0
    for start, end, label in _spans(text):
        out.append(text[pos:start])
        out.append(f"[{label} REDACTED]")
        counts[label] = counts.get(label, 0) + 1
        pos = end
    out.append(text[pos:])
    return Redaction("".join(out), counts)


def redact_content(content: object) -> tuple[object, dict[str, int]]:
    """Redact a message's content, which may be a string or a list of content blocks."""
    if isinstance(content, str):
        r = redact(content)
        return r.text, r.counts
    if isinstance(content, list):
        out, counts = [], {}
        for block in content:
            if isinstance(block, str):
                r = redact(block)
                out.append(r.text)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                r = redact(block["text"])
                out.append({**block, "text": r.text})
            else:
                out.append(block)
                continue
            _merge(counts, r.counts)
        return out, counts
    return content, {}


def redact_value(value: Any) -> tuple[Any, dict[str, int]]:
    """Redact every string inside a JSON-like value (tool-call arguments)."""
    if isinstance(value, str):
        r = redact(value)
        return r.text, r.counts
    counts: dict[str, int] = {}
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            out[k], c = redact_value(v)
            _merge(counts, c)
        return out, counts
    if isinstance(value, (list, tuple)):
        items = []
        for v in value:
            nv, c = redact_value(v)
            items.append(nv)
            _merge(counts, c)
        return type(value)(items), counts
    return value, counts


def redact_tool_calls(msg: Any) -> tuple[Any, dict[str, int]]:
    """Redact tool-call arguments on an AIMessage, including the provider's raw copy."""
    counts: dict[str, int] = {}
    calls = getattr(msg, "tool_calls", None) or []
    if not calls:
        return msg, counts
    new_calls = []
    for tc in calls:
        args, c = redact_value(tc.get("args", {}))
        _merge(counts, c)
        new_calls.append({**tc, "args": args})
    update: dict[str, Any] = {"tool_calls": new_calls}
    raw = (getattr(msg, "additional_kwargs", None) or {}).get("tool_calls")
    if raw:
        fixed = []
        for rc in raw:
            fn = dict(rc.get("function") or {})
            if isinstance(fn.get("arguments"), str):
                try:
                    args, _ = redact_value(json.loads(fn["arguments"]))
                    fn["arguments"] = json.dumps(args)
                except json.JSONDecodeError:
                    fn["arguments"] = redact(fn["arguments"]).text
            fixed.append({**rc, "function": fn})
        update["additional_kwargs"] = {**msg.additional_kwargs, "tool_calls": fixed}
    if not counts:
        return msg, counts
    return msg.model_copy(update=update), counts


class StreamRedactor:
    """Redacts text that arrives in chunks without ever releasing raw PII.

    Text is released only up to a safe cut: at least ``STREAM_HOLDBACK``
    characters before the end of the buffer, moved back to a sentence end (so
    NER sees whole sentences), and never inside a detected span.
    """

    def __init__(self, holdback: int = STREAM_HOLDBACK) -> None:
        self.buf = ""
        self.holdback = holdback
        self.counts: dict[str, int] = {}

    def _cut(self) -> int:
        limit = len(self.buf) - self.holdback
        if limit <= 0:
            return 0
        window = self.buf[:limit]
        ends = [m.end() for m in re.finditer(r"[.!?\n]\s+", window)]
        if ends:
            cut = ends[-1]
        elif engine_name() != "presidio" or len(self.buf) > 4 * self.holdback:
            # Patterns don't need sentence context, so any whitespace is a safe cut.
            # NER does: wait for a sentence end unless the buffer is getting long.
            cut = window.rfind(" ") + 1
        else:
            cut = 0
        for start, end, _ in _spans(self.buf):
            if start < cut < end:
                cut = start
        return cut

    def feed(self, text: str) -> str:
        self.buf += text
        cut = self._cut()
        if cut <= 0:
            return ""
        out, self.buf = redact(self.buf[:cut]), self.buf[cut:]
        _merge(self.counts, out.counts)
        return out.text

    def flush(self) -> str:
        out, self.buf = redact(self.buf), ""
        _merge(self.counts, out.counts)
        return out.text
