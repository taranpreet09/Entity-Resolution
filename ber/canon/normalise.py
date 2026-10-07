"""
ber.canon.normalise — Control Baseline Canonicalisation
=========================================================
Deliberately lossy baseline that mimics the old "exact keys + size gate"
blocking pipeline.

Algorithm (applied in order):
  1. NFKD Unicode decomposition
  2. ASCII encoding with 'ignore' error mode (drops all non-ASCII)
  3. Lowercase
  4. Strip punctuation (keep letters, digits, spaces)
  5. Collapse consecutive whitespace to single space
  6. Strip leading/trailing whitespace

This is the CONTROL canonicalisation.  More sophisticated forms (phonetic,
abbreviation expansion, legal-form stripping) belong in later phases.

Key outputs:
  - ``canonicalise(text)``  — single string → canonical string
  - ``canon_stats(records)`` — count empty-after-canon vs non-empty-raw
  - ``outward_code(postcode)`` — first part of a UK postcode (e.g. "SW1A")
  - ``first_token(text)`` — first whitespace-separated token of canonical form
  - ``first_n_chars(text, n)`` — first n chars of canonical form

Usage::

    from ber.canon.normalise import canonicalise, canon_stats, outward_code

    canonicalise("ACME Ltd.") -> "acme ltd"
    outward_code("SW1A 2AA") -> "SW1A"
    canon_stats([{"name": "ACME Ltd.", "addr": ""}]) -> {...}
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

# Punctuation stripper: keep letters [a-z], digits [0-9], spaces
_PUNCT_RE = re.compile(r"[^a-z0-9 ]")
_SPACE_RE = re.compile(r"\s+")

# UK postcode: outward code is everything before the last 3 characters
# Pattern: 1-2 letters + 1-2 digits (+ optional letter)
_POSTCODE_OUTWARD_RE = re.compile(
    r"^([A-Z]{1,2}[0-9][0-9A-Z]?)", re.IGNORECASE
)


def canonicalise(text: str | None) -> str:
    """Apply the control canonicalisation pipeline to a single string.

    Parameters
    ----------
    text:
        Raw field value.  None and empty string both return "".

    Returns
    -------
    str
        Canonical form.  May be empty if the input contained only
        non-ASCII or punctuation characters.
    """
    if not text:
        return ""
    # Step 1: NFKD decomposition
    s = unicodedata.normalize("NFKD", text)
    # Step 2: ASCII-only (drop non-ASCII accents and symbols)
    s = s.encode("ascii", errors="ignore").decode("ascii")
    # Step 3: lowercase
    s = s.lower()
    # Step 4: strip punctuation (keep alnum + space)
    s = _PUNCT_RE.sub(" ", s)
    # Step 5 & 6: collapse whitespace + strip
    s = _SPACE_RE.sub(" ", s).strip()
    return s


def outward_code(postcode: str | None) -> str:
    """Extract the outward (district) code from a UK postcode.

    Examples
    --------
    >>> outward_code("SW1A 2AA") -> "SW1A"
    >>> outward_code("EC1A1BB")  -> "EC1A"
    >>> outward_code("")         -> ""
    >>> outward_code("INVALID")  -> ""
    """
    if not postcode:
        return ""
    # Normalise: strip spaces, uppercase
    cleaned = postcode.strip().upper().replace(" ", "")
    # UK postcodes: inward code is always 3 chars (digit + 2 letters)
    # outward = everything except last 3 chars
    if len(cleaned) >= 5:
        outward = cleaned[: len(cleaned) - 3]
        m = _POSTCODE_OUTWARD_RE.match(outward)
        if m:
            return m.group(1)
    # Fallback: try matching the full string for truncated postcodes
    m = _POSTCODE_OUTWARD_RE.match(cleaned)
    return m.group(1) if m else ""


def postcode_district(postcode: str | None) -> str:
    """Extract the postcode district (first letter(s) + digit(s)).

    Examples
    --------
    >>> postcode_district("SW1A 2AA") -> "SW1"
    >>> postcode_district("EC1A 1BB") -> "EC1"
    >>> postcode_district("M1 1AE")   -> "M1"
    """
    if not postcode:
        return ""
    cleaned = postcode.strip().upper().replace(" ", "")
    m = re.match(r"^([A-Z]{1,2}[0-9])", cleaned)
    return m.group(1) if m else ""


def first_token(text: str) -> str:
    """Return the first whitespace-separated token of the canonical form."""
    canon = canonicalise(text)
    if not canon:
        return ""
    return canon.split()[0]


def first_n_chars(text: str, n: int) -> str:
    """Return the first n characters of the canonical form."""
    canon = canonicalise(text)
    return canon[:n]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def canon_stats(
    records: list[dict[str, Any]],
    fields: list[str],
) -> dict[str, dict[str, int]]:
    """Count empty-after-canon vs non-empty-raw for each specified field.

    Parameters
    ----------
    records:
        List of record dicts.  Each dict may have the field as a direct key,
        or nested inside ``raw_fields`` (parsed as JSON if needed).
    fields:
        List of field names to check.

    Returns
    -------
    dict mapping field_name → {"raw_non_empty": int, "canon_empty_from_non_empty": int}
        ``raw_non_empty``: records where the raw field was non-empty.
        ``canon_empty_from_non_empty``: subset where canon output is "".
        These are the CANONICALIZATION-stage losses.
    """
    import json as _json

    counts: dict[str, dict[str, int]] = {
        f: {"raw_non_empty": 0, "canon_empty_from_non_empty": 0}
        for f in fields
    }

    for rec in records:
        for field in fields:
            # Try direct key first, then inside raw_fields JSON
            if field in rec:
                raw_val = rec[field]
            else:
                raw_fields = rec.get("raw_fields", "{}")
                if isinstance(raw_fields, str):
                    try:
                        raw_fields = _json.loads(raw_fields)
                    except Exception:
                        raw_fields = {}
                raw_val = raw_fields.get(field, "")

            raw_str = str(raw_val) if raw_val is not None else ""
            if raw_str.strip():  # raw non-empty
                counts[field]["raw_non_empty"] += 1
                if not canonicalise(raw_str):
                    counts[field]["canon_empty_from_non_empty"] += 1

    return counts


# ---------------------------------------------------------------------------
# Legal-suffix stripping (control_v2 K4)
# ---------------------------------------------------------------------------

import json as _json_mod
from functools import lru_cache

_SUFFIX_CONFIG_PATH = (
    __import__("pathlib").Path(__file__).resolve().parent.parent.parent
    / "configs" / "legal_suffixes.json"
)


@lru_cache(maxsize=1)
def _load_suffix_config() -> tuple[list[str], list[str], "re.Pattern"]:
    """Load suffix list and build a compiled end-of-string regex.

    Returns (and_aliases_canon, suffixes_canon, suffix_regex).
    suffix_regex matches trailing suffix tokens at end of string.
    """
    try:
        cfg = _json_mod.loads(_SUFFIX_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return ([], [], re.compile(r"$^"))  # never matches

    and_aliases = [canonicalise(a) for a in cfg.get("and_aliases", [])]
    suffixes = [canonicalise(s) for s in cfg.get("suffixes", [])]
    # Sort longest first so multi-word suffixes match before their sub-terms
    suffixes_sorted = sorted(suffixes, key=lambda s: len(s.split()), reverse=True)
    # Build regex: one or more (space + suffix) occurrences at end of string
    parts = [re.escape(s) for s in suffixes_sorted if s]
    pattern = r"(?:\s+(?:" + "|".join(parts) + r"))+$" if parts else r"$^"
    suffix_regex = re.compile(pattern)
    return (and_aliases, suffixes_sorted, suffix_regex)


def strip_legal_suffixes(text: str | None) -> str:
    """Return the canonical name with UK legal-form suffixes removed.

    Algorithm:
    1. Apply standard canonicalisation (NFKD+ASCII+lower+strip-punct+collapse).
    2. Replace any surviving and_alias tokens with 'and'.
    3. Use a pre-compiled regex to strip all trailing suffix tokens in a loop
       until no more match (handles stacked suffixes like "limited partnership").

    Note: '&' is stripped as punctuation by canonicalise() before this
    function sees it, so 'and_aliases' only helps when 'and' is spelled out.

    Examples
    --------
    >>> strip_legal_suffixes("ACME LIMITED")      -> "acme"
    >>> strip_legal_suffixes("BARCLAYS PLC")      -> "barclays"
    >>> strip_legal_suffixes("BARCLAYS & CO LTD") -> "barclays"
    >>> strip_legal_suffixes("LTD")               -> ""
    >>> strip_legal_suffixes("")                  -> ""
    """
    canon = canonicalise(text)
    if not canon:
        return ""

    and_aliases, _, suffix_regex = _load_suffix_config()

    # Normalise any surviving and-alias tokens to 'and'
    if and_aliases:
        tokens = canon.split()
        tokens = ["and" if t in and_aliases else t for t in tokens]
        canon = " ".join(tokens)

    # Strip all trailing suffix tokens; loop handles stacked suffixes.
    # Also handle the degenerate case where the entire string is a suffix.
    prev = None
    while canon != prev:
        prev = canon
        canon = suffix_regex.sub("", canon).rstrip()
        # Full-string-is-suffix: check if what remains is itself a suffix
        if not canon:
            break
        # Check exact match (no leading space possible on remaining string)
        _, suffixes_sorted, _ = _load_suffix_config()
        if canon in set(suffixes_sorted):
            canon = ""
            break

    return canon
