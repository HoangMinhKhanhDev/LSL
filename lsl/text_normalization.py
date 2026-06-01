"""Unicode and Vietnamese text normalization helpers for LSL."""
from __future__ import annotations

import re
import unicodedata

_SUSPICIOUS_MOJIBAKE = (
    "ÃƒÆ’",
    "Ãƒâ€ž",
    "Ãƒâ€š",
    "ÃƒÂ¡Ã‚Âº",
    "ÃƒÂ¡Ã‚Â»",
    "Ãƒâ€ ",
    "áº",
    "á»",
    "Ã‚",
    "Ãƒ",
    "Â",
    "Ã",
    "Æ",
    "â",
    "Ã¢â‚¬â„¢",
    "Ã¢â‚¬Å“",
    "Ã¢â‚¬Â",
)
_INVISIBLE_CHARS = {
    "\ufeff",
    "\u00ad",
    "\u034f",
    "\u180e",
    "\u200b",
    "\u200c",
    "\u200d",
    "\u2060",
}
_VIETNAMESE_HINT_CHARS = "".join(
    [
        "\u0103",
        "\u00e2",
        "\u0111",
        "\u00ea",
        "\u00f4",
        "\u01a1",
        "\u01b0",
        "\u00e1",
        "\u00e0",
        "\u1ea3",
        "\u00e3",
        "\u1ea1",
        "\u1eaf",
        "\u1eb1",
        "\u1eb3",
        "\u1eb5",
        "\u1eb7",
        "\u1ea5",
        "\u1ea7",
        "\u1ea9",
        "\u1eab",
        "\u1ead",
        "\u00e9",
        "\u00e8",
        "\u1ebb",
        "\u1ebd",
        "\u1eb9",
        "\u1ebf",
        "\u1ec1",
        "\u1ec3",
        "\u1ec5",
        "\u1ec7",
        "\u00ed",
        "\u00ec",
        "\u1ec9",
        "\u0129",
        "\u1ecb",
        "\u00f3",
        "\u00f2",
        "\u1ecf",
        "\u00f5",
        "\u1ecd",
        "\u1ed1",
        "\u1ed3",
        "\u1ed5",
        "\u1ed7",
        "\u1ed9",
        "\u1edb",
        "\u1edd",
        "\u1edf",
        "\u1ee1",
        "\u1ee3",
        "\u00fa",
        "\u00f9",
        "\u1ee7",
        "\u0169",
        "\u1ee5",
        "\u1ee9",
        "\u1eeb",
        "\u1eed",
        "\u1eef",
        "\u1ef1",
        "\u00fd",
        "\u1ef3",
        "\u1ef7",
        "\u1ef9",
        "\u1ef5",
    ]
)
_VIETNAMESE_HINT_RE = re.compile(f"[{re.escape(_VIETNAMESE_HINT_CHARS)}]")


def repair_utf8_mojibake(text: str) -> str:
    """Repair common UTF-8-as-Latin-1 mojibake when it is obvious."""
    value = str(text)
    if not any(marker in value for marker in _SUSPICIOUS_MOJIBAKE):
        return value
    try:
        repaired = value.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value
    original_score = sum(value.count(marker) for marker in _SUSPICIOUS_MOJIBAKE)
    repaired_score = sum(repaired.count(marker) for marker in _SUSPICIOUS_MOJIBAKE)
    return repaired if repaired_score < original_score else value


def strip_diacritics(text: str) -> str:
    """Remove combining marks while preserving the base characters."""
    decomposed = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _strip_invisible_chars(value: str) -> str:
    out = []
    for ch in value:
        if ch in "\n\t":
            out.append(ch)
            continue
        if ch in _INVISIBLE_CHARS:
            continue
        if unicodedata.category(ch).startswith("C"):
            continue
        out.append(ch)
    return "".join(out)


def normalize_text(
    text: str,
    *,
    normalize_unicode: bool = True,
    normalization_form: str = "NFC",
    compatibility_normalization: bool = False,
    vietnamese_normalization: bool = False,
    repair_mojibake: bool = True,
    lowercase: bool = False,
    strip_invisible: bool = True,
) -> str:
    """Normalize text while preserving Vietnamese diacritics and spacing."""
    value = str(text)
    if repair_mojibake:
        value = repair_utf8_mojibake(value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    if normalize_unicode:
        if compatibility_normalization:
            value = unicodedata.normalize("NFKC", value)
        value = unicodedata.normalize(str(normalization_form or "NFC"), value)
    if strip_invisible:
        value = _strip_invisible_chars(value)
    if vietnamese_normalization:
        value = re.sub(r"[ \t]+([.,!?;:)\]])", r"\1", value)
        value = re.sub(r"([(])[\t ]+", r"\1", value)
        value = re.sub(r"[ \t]+", " ", value)
        value = re.sub(r"\n{3,}", "\n\n", value)
    if lowercase:
        value = value.lower()
    return value


def lexical_key(
    text: str,
    *,
    normalize_unicode: bool = True,
    compatibility_normalization: bool = True,
    repair_mojibake: bool = True,
    lowercase: bool = True,
    strip_accents: bool = True,
) -> str:
    """Canonical lexical key used for multilingual alias matching."""
    value = normalize_text(
        text,
        normalize_unicode=normalize_unicode,
        normalization_form="NFC",
        compatibility_normalization=compatibility_normalization,
        vietnamese_normalization=True,
        repair_mojibake=repair_mojibake,
        lowercase=lowercase,
        strip_invisible=True,
    )
    if strip_accents:
        value = strip_diacritics(value)
    value = re.sub(r"[^\w\s]+", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip()
    return "_".join(part for part in value.split(" ") if part)


def token_variants(
    text: str,
    *,
    lowercase: bool = True,
) -> tuple[str, ...]:
    """Return stable lexical variants useful for alias and morphology matching."""
    exact = normalize_text(
        text,
        normalize_unicode=True,
        normalization_form="NFC",
        compatibility_normalization=True,
        vietnamese_normalization=True,
        repair_mojibake=True,
        lowercase=lowercase,
        strip_invisible=True,
    )
    accentless = strip_diacritics(exact)
    compact_exact = lexical_key(exact, strip_accents=False)
    compact_accentless = lexical_key(exact, strip_accents=True)
    variants = []
    for candidate in (exact, accentless, compact_exact, compact_accentless):
        if candidate and candidate not in variants:
            variants.append(candidate)
    return tuple(variants)


def looks_vietnamese(text: str) -> bool:
    value = normalize_text(
        text,
        normalize_unicode=True,
        normalization_form="NFC",
        compatibility_normalization=True,
        vietnamese_normalization=False,
        repair_mojibake=True,
        lowercase=True,
        strip_invisible=True,
    )
    if _VIETNAMESE_HINT_RE.search(value):
        return True
    return any(marker in value for marker in ("\u0103", "\u00e2", "\u0111", "\u00ea", "\u00f4", "\u01a1", "\u01b0"))
