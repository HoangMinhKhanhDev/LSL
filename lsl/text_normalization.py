"""Unicode and Vietnamese text normalization helpers for LSL."""
from __future__ import annotations

import re
import unicodedata


_SUSPICIOUS_MOJIBAKE = ("Ã", "Ä", "Â", "áº", "á»", "Æ")


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


def normalize_text(
    text: str,
    *,
    normalize_unicode: bool = True,
    normalization_form: str = "NFC",
    vietnamese_normalization: bool = False,
    repair_mojibake: bool = True,
    lowercase: bool = False,
) -> str:
    """Normalize text while preserving Vietnamese diacritics."""
    value = str(text)
    if repair_mojibake:
        value = repair_utf8_mojibake(value)
    if normalize_unicode:
        value = unicodedata.normalize(str(normalization_form or "NFC"), value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = "".join(ch for ch in value if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    if vietnamese_normalization:
        value = re.sub(r"[ \t]+([.,!?;:)\]])", r"\1", value)
        value = re.sub(r"([(])[\t ]+", r"\1", value)
        value = re.sub(r"[ \t]+", " ", value)
        value = re.sub(r"\n{3,}", "\n\n", value)
    if lowercase:
        value = value.lower()
    return value


def looks_vietnamese(text: str) -> bool:
    value = str(text).lower()
    return bool(re.search(r"[ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]", value))
