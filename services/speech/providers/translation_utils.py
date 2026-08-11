"""Glossary substitution and heuristic confidence scoring for translation."""

import re

_MAX_GLOSSARY_TERMS = 200


def apply_glossary(text: str, glossary: dict[str, str] | None) -> tuple[str, list[str]]:
    """Apply whole-word glossary substitutions to ``text``.

    Returns ``(transformed_text, matched_terms)``. Terms are matched
    case-insensitively on word boundaries; longer terms take precedence so
    multi-word phrases are replaced before their individual words.
    """
    if not glossary:
        return text, []

    terms = [t for t in glossary if t]
    if not terms:
        return text, []

    # Limit to avoid pathological regexes.
    terms = terms[:_MAX_GLOSSARY_TERMS]
    terms.sort(key=len, reverse=True)

    matched: list[str] = []
    transformed = text
    for term in terms:
        replacement = glossary[term]
        pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
        new_text, count = pattern.subn(replacement, transformed)
        if count:
            matched.append(term)
            transformed = new_text
    return transformed, matched


def compute_translation_confidence(
    source_text: str,
    translated_text: str,
    source_language_code: str,
    target_language_code: str,
    glossary_matches: list[str] | None = None,
) -> float:
    """Return a heuristic confidence score in ``[0.0, 1.0]``.

    Sarvam does not return translation confidence, so this is a deterministic
    heuristic based on whether a translation actually occurred, target/source
    length sanity, and glossary coverage.
    """
    if not translated_text:
        return 0.0

    glossary_matches = glossary_matches or []
    score = 0.55

    # A translation must have happened: source != target and text changed.
    if source_language_code == "auto" or source_language_code != target_language_code:
        if source_text.strip() != translated_text.strip():
            score += 0.2
        else:
            score += 0.05
    else:
        # Same source and target: no-op translation is low confidence.
        return 0.3

    # Length sanity: translation length should be in a sane band vs source.
    src_len = max(1, len(source_text))
    ratio = len(translated_text) / src_len
    if 0.3 <= ratio <= 3.0:
        score += 0.15
    else:
        score -= 0.15

    # Glossary coverage adds confidence proportional to term coverage.
    if glossary_matches:
        covered = sum(len(t) for t in glossary_matches)
        coverage = min(1.0, covered / max(1, len(source_text)))
        score += 0.1 * coverage

    return round(min(1.0, max(0.0, score)), 2)
