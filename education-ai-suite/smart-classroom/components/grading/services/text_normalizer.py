import re
import unicodedata
from typing import Optional, Dict, Any


def normalize_text(text: str, preserve_original: bool = True) -> str:
    """Apply NFKC to unify full-width/half-width characters"""
    if not text:
        return text
    return unicodedata.normalize('NFKC', text)


# LaTeX math delimiters and typographic-noise commands stripped before
# comparing answers. Only used for matching, never rewrites stored text.
_LATEX_DELIMITERS = [r'\(', r'\)', r'\[', r'\]', '$$', '$']
_LATEX_NOISE = [r'\quad', r'\qquad', r'\,', r'\;', r'\!', r'\left', r'\right']


def normalize_for_match(text: str) -> str:
    """Aggressively normalize a string for answer comparison.

    NFKC -> strip LaTeX delimiters -> unwrap \\text{X} -> drop typographic
    noise commands -> remove all whitespace. Punctuation is preserved.
    """
    if not text:
        return ''
    s = normalize_text(text)
    for delim in _LATEX_DELIMITERS:
        s = s.replace(delim, '')
    s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)
    for cmd in _LATEX_NOISE:
        s = s.replace(cmd, '')
    s = re.sub(r'\s+', '', s)
    return s


def normalize_with_metadata(text: str) -> Dict[str, Any]:
    """Normalize and return dict with 'normalized', 'original', 'changed'"""
    if not text:
        return {'normalized': text, 'original': text, 'changed': False}

    normalized = unicodedata.normalize('NFKC', text)
    return {
        'normalized': normalized,
        'original': text,
        'changed': normalized != text
    }


def should_normalize(text: str, context: Optional[Dict[str, Any]] = None) -> bool:
    """Check if text should be normalized based on context"""
    if not context:
        return True

    region_type = context.get('region_type', '')
    if region_type in ['display_formula', 'inline_formula']:
        return False

    return True

