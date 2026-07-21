import unicodedata
from typing import Optional, Dict, Any


def normalize_text(text: str, preserve_original: bool = True) -> str:
    """Apply NFKC to unify full-width/half-width characters"""
    if not text:
        return text
    return unicodedata.normalize('NFKC', text)


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

