import re

def extract_sentence(text: str) -> str | None:
    """
    Returns the first complete Arabic/English sentence found in the buffer,
    or None if no complete sentence exists yet.
    Triggers on: . ! ? ؟ (Arabic question mark) newline
    Requires at least 4 characters before the punctuation to avoid false triggers.
    """
    match = re.search(r'[^.!?؟\n]{4,}[.!?؟\n]', text)
    return match.group(0) if match else None