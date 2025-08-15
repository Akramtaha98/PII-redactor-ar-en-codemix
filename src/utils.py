import unicodedata

ARABIC_INDIC_DIGITS = dict(zip("٠١٢٣٤٥٦٧٨٩", "0123456789"))
EASTERN_ARABIC_INDIC_DIGITS = dict(zip("۰۱۲۳۴۵۶۷۸۹", "0123456789"))

def normalize_arabic_digits(text: str) -> str:
    def map_digit(ch):
        if ch in ARABIC_INDIC_DIGITS:
            return ARABIC_INDIC_DIGITS[ch]
        if ch in EASTERN_ARABIC_INDIC_DIGITS:
            return EASTERN_ARABIC_INDIC_DIGITS[ch]
        return ch
    return "".join(map_digit(c) for c in text)

def strip_tatweel(s: str) -> str:
    return s.replace("\u0640", "")  # tatweel

def normalize_text_basic(s: str) -> str:
    s = normalize_arabic_digits(s)
    s = strip_tatweel(s)
    s = unicodedata.normalize("NFKC", s)
    return s