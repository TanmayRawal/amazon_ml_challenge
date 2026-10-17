"""Name and address normalization (pure Python, applied to unique strings in parallel).

Design rules, from the data audit:
- fold accents only on Latin letters; Indic combining signs must survive;
- never discard information that can separate businesses: legal forms are kept in
  their own field, digits in names are kept, blank addresses stay blank;
- remove only injected noise: junk prefixes, template tags, appended ids, pipe
  websites (kept separately as a domain), PO boxes, placeholder tokens.
"""
from __future__ import annotations

import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor

_MOJIBAKE = re.compile("[âÂ][\u0080-\u009f]{1,2}")
_C1 = re.compile("[\u0080-\u009f]")
_INDIC = re.compile("[ऀ-෿]")


def fold(text: str) -> str:
    """Lowercase; strip accents from Latin letters only (keeps Indic vowel signs/viramas)."""
    s = text.lower()
    if s.isascii():
        return s
    s = _C1.sub(" ", _MOJIBAKE.sub(" ", s))
    out = []
    prev_latin = False
    for ch in unicodedata.normalize("NFKD", s):
        if unicodedata.combining(ch):
            if not prev_latin:
                out.append(ch)
            continue
        out.append(ch)
        prev_latin = ch.isalpha() and ch < "ɐ"
    return "".join(out)


def script_of(raw: str) -> str:
    """Classify a raw string as "latin" (ASCII), "indic", "mixed" (Indic + Latin) or "accented"."""
    if raw.isascii():
        return "latin"
    indic = _INDIC.search(raw) is not None
    latin = any(("a" <= c <= "z") or ("A" <= c <= "Z") for c in raw)
    if indic:
        return "mixed" if latin else "indic"
    return "accented"


# ----------------------------------------------------------------------------- names

LEGAL = {
    "llc": "llc", "lllp": "llp", "llp": "llp", "lp": "lp", "pllc": "pllc",
    "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp",
    "ltd": "ltd", "limited": "ltd", "pvt": "pvt", "private": "pvt", "public": "public",
    "pc": "pc", "co": "co", "company": "co", "plc": "plc", "group": "group",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl",
    "sci": "sci", "snc": "snc", "ei": "ei", "gmbh": "gmbh",
}
NAME_STOP = {"the", "and", "of", "an", "et", "de", "du", "des", "la", "le", "les",
             "d", "l", "dr", "shri", "sri", "smt", "mr", "mrs", "messrs"}

_TOKEN = re.compile(r"[a-z0-9ß-ɏऀ-෿]+")
_TEMPLATE = re.compile(r"<[^<>]{1,24}>")
_APPENDED_ID = re.compile(r"\(\s*id\s*:?\s*\d+\s*\)|(?:^|\s)-\s*\d{5,}\s*$|#\s*\d{4,}")
_MS = re.compile(r"\bm\s*/\s*s\b\.?")
_DOTTED = re.compile(r"\b(?:[a-z]\.){2,}")
_APOS_S = re.compile("['’`´]s\\b")
_APOS = re.compile("['’`´]")
_DOMAIN = re.compile(
    r"(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*(?:\.[a-z0-9\-]+)*)"
    r"\.(?:com|in|fr|net|org|co|io|biz|info)\b"
)
_HANDLE = re.compile(r"^[@#]([a-z][a-z0-9_.]{2,})$")
_ORDINAL_TOKEN = re.compile(r"^\d+(?:st|nd|rd|th|hr|hrs|am|pm|k|x)$")
_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s",
                       "6": "g", "7": "t", "8": "b"})


def fix_leet(tok: str) -> str:
    """Undo digit-for-letter substitutions in mixed tokens (L0TUS -> lotus)."""
    if tok.isdigit() or tok.isalpha() or not tok.isascii() or _ORDINAL_TOKEN.match(tok):
        return tok
    digits = sum(c.isdigit() for c in tok)
    if digits <= 2 and len(tok) - digits >= 3:
        return tok.translate(_LEET)
    return tok


def norm_name(raw: str) -> tuple:
    """Return (core, core_key, legal, concat, initials, acro_self, domain, is_domain, script).

    core      space-joined content tokens in original order (legal/stop words removed)
    core_key  sorted unique core tokens (order-insensitive exact key)
    legal     sorted canonical legal forms found anywhere in the name
    concat    concatenated core tokens, or the domain/handle label for domain-only names
    initials  first letters of Latin core tokens (only when there are >= 2 tokens)
    acro_self the name itself when it is an all-caps 2-5 letter acronym
    domain    first website label found in the name (e.g. "| www.shivshakti.com")
    """
    s = fold(raw)
    s = _TEMPLATE.sub(" ", s)
    s = _APPENDED_ID.sub(" ", s)
    s = _MS.sub(" ", s)
    s = _DOTTED.sub(lambda m: m.group(0).replace(".", ""), s)
    s = _APOS.sub("", _APOS_S.sub("", s))
    domains: list[str] = []

    def _grab(m: re.Match) -> str:
        """Record the matched website label (dots/hyphens removed) and blank it out."""
        domains.append(m.group(1).replace(".", "").replace("-", ""))
        return " "

    s = _DOMAIN.sub(_grab, s).replace("|", " ")
    handle = _HANDLE.match(s.strip())
    if handle:
        domains.append(handle.group(1).replace(".", "").replace("_", ""))
        s = ""

    core: list[str] = []
    legal: set[str] = set()
    prev = None
    for tok in _TOKEN.findall(s):
        tok = fix_leet(tok)
        canon = LEGAL.get(tok)
        if canon is not None:
            legal.add(canon)
            continue
        if tok in NAME_STOP or tok == prev:
            continue
        core.append(tok)
        prev = tok

    is_domain = not core and bool(domains)
    if is_domain:
        concat = domains[0]
    else:
        concat = "".join(t for t in core if t.isascii())
    latin_core = [t for t in core if t.isascii() and t[0].isalpha()]
    initials = "".join(t[0] for t in latin_core) if len(latin_core) >= 2 else ""
    stripped = raw.strip()
    acro_self = ""
    if (len(core) == 1 and 2 <= len(core[0]) <= 5 and core[0].isascii()
            and core[0].isalpha() and stripped.isupper()):
        acro_self = core[0]
    return (
        " ".join(core),
        " ".join(sorted(set(core))),
        " ".join(sorted(legal)),
        concat,
        initials,
        acro_self,
        domains[0] if domains else "",
        is_domain,
        script_of(raw),
    )


NAME_FIELDS = ["core", "core_key", "legal", "concat", "initials", "acro_self",
               "domain", "is_domain", "script"]


# --------------------------------------------------------------------------- addresses

# Language-agnostic abbreviation canonicalization (no country is hard-coded):
# long and short forms map to one code; "street"/"saint" share "st" on purpose.
ADDR_ABBR = {
    "street": "st", "str": "st", "saint": "st", "sainte": "ste",
    "avenue": "av", "ave": "av", "avn": "av",
    "road": "rd", "route": "rte", "boulevard": "bd", "blvd": "bd", "boul": "bd",
    "drive": "dr", "drv": "dr", "lane": "ln", "court": "ct", "crt": "ct",
    "place": "pl", "plaza": "plz", "circle": "cir", "trail": "trl",
    "parkway": "pkwy", "pky": "pkwy", "highway": "hwy", "terrace": "ter",
    "square": "sq", "impasse": "imp", "allee": "all", "chemin": "ch", "che": "ch",
    "rue": "rue", "r": "rue", "mount": "mt", "fort": "ft", "point": "pt",
    "cove": "cv", "way": "wy", "crossing": "xing",
}
STRUCTURAL = set(ADDR_ABBR.values()) | {"main", "cross", "nagar", "marg", "colony", "sector",
                                        "phase", "block", "layout", "extension", "industrial",
                                        "area", "estate", "complex", "tower", "wing"}
ADDR_STOP = {
    "no", "nos", "hno", "dno", "house", "plot", "flat", "door", "unit", "suite", "shop",
    "office", "room", "floor", "flr", "fl", "bldg", "building", "batiment", "apt",
    "apartment", "appt", "appart", "near", "nr", "opp", "opposite", "behind", "beside",
    "po", "box", "bp", "null", "nan", "none", "na", "nil", "the", "of", "and", "de", "du",
    "des", "la", "le", "les", "city", "town", "township", "cdp", "county", "landmark",
}
_POBOX = re.compile(
    r"\b(?:p\s*\.?\s*o\s*\.?\s*box|post\s*box|b\s*\.?\s*p)\s*(?:no\s*\.?)?\s*[:#.\-]?\s*\d+"
)
_ORD = re.compile(r"(\d)(?:st|nd|rd|th)\b")
_ADDR_TOKEN = re.compile(r"[a-zß-ɏऀ-෿]+|\d+")
# Spelled-out ordinals become the number, like "1st" -> "1" above ("1510 11th Street" vs
# "1510 ELEVENTH ST"; "449 1st St" vs "0449 First St" were blocking misses in v11).
ORDINAL_WORDS = {w: str(i) for i, w in enumerate(
    ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
     "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
     "eighteenth", "nineteenth", "twentieth"], start=1)}


def norm_addr(raw: str) -> tuple:
    """Return (nums, toks, blank).

    nums  house/unit numbers with leading zeros stripped, first 5 unique, in order
    toks  canonical alphabetic tokens (placeholders, PO boxes, filler words removed)
    blank True when nothing usable remains (never imputed)
    """
    s = fold(raw)
    s = _POBOX.sub(" ", s)
    s = _ORD.sub(r"\1", s)
    nums: list[str] = []
    toks: list[str] = []
    seen: set[str] = set()
    for tok in _ADDR_TOKEN.findall(s):
        tok = ORDINAL_WORDS.get(tok, tok)
        if tok[0].isdigit():
            n = str(int(tok))
            if n != "0" and n not in seen and len(nums) < 5:
                seen.add(n)
                nums.append(n)
            continue
        tok = ADDR_ABBR.get(tok, tok)
        if len(tok) < 2 or tok in ADDR_STOP or tok in seen:
            continue
        seen.add(tok)
        toks.append(tok)
    return " ".join(nums), " ".join(toks), not nums and not toks


ADDR_FIELDS = ["addr_nums", "addr_toks", "addr_blank"]


# ------------------------------------------------------------- cross-script skeletons

# Indic-script target names are English business names written in 9 Indian scripts
# (e.g. "राम मार्केटिंग प्राइवेट लिमिटेड" = "Ram Marketing Private Limited"). Both sides are
# reduced to a phonetic consonant skeleton: transliterate Indic runs to Latin (IAST,
# indic_transliteration, MIT), fold accents, drop aspiration h and vowels after the
# first letter, merge voiced/unvoiced pairs (Tamil does not distinguish them), drop
# legal words. On train, skeleton similarity separates true Indic pairs from decoys
# with AUC 0.97.
_SCRIPT_BLOCKS = [(0x0900, "devanagari"), (0x0980, "bengali"), (0x0A00, "gurmukhi"), (0x0A80, "gujarati"),
                  (0x0B00, "oriya"), (0x0B80, "tamil"), (0x0C00, "telugu"), (0x0C80, "kannada"),
                  (0x0D00, "malayalam")]
_INDIC_RUN = re.compile("[ऀ-ൿ]+")
_CHILLU = str.maketrans({"ൽ": "l", "ർ": "r", "ൺ": "n", "ൾ": "l", "ൻ": "n", "ൿ": "k"})
_SK_MERGE = str.maketrans({"g": "k", "c": "k", "q": "k", "d": "t", "b": "p", "v": "p", "w": "p", "f": "p",
                           "z": "s", "j": "s"})
_SK_VOWELS = set("aeiouy")


def to_latin(text: str) -> str:
    """Transliterate Indic-script runs to Latin and fold everything to lowercase, near-ASCII text."""
    from indic_transliteration.sanscript import IAST, transliterate

    def conv(m: re.Match) -> str:
        """Transliterate one Indic-script run to IAST using the scheme of its Unicode block."""
        run = m.group(0)
        cp = ord(run[0])
        scheme = next(name for base, name in reversed(_SCRIPT_BLOCKS) if cp >= base)
        return " " + transliterate(run.translate(_CHILLU), scheme, IAST) + " "

    s = _INDIC_RUN.sub(conv, text)
    s = s.replace("ṟṟ", "tt")                                   # Malayalam double r = tt
    s = re.sub("ṃ(?=[pbm])", "m", s).replace("ṃ", "n")         # anusvara
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def skel_word(w: str) -> str:
    """Consonant skeleton of a Latin word: drop aspiration h and inner vowels, merge similar
    consonants and collapse repeats, so transliteration variants map to the same key."""
    w = w.replace("x", "ks")
    out, prev_cons = [], False
    for ch in w:
        if ch == "h" and prev_cons:          # aspiration: kh gh ch jh th dh ph bh sh
            continue
        prev_cons = ch not in _SK_VOWELS
        out.append(ch)
    if not out:
        return ""
    first = "a" if out[0] in _SK_VOWELS else out[0]
    s = (first + "".join(c for c in out[1:] if c not in _SK_VOWELS)).translate(_SK_MERGE)
    return re.sub(r"(.)\1+", r"\1", s)


_LEGAL_SK = {skel_word(w) for w in list(LEGAL) + ["pvt", "ltd", "pra", "li", "prai"]}


def name_skeleton(raw: str) -> str:
    """Space-joined skeleton tokens of a business name (any script), legal words removed."""
    words = re.findall(r"[a-z]+", to_latin(_TEMPLATE.sub(" ", raw)))
    sk = (skel_word(w) for w in words if w not in NAME_STOP)
    return " ".join(s for s in sk if len(s) >= 2 and s not in _LEGAL_SK)


# ------------------------------------------------------------------------ parallel map

def _apply_chunk(args: tuple) -> list:
    """Worker: apply the normalizer named by args[0] ("name", "addr", "skel") to the values in args[1]."""
    fn_name, values = args
    fn = {"name": norm_name, "addr": norm_addr, "skel": name_skeleton}[fn_name]
    return [fn(v) for v in values]


def parallel_normalize(kind: str, values: list[str], n_jobs: int, chunk: int = 20_000) -> list:
    """Apply norm_name/norm_addr to `values` with a process pool, preserving order."""
    chunks = [(kind, values[i:i + chunk]) for i in range(0, len(values), chunk)]
    if n_jobs <= 1 or len(chunks) <= 1:
        return [row for part in map(_apply_chunk, chunks) for row in part]
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        return [row for part in pool.map(_apply_chunk, chunks) for row in part]
