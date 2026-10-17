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
    """Classify a raw string's script: "latin" (pure ASCII), "indic", "mixed" (Indic + Latin) or "accented"."""
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
    "cie": "co", "compagnie": "co", "groupe": "group",
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
        """Record a matched website label (dots/hyphens removed) and blank it out of the name."""
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

# V15: the street part of an address. norm_addr keeps "155 K Rue du Levant" and "155 J R. du Levant" apart
# only by a letter it drops, and pools street, city and region words, so "27 Rue Francois Marceau" and
# "27 Rue Commandant Arnould" in one city look alike. Street segment = the comma-separated part holding the
# first house number (else the first part with a street type); its suffix is a letter / bis / ter right
# after that number; its core drops the number, suffix, street types and filler words.
STREET_TYPES = set(ADDR_ABBR) | set(ADDR_ABBR.values())
# Audit 2: the suffix belongs to the segment's FIRST number only ("100 Main St Unit 4C" has none, "10E ... Suite 1A"
# has E) and is read before ordinals are stripped ("64th Ter" is a terrace, not house 64 + "ter"). A unit tail
# ("suite 200", "apt 4c", "# 12") after the street is cut, so its number is neither the house number nor street core.
_SFX = re.compile(r"^\D*(\d+)\s*(bis|ter|quater|[a-z])(?![a-z0-9])")
_UNIT_TAIL = re.compile(r"\s(?:suite|ste|apt|apartment|unit|bureau|#)\s*(?:no\s*)?[a-z]?\d.*$")
_HOUSE = re.compile(r"\d+(?!\d|st\b|nd\b|rd\b|th\b)")


# V19 (audit): "ter" is also the Terrace abbreviation, so a French "12 ter" lost its suffix; the first numbered
# part could be a unit ("Suite 4, 12 Rue ...": empty street) or a postal code ("75001 Paris, 12 Rue ...",
# "Mack Rd, Haltom City, Texas 76117": street "paris" / "texas"). Unit parts and postal parts (one 5-6 digit
# number, no street type) are skipped when another part can hold the street. "#12" stays a house number (India).
SFX_WORDS = {"bis", "ter", "quater"}
_UNIT = re.compile(r"^\s*(?:suite|ste|apt|apartment|unit|bureau|bat|batiment|etage|floor|fl)\b")


def _street_part(parts: list) -> int | None:
    """Index of the street segment among the comma-separated parts (None: no part)."""
    def types(p):
        """Street-type words (rue, road, st, ...) present in address part `p`."""
        return STREET_TYPES & set(_ADDR_TOKEN.findall(p))

    def aside(p):   # short unit part ("suite 4") or short postal part ("75001 paris", "texas 76117"); India pads
        """True if part `p` is a short unit or postal-code part that cannot hold the street."""
        toks =_ADDR_TOKEN.findall(p)   # house numbers to 5 digits ("door no 00413 tanna house"), so only short parts
        nums = [t for t in toks if t[0].isdigit()]
        postal = (len(nums) == 1 and re.fullmatch(r"\d{5,6}", nums[0]) is not None and not nums[0].startswith("00")
                  and not types(p))   # "flat no 00408": India pads house numbers with zeros; no postcode starts 00
        return len(toks) <= 3 and (bool(_UNIT.match(p)) or postal)

    numbered = [i for i, p in enumerate(parts) if re.search(r"\d", p)]
    seg = next((i for i in numbered if not aside(parts[i])), None)
    if seg is None:
        seg = next((i for i, p in enumerate(parts) if types(p)), numbered[0] if numbered else None)
    return seg


def norm_street(raw: str) -> tuple:
    """Return (suffix, core, house): house-number suffix ("" if none), street-name words of the street segment and
    its house number (first non-ordinal number, leading zeros stripped; "" if none), so "Unit 1416, 6515 Belcrest Road" and
    "6515 Belcrest Road, Unit 1416" both give house 6515."""
    parts = _POBOX.sub(" ", fold(raw)).split(",")
    i = _street_part([_ORD.sub(r"\1", p) for p in parts])
    raw_seg = "" if i is None else _UNIT_TAIL.sub("", parts[i].strip())
    seg = _ORD.sub(r"\1", raw_seg)
    m = _SFX.match(raw_seg)
    sfx = m.group(2) if m and (m.group(2) not in STREET_TYPES or m.group(2) in SFX_WORDS) else ""
    toks = _ADDR_TOKEN.findall(seg)
    h = _HOUSE.search(raw_seg)   # an ordinal ("64th Ter") is a street name, never the house number
    house = str(int(h.group())) if h and int(h.group()) else ""
    if h:
        toks.remove(h.group())
    core = []
    for t in toks:
        t = ADDR_ABBR.get(t, t)
        if (t == sfx or t in STREET_TYPES or t in ADDR_STOP or len(t) < 2) and not t.isdigit():
            continue
        if t not in core:
            core.append(t)
    return sfx, " ".join(core), house


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
                           "z": "s", "j": "s", "m": "n"})
_SK_VOWELS = set("aeiouy")
# V12: candra vowels (ऑ ॉ ऍ ॅ, Gujarati ઑ ૉ ૅ) are not in IAST and survived as raw Indic signs,
# splitting "टेक्नोलॉजी" into "teknola" + "ji"; map them to the plain o/e vowels. Zero-width
# joiners split Telugu/Kannada words ("kan‌straksan"): drop them.
_CANDRA = str.maketrans({"ऑ": "ओ", "ॉ": "ो", "ऍ": "ए", "ॅ": "े",
                         "ઑ": "ઓ", "ૉ": "ો", "ૅ": "ે",
                         "‌": None, "‍": None})
# V12: English spellings -> sounds, so both scripts give one skeleton: soft c/g before e/i/y
# ("services"/"sarvisej", "energy"/"enarji"), -tion/-sion ("solutions"/"solyushans").
_SK_SOUND = [(re.compile(r"[ts]ion"), "shan"), (re.compile(r"ture"), "char"), (re.compile(r"gh(?=t)"), ""),
             (re.compile(r"ng(?=a?$)"), "n"), (re.compile(r"c(?=[eiy])"), "s"), (re.compile(r"g(?=[eiy])"), "j")]
_SK_STOP = NAME_STOP | {"sree", "shree", "shri", "sri"}


def to_latin(text: str) -> str:
    """Transliterate Indic-script runs to Latin and fold everything to lowercase, near-ASCII text."""
    from indic_transliteration.sanscript import IAST, transliterate

    def conv(m: re.Match) -> str:
        """Transliterate one Indic-script run to IAST, choosing the scheme from its Unicode block."""
        run =m.group(0)
        cp = ord(run[0])
        scheme = next(name for base, name in reversed(_SCRIPT_BLOCKS) if cp >= base)
        return " " + transliterate(run.translate(_CHILLU), scheme, IAST) + " "

    s = _INDIC_RUN.sub(conv, text.translate(_CANDRA))
    s = s.replace("ṟṟ", "tt")                                   # Malayalam double r = tt
    s = re.sub("ṃ(?=[pbm])", "m", s).replace("ṃ", "n")         # anusvara
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def skel_word(w: str) -> str:
    """Phonetic consonant skeleton of one Latin word.

    Applies English sound rules, drops aspiration h and all vowels after the first letter
    (a leading vowel becomes "a"), merges voiced/unvoiced consonants and collapses repeats.
    """
    w =w.replace("x", "ks")
    for pat, rep in _SK_SOUND:
        w = pat.sub(rep, w)
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
    sk = (skel_word(w) for w in words if w not in _SK_STOP)
    return " ".join(s for s in sk if len(s) >= 2 and s not in _LEGAL_SK)


# ------------------------------------------------------------------------ parallel map

def _apply_chunk(args: tuple) -> list:
    """Worker: apply the normalizer named by args[0] ("name", "addr", "skel", "street") to each value in args[1]."""
    fn_name, values = args
    fn = {"name": norm_name, "addr": norm_addr, "skel": name_skeleton, "street": norm_street}[fn_name]
    return [fn(v) for v in values]


def parallel_normalize(kind: str, values: list[str], n_jobs: int, chunk: int = 20_000) -> list:
    """Apply norm_name/norm_addr to `values` with a process pool, preserving order."""
    chunks = [(kind, values[i:i + chunk]) for i in range(0, len(values), chunk)]
    if n_jobs <= 1 or len(chunks) <= 1:
        return [row for part in map(_apply_chunk, chunks) for row in part]
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        return [row for part in pool.map(_apply_chunk, chunks) for row in part]
