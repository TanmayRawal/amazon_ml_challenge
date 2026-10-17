"""Golden cases taken from real records seen in the data audit."""
from normalize import fix_leet, fold, norm_addr, norm_name


def name(raw):
    """Run norm_name on a raw string and return its output fields as a dict keyed by field name."""
    core, core_key, legal, concat, initials, acro, domain, is_domain, script = norm_name(raw)
    return dict(core=core, core_key=core_key, legal=legal, concat=concat, initials=initials,
                acro=acro, domain=domain, is_domain=is_domain, script=script)


def test_accents_and_moved_legal_form():
    """Accents are folded, a leading legal form is extracted, and the script is tagged 'accented'."""
    n = name("LLC Moncada Léarning Center")
    assert n["core"] == "moncada learning center"
    assert n["legal"] == "llc"
    assert n["script"] == "accented"


def test_junk_prefix_and_inner_legal_token():
    """Leading punctuation junk is stripped and a legal token in the middle of the name is extracted."""
    n = name("-- Holloway Peak Inc Seafood")
    assert n["core"] == "holloway peak seafood"
    assert n["legal"] == "inc"


def test_pipe_website_duplicate_token():
    """A '| website' suffix becomes the domain field and a repeated word is collapsed in the core name."""
    n = name("SHIVSHAKTI VIDYALAYA VIDYALAYA OVERSEAS CORPORATION | www.shivshakti.com")
    assert n["core"] == "shivshakti vidyalaya overseas"
    assert n["legal"] == "corp"
    assert n["domain"] == "shivshakti"
    assert not n["is_domain"]


def test_domain_only_and_handle_names():
    """Names that are just a domain or a #handle are flagged is_domain with the joined label in concat."""
    n = name("wilfordhancock.com")
    assert n["is_domain"] and n["concat"] == "wilfordhancock" and n["core"] == ""
    h = name("#centraleducation")
    assert h["is_domain"] and h["concat"] == "centraleducation"


def test_leet_and_appended_id():
    """Leetspeak digits are repaired (real ordinals kept) and trailing '(ID: n)' / '#n' tags are removed."""
    assert name("L0TUS IT LTD")["core"] == "lotus it"
    assert name("Ratnagiri Capital Capital (ID: 68415)")["core"] == "ratnagiri capital"
    assert name("Cardiology Heartland Cáre Associates #98825")["core"] == "cardiology heartland care associates"
    assert fix_leet("24hr") == "24hr" and fix_leet("3rd") == "3rd" and fix_leet("5uperior") == "superior"


def test_dotted_legal_forms_and_initials():
    """Dotted legal forms like 'P.C.' are recognised, '&' is dropped, and initials/acronym fields are filled."""
    n = name("Foot & Ankle Physicians of Santaquin City P.C.")
    assert n["legal"] == "pc"
    assert "and" not in n["core"].split()
    t = name("Talava Certified Ishares, Inc.")
    assert t["initials"] == "tci"
    assert name("TCI")["acro"] == "tci"


def test_indic_names_keep_vowel_signs():
    """Indic names keep their vowel signs and word count, and are tagged 'indic' (or 'mixed' alongside Latin)."""
    n = name("राम मार्केटिंग प्राइवेट लिमिटेड")
    assert n["script"] == "indic"
    assert n["core"].split()[0] == "राम"
    assert len(n["core"].split()) == 4
    assert name("Sun पावर Provision")["script"] == "mixed"


def test_fold_keeps_indic_marks_and_fixes_mojibake():
    """fold strips Latin accents, leaves Indic combining marks intact, and removes mojibake control bytes."""
    assert fold("Cáre") == "care"
    assert fold("कृष्णा") == "कृष्णा".lower()
    assert "\u0080" not in fold("Vspâ\x80\x99S Bhavana")


def test_cross_script_skeletons_agree():
    """name_skeleton gives Latin and Indic-script spellings the same first consonant skeleton and drops legal words."""
    from normalize import name_skeleton
    pairs = [
        ("Ram Marketing Private Limited", "राम मार्केटिंग प्राइवेट लिमिटेड"),                 # Devanagari
        ("Krishna Impex Limited", "కృష్ణా ఇంపెక్స్ లిమిటెడ్"),                            # Telugu
        ("Eastern Consultancy Private Limited", "ஈஸ்டர்ன் கன்சல்டன்சி பிரைவேட் லிமிடெட்"),  # Tamil
        ("Sky Arihant Global", "ਸਕਾਈ ਅਰਿਹੰਤ ਗਲੋਬਲ ਪ੍ਰਾ. ਲਿ."),                           # Gurmukhi
    ]
    for latin, indic in pairs:
        a, b = name_skeleton(latin).split(), name_skeleton(indic).split()
        assert a and b and a[0] == b[0], (latin, a, b)
    assert name_skeleton("Ram Marketing Private Limited") == "rm mrktnk"
    assert "prpt" not in name_skeleton("Silver Consultancy Private Limited").split()   # legal words dropped


def test_address_zero_padding_and_city_filler():
    """norm_addr strips leading zeros from numbers and the filler word 'city' from address tokens."""
    nums, toks, blank = norm_addr("05131 COPPER MEADOW LN, WEST JORDAN CITY, UT")
    assert nums == "5131"
    assert toks == "copper meadow ln west jordan ut"
    assert not blank


def test_address_placeholders_pobox_ordinals():
    """norm_addr drops 'null' placeholders and PO Box parts, splits fractions, and keeps ordinal street numbers."""
    nums, toks, _ = norm_addr("22120 1/2 COYOTE CAVE TRL, null, SPICEWOOD, TX")
    assert nums == "22120 1 2" and "null" not in toks.split()
    nums, toks, _ = norm_addr("1 Ivanhoe Ave, PO Box 6009, Cincinnati, Ohio")
    assert nums == "1" and toks == "ivanhoe av cincinnati ohio"
    nums, _, _ = norm_addr("013623 24ND LN, SHAW BUTTE, AZ")
    assert nums == "13623 24"


def test_address_french_and_blank():
    """norm_addr handles 'N°' numbers and the 'R.' = rue abbreviation, and flags empty or placeholder-only addresses as blank."""
    nums, toks, _ = norm_addr("N°220222 GRANDE R., ROUBAIX, Nord")
    assert nums == "220222" and toks == "grande rue roubaix nord"
    assert norm_addr("")[2] is True
    assert norm_addr("NULL, N/A")[2] is True
