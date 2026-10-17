"""V13 retrieval channels and number features on the error examples that motivated them."""
import polars as pl

from features import _extended
from ranker import _addr_pair_keys, _domain_index, _fuzzy_keys, _truncated_numbers

S1 = pl.DataFrame({"s1": [0, 1], "country": ["US", "US"], "core": ["atlantic galata", "desert alliance"],
                   "addr_nums": ["1684 12", "120"], "addr_toks": ["clover hill dr portsmouth", "misty lake knoxville"],
                   "addr_blank": [False, False]})
TG = pl.DataFrame({"t": [0, 1, 2], "country": ["US"] * 3, "core": ["atlantic galaoat", "desert afligacne", "urologypartnerscom"],
                   "domain": ["", "", ""], "addr_toks": ["clover hill portsmouth", "", ""], "addr_blank": [False, True, True]})


def _hits(s, t):
    """Join source and target blocking-key frames on key and return the set of (s1, t) pairs they produce."""
    return set(s.join(t, on="key").select("s1", "t").unique().rows())


def test_dropped_digit_either_end():
    """_truncated_numbers yields each address number with its first or last digit dropped."""
    assert _truncated_numbers(S1)["addr_nums"].to_list() == ["168 684", "12 20"]


def test_typo_and_address_pair_keys():
    """Fuzzy name keys link typo'd names, and address-pair keys link records sharing street words."""
    assert _hits(_fuzzy_keys(S1, "s1", True), _fuzzy_keys(TG, "t", False)) == {(0, 0), (1, 1)}
    assert _hits(_addr_pair_keys(S1, "s1"), _addr_pair_keys(TG, "t")) == {(0, 0)}


def test_dotless_website_label():
    """_domain_index links a spaced name to a target whose core is the dotless website label."""
    s = pl.DataFrame({"s1": [0], "country": ["US"], "core": ["urology partners"]})
    tk, sk = _domain_index(s, TG, pl.DataFrame({"country": ["US"], "N": [3]}), 2000)
    assert sk.join(tk, on="key").group_by("s1", "t").len().rows() == [(0, 2, 2)]


def test_truncation_vs_decoy_shift():
    """_extended tells a truncated number from a shifted (decoy) one, and matches house numbers despite reordering."""
    df = pl.DataFrame({"s_num": [["1684"], ["73"]], "t_num": [["684"], ["74"]], "cheap": [1.0] * 2, "t_n_s1": [1] * 2,
                       "t_max_rk": [1.0] * 2, "s_core": ["a"] * 2, "t_core": ["a"] * 2, "_t_dom": [""] * 2,
                       "s_addr_toks": ["x"] * 2, "t_addr_toks": ["x"] * 2, "s_name_sk": ["a"] * 2, "t_name_sk": ["a"] * 2,
                       "s_street_house": ["", "73"], "t_street_house": [""] * 2,
                       "s_ctok": [["a"]] * 2, "t_ctok": [["a"]] * 2, "t_addr_blank": [False] * 2, "s_legal": [""] * 2,
                       "t_legal": [""] * 2})
    assert _extended(df).select("num_trunc", "num_shift").rows() == [(True, False), (False, True)]
    assert _extended(df)["num_delta"].to_list() == [-20.0, 1.0]   # V20: signed change of the one differing number
    # audit 1: "Unit 1416, 6515 Belcrest Road" vs "6515 Belcrest Road, Unit 1416": same house number
    df = df.with_columns(s_num=pl.Series([["1416", "6515"], ["73"]]), t_num=pl.Series([["6515", "1416"], ["74"]]),
                         s_street_house=pl.Series(["6515", "73"]))
    assert _extended(df)["num_first_eq"].to_list() == [True, False] and _extended(df)["num_first_raw"].to_list() == [False, False]


def test_raw_name_tie_break_features():
    """raw_name_context gives raw-name exact/legal-form/similarity features and their differences between rival S1s."""
    from stack import raw_name_context
    scores = pl.DataFrame({"s1": [0, 1, 0], "t": [0, 0, 1], "p": [0.5, 0.5, 0.01]})
    s1 = pl.DataFrame({"s1": [0, 1], "name": ["Sterling Data LLC", "Sterling Data Inc"], "legal": ["llc", "inc"]})
    tg = pl.DataFrame({"t": [0, 1], "name": ["STERLING DATA LLC", "x"], "legal": ["llc", ""]})
    r = raw_name_context(scores, s1, tg).sort("s1")
    assert r["s1"].to_list() == [0, 1]                       # p <= 0.05 pairs are left out
    assert r["raw_exact"].to_list() == [1.0, 0.0] and r["d_raw_exact"].to_list() == [1.0, -1.0]
    assert r["d_leg_eq"].to_list() == [1.0, -1.0] and r["d_raw_sim"][0] > 0 > r["d_raw_sim"][1]


def test_sibling_view_features():
    """sibling_features scores whether a candidate's address number is backed by a confident sibling record or is new."""
    from stack import SIBLING, sibling_features
    # S1 0 says "144 tagore park"; its confident src2 copy says 944. The uncertain src2 record with 944 is
    # backed by that view; the one with 188 is new to everybody (decoy-like). The confident pair gets -1.
    tg = pl.DataFrame({"addr_low": ["944 tagore park delhi", "944 tagore park delhi", "188 tagore park delhi"],
                       "name_low": ["indo trading", "ind0trading.com", "indo trading"],
                       "addr_toks": ["tagore park delhi"] * 3, "addr_nums": ["944", "944", "188"]})
    s1t = pl.DataFrame({"addr_toks": ["tagore park delhi"], "addr_nums": ["144"]})
    x = pl.DataFrame({"s1": [0, 0, 0], "t": [0, 1, 2], "p": [0.99, 0.5, 0.5], "t_src": [2, 2, 2]})
    out = sibling_features(x, tg, s1t).sort("t")
    assert out["num_view"].to_list() == [-1.0, 1.0, 0.0]
    assert out["num_new_min"].to_list() == [-1.0, 0.0, 1.0]
    assert out["addr_eq_same"].to_list() == [-1.0, 1.0, 0.0]
    assert out["n_sib_same"].to_list() == [-1.0, 1.0, 1.0] and out["n_sib_other"][1] == 0.0
    assert set(SIBLING) <= set(out.columns)


def test_density_matching_hides_and_reindexes():
    """density_keep/hide_s1 subsample train S1 per country to test density, keeping folds and owners and re-indexing."""
    from normalize import norm_name
    from prepare import density_keep, hide_s1
    train = pl.DataFrame({"country": ["US", "India"], "n_s1": [100, 100], "n_t": [500, 500]})
    test = pl.DataFrame({"country": ["US", "France"], "n_s1": [40, 10], "n_t": [250, 50]})
    keep = density_keep(train, test)
    assert abs(keep["US"] - 0.8) < 1e-9 and keep["India"] == 1.0          # India absent from test: kept whole
    s1 = pl.DataFrame({"s1": pl.Series(range(2000), dtype=pl.UInt32), "entity_id": [f"S1-{i}" for i in range(2000)],
                       "country": ["US"] * 1000 + ["India"] * 1000})
    gt = pl.DataFrame({"s1": pl.Series([5, 1500], dtype=pl.UInt32), "t": pl.Series([7, 8], dtype=pl.UInt32)})
    folds = s1.select("s1", fold=(pl.col("s1") % 5).cast(pl.UInt8))
    new, other = hide_s1(s1, {"gt": gt, "folds": folds}, keep, seed=1)
    kept_fold = other["folds"].join(new, on="s1").select("entity_id", "fold")   # every S1 keeps its fold
    assert kept_fold.height == new.height
    assert (kept_fold["entity_id"].str.slice(3).cast(pl.Int64) % 5 == kept_fold["fold"].cast(pl.Int64)).all()
    us = new.filter(pl.col("country") == "US").height
    assert 750 < us < 850 and new.filter(pl.col("country") == "India").height == 1000
    assert new["s1"].to_list() == list(range(new.height))                   # re-indexed 0..n-1
    moved = other["gt"].join(new, on="s1").select("entity_id", "t")       # surviving pairs keep their owner
    assert set(moved["entity_id"]) <= {"S1-5", "S1-1500"} and "S1-1500" in set(moved["entity_id"])
    assert norm_name("Comité de Boutique Cie")[2] == "co" and norm_name("Nord Groupe SAS")[2] == "group sas"


def test_v15_street_suffix_and_idf_features():
    """norm_street parses house suffix/street core/house number across formats, and _v15 builds suffix and IDF features."""
    from features import _toks, _v15, idf_table
    from normalize import norm_street
    assert norm_street("155 K Rue du Levant, Tourcoing, Hauts-de-France")[:2] == ("k", "levant")
    assert norm_street("155 J R. DU LEVANT, TOURCOING")[:2] == ("j", "levant")
    assert norm_street("12 R GEORGES JUZAN, Bordeaux")[:2] == ("", "georges juzan")        # "R" is rue, not a suffix
    assert norm_street("Bordeaux, Nouvelle-Aquitaine, 12 Rue Georges Juzan")[:2] == ("", "georges juzan")
    assert norm_street("12 bis rue de la Paix")[0] == "bis" and norm_street("Plot No 12, Sector 5")[:2] == ("", "")
    # V19 audit fixes: "ter" suffix, unit / postal-code parts before the street; US 5-digit house numbers kept
    assert norm_street("12 ter Rue Victor Hugo, Paris")[:2] == ("ter", "victor hugo")
    assert norm_street("75001 Paris, 12 Rue Victor Hugo")[:2] == ("", "victor hugo")
    assert norm_street("Suite 4, 12 Rue Victor Hugo, Paris")[:2] == ("", "victor hugo")
    assert norm_street("Suite 200, 1795 Westchester Drive, High Point, NC")[:2] == ("", "westchester")
    assert norm_street("1795 Westchester Drive, Suite 200, High Point, NC")[:2] == ("", "westchester")
    assert norm_street("Mack Rd, Haltom City, Texas 76117")[:2] == ("", "mack")
    assert norm_street("17560 Ellis Road, Tahlequah, OK")[:2] == ("", "ellis")
    assert norm_street("17560 Ellis, Tahlequah, OK")[1] == "ellis"          # no street type: still the house number
    assert norm_street("#12, 3rd Cross, Bangalore 560001")[1] == norm_street("#12, 3rd Cross")[1]
    assert norm_street("1001 64TH TER, KANSAS CITY, MO")[0] == ""            # terrace after an ordinal street
    assert norm_street("Quail Ct, Unit 8, Louisville, Kentucky")[:2] == ("", "quail")
    long_unit = "UNIT NO A111 BUILDING NO 5A SANJAY BUILDING MITTAL INDUSTRIAL ESTATE, ANDHERI"
    assert "sanjay" in norm_street(long_unit)[1]                              # long parts are not unit parts
    assert norm_street("Door No 00413 Tanna House, Ground Floor, 11/A, Nathalal Parekh Marg")[1] == "tanna"   # padded house no.
    # audit 2: unit numbers and ordinals are neither the house number, its suffix nor street words
    assert norm_street("Unit 1416, 6515 Belcrest Road")[2] == norm_street("6515 Belcrest Road, Unit 1416")[2] == "6515"
    assert norm_street("100 Main St Unit 4C") == norm_street("100 Main St, Unit 4 C") == ("", "main", "100")
    assert norm_street("10E Ho Chi Minh Sarani Suite 1A, Kolkata")[:2] == ("e", "ho chi minh sarani")
    assert norm_street("64th Ter, Miami, FL") == ("", "64", "") and norm_street("1001 64TH TER, KANSAS CITY") == ("", "64", "1001")
    assert norm_street("12 Rue Ste Catherine, Bordeaux")[1] == "catherine"      # "Ste" + word is Sainte, not a suite
    assert norm_street("Door No 00413 Tanna House")[2] == "413" and norm_street("#12, 3rd Cross")[2] == "12"
    assert norm_street("FLAT NO.00408 , S NO.145, MGR ESTATES")[2] == norm_street("Flat No.408, S No.145, Mgr Estates")[2] == "408"
    recs = pl.DataFrame({"country": ["France"] * 5, "addr_toks": ["rue levant tourcoing hauts france", "rue gand lille hauts france",
                                                                  "rue kleber lille hauts france", "rue paix nantes hauts france",
                                                                  "rue levant tourcoing"]})
    idf = {"addr": idf_table(recs, "addr_toks"), "name": idf_table(pl.DataFrame({"country": ["France"] * 2, "core": ["lille union", "lille sante"]}), "core")}
    a = idf["addr"].filter(pl.col("tok") == "rue")["idf"][0]
    assert abs(a) < 1e-6                                                          # every record has "rue": no weight
    df = pl.DataFrame({"s_country": ["France"] * 3, "s_street_sfx": ["k", "", "k"], "t_street_sfx": ["j", "", ""],
                       "s_street_core": ["levant", "francois marceau", "levant"], "t_street_core": ["levant", "commandant arnould", ""],
                       "s_addr_toks": ["rue levant tourcoing hauts france"] * 3, "t_addr_toks": ["rue levant tourcoing"] * 3,
                       "s_core": ["lille union"] * 3, "t_core": ["lille union", "lille sante", "x"], "s_nm_min_df": [0.5] * 3})
    df = df.with_columns(s_atok=_toks("s_addr_toks"), t_atok=_toks("t_addr_toks"), s_ctok=_toks("s_core"), t_ctok=_toks("t_core"))
    idc = {"addr": idf["addr"], "name": idf["name"]}
    tot = lambda col: pl.col(col).list.eval(pl.element().replace_strict(dict(idf["addr"].select("tok", "idf").iter_rows()), default=0.0)).list.sum()  # noqa: E731
    df = df.with_columns(s_addr_idf=tot("s_atok"), t_addr_idf=tot("t_atok"))
    out = _v15(df, idc)
    assert out["sfx_conflict"].to_list() == [True, False, False] and out["sfx_one"].to_list() == [False, False, True]
    assert out["st_tset"][0] == 1.0 and out["st_tset"][1] < 0.5 and out["st_tset"][2] == -1.0
    assert out["st_cont"].to_list()[:2] == [1.0, 0.0]
    assert 0.75 < out["addr_wjac"][0] < 1.0     # plain Jaccard 0.6: the one-sided region words are common, so they weigh little
    assert out["nm_shared_idf"][0] > 0.0 == out["nm_shared_idf"][1]   # sharing only the city word "lille" is no evidence


def test_v17_name_word_by_number_keys():
    """_word_num_keys links records only when they share a name word AND an address number within the same country."""
    from ranker import _word_num_keys
    s = pl.DataFrame({"s1": [0], "country": ["India"], "core": ["alpha impex"], "addr_nums": ["116 2"]})
    t = pl.DataFrame({"t": [0, 1, 2], "country": ["India"] * 3, "core": ["alpha center", "alpha impex", "beta impex"],
                      "addr_nums": ["116", "999", "116"]})
    hits = set(_word_num_keys(s, "s1", "core", 3).join(_word_num_keys(t, "t", "core", 3), on="key")["t"].to_list())
    assert hits == {0, 2}          # a shared word WITH a shared number; t=1 shares words but not the number
    us = t.with_columns(country=pl.lit("US"))
    assert _word_num_keys(s, "s1", "core", 3).join(_word_num_keys(us, "t", "core", 3), on="key").height == 0


def test_v21_skeleton_address_label_keys_and_hop_filter():
    """V21 skeleton+address and website-label keys link the right targets, and hop_keep filters multi-hop additions."""
    from ranker import _label_keys, _skf_addr_keys, hop_keep
    s = pl.DataFrame({"s1": [0], "country": ["India"], "name_sk": ["prtkts mn"], "concat": ["maaproducts"],
                      "addr_nums": ["1 2 178 305"], "addr_toks": ["srimalle hyderabad"]})
    t = pl.DataFrame({"t": [0, 1, 2], "country": ["India"] * 3, "name_sk": ["mn prtkts", "mn prtkts", "prtkts"],
                      "domain": ["", "maaproducts", ""], "addr_nums": ["772 1", "9", "1"], "addr_toks": ["hyderabad", "pune", "hyderabad"]})
    assert set(_skf_addr_keys(s, "s1").join(_skf_addr_keys(t, "t"), on="key")["t"]) == {0}   # same skeleton AND a shared number/word
    assert set(_label_keys(s, "s1").join(_label_keys(t, "t"), on="key")["t"]) == {1}         # concat == website label
    add = pl.DataFrame({"s_core": ["maa products"] * 3, "t_core": ["మా ప్రొడక్ట్స్"] * 2 + [""], "s_sk": ["n prtkts"] * 3,
                        "t_sk": ["n prtkts"] * 3, "hop": [1.0, 4.0, 6.0], "v21": [False, False, True]})
    assert hop_keep(add).to_list() == [True, False, True]   # Indic core scores 0: address hops pass on the skeleton
