"""Stage 6 decoy features on a hand-made candidate list."""
import polars as pl
import pytest

from stack import decoy_features


def test_decoy_features_flag_swap_offset_and_consensus():
    """decoy_features flags legal-form swaps/losses, number offsets, and number/core consensus among candidates."""
    # S1: "Krishna Energy Private Limited", number 159. Two true copies agree on 159; the decoy
    # swaps Private -> Public and shifts the number to 161; a third record lost its legal form.
    x = pl.DataFrame({
        "s1": [0, 0, 0, 0],
        "t": [10, 11, 12, 13],
        "p": [0.9, 0.8, 0.6, 0.5],
        "s_leg": ["ltd pvt"] * 4,
        "t_leg": ["ltd pvt", "ltd pvt group", "ltd public", ""],
        "s_nums": ["159 93"] * 4,
        "t_nums": ["159 93", "159", "161 93", ""],
        "t_core": ["krishna energy", "krishna energy", "krishna energy", "krishna"],
    }).with_columns(s_p_sum=pl.col("p").sum().over("s1"))
    out = {r["t"]: r for r in decoy_features(x).iter_rows(named=True)}

    assert not out[10]["leg_swap"] and not out[11]["leg_swap"]      # "group" is ignored
    assert out[12]["leg_swap"] and out[13]["leg_lost"]
    assert out[12]["num_s_only"] == 1 and out[12]["num_t_only"] == 1
    assert out[12]["num_off"] == pytest.approx(1.0986, abs=1e-3)    # log1p(|159 - 161|)
    assert out[10]["num_off"] == -1.0                               # nothing unmatched
    # Consensus: 159 is backed by the other true copy; 161 by nobody.
    assert out[10]["sup_num_p"] == pytest.approx(0.8)
    assert out[12]["sup_num_p"] == pytest.approx(0.0)
    assert out[10]["t_num_mode"] and not out[12]["t_num_mode"] and not out[13]["t_num_mode"]
    assert out[10]["s_num_share"] == pytest.approx(1.7 / 2.8)
    assert out[13]["sup_num_p"] == 0.0 and out[13]["sup_core_p"] == 0.0
    assert out[12]["sup_core_p"] == pytest.approx(1.7)


def test_vocab_features_rare_typo_vs_vocabulary_swap():
    """vocab_features scores new/removed name words by target document frequency, separating rare typos from real word swaps."""
    import math

    from stack import token_df, vocab_features

    tg = pl.DataFrame({"t": [0, 1, 2, 3], "country": ["IN"] * 4,
                       "core": ["ntt lab", "ntt fab", "fab world", "ntt lbx"]})
    tdf = token_df(tg)          # fab: 2 targets, lbx: 1, lab: 1, ntt: 3
    x = pl.DataFrame({"country": ["IN"] * 3, "s_core": ["ntt lab"] * 3, "t_core": ["ntt fab", "ntt lbx", "ntt lab"]})
    out = vocab_features(x, tdf).to_dicts()
    assert out[0]["nw_n_new"] == 1 and out[0]["nw_new_df_min"] == pytest.approx(math.log1p(2))
    assert out[1]["nw_new_df_min"] == pytest.approx(math.log1p(1))
    assert out[0]["nw_gone_df_min"] == pytest.approx(math.log1p(1))
    assert out[2]["nw_n_new"] == 0 and out[2]["nw_new_df_min"] == -1.0 and out[2]["nw_gone_df_min"] == -1.0
    assert [r["t_core"] for r in out] == ["ntt fab", "ntt lbx", "ntt lab"]   # row order kept


def test_source_features_flag_empty_source():
    """source_features counts confident candidates and probability mass per source, plus per-source rank and gap."""
    from stack import source_features

    # S1 0: two confident S2 records, S3 candidates weak -> the best S3 one sees mass in the other source.
    x = pl.DataFrame({"s1": [0, 0, 0, 0], "t": [1, 2, 3, 4], "t_src": [2, 2, 3, 3], "p": [0.9, 0.8, 0.4, 0.1]})
    out = {r["t"]: r for r in source_features(x).iter_rows(named=True)}
    assert out[3]["src_n_conf_same"] == 0 and out[3]["src_n_conf_other"] == 2
    assert out[3]["src_mass_other"] == pytest.approx(1.7) and out[3]["src_mass_same"] == pytest.approx(0.1)
    assert out[3]["src_rank"] == 1 and out[4]["src_rank"] == 2
    assert out[3]["src_gap"] == 0.0 and out[4]["src_gap"] == pytest.approx(-0.3)
    assert out[1]["src_n_conf_same"] == 1 and out[1]["src_n_conf_other"] == 0
