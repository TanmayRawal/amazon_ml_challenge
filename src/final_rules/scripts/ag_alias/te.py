"""French alias rule (v24), step 3: application to France test.

Loads the unique-candidate alias pairs from build.py (cand_test.parquet) and compares them with every pair already in
the submission (final_pairs plus the acronym, tie, ensemble, domain, R1 and tie3 add files under w20/). Reports how
many alias targets are unclaimed, whether the claimed ones agree with the rule's S1, and the breakdown by full-address
match (token_sort_ratio >= 95) and number of name words. The unclaimed, full-address, single-word pairs form
ag_alias_add.parquet (+2,125 pairs in v24). Writes u_test.parquet.
"""
import polars as pl
from rapidfuzz import fuzz
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(50)
W = 'work/w20/'
fs = ['final_pairs', 'acro_add', 'fr_tie_add', 'fr_tie2_add', 'ens_add_w0.7_cap', 'q_dom_usin_add_tight', 'q_r1_add', 'q_fr_tie3_add']
cl = pl.concat([pl.read_parquet(W + f + '.parquet').select(pl.col('s1').cast(pl.UInt32), pl.col('t').cast(pl.UInt32)) for f in fs])
print('claimed pairs', cl.height, cl['t'].n_unique())
c = pl.read_parquet('cand_test.parquet')
u = c.filter(pl.col('ncand') == 1)
u = u.join(cl.rename({'s1': 'cs1'}).unique('t'), on='t', how='left').with_columns(
    claimed=pl.col('cs1').is_not_null(), same=pl.col('cs1') == pl.col('s1'))
print(u.group_by('claimed', 'same').len())
s = pl.read_parquet('work/test/s1.parquet', columns=['s1', 'addr_toks'])
t = pl.read_parquet('work/test/tgt.parquet', columns=['t', 'addr_toks', 'core'])
u = u.join(s, on='s1').join(t, on='t', suffix='_t').with_columns(
    full=pl.struct('addr_toks', 'addr_toks_t').map_elements(lambda r: fuzz.token_sort_ratio(r['addr_toks'], r['addr_toks_t']) >= 95, return_dtype=pl.Boolean),
    nw=pl.col('core_t').str.count_matches(' ') + 1)
print(u.group_by('claimed', 'full', 'nw').len().sort('claimed', 'full', 'nw'))
u.write_parquet('u_test.parquet')
print(u.filter(~pl.col('claimed'), pl.col('full'), pl.col('nw') == 1).sample(20, seed=0).select('s1_name', 's1_addr', 'business_name', 'business_address', 'src'))
# S1 claimed count
s1n = cl.group_by('s1').agg(s1_nclaim=pl.len())
x = u.filter(~pl.col('claimed')).join(s1n, on='s1', how='left').with_columns(pl.col('s1_nclaim').fill_null(0))
print(x.group_by(pl.col('s1_nclaim').clip(0, 4)).len().sort('s1_nclaim'))
