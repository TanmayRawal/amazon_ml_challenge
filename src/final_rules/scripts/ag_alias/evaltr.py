"""French alias rule (v24), step 2: validation on US/India train.

Takes the unique-candidate alias pairs from build.py (cand_train.parquet, ncand == 1) and reports their precision
against train labels (gt.parquet), overall and broken down by country, source, exact-address group size, the target's
best model p (scores_c.parquet), the S1's number of predicted pairs, and on the "unclaimed" subset (max p < 0.5), which
is the part the rule adds on test. Result: 98.3% true (US), 98.6% (India). Writes u_train.parquet to the scratch folder.
"""
import polars as pl
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(50)
W = r'work'
O = r'work/scratch/ag_alias'
c = pl.read_parquet(f'{O}\\cand_train.parquet')
al = pl.read_parquet(f'{O}\\alias_train.parquet')
gt = pl.read_parquet(f'{W}\\train\\gt.parquet')
sc = pl.scan_parquet(f'{W}\\train\\scores_c.parquet').select('s1', 't', 'p')
sct = sc.join(al.lazy().select('t'), on='t', how='semi').collect()
tmax = sct.group_by('t').agg(tmaxp=pl.col('p').max())
# s1 "claimed" count: s1 with pairs p>0.5
s1c = sc.filter(pl.col('p') > 0.5).group_by('s1').agg(s1_npred=pl.len()).collect()
own = gt.join(al.select('t'), on='t', how='semi')
nown = own.group_by('t').agg(n_own=pl.len())
u = c.filter(pl.col('ncand') == 1)
u = (u.join(gt.with_columns(y=pl.lit(1)), on=['s1', 't'], how='left').with_columns(pl.col('y').fill_null(0))
     .join(nown, on='t', how='left').with_columns(pl.col('n_own').fill_null(0))
     .join(sct, on=['s1', 't'], how='left').join(tmax, on='t', how='left')
     .join(s1c, on='s1', how='left').with_columns(pl.col('s1_npred').fill_null(0), pl.col('tmaxp').fill_null(0)))
# true copies of S1 at its address (other true targets of this s1): from gt count
s1true = gt.group_by('s1').agg(s1_ntrue=pl.len())
u = u.join(s1true, on='s1', how='left').with_columns(pl.col('s1_ntrue').fill_null(0))
def rep(df, by):
    """Print pair count, precision (mean y) and orphan share (targets with no true owner) grouped by `by`."""
    print(df.group_by(by).agg(n=pl.len(), prec=pl.col('y').mean(), orphan=(pl.col('n_own') == 0).mean()).sort(by))
print('overall', u.height, u['y'].mean(), (u['n_own'] == 0).mean())
rep(u, 'country')
rep(u, ['country', 'src'])
rep(u.with_columns(ne=pl.col('n_exact').clip(1, 3)), ['country', 'ne'])
rep(u.with_columns(pb=(pl.col('tmaxp') * 10).floor().clip(0, 9)), ['country', 'pb'])
rep(u.with_columns(sp=pl.col('s1_npred').clip(0, 4)), ['country', 'sp'])
rep(u.with_columns(sp=pl.col('p').is_null()), ['country', 'sp'])
# unclaimed proxy
un = u.filter(pl.col('tmaxp') < 0.5)
print('unclaimed proxy'); rep(un, 'country')
rep(un.with_columns(ne=pl.col('n_exact').clip(1, 3)), ['country', 'ne'])
rep(un.with_columns(sp=pl.col('s1_npred').clip(0, 4)), ['country', 'sp'])
rep(un.with_columns(r=pl.col('r') == 100), ['country', 'r'])
u.write_parquet(f'{O}\\u_train.parquet')
print(un.filter(pl.col('y') == 0, pl.col('country') == 'US').sample(15, seed=0).select('s1_name', 's1_addr', 'business_name', 'business_address', 'n_own'))
