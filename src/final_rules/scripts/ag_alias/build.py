"""French alias rule (v24), step 1: candidate pairs for "alias" targets.

An alias target has a made-up brand name: none of its core name words occurs in the country's S1 name vocabulary
(acronyms of 2-5 letters, domains and blank addresses excluded). Each alias target is paired with the S1s of the same
country that share its first house number, a non-generic street token, and a street token_set_ratio >= 85.

Usage: build.py train|test (train = US/India for validation, test = France).
Inputs: W/<split>/s1.parquet and tgt.parquet. Outputs (OUT): cand_<split>.parquet (candidate pairs with ncand =
candidates per target, n_exact = S1s at the same exact address) and alias_<split>.parquet (the alias targets).
evaltr.py measures the rule on train; te.py applies it to France test (ag_alias_add.parquet, +2,125 pairs in v24).
"""
import polars as pl, re, sys, unicodedata
from rapidfuzz import fuzz

W = r'work'
OUT = r'work/scratch/ag_alias'
split = sys.argv[1]  # train / test
countries = ['US', 'India'] if split == 'train' else ['France']

def strip_acc(x):
    """Lower-case `x` and drop accents / non-ASCII characters."""
    return unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode().lower()

def street_toks(addr, toks):
    """Street tokens of an address: `toks` without digits and without words that only occur in comma segments not
    holding the first house number (city, region, postcode). Returns a space-joined string ("" if missing)."""
    # remove tokens from comma segments that don't contain the first number
    if addr is None or toks is None:
        return ''
    segs = addr.split(',')
    m = re.search(r'\d+', addr)
    other = set()
    for sg in segs:
        if m and m.group(0) in sg:
            continue
        other.update(re.findall(r'[a-z0-9]+', strip_acc(sg)))
    num_seg = set()
    for sg in segs:
        if m and m.group(0) in sg:
            num_seg.update(re.findall(r'[a-z0-9]+', strip_acc(sg)))
    out = [w for w in toks.split() if not w.isdigit() and (w not in other or w in num_seg)]
    return ' '.join(out)

cols = ['core', 'is_domain', 'addr_nums', 'addr_toks', 'addr_blank', 'business_address', 'business_name', 'country']
s = pl.read_parquet(f'{W}\\{split}\\s1.parquet', columns=['s1'] + cols).filter(pl.col('country').is_in(countries))
t = pl.read_parquet(f'{W}\\{split}\\tgt.parquet', columns=['t', 'src'] + cols).filter(pl.col('country').is_in(countries))

# country S1 core vocab
vocab = s.select('country', w=pl.col('core').str.split(' ')).explode('w').filter(pl.col('w').str.len_chars() > 0).unique()
tw = t.select('t', 'country', w=pl.col('core').str.split(' ')).explode('w').filter(pl.col('w').str.len_chars() > 0)
hit = tw.join(vocab, on=['country', 'w'], how='semi').select('t').unique()
core_nospace = pl.col('core').str.replace_all(' ', '')
alias = (t.join(hit, on='t', how='anti')
         .filter(pl.col('core').str.len_chars() > 0, ~pl.col('is_domain'), ~pl.col('addr_blank'),
                 pl.col('addr_nums').str.len_chars() > 0)
         .filter(~((~pl.col('core').str.contains(' ')) & core_nospace.str.len_chars().is_between(2, 5)))
         .with_columns(num=pl.col('addr_nums').str.split(' ').list.first()))
print('alias targets', alias.group_by('country').len())

s = s.filter(~pl.col('addr_blank'), pl.col('addr_nums').str.len_chars() > 0).with_columns(
    num=pl.col('addr_nums').str.split(' ').list.first())
s = s.with_columns(st=pl.struct('business_address', 'addr_toks').map_elements(
    lambda r: street_toks(r['business_address'], r['addr_toks']), return_dtype=pl.Utf8))
alias = alias.with_columns(st=pl.struct('business_address', 'addr_toks').map_elements(
    lambda r: street_toks(r['business_address'], r['addr_toks']), return_dtype=pl.Utf8))

# block: same country+num and share a non-generic street token
stf = s.select('country', w=pl.col('st').str.split(' ')).explode('w').group_by('country', 'w').len()
generic = stf.sort('len', descending=True).group_by('country').head(40).select('country', 'w')
def tokx(df, key):
    """Explode `df` into unique (key, country, num, street word) rows, keeping words of 3+ letters that are not among
    the country's 40 most frequent (generic) street words. Used as the blocking key for candidate pairs."""
    return (df.select(key, 'country', 'num', w=pl.col('st').str.split(' ')).explode('w')
            .filter(pl.col('w').str.len_chars() >= 3).join(generic, on=['country', 'w'], how='anti').unique())
cand = (tokx(alias, 't').join(tokx(s, 's1'), on=['country', 'num', 'w']).select('t', 's1').unique())
print('cand pairs', cand.height)
cand = (cand.join(alias.select('t', st_t='st'), on='t').join(s.select('s1', 'st', 'core'), on='s1'))
cand = cand.with_columns(r=pl.struct('st_t', 'st').map_elements(lambda r: fuzz.token_set_ratio(r['st_t'], r['st']), return_dtype=pl.Float64))
cand = cand.filter(pl.col('r') >= 85)
# exact address group size: s1s sharing same normalized addr_toks+num
s = s.with_columns(akey=pl.col('num') + '|' + pl.col('st'))
asz = s.group_by('country', 'akey').agg(n_exact=pl.len())
s = s.join(asz, on=['country', 'akey'])
cand = cand.join(s.select('s1', 'n_exact'), on='s1')
nc = cand.group_by('t').agg(ncand=pl.len())
cand = cand.join(nc, on='t').join(alias.select('t', 'src', 'country', 'business_name', 'business_address'), on='t')
cand = cand.join(s.select('s1', s1_name='business_name', s1_addr='business_address'), on='s1')
cand.write_parquet(f'{OUT}\\cand_{split}.parquet')
alias.select('t', 'country', 'src').write_parquet(f'{OUT}\\alias_{split}.parquet')
print(cand.group_by('country', 'ncand').len().sort('country', 'ncand').head(20))
