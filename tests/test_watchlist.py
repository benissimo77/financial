"""Watchlist: expanding include/exclude rules against the registry, and named sub-lists.
Runs against temporary files; it never touches the real data. Run: python tests/test_watchlist.py"""
import os, sys, tempfile, io, contextlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import pandas as pd
from pipeline.Watchlist import Watchlist

T = tempfile.mkdtemp()
ok = lambda name, cond: print(('PASS' if cond else 'FAIL'), name)
REG, WATCH = os.path.join(T, 'registry.csv'), os.path.join(T, 'watchlist.csv')
pd.DataFrame([('CHIP1', 'Semiconductors', 'Tech', ''), ('CHIP2', 'Semiconductors', 'Tech', ''), ('CHIP3', 'Semiconductors', 'Tech', 'DELISTED'),
              ('SOFT1', 'Computer Software', 'Tech', ''), ('BRK.B', 'Insurance', 'Finance', ''), ('SHOP1', 'Retail', 'Retail', ''),
              ('TINY1', 'Semiconductors', 'Tech', '')],
             columns=['ticker', 'industry', 'sector', 'status']).to_csv(REG, index=False)
# The watchlist as the build cell would have written it: TINY1 failed the liquidity screen.
pd.DataFrame({'ticker': ['BRK-B', 'CHIP1', 'CHIP2', 'SHOP1', 'SOFT1']}).to_csv(WATCH, index=False)


def rules(text, name='rules.csv'):
    path = os.path.join(T, name)
    with open(path, 'w') as f:
        f.write(text)
    return Watchlist(rules_file=path, registry_file=REG, tickers_file=WATCH)


wl = rules("""action,level,value,list
# a comment line
include,industry,Semiconductors,semis
include,sector,Tech,
include,ticker,CHIP1,holdings
include,ticker,BRK.B,holdings
include,ticker,SHOP1,holdings
include,ticker,CHIP1,ideas
include,ticker,SOFT1,ideas
exclude,ticker,SHOP1,
""")
r = wl.resolve().query("matched_by != 'benchmark'")
ok('resolve: one row per ticker with the first rule that matched', list(r.columns) == ['ticker', 'matched_by'] and r['ticker'].is_unique
   and r.set_index('ticker').at['CHIP1', 'matched_by'] == 'industry:Semiconductors')
ok('resolve: delisted tickers are never candidates, share classes use a dash', 'CHIP3' not in set(r['ticker']) and 'BRK-B' in set(r['ticker']))
ok('resolve: an excluded ticker is dropped, a screen failure is still a candidate', 'SHOP1' not in set(r['ticker']) and 'TINY1' in set(r['ticker']))

ok('a list defined by an industry rule', wl.members('semis') == ['CHIP1', 'CHIP2'])
ok('a list defined ticker by ticker', wl.members('holdings') == ['BRK-B', 'CHIP1'])
ok('a ticker can be in several lists', all('CHIP1' in wl.members(n) for n in ('semis', 'holdings', 'ideas')) and wl.members('ideas') == ['CHIP1', 'SOFT1'])
ok('a ticker that failed the liquidity screen is in no list', 'TINY1' not in wl.members('semis'))
ok('an excluded ticker is in no list', 'SHOP1' not in wl.members('holdings'))
ok('unlabelled rules add to the watchlist but to no list', 'SOFT1' in set(r['ticker']) and 'SOFT1' not in wl.members('semis'))
ok('lists() shows each name with its member count', wl.lists().to_dict() == {'holdings': 2, 'ideas': 2, 'semis': 2})
try: wl.members('holdngs'); raised = None
except ValueError as e: raised = str(e)
ok('unknown list name raises, showing the available names', raised is not None and 'holdings' in raised and 'semis' in raised)

old = rules("action,level,value\ninclude,industry,Semiconductors\ninclude,ticker,SOFT1\n", 'old.csv')
ok('a three-column rules file still works and has no lists', set(old.resolve().query("matched_by != 'benchmark'")['ticker']) == {'CHIP1', 'CHIP2', 'TINY1', 'SOFT1'} and old.lists().empty)
for name, text in (('exclude above ticker level', "action,level,value\nexclude,industry,Semiconductors\n"), ('unknown action', "action,level,value\nremove,ticker,CHIP1\n")):
    try: rules(text, 'bad.csv').resolve().query("matched_by != 'benchmark'"); raised = False
    except ValueError: raised = True
    ok(f'invalid rule rejected: {name}', raised)
try: rules("action,level,value\ninclude,industry,Semiconducters\n", 'typo.csv').resolve().query("matched_by != 'benchmark'"); raised = None
except ValueError as e: raised = str(e)
ok('a misspelt industry raises with a suggestion', raised is not None and 'Semiconductors' in raised)
with contextlib.redirect_stdout(io.StringIO()) as out:
    got = rules("action,level,value\ninclude,ticker,NOPE1\ninclude,ticker,CHIP1\n", 'unknown.csv').resolve().query("matched_by != 'benchmark'")
ok('a ticker missing from the registry is skipped with a warning', list(got['ticker']) == ['CHIP1'] and 'NOPE1' in out.getvalue())

# --- Sample rule: random, proportional to sector size, the same every time
import numpy as np
big = pd.DataFrame({'ticker': [f'A{i:03d}' for i in range(60)] + [f'B{i:03d}' for i in range(30)] + [f'C{i:03d}' for i in range(9)] + ['DEAD1', 'NOSEC'],
                    'sector': ['Alpha'] * 60 + ['Beta'] * 30 + ['Gamma'] * 9 + ['Alpha', None],
                    'industry': 'Misc', 'status': [''] * 99 + ['DELISTED', ''], 'cik': range(101)})
BIG = os.path.join(T, 'big-registry.csv'); big.to_csv(BIG, index=False)
def sampled(text, name='sample.csv', watch=None):
    path = os.path.join(T, name)
    with open(path, 'w') as f:
        f.write(text)
    return Watchlist(rules_file=path, registry_file=BIG, tickers_file=watch or WATCH)
draw = sampled("action,level,value,list\ninclude,sample,20,broad\n").resolve().query("matched_by != 'benchmark'")
by_sector = draw['ticker'].str[0].value_counts().to_dict()
ok('sample: the requested number, split across sectors in proportion to their size', len(draw) == 20 and by_sector.get('A') == 12 and by_sector.get('B') == 6 and by_sector.get('C') == 2)
ok('sample: delisted companies are never drawn', 'DEAD1' not in set(draw['ticker']))
ok('sample: the same draw every time', sampled("action,level,value,list\ninclude,sample,20,broad\n").resolve().query("matched_by != 'benchmark'")['ticker'].tolist() == draw['ticker'].tolist())
ok('sample: recorded as the matching rule', set(draw['matched_by']) == {'sample:20'})
other = sampled("action,level,value\ninclude,sample,21\n", 'sample2.csv').resolve().query("matched_by != 'benchmark'")['ticker']
ok('sample: a different size is a different draw, but mostly overlapping', len(other) == 21 and len(set(other) & set(draw['ticker'])) >= 15)
ok('sample: asking for more than exist gives all usable companies', len(sampled("action,level,value\ninclude,sample,5000\n", 'sample3.csv').resolve().query("matched_by != 'benchmark'")) == 100)
watch2 = os.path.join(T, 'watch2.csv'); pd.DataFrame({'ticker': draw['ticker'].tolist()[:10]}).to_csv(watch2, index=False)
mixed = sampled("action,level,value,list\ninclude,sample,20,broad\ninclude,ticker,C000,holdings\nexclude,ticker," + draw['ticker'].iloc[0] + ",\n", 'sample4.csv', watch2)
ok('sample: works with the other rules, sub-lists and exclusions', draw['ticker'].iloc[0] not in mixed.members('broad') and len(mixed.members('broad')) == 9)
for bad in ('0', 'ten', '-5'):
    try: sampled(f"action,level,value\ninclude,sample,{bad}\n", 'bad-sample.csv').resolve().query("matched_by != 'benchmark'"); raised = False
    except ValueError: raised = True
    ok(f'sample: an invalid size is rejected ({bad})', raised)

# One ticker per company: two share classes of one company count once, and the plain ticker is the one drawn
two = pd.DataFrame({'ticker': ['ONE', 'ONE-B', 'TWO-A', 'TWO-B'], 'sector': 'Alpha', 'industry': 'Misc', 'status': '', 'cik': [1, 1, 2, 2]})
TWO_REG = os.path.join(T, 'two-class.csv'); two.to_csv(TWO_REG, index=False)
path = os.path.join(T, 'two-rules.csv')
with open(path, 'w') as f:
    f.write("action,level,value\ninclude,sample,10\n")
drawn = Watchlist(rules_file=path, registry_file=TWO_REG, tickers_file=WATCH).resolve().query("matched_by != 'benchmark'")['ticker'].tolist()
ok('sample: one ticker per company, preferring the plain ticker', drawn == ['ONE', 'TWO-A'])

ok('the benchmark is always a candidate, outside the registry and any rule', 'SPY' in set(wl.resolve()['ticker']) and
   wl.resolve().set_index('ticker').at['SPY', 'matched_by'] == 'benchmark' and 'SPY' not in set(pd.read_csv(REG)['ticker']))
