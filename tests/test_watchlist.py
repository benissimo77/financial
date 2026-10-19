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
pd.DataFrame([('CHIP1', 'Semiconductors', 'Tech', 'HEALTHY'), ('CHIP2', 'Semiconductors', 'Tech', 'INCOMPLETE'), ('CHIP3', 'Semiconductors', 'Tech', 'DELISTED'),
              ('SOFT1', 'Computer Software', 'Tech', 'HEALTHY'), ('BRK.B', 'Insurance', 'Finance', 'HEALTHY'), ('SHOP1', 'Retail', 'Retail', 'HEALTHY'),
              ('TINY1', 'Semiconductors', 'Tech', 'HEALTHY')],
             columns=['ticker', 'zacks_x_ind_desc', 'sector', 'status']).to_csv(REG, index=False)
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
r = wl.resolve()
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
ok('a three-column rules file still works and has no lists', set(old.resolve()['ticker']) == {'CHIP1', 'CHIP2', 'TINY1', 'SOFT1'} and old.lists().empty)
for name, text in (('exclude above ticker level', "action,level,value\nexclude,industry,Semiconductors\n"), ('unknown action', "action,level,value\nremove,ticker,CHIP1\n")):
    try: rules(text, 'bad.csv').resolve(); raised = False
    except ValueError: raised = True
    ok(f'invalid rule rejected: {name}', raised)
try: rules("action,level,value\ninclude,industry,Semiconducters\n", 'typo.csv').resolve(); raised = None
except ValueError as e: raised = str(e)
ok('a misspelt industry raises with a suggestion', raised is not None and 'Semiconductors' in raised)
with contextlib.redirect_stdout(io.StringIO()) as out:
    got = rules("action,level,value\ninclude,ticker,NOPE1\ninclude,ticker,CHIP1\n", 'unknown.csv').resolve()
ok('a ticker missing from the registry is skipped with a warning', list(got['ticker']) == ['CHIP1'] and 'NOPE1' in out.getvalue())
