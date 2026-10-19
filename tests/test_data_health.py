"""DataHealth: each price-quality check, the score and bands, problems, summary and the two charts.
Runs against temporary files; it never touches the real data. Run: python tests/test_data_health.py"""
import os, sys, tempfile, hashlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd
from pipeline.DataHealth import DataHealth

T = tempfile.mkdtemp()
ok = lambda name, cond: print(('PASS' if cond else 'FAIL'), name)
days = pd.bdate_range(end='2026-09-30', periods=300)
AS_OF = '2026-10-07'   # seven days after the last trading day, so a complete ticker is exactly up to date


def rows(ticker, dates, base=50.0):
    close = base + np.arange(len(dates)) * 0.1
    return pd.DataFrame({'ticker': ticker, 'date': dates.strftime('%Y-%m-%d'), 'open': close, 'high': close + 1, 'low': close - 1,
                         'close': close, 'volume': 1000.0, 'adj_close': close})


t = {'SPY': rows('SPY', days), 'GOOD': rows('GOOD', days), 'LATE': rows('LATE', days[:-5]), 'GAPPY': rows('GAPPY', days.delete(range(100, 110))),
     'YOUNG': rows('YOUNG', days[200:]), 'BADBAR': rows('BADBAR', days), 'NOVOL': rows('NOVOL', days), 'FLAT': rows('FLAT', days), 'ADJ': rows('ADJ', days),
     'OFFLIST': rows('OFFLIST', days[:50])}
t['BADBAR'].loc[150, 'high'] = t['BADBAR'].loc[150, 'low'] - 1            # high below low
t['NOVOL'].loc[t['NOVOL'].index[-60:], 'volume'] = 0.0                     # 60 of the last 252 days with no volume
flat = t['FLAT'].index[-40:]                                                # the price does not move for 40 days, with volume
t['FLAT'].loc[flat, ['open', 'close', 'adj_close']] = 70.0; t['FLAT'].loc[flat, 'high'] = 71.0; t['FLAT'].loc[flat, 'low'] = 69.0
t['ADJ'].loc[200:, ['open', 'high', 'low', 'close']] *= 2                  # raw prices double overnight, the adjusted price does not
P, M, W, R = (os.path.join(T, f) for f in ('prices.csv', 'manifest.csv', 'watch.csv', 'registry.csv'))
pd.concat(t.values()).to_csv(P, index=False)
watch = [k for k in t if k != 'OFFLIST'] + ['MISSING']
pd.DataFrame({'ticker': watch}).to_csv(W, index=False)
pd.DataFrame({'ticker': watch, 'last_date': '2026-09-30', 'last_attempt': '2026-10-07 09:00:00', 'outcome': ['ok'] * (len(watch) - 1) + ['transient_error'], 'rows_fetched': 1}).to_csv(M, index=False)
pd.DataFrame({'ticker': watch, 'sector': 'Tech', 'zacks_x_ind_desc': ['Software'] * 5 + ['Chips'] * (len(watch) - 5)}).to_csv(R, index=False)
digest = lambda: [hashlib.md5(open(f, 'rb').read()).hexdigest() for f in (P, M, W, R)]
before = digest()

h = DataHealth(data_file=P, manifest_file=M, tickers_file=W, registry_file=R)
s = h.score(as_of=AS_OF)
ok('one row per watchlist ticker, off-watchlist tickers ignored', list(s.index) == watch and 'OFFLIST' not in s.index)
ok('complete ticker scores 100, green', s.at['GOOD', 'score'] == 100 and s.at['GOOD', 'band'] == 'green' and s.at['GOOD', 'days_behind'] == 0 and s.at['GOOD', 'issues'] == '')
ok('a ticker with a short history is not penalised for it', s.at['YOUNG', 'score'] == 100 and s.at['YOUNG', 'rows'] == 100)
ok('five days behind costs 2 points a day', s.at['LATE', 'days_behind'] == 5 and s.at['LATE', 'score'] == 90 and s.at['LATE', 'band'] == 'amber')
ok('a ten-day hole lowers coverage and counts as the longest gap', s.at['GAPPY', 'longest_gap'] == 10 and abs(s.at['GAPPY', 'coverage_pct'] - 100 * 290 / 300) < 1e-9
   and abs(s.at['GAPPY', 'score'] - (100 - 2 * (100 - 100 * 290 / 300) - 2 * (10 - 3))) < 1e-9 and s.at['GAPPY', 'band'] == 'red')
ok('an impossible bar (high below low) is flagged', s.at['BADBAR', 'bad_ohlc_rows'] == 1 and s.at['BADBAR', 'score'] == 90)
ok('too many zero-volume days are flagged', round(s.at['NOVOL', 'zero_volume_pct'], 1) == round(100 * 60 / 252, 1) and s.at['NOVOL', 'score'] == 95)
ok('a close that does not move is flagged', s.at['FLAT', 'flat_close_pct'] > 5 and s.at['FLAT', 'score'] == 95)
ok('raw price jumping without the adjusted price is flagged', s.at['ADJ', 'adj_inconsistency'] == 1 and s.at['ADJ', 'score'] == 90)
ok('a watchlist ticker with no prices scores 0, red', s.at['MISSING', 'score'] == 0 and s.at['MISSING', 'band'] == 'red' and not s.at['MISSING', 'in_database'] and 'not in database' in s.at['MISSING', 'issues'])
ok('sector and industry come from the registry', s.at['GOOD', 'sector'] == 'Tech' and s.at['FLAT', 'industry'] == 'Chips')

late = h.score(as_of='2026-10-21')   # two weeks on, nothing new fetched for anyone
ok('a database-wide stall still shows every ticker as behind', late.at['GOOD', 'days_behind'] == 10 and late.at['SPY', 'days_behind'] == 10 and late.at['GOOD', 'band'] == 'amber')

p = h.problems(s)
ok('problems lists amber and red only, worst first, each with an action', p.index[0] == 'MISSING' and 'GOOD' not in p.index and set(p['band']) <= {'amber', 'red'} and p['action'].str.len().gt(0).all())
ok('problems is an empty table when everything is green', h.problems(s[s['band'] == 'green']).empty)
sm = h.summary(s)
ok('summary counts', sm['watchlist tickers'] == len(watch) and sm['in database'] == len(watch) - 1 and sm['behind 1-5d'] == 1 and sm['red'] == 2 and sm['green'] == 5
   and sm['last scrape outcomes'] == {'ok': len(watch) - 1, 'transient_error': 1})
tm, hm = h.treemap(s), h.heatmap(s)
grid = pd.DataFrame(hm.data[0].z, index=hm.data[0].y, columns=hm.data[0].x)
ok('treemap has a tile per ticker, grouped by sector and industry', sum(str(i).count('/') == 2 for i in tm.data[0].ids) == len(watch))
ok('heatmap: full quarters show 100, the hole shows less, quarters before a first date are blank', grid.loc['GOOD'].dropna().eq(100).all() and grid.loc['GAPPY'].min() < 100
   and grid.loc['YOUNG'].isna().any() and 'MISSING' not in grid.index)
ok('nothing written', digest() == before)
