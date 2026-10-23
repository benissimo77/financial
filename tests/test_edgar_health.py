"""EdgarHealth: scoring, bands, problems, summary and the two charts.
Runs against temporary files with mocked downloads; it never touches the real data. Run: python tests/test_edgar_health.py"""
import os, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import sys, os, shutil, hashlib, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
import pandas as pd, numpy as np
from pipeline.EdgarHealth import EdgarHealth
T = tempfile.mkdtemp() + '/tmphealth'; shutil.rmtree(T, ignore_errors=True); os.makedirs(T + '/EDGAR')
ok = lambda name, cond: print(('PASS' if cond else 'FAIL'), name)
METRICS = ['revenue', 'op_income', 'pretax_income', 'tax', 'net_income', 'cfo', 'capex', 'assets', 'equity', 'cash', 'debt_long', 'debt_short', 'shares']
AS_OF = '2026-10-10'


def rows(ticker, ends, revenue=100.0, currency='USD'):
    d = pd.DataFrame({'ticker': ticker, 'period_end': ends, 'filed': pd.DatetimeIndex(ends) + pd.Timedelta(days=40), 'currency': currency})
    for m in METRICS: d[m] = 10.0
    d['revenue'] = revenue; d['assets'] = 1000.0
    return d


def year_rows(q):   # annual rows that agree with the quarters, for calendar years that have all four
    inside = lambda e: q[(q.period_end > e - pd.Timedelta(days=350)) & (q.period_end <= e)]
    a = q[q.period_end.dt.month == 12].copy()
    a = a[[len(inside(e)) == 4 for e in a.period_end]].copy()
    for m in ('revenue', 'net_income', 'cfo'): a[m] = [inside(e)[m].sum() for e in a.period_end]
    return a


Q20 = pd.date_range(end='2026-06-30', periods=20, freq='QE')
q, a = {}, {}
q['GOOD'] = rows('GOOD', Q20, 2e7); a['GOOD'] = year_rows(q['GOOD'])
q['LATE'] = rows('LATE', pd.date_range(end='2025-12-31', periods=18, freq='QE'), 2e7); a['LATE'] = year_rows(q['LATE'])   # 283 days old: 98 overdue
q['HOLE'] = rows('HOLE', Q20.delete(10), 2e7); q['HOLE'].loc[q['HOLE'].index[:2], 'cfo'] = np.nan; a['HOLE'] = year_rows(q['HOLE'])
q['MIS'] = rows('MIS', Q20, 2e7); a['MIS'] = year_rows(q['MIS']); a['MIS'].loc[a['MIS'].index[-1], 'revenue'] *= 1.2
q['NEG'] = rows('NEG', Q20, 2e7); q['NEG'].loc[q['NEG'].index[5], 'revenue'] = -5e6; a['NEG'] = year_rows(q['NEG'])
q['JUMP'] = rows('JUMP', Q20, 2e7); q['JUMP'].loc[q['JUMP'].index[12]:, 'revenue'] = 2e8; a['JUMP'] = year_rows(q['JUMP'])
a['FRGN'] = rows('FRGN', pd.date_range(end='2025-12-31', periods=5, freq='YE'), 5e9, 'TWD')
quarterly = pd.concat(q.values(), ignore_index=True); annual = pd.concat(a.values(), ignore_index=True)
quarterly.to_csv(T + '/EDGAR/edgar-quarterly.csv', index=False, date_format='%Y-%m-%d'); annual.to_csv(T + '/EDGAR/edgar-annual.csv', index=False, date_format='%Y-%m-%d')
names = ['GOOD', 'LATE', 'HOLE', 'MIS', 'NEG', 'JUMP', 'FRGN', 'ETF', 'LOST']
pd.DataFrame({'ticker': names}).to_csv(T + '/watch.csv', index=False)
pd.DataFrame({'ticker': names, 'outcome': ['ok'] * 7 + ['no_facts', 'ok']}).to_csv(T + '/EDGAR/edgar-manifest.csv', index=False)
pd.DataFrame({'ticker': names, 'sector': 'Tech', 'industry': 'Software'}).to_csv(T + '/reg.csv', index=False)
digest = lambda: [hashlib.md5(open(f'{T}/EDGAR/{f}', 'rb').read()).hexdigest() for f in sorted(os.listdir(T + '/EDGAR'))]
before = digest()

h = EdgarHealth(data_dir=T + '/EDGAR', tickers_file=T + '/watch.csv', registry_file=T + '/reg.csv')
s = h.score(as_of=AS_OF)
ok('complete quarterly filer scores 100, green', s.at['GOOD', 'score'] == 100 and s.at['GOOD', 'band'] == 'green' and s.at['GOOD', 'basis'] == 'quarterly')
ok('overdue filer loses 5 points per 30 days', s.at['LATE', 'days_late'] == 98 and abs(s.at['LATE', 'score'] - (100 - 98 / 30 * 5)) < 1e-9 and s.at['LATE', 'band'] == 'amber')
ok('missing quarter and missing core metric both counted', s.at['HOLE', 'missing_periods'] == 1 and abs(s.at['HOLE', 'core_pct'] - 100 * 17 / 19) < 1e-9 and abs(s.at['HOLE', 'score'] - (100 - (100 - 100 * 17 / 19) - 5)) < 1e-9)   # the 5-year window holds 19 expected quarters
ok('restated year is reported but not scored', s.at['MIS', 'sum_mismatches'] == 1 and s.at['MIS', 'score'] == 100 and s.at['MIS', 'band'] == 'green' and 'restated year' in s.at['MIS', 'issues'] and s.at['GOOD', 'sum_mismatches'] == 0)
ok('negative revenue flagged (and the jumps around it)', s.at['NEG', 'negative_revenue'] == 1 and s.at['NEG', 'score'] <= 90)
ok('revenue jump flagged', s.at['JUMP', 'revenue_jumps'] == 1 and s.at['JUMP', 'score'] == 95)
ok('annual-only foreign filer scored on annual rows, green', s.at['FRGN', 'basis'] == 'annual' and s.at['FRGN', 'score'] == 100 and s.at['FRGN', 'currency'] == 'TWD')
ok('ETF is not applicable: no score, never a problem', s.at['ETF', 'band'] == 'n/a' and pd.isna(s.at['ETF', 'score']) and s.at['ETF', 'issues'] == '')
ok('expected statements that are absent score 0, red', s.at['LOST', 'score'] == 0 and s.at['LOST', 'band'] == 'red' and 'no statements' in s.at['LOST', 'issues'])
p = h.problems(s)
ok('problems lists amber and red only, worst first, with actions', list(p.index[:1]) == ['LOST'] and 'ETF' not in p.index and 'GOOD' not in p.index and p['action'].str.len().gt(0).all())
sm = h.summary(s)
ok('summary counts', sm['quarterly filers'] == 6 and sm['annual only'] == 1 and sm['not applicable (no statements at the SEC)'] == 1 and sm['no statements but expected'] == 1 and sm['green'] == 4 and sm['tickers with a restated year (reported, not scored)'] == 1)
t, hm = h.treemap(s), h.heatmap(s)
grid = pd.DataFrame(hm.data[0].z, index=hm.data[0].y, columns=hm.data[0].x)
ok('heatmap: missing quarter shows 0, annual filer blank between year ends', grid.loc['HOLE'].eq(0).sum() == 1 and grid.loc['FRGN'].notna().sum() == 5 and grid.loc['GOOD'].eq(100).all())
ok('treemap leaves out not-applicable tickers', not any('ETF' in str(i) for i in t.data[0].ids) and any('LOST' in str(i) for i in t.data[0].ids))
ok('nothing written', digest() == before and sorted(os.listdir(T + '/EDGAR')) == ['edgar-annual.csv', 'edgar-manifest.csv', 'edgar-quarterly.csv'])

# Saved scores
h.save(s)
back = h.saved()
ok('save() and saved() round-trip the scores', os.path.exists(T + '/EDGAR/edgar-health.csv') and back['band'].tolist() == s['band'].tolist())
hi = h.heatmap(s, by='industry')
ok('industry heatmap: one row per industry, with ticker counts', list(hi.data[0].y) == [f"Software ({(s['basis'] != 'none').sum()})"] and np.asarray(hi.data[0].z).shape[0] == 1)
