"""FundamentalData: period types, trailing twelve months, point-in-time filtering, and who is included by default.
Runs against temporary statement files; it never touches the real data. Run: python tests/test_fundamental_data.py"""
import os, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd
from pipeline.EdgarHealth import EdgarHealth
from FundamentalData import FundamentalData

T = tempfile.mkdtemp(); os.makedirs(T + '/EDGAR')
ok = lambda name, cond: print(('PASS' if cond else 'FAIL'), name)
METRICS = ['revenue', 'op_income', 'pretax_income', 'tax', 'net_income', 'cfo', 'capex', 'assets', 'equity', 'cash', 'debt_long', 'debt_short', 'shares']
today = pd.Timestamp.today().normalize()


def rows(ticker, ends, currency='USD'):
    d = pd.DataFrame({'ticker': ticker, 'period_end': pd.DatetimeIndex(ends), 'filed': pd.DatetimeIndex(ends) + pd.Timedelta(days=40), 'currency': currency})
    for m in METRICS: d[m] = 10.0
    d['revenue'] = [1e7 + 1e6 * i for i in range(len(d))]   # 10, 11, 12 ... million, so sums are easy to check
    d['assets'] = [5e8 + 1e7 * i for i in range(len(d))]
    return d


def years(q):   # annual rows consistent with the quarters, for calendar years that have all four
    inside = lambda e: q[(q.period_end > e - pd.Timedelta(days=350)) & (q.period_end <= e)]
    a = q[q.period_end.dt.month == 12].copy()
    a = a[[len(inside(e)) == 4 for e in a.period_end]].copy()
    for m in ('revenue', 'net_income', 'cfo'): a[m] = [inside(e)[m].sum() for e in a.period_end]
    return a


recent = pd.date_range(end=today - pd.offsets.QuarterEnd(1), periods=16, freq='QE')   # quarters up to the last one ended
stale = pd.date_range(end=today - pd.Timedelta(days=330), periods=16, freq='QE')      # stopped filing about a year ago: red
q = {'GOOD': rows('GOOD', recent), 'GAPY': rows('GAPY', recent.delete(8)), 'TWDC': rows('TWDC', recent, 'TWD'),
     'REDX': rows('REDX', stale), 'OFFW': rows('OFFW', recent)}
a = {t: years(d) for t, d in q.items()}
a['ANNL'] = rows('ANNL', pd.date_range(end=today - pd.offsets.YearEnd(1), periods=5, freq='YE'))
pd.concat(q.values()).to_csv(T + '/EDGAR/edgar-quarterly.csv', index=False, date_format='%Y-%m-%d')
pd.concat(a.values()).to_csv(T + '/EDGAR/edgar-annual.csv', index=False, date_format='%Y-%m-%d')
watch = ['GOOD', 'GAPY', 'TWDC', 'REDX', 'ANNL', 'ETFX']   # OFFW has statements but is not on the watchlist
pd.DataFrame({'ticker': watch}).to_csv(T + '/watch.csv', index=False)
pd.DataFrame({'ticker': watch, 'outcome': ['ok'] * 5 + ['no_facts']}).to_csv(T + '/EDGAR/edgar-manifest.csv', index=False)
health = lambda: EdgarHealth(data_dir=T + '/EDGAR', tickers_file=T + '/watch.csv', registry_file=T + '/none.csv')
fd = FundamentalData(data_dir=T + '/EDGAR', health=health())

f = fd.filers()
ok('filers table: basis, band and currency per watchlist ticker', f.at['GOOD', 'basis'] == 'quarterly' and f.at['ANNL', 'basis'] == 'annual' and f.at['REDX', 'band'] == 'red'
   and f.at['TWDC', 'currency'] == 'TWD' and f.at['ETFX', 'basis'] == 'none' and 'OFFW' not in f.index)
ok('quarterly view: quarterly filers only, no red, no other currency, watchlist only', fd.get_tickers('quarterly') == ['GAPY', 'GOOD'])
ok('annual view also holds annual-only filers', fd.get_tickers('annual') == ['ANNL', 'GAPY', 'GOOD'])
ok('red tickers on request', 'REDX' in FundamentalData(data_dir=T + '/EDGAR', health=health(), include_red=True).get_tickers('annual') and 'REDX' not in fd.get_tickers('annual'))
ok('other currencies on request', FundamentalData(data_dir=T + '/EDGAR', health=health(), currency=None).get_tickers() == ['GAPY', 'GOOD', 'TWDC'])

w = fd.get_data_wide('revenue')
ok('wide format: period ends down, tickers across', list(w.columns) == ['GAPY', 'GOOD'] and isinstance(w.index, pd.DatetimeIndex) and len(w) == 16 and w['GOOD'].iloc[0] == 1e7)
ok('a missing quarter is a gap, not an error', w['GAPY'].isna().sum() == 1)
ttm = fd.get_data_wide('revenue', period='ttm')
ok('trailing twelve months sums four quarters', ttm['GOOD'].iloc[3] == 1e7 + 1.1e7 + 1.2e7 + 1.3e7 and ttm['GOOD'].iloc[:3].isna().all() and ttm['GOOD'].iloc[-1] == sum(1e7 + 1e6 * i for i in range(12, 16)))
ok('trailing twelve months is blank across a gap, then recovers', ttm['GAPY'].notna().sum() == 16 - 1 - 3 - 3 and pd.notna(ttm['GAPY'].iloc[-1]))
ok('balances are the latest value, not a sum', fd.get_data_wide('assets', period='ttm')['GOOD'].iloc[-1] == 5e8 + 1e7 * 15)
ann = fd.get_data_wide('revenue', period='annual')
ok('annual view', set(ann.columns) == {'ANNL', 'GAPY', 'GOOD'} and ann['ANNL'].notna().sum() == 5)

cut = recent[9] + pd.Timedelta(days=10)    # ten days after a quarter ends, its figures are not filed yet (filed after 40)
pit = fd.get_data_wide('revenue', as_of=cut)
ok('point-in-time: a quarter not yet filed on the date is hidden', pit.index.max() == recent[8] and len(pit) == 9)
ok('point-in-time applies to long format too', fd.get_data_long(as_of=cut)['filed'].max() <= cut)
ok('ticker filter narrows, but cannot bring back an excluded ticker', list(fd.get_data_wide('revenue', tickers=['GOOD', 'REDX']).columns) == ['GOOD'])
ok('long format carries every column', list(fd.get_data_long().columns) == ['ticker', 'period_end', 'filed', 'currency'] + METRICS)

for name, call in (('unknown metric', lambda: fd.get_data_wide('profit')), ('unknown period', lambda: fd.get_data_wide('revenue', period='monthly'))):
    try: call(); raised = None
    except ValueError as e: raised = str(e)
    ok(f'{name} raises with the allowed values', raised is not None and ('revenue' in raised or 'quarterly' in raised))
try: FundamentalData(data_dir=T + '/nowhere', health=health()); raised = None
except Exception as e: raised = e
ok('missing statement file raises, naming the file', isinstance(raised, FileNotFoundError) and 'edgar-quarterly.csv' in str(raised))
