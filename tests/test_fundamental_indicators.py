"""FundamentalIndicators: each indicator against hand-checkable numbers, the daily (as known) form, the snapshot,
and blanks instead of errors for unusable inputs. Uses temporary files only. Run: python tests/test_fundamental_indicators.py"""
import os, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd
from pipeline.EdgarHealth import EdgarHealth
from FundamentalData import FundamentalData
from FundamentalIndicators import FundamentalIndicators
from PriceData import PriceData

T = tempfile.mkdtemp(); os.makedirs(T + '/EDGAR')
ok = lambda name, cond: print(('PASS' if cond else 'FAIL'), name)
near = lambda a, b: abs(a - b) < 1e-9
METRICS = ['revenue', 'op_income', 'pretax_income', 'tax', 'net_income', 'cfo', 'capex', 'assets', 'equity', 'cash', 'debt_long', 'debt_short', 'shares']
today = pd.Timestamp.today().normalize()
recent = pd.date_range(end=today - pd.offsets.QuarterEnd(1), periods=12, freq='QE')


def rows(ticker, ends, **over):
    """Per quarter: revenue 100 (doubling after the first 4 quarters when grow=True), op income 20, pre-tax 20, tax 5, net 15,
    cash flow 30, capex 10; equity 500, cash 100, long debt 150, short debt 50; 10 shares."""
    n = len(ends)
    d = pd.DataFrame({'ticker': ticker, 'period_end': pd.DatetimeIndex(ends), 'filed': pd.DatetimeIndex(ends) + pd.Timedelta(days=40), 'currency': 'USD'})
    base = dict(revenue=100.0, op_income=20.0, pretax_income=20.0, tax=5.0, net_income=15.0, cfo=30.0, capex=10.0,
                assets=1000.0, equity=500.0, cash=100.0, debt_long=150.0, debt_short=50.0, shares=10.0)
    for m in METRICS: d[m] = base[m]
    for m, v in over.items(): d[m] = v
    return d


def years(q, n=4):
    a = q[q.period_end.dt.month == 12].copy()
    for m in ('revenue', 'op_income', 'pretax_income', 'tax', 'net_income', 'cfo', 'capex'): a[m] = a[m] * n
    return a


q = {'BASE': rows('BASE', recent),
     'GROW': rows('GROW', recent, revenue=[100.0] * 4 + [150.0] * 8),          # trailing revenue 400, then rising to 600
     'NODEBT': rows('NODEBT', recent, debt_long=np.nan, debt_short=np.nan),   # no debt figure at all
     'NEGCAP': rows('NEGCAP', recent, equity=10.0, cash=500.0),               # invested capital below zero
     'NOREV': rows('NOREV', recent, revenue=0.0)}                             # revenue of zero
a = {t: years(d) for t, d in q.items()}
a['ANNL'] = years(rows('ANNL', pd.date_range(end=today - pd.offsets.YearEnd(1), periods=4, freq='YE')), 1)
pd.concat(q.values()).to_csv(T + '/EDGAR/edgar-quarterly.csv', index=False, date_format='%Y-%m-%d')
pd.concat(a.values()).to_csv(T + '/EDGAR/edgar-annual.csv', index=False, date_format='%Y-%m-%d')
watch = list(q) + ['ANNL']
pd.DataFrame({'ticker': watch}).to_csv(T + '/watch.csv', index=False)
pd.DataFrame({'ticker': watch, 'outcome': 'ok'}).to_csv(T + '/EDGAR/edgar-manifest.csv', index=False)
# Prices: every ticker closes at 20 on every weekday, so market value = 20 x 10 shares = 200.
days = pd.bdate_range(recent[0] - pd.Timedelta(days=5), today)
px = pd.DataFrame([(t, d.strftime('%Y-%m-%d'), 20.0, 20.0, 20.0, 20.0, 1000.0, 18.0) for t in watch if t != 'NOREV' for d in days],
                  columns=['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'adj_close'])
px.to_csv(T + '/prices.csv', index=False)
fd = FundamentalData(data_dir=T + '/EDGAR', health=EdgarHealth(data_dir=T + '/EDGAR', tickers_file=T + '/watch.csv', registry_file=T + '/none.csv'), include_red=True)
ind = FundamentalIndicators(fd, PriceData(T + '/prices.csv', tickers_file=None))
last = lambda table, t: table[t].dropna().iloc[-1]

om = ind.operating_margin()
ok('one metric, one wide table: period ends down, tickers across', isinstance(om.index, pd.DatetimeIndex) and 'BASE' in om.columns and 'ANNL' not in om.columns)
ok('operating margin = 80 / 400', near(last(om, 'BASE'), 0.20))
ok('net margin = 60 / 400', near(last(ind.net_margin(), 'BASE'), 0.15))
ok('free cash flow margin = (120 - 40) / 400', near(last(ind.fcf_margin(), 'BASE'), 0.20))
g = ind.revenue_growth()
ok('revenue growth: flat company is 0, and it needs a year of history first', near(last(g, 'BASE'), 0.0) and g['BASE'].notna().sum() == 12 - 3 - 4)
ok('revenue growth = 600 / 400 - 1 once the faster year is complete', near(g['GROW'].dropna().iloc[0], 0.5) and near(last(g, 'GROW'), 0.0))
ok('ROIC = 80 x (1 - 5/20) / (500 + 150 + 50 - 100)', near(last(ind.roic(), 'BASE'), 80 * 0.75 / 600))
ok('ROIC with no debt figure counts debt as zero', near(last(ind.roic(), 'NODEBT'), 80 * 0.75 / 400))
ok('free cash flow yield = 80 / (20 x 10)', near(last(ind.fcf_yield(), 'BASE'), 0.40))

ok('unusable inputs give blanks, not errors: negative invested capital', ind.roic()['NEGCAP'].isna().all())
ok('unusable inputs give blanks, not errors: zero revenue', ind.operating_margin()['NOREV'].isna().all() and ind.revenue_growth()['NOREV'].isna().all())
ok('a ticker with no prices is blank for yield and fine elsewhere', ind.fcf_yield()['NOREV'].isna().all() and 'NOREV' in ind.fcf_yield().columns)
ok('annual period includes the annual-only filer', near(last(ind.operating_margin(period='annual'), 'ANNL'), 0.20))
ok('ticker filter', list(ind.roic(tickers=['BASE', 'GROW']).columns) == ['BASE', 'GROW'])

d = ind.operating_margin(daily=True)
first_known = recent[3] + pd.Timedelta(days=40)     # the fourth quarter completes the first trailing year; it is filed 40 days later
ok('daily form is indexed by trading day', d.index.equals(pd.DatetimeIndex(days)) and list(d.columns) == list(om.columns))
ok('daily form shows nothing before the figures were filed', d.loc[d.index < first_known, 'BASE'].isna().all() and near(d.loc[d.index >= first_known, 'BASE'].iloc[0], 0.20))
gd = ind.revenue_growth(daily=True)['GROW']
switch = recent[7] + pd.Timedelta(days=40)          # the quarter that completes the faster year, plus the filing delay
ok('daily form changes on the filing date, not the period end', gd[gd.index < switch].isna().all() and near(gd[gd.index >= switch].iloc[0], 0.5) and gd[gd.index >= recent[7]].isna().any())
ok('daily yield uses that day\'s price', near(ind.fcf_yield(daily=True)['BASE'].dropna().iloc[-1], 0.40))
ok('as_of limits the daily form too', ind.operating_margin(daily=True, as_of=recent[8]).index.max() <= recent[8])

s = ind.snapshot()
ok('snapshot: one row per ticker, every indicator, all tickers by default', list(s.columns) == ['basis', 'period_end'] + FundamentalIndicators.INDICATORS and set(s.index) == set(watch))
ok('snapshot mixes trailing-year and annual filers, labelled', s.at['BASE', 'basis'] == 'quarterly' and s.at['ANNL', 'basis'] == 'annual' and near(s.at['ANNL', 'operating_margin'], 0.20))
ok('snapshot holds the latest values', near(s.at['BASE', 'roic'], 0.10) and near(s.at['BASE', 'fcf_yield'], 0.40) and s.at['BASE', 'period_end'] == recent[-1])
ok('snapshot keeps blanks for unusable inputs', pd.isna(s.at['NEGCAP', 'roic']) and pd.isna(s.at['NOREV', 'net_margin']))
early = ind.snapshot(['revenue_growth', 'roic'], tickers=['GROW', 'BASE'], as_of=recent[8] + pd.Timedelta(days=10))   # the quarter just ended is not filed yet
ok('snapshot narrowed by indicator, ticker and date', list(early.columns) == ['basis', 'period_end', 'revenue_growth', 'roic'] and list(early.index) == ['BASE', 'GROW']
   and early.at['GROW', 'period_end'] == recent[7] and near(early.at['GROW', 'revenue_growth'], 0.5))
for name, call in (('unknown indicator', lambda: ind.snapshot(['magic'])), ('unknown period', lambda: ind.roic(period='quarterly'))):
    try: call(); raised = False
    except ValueError: raised = True
    ok(f'{name} raises', raised)
