"""Backtest engine and strategy helpers: trading rules, the point-in-time universe, and robustness.
Runs on a small synthetic price file; it never touches the real data. Run: python tests/test_backtest.py"""
import os, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd
from PriceData import PriceData
from Backtest import Backtest
from Strategy import hold, top, weights

T = tempfile.mkdtemp()
ok = lambda name, cond: print(('PASS' if cond else 'FAIL'), name)
near = lambda a, b, tol=1e-9: abs(a - b) < tol
days = pd.bdate_range('2019-09-02', '2020-06-30')          # history from September, the test starts in January
START = '2020-01-02'
n = len(days)
grow = lambda rate: 100 * (1 + rate) ** np.arange(n)
series = {
    'UP': grow(0.001),                                     # rises 0.1% a day
    'DOWN': grow(-0.001),                                  # falls 0.1% a day
    'FLAT': np.full(n, 50.0),
    'GAPPY': grow(0.002),                                  # rises steadily but has missing days (removed below)
    'THIN': grow(0.001),                                   # too little trading until March
    'PENNY': np.full(n, 1.5),                              # below the $2 minimum price
    'DIES': grow(0.001),                                   # stops trading in April
    'RED': grow(0.001),                                    # red in the price health report
    'LATE': grow(0.001),                                   # first trades in March
    'SPY': grow(0.0005),                                   # the benchmark
}
rows = []
for t, px in series.items():
    for i, d in enumerate(days):
        if t == 'GAPPY' and pd.Timestamp('2020-02-10') <= d <= pd.Timestamp('2020-02-14'): continue
        if t == 'DIES' and d > pd.Timestamp('2020-04-15'): continue
        if t == 'LATE' and d < pd.Timestamp('2020-03-02'): continue
        vol = 1e3 if (t == 'THIN' and d < pd.Timestamp('2020-03-01')) else 1e6
        rows.append((t, d.strftime('%Y-%m-%d'), px[i], px[i], px[i], px[i], vol, px[i]))
path = os.path.join(T, 'prices.parquet')
pd.DataFrame(rows, columns=['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']).to_parquet(path, index=False)
prices = PriceData(path, tickers_file=None)


class Health:   # stands in for DataHealth: RED is red, everything else green
    def score(self):
        return pd.DataFrame({'band': ['red' if t == 'RED' else 'green' for t in series]}, index=list(series))


bt = lambda strategy, **kw: Backtest(strategy, prices=prices, health=Health(), start=START, **{'cost': 0.0, **kw}).run()
px = prices.get_data_wide()
test_days = px.index[px.index >= START]
first_trade = test_days[1]                                  # decided on the first test day, traded at the next close

# ---- the universe
u = Backtest(lambda d: None, prices=prices, health=Health()).universe()
ok('benchmark and red tickers are never in the universe', 'SPY' not in u.columns and 'RED' not in u.columns and 'UP' in u.columns)
ok('liquidity needs 40 days of history first', not u['UP'].iloc[:40].any() and u['UP'].iloc[41])
ok('a price under $2 is never eligible', not u['PENNY'].any())
ok('thin trading becomes eligible only once enough days of real volume have passed',
   not u.loc[:'2020-03-20', 'THIN'].any() and u.loc['2020-06-01':, 'THIN'].all())
ok('a late lister is not eligible before it trades, nor for its first 40 days', not u.loc[:'2020-04-24', 'LATE'].any() and u.loc['2020-05-01', 'LATE'])
ok('a gap in prices does not make a ticker ineligible, during or after it', u.loc['2020-02-11', 'GAPPY'] and u.loc['2020-02-17':, 'GAPPY'].all())
ok('a ticker is not eligible after its prices stop', not u.loc['2020-04-16':, 'DIES'].any())

# ---- trading rules
r = bt(lambda d: weights(pd.DataFrame({'UP': 1.0}, index=d.prices.index)))
ok('holding one stock earns exactly its return from the day it was bought', near(r.equity.iloc[-1], px['UP'].iloc[-1] / px['UP'].loc[first_trade]))
ok('the test starts in cash, and buys at the next close after the first decision', r.equity.loc[first_trade] == 1.0 and r.holdings.loc[test_days[0]].sum() == 0
   and near(r.holdings.loc[first_trade, 'UP'], 1.0))
flip = pd.Timestamp('2020-03-16')
signal = lambda d: hold(pd.DataFrame({'UP': d.prices.index >= flip}, index=d.prices.index))
r = bt(signal, rebalance='daily')
nxt = r.holdings.index[r.holdings.index.get_loc(flip) + 1]
ok('one day of delay: a signal on a day is traded at the next close', r.holdings.loc[flip, 'UP'] == 0 and near(r.holdings.loc[nxt, 'UP'], 1.0)
   and near(r.equity.loc[nxt], 1.0) and r.equity.iloc[-1] > 1.0)

half = lambda d: weights(pd.DataFrame({'UP': 0.5, 'DOWN': 0.5}, index=d.prices.index))
r = bt(half)                                                # monthly
jan_end = r.holdings.loc[:'2020-01-31'].index[-1]
u_grow, d_grow = px['UP'].loc[jan_end] / px['UP'].loc[first_trade], px['DOWN'].loc[jan_end] / px['DOWN'].loc[first_trade]
ok('holdings drift with prices between rebalances', near(r.holdings.loc[jan_end, 'UP'], u_grow / (u_grow + d_grow)))
feb_trade = r.holdings.index[r.holdings.index.get_loc(jan_end) + 1]
ok('they are reset to target at the next rebalance, and only then', near(r.holdings.loc[feb_trade, 'UP'], 0.5) and (r.turnover > 0).sum() == 6)
cost = bt(half, cost=0.001)
ok('costs are 0.1% of the value traded', near(cost.costs.sum(), 0.001 * cost.turnover.sum()) and near(cost.costs.loc[first_trade], 0.001)
   and cost.equity.iloc[-1] < r.equity.iloc[-1])
ok('buying from all cash turns over the whole portfolio once', near(cost.turnover.loc[first_trade], 1.0))

r = bt(lambda d: weights(pd.DataFrame({'GAPPY': 1.0}, index=d.prices.index)))
ok('a gap in prices keeps its return', near(r.equity.iloc[-1], px['GAPPY'].iloc[-1] / px['GAPPY'].loc[first_trade], 1e-9))
r = bt(lambda d: weights(pd.DataFrame({'DIES': 0.5, 'UP': 0.5}, index=d.prices.index)), rebalance='quarterly')
after = r.holdings.loc['2020-04-16':]
ok('a holding whose prices stop is sold at its last price and becomes cash', (after['DIES'] == 0).all() and r.equity.notna().all()
   and near(r.equity.loc['2020-04-16'], r.equity.loc['2020-04-15'] * (1 + 0.5 * 0 + 0), 0.01))

# ---- helpers
r = bt(lambda d: hold(d.prices > 0), rebalance='monthly')
june = r.holdings.loc['2020-06-01':].iloc[0]
held = june[june > 0]
ok('hold(): equal weights across eligible tickers only', set(held.index) == {'UP', 'DOWN', 'FLAT', 'GAPPY', 'THIN', 'LATE'} and np.allclose(held, 1 / 6))
r = bt(lambda d, k: top(d.prices.pct_change(20) * pd.Series({'UP': 1.5, 'DOWN': -1}).reindex(d.prices.columns).fillna(1), n=k), k=2)
last = r.holdings.loc[r.turnover[r.turnover > 0].index[-1]]   # on the last trading day, before any drift
ok('top(): the n highest scores, equally weighted; parameters reach the strategy', set(last[last > 0].index) == {'GAPPY', 'UP'} and np.allclose(last[last > 0], 0.5))
r = bt(lambda d: top(d.prices.pct_change(20)[['FLAT']], n=3))   # a score for one ticker only
ok('top(): fewer scores than n holds those available', set(r.holdings.iloc[-1][r.holdings.iloc[-1] > 0].index) == {'FLAT'} and near(r.holdings.iloc[-1].sum(), 1.0))
r = bt(lambda d: hold(d.prices < 0))
ok('nothing chosen means all cash', (r.holdings.sum(axis=1) == 0).all() and (r.equity == 1.0).all())
r = bt(lambda d: pd.DataFrame({'UP': 0.6, 'THIN': 0.4, 'NOPE': 0.3}, index=d.prices.index))
first = r.holdings.loc[first_trade]
ok('a plain table is taken as weights; ineligible and unknown tickers get zero, the rest is cash',
   near(first['UP'], 0.6) and first['THIN'] == 0 and 'NOPE' not in first.index and near(first.sum(), 0.6))
r = bt(lambda d: weights(pd.DataFrame({'UP': 1.0, 'DOWN': -1.0}, index=d.prices.index)))
ok('shorts: a long-short position runs, and the short side profits as its price falls', r.equity.iloc[-1] > 1.0 and near(r.holdings.loc[first_trade, 'DOWN'], -1.0))

# ---- robustness and access to data
bad = lambda d: weights(d.prices.pct_change().replace(0, np.inf).assign(FLAT=np.nan) * 0 + 0.1)
ok('infinite and blank weights do not break a run', bt(bad).equity.notna().all())
def uses_benchmark(d):
    above = d.benchmark > d.benchmark.rolling(20).mean()
    return hold(pd.DataFrame({t: above for t in ['UP']}, index=d.prices.index))
ok('the benchmark is available to strategies as a market filter', bt(uses_benchmark).equity.iloc[-1] > 1.0)
ok('price indicators are available to strategies', bt(lambda d: hold(d.prices > d.price_ind.sma(d.prices, 50))).equity.notna().all())
r = bt(half, tickers=['UP'])
ok('tickers= narrows the universe', list(r.holdings.columns) == ['UP'] and near(r.holdings.loc[first_trade, 'UP'], 0.5))
r = bt(half, rebalance='weekly'); trades = r.turnover[r.turnover > 0].index
ok('weekly rebalancing trades on the first trading day of each week', len(trades) > 20 and all(d.dayofweek == 0 or d == first_trade for d in trades[1:4]))
ok('the result records its settings', r.settings['rebalance'] == 'weekly' and r.settings['start'] == test_days[0] and r.name == '<lambda>')
for name, call in (('unknown rebalance', lambda: Backtest(half, rebalance='yearly', prices=prices, health=Health())), ('top without n', lambda: top(px, n=0))):
    try: call(); raised = False
    except ValueError: raised = True
    ok(f'{name} raises', raised)

# ---- baselines
from Backtest import BacktestResult
r = bt(lambda d: top(d.prices.pct_change(20), n=2), cost=0.001)
ok('every run carries the two baselines', list(r.baselines) == ['Equal-weight universe', 'SPY'])
ew = r.baselines['Equal-weight universe']
own_all = bt(lambda d: hold(d.prices.notna()), cost=0.001)
ok('equal-weight baseline = owning every eligible ticker equally, same schedule and costs', np.allclose(ew.equity, own_all.equity)
   and (ew.turnover > 0).sum() == (r.turnover > 0).sum())
spy = r.baselines['SPY']
ok('SPY baseline: bought at the strategy\'s first trade, with the same cost, then held', spy.equity.loc[test_days[0]] == 1.0
   and near(spy.equity.iloc[-1], px['SPY'].iloc[-1] / px['SPY'].loc[first_trade] * 0.999) and (spy.turnover > 0).sum() == 1)
no_spy_path = os.path.join(T, 'prices-no-spy.parquet')
pd.read_parquet(path).query("ticker != 'SPY'").to_parquet(no_spy_path, index=False)
r2 = Backtest(lambda d: hold(d.prices.notna()), prices=PriceData(no_spy_path, tickers_file=None), health=Health(), start=START).run()
ok('without benchmark prices the run still works, with only the equal-weight baseline', list(r2.baselines) == ['Equal-weight universe'])

# ---- measures, on a hand-built record
idx = pd.DatetimeIndex(['2021-01-04', '2021-01-05', '2021-01-06', '2021-01-07', '2022-01-04'])
eq = pd.Series([1.0, 1.1, 0.99, 1.089, 1.21], index=idx)
hb = pd.DataFrame({'A': [0, 1, 1, 0.5, 0.5], 'B': [0, 0, 0, 0.5, 0.5]}, index=idx)
to = pd.Series([0, 1.0, 0, 1.0, 0], index=idx)
hand = BacktestResult('hand', eq, hb, to, to * 0.001, {})
m = hand.metrics().loc['hand']
rets = eq.pct_change().fillna(0); years = 365 / 365.25
ok('total and annual return', near(m['total_return'], 0.21) and near(m['annual_return'], 1.21 ** (1 / years) - 1))
ok('volatility and Sharpe, annualised from daily returns', near(m['volatility'], rets.std() * np.sqrt(252)) and near(m['sharpe'], rets.mean() * 252 / (rets.std() * np.sqrt(252))))
ok('worst drawdown and the longest stretch below a previous peak', near(m['worst_drawdown'], 0.99 / 1.1 - 1) and m['longest_drawdown_days'] == 2)
ok('turnover and cost per year', near(m['turnover_per_year'], 2.0 / years) and near(m['cost_per_year'], 0.002 / years))
ok('average holdings counted from the first trade', near(m['avg_holdings'], (1 + 1 + 2 + 2) / 4))
table = r.metrics()
ok('metrics(): one row per run, strategy first, one column per measure', list(table.index) == ['<lambda>', 'Equal-weight universe', 'SPY']
   and list(table.columns) == ['total_return', 'annual_return', 'volatility', 'sharpe', 'worst_drawdown', 'longest_drawdown_days',
                               'turnover_per_year', 'cost_per_year', 'avg_holdings'] and table.notna().all().all())
cashy = bt(lambda d: hold(d.prices < 0)).metrics().iloc[0]
ok('a strategy that never invests gives zeros and blanks, not errors', cashy['total_return'] == 0 and cashy['avg_holdings'] == 0 and pd.isna(cashy['sharpe']))

# ---- tear sheet
import io, contextlib
import plotly.graph_objects as go
r = bt(lambda d: weights(pd.DataFrame({'UP': 0.5, 'DOWN': 0.5}, index=d.prices.index)), cost=0.001)
cal = r.calendar_returns()
e = r.equity
ok('calendar returns: one row per year, one column per run, chaining to the total', list(cal.index) == [2020] and list(cal.columns) == ['<lambda>', 'Equal-weight universe', 'SPY']
   and near((1 + cal['<lambda>']).prod() - 1, e.iloc[-1] / e.iloc[0] - 1))
hand_cal = BacktestResult('hand', pd.Series([1.0, 1.1, 1.21], index=pd.DatetimeIndex(['2020-06-01', '2020-12-31', '2021-06-30'])),
                          pd.DataFrame({'A': [1, 1, 1]}), pd.Series([1.0, 0, 0]), pd.Series([0.0, 0, 0]), {}).calendar_returns()
ok('calendar returns across a year end', np.allclose(hand_cal['hand'].values, [0.1, 0.1]))
c = r.contributions()
ok('contributions: the rising stock added, the falling one lost, nothing else appears', set(c.index) == {'UP', 'DOWN'} and c['UP'] > 0 > c['DOWN']
   and np.allclose(r.contribution.sum(axis=1)[r.turnover == 0], r.returns[r.turnover == 0]))   # on days without trading they add up to the day's return
ok('baselines without a contribution record give an empty table, not an error', r.baselines['SPY'].contributions().empty)
notes = r.caveats()
ok('caveats state the period, schedule, costs and the hindsight in the universe', 'monthly' in notes[0] and '0.10%' in notes[0] and any('failed or were delisted' in n for n in notes))
one = bt(lambda d, x: weights(pd.DataFrame({'UP': 0.9, 'FLAT': 0.1}, index=d.prices.index)), x=1)
ok('caveats warn about a single dominant stock, few holdings, and a single setting', any('UP made' in n for n in one.caveats()) and any('only 2.0 stocks' in n for n in one.caveats())
   and any('x=1' in n for n in one.caveats()))
fig = r.figure()
ok('figure: growth, drawdown and rolling return for all three runs, growth on a log scale', len(fig.data) == 9 and fig.layout.yaxis.type == 'log'
   and [t.name for t in fig.data[::3]] == ['<lambda>', 'Equal-weight universe', 'SPY'])
shown = []
go.Figure.show = lambda self, *a, **k: shown.append(self)
with contextlib.redirect_stdout(io.StringIO()) as out:
    r.tear_sheet()
ok('tear_sheet() runs outside a notebook, shows the figure once, and prints the tables and caveats', len(shown) == 1 and 'Annual return' in out.getvalue()
   and 'Calendar-year returns' in out.getvalue() and 'Contributors' in out.getvalue() and 'Caveats' in out.getvalue())

# Without a health object the backtest reads the saved price health scores
pd.DataFrame({'ticker': list(series), 'band': ['red' if t == 'RED' else 'green' for t in series]}).to_csv(os.path.join(T, 'price-health.csv'), index=False)
u2 = Backtest(lambda d: None, prices=prices).universe()
ok('saved price health scores are used by default: red tickers left out', 'RED' not in u2.columns and 'UP' in u2.columns)
