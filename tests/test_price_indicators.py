"""PriceIndicators: each indicator against hand-checkable numbers, and the standard pattern
(one wide table in, one wide table out; a ticker with missing prices gets blanks and does not disturb the others).
No files needed. Run: python tests/test_price_indicators.py"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd
from PriceIndicators import PriceIndicators

ok = lambda name, cond: print(('PASS' if cond else 'FAIL'), name)
near = lambda a, b: abs(a - b) < 1e-9
idx = pd.bdate_range('2026-01-05', periods=60)
rng = np.random.default_rng(7)
walk = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 60)))
late = np.where(np.arange(60) < 25, np.nan, 50 + np.arange(60) * 1.0)   # a ticker that only starts trading on day 25
prices = pd.DataFrame({'UP': 100.0 + np.arange(60), 'WALK': walk, 'LATE': late, 'FLAT': 20.0}, index=idx)
pi = PriceIndicators()


def wilder_rsi(values, n):
    """Textbook RSI: first averages are simple means of the first n changes, then each is (previous x (n-1) + new) / n."""
    values = [v for v in values if not np.isnan(v)]
    changes = np.diff(values); gains, losses = np.clip(changes, 0, None), np.clip(-changes, 0, None)
    out = [np.nan] * n
    ag, al = gains[:n].mean(), losses[:n].mean()
    out.append(100.0 if al == 0 and ag > 0 else np.nan if al == 0 else 100 - 100 / (1 + ag / al))
    for g, l in zip(gains[n:], losses[n:]):
        ag, al = (ag * (n - 1) + g) / n, (al * (n - 1) + l) / n
        out.append(100.0 if al == 0 and ag > 0 else np.nan if al == 0 else 100 - 100 / (1 + ag / al))
    return np.array(out)


for name, fn in (('sma', lambda: pi.sma(prices, 5)), ('ema', lambda: pi.ema(prices, 5)), ('daily_returns', lambda: pi.daily_returns(prices)),
                 ('cumulative_returns', lambda: pi.cumulative_returns(prices)), ('rolling_volatility', lambda: pi.rolling_volatility(prices, 5)),
                 ('rsi', lambda: pi.rsi(prices, 14)), ('momentum', lambda: pi.momentum(prices, 10))):
    out = fn()
    ok(f'{name}: same dates and tickers out as in', out.index.equals(prices.index) and list(out.columns) == list(prices.columns))

ok('sma is the mean of the last 5 prices', near(pi.sma(prices, 5)['UP'].iloc[10], np.mean(100.0 + np.arange(6, 11))) and pi.sma(prices, 5)['UP'].iloc[:4].isna().all())
ema = pi.ema(prices, 5)['UP']
ok('ema follows its recursion (alpha = 2 / 6)', near(ema.iloc[0], 100.0) and near(ema.iloc[1], 100.0 + (101.0 - 100.0) * 2 / 6))
ok('daily return', near(pi.daily_returns(prices)['UP'].iloc[1], 0.01) and np.isnan(pi.daily_returns(prices)['UP'].iloc[0]))
ok('cumulative return from the first day', near(pi.cumulative_returns(prices)['UP'].iloc[-1], 159.0 / 100.0 - 1))
ok('momentum over 10 days', near(pi.momentum(prices, 10)['UP'].iloc[10], 110.0 / 100.0 - 1))
ok('rolling volatility is the standard deviation of daily returns', near(pi.rolling_volatility(prices, 5)['WALK'].iloc[20], prices['WALK'].pct_change().iloc[16:21].std()))
ok('volatility of a flat price is zero', near(pi.rolling_volatility(prices, 5)['FLAT'].iloc[-1], 0.0))

rsi = pi.rsi(prices, 14)
ref = wilder_rsi(prices['WALK'].values, 14)
ok('rsi matches the textbook (Wilder) calculation on every day', np.allclose(rsi['WALK'].values, ref, equal_nan=True))
ok('rsi needs 14 changes before its first value', rsi['WALK'].iloc[:14].isna().all() and pd.notna(rsi['WALK'].iloc[14]))
ok('rsi of a price that only rises is 100', near(rsi['UP'].iloc[-1], 100.0))
ok('rsi stays between 0 and 100', rsi['WALK'].dropna().between(0, 100).all())
ok('rsi of a flat price is blank, not an error', rsi['FLAT'].isna().all())
ok('a ticker that starts late gets its own warm-up and leaves the others untouched',
   rsi['LATE'].iloc[:25 + 14].isna().all() and near(rsi['LATE'].iloc[25 + 14], 100.0) and np.allclose(pi.rsi(prices[['WALK']], 14)['WALK'].values, rsi['WALK'].values, equal_nan=True))
ok('late starter is blank before it trades in every indicator', pi.sma(prices, 5)['LATE'].iloc[:25].isna().all() and pi.momentum(prices, 10)['LATE'].iloc[:35].isna().all())

corr = pi.correlation(prices[['UP', 'WALK', 'LATE']])
ok('correlation is a square table with ones on the diagonal', list(corr.index) == list(corr.columns) == ['UP', 'WALK', 'LATE'] and near(corr.at['UP', 'UP'], 1.0) and near(corr.at['UP', 'LATE'], 1.0))
