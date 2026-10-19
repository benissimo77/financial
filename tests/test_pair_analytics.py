"""PairAnalytics: the hedge ratio, the spread built on it, and signals that only open on an entry.
No files needed. Run: python tests/test_pair_analytics.py"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd
from PairAnalytics import PairAnalytics

ok = lambda name, cond: print(('PASS' if cond else 'FAIL'), name)
rng = np.random.default_rng(11)
n = 600
idx = pd.bdate_range('2024-01-01', periods=n)
b = 50 + np.cumsum(rng.normal(0, 0.5, n))                 # a random walk
noise = rng.normal(0, 1.0, n)                             # stationary: the pair's true spread
pair = pd.DataFrame({'BIG': 5 + 10 * b + noise, 'SMALL': b}, index=idx)          # BIG moves 10 for every 1 of SMALL
other = pd.DataFrame({'X': 100 + np.cumsum(rng.normal(0, 1, n)), 'Y': 100 + np.cumsum(rng.normal(0, 1, n))}, index=idx)   # unrelated walks
pa = PairAnalytics()

ratio = pa.hedge_ratio(pair)
ok('hedge ratio recovers the true ratio of 10', abs(ratio.iloc[-1] - 10) < 0.1 and ratio.index.equals(pair.index))
ok('whole-history ratio is one constant', ratio.nunique() == 1)
spread = pa.spread(pair)
ok('spread removes the common movement, leaving the noise', abs(spread.std() - noise.std()) < 0.1 and spread.index.equals(pair.index))
ok('spread is far steadier than the raw price difference', spread.std() < 0.2 * (pair['BIG'] - pair['SMALL']).std())

shift = pair.copy(); shift.loc[idx[300]:, 'BIG'] = 5 + 20 * b[300:] + noise[300:]   # the relationship doubles half way
rolling = pa.hedge_ratio(shift, window=60)
ok('rolling ratio tracks a change in the relationship', abs(rolling.iloc[250] - 10) < 1 and abs(rolling.iloc[-1] - 20) < 1)
ok('rolling ratio needs its window first', rolling.iloc[:59].isna().all() and pd.notna(rolling.iloc[59]))
ok('rolling ratio on a day uses only prices up to that day', abs(pa.hedge_ratio(shift.iloc[:200], window=60).iloc[-1] - rolling.iloc[199]) < 1e-9)
ok('rolling spread is available wherever the ratio is', pa.spread(shift, hedge_window=60).notna().sum() == n - 59)

z = pa.zscore(pair, window=30)
ok('z-score is centred with unit scale', abs(z.mean()) < 0.2 and 0.7 < z.std() < 1.3 and z.iloc[:29].isna().all())

score, pvalue = pa.cointegration_test(pair)
ok('cointegration test: related pair passes, unrelated walks do not', pvalue < 0.01 and pa.cointegration_test(other)[1] > 0.05)

path = [np.nan, 0.0, 2.5, 1.5, 0.3, 0.8, 1.2, 0.1, -0.9, -2.2, -1.0, np.nan, -0.7, 2.4, 0.2]
want = [0,      0,   -1,  -1,  0,   0,   0,   0,   0,    1,    1,    1,      1,    -1,  0]
zs = pd.Series(path, index=idx[:len(path)])
stub = PairAnalytics(); stub.zscore = lambda prices, window=30, hedge_window=None: zs
sig = stub.generate_signals(pair.iloc[:len(path)])
ok('signals: columns and index', list(sig.columns) == ['zscore', 'signal'] and sig.index.equals(zs.index))
ok('signals: open only on an entry, close inside the exit band, stay closed until the next entry', sig['signal'].tolist() == want)
ok('signals: after an exit, drifting back between the two levels does not reopen the position', sig['signal'].iloc[5:9].eq(0).all())
ok('signals: a jump straight to the opposite entry level reverses the position', sig['signal'].iloc[12] == 1 and sig['signal'].iloc[13] == -1)
ok('signals: a missing z-score holds the current position', sig['signal'].iloc[11] == 1 and sig['signal'].iloc[0] == 0)
real = pa.generate_signals(pair, window=30, entry_z=2.0, exit_z=0.5)
ok('signals on a real pair take only the values -1, 0 and 1 and do trade', set(real['signal'].unique()) <= {-1, 0, 1} and (real['signal'] != 0).any())

for name, call in (('spread', lambda: pa.spread(pair.assign(THIRD=1.0))), ('hedge ratio', lambda: pa.hedge_ratio(pair[['BIG']])), ('cointegration', lambda: pa.cointegration_test(pair.assign(THIRD=1.0)))):
    try: call(); raised = False
    except ValueError: raised = True
    ok(f'{name} insists on exactly two tickers', raised)
