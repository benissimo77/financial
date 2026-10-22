"""YF.daily_update: two-step delisting, the write guard, and force refresh keeping stored rows.
Runs against temporary files with mocked downloads; it never touches the real data. Run: python tests/test_yf_update.py"""
import os, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import sys, os, io, contextlib, shutil, hashlib, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
import pandas as pd
from pipeline.YF import YF
real = hashlib.md5(open('data/prices/yf-prices.parquet', 'rb').read()).hexdigest(); real_reg = hashlib.md5(open('data/registry/tickers.csv', 'rb').read()).hexdigest()
T = tempfile.mkdtemp() + '/tmpdata'; shutil.rmtree(T, ignore_errors=True); os.makedirs(T)
P, M, W, R = T + '/prices.parquet', T + '/manifest.csv', T + '/watch.csv', T + '/reg.csv'
cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
def rows(t, dates, px=10.0): return pd.DataFrame([[t, d, px, px, px, px, 100.0, px] for d in dates], columns=cols)
old_dates = [d.strftime('%Y-%m-%d') for d in pd.bdate_range('2026-08-03', '2026-08-28')]
pd.concat([rows('AAA', old_dates), rows('DEAD', old_dates), rows('ORPH', old_dates)]).to_parquet(P, index=False)
pd.DataFrame({'ticker': ['AAA', 'DEAD']}).to_csv(W, index=False)
pd.DataFrame({'ticker': ['AAA', 'DEAD', 'ORPH'], 'status': ['HEALTHY'] * 3, 'last_update': ['2026-01-01'] * 3}).to_csv(R, index=False)
calls = []
def fake(self, ticker, start_date="2000-01-01", end_date=None):
    calls.append((ticker, start_date))
    if ticker == 'DEAD': self.last_outcome = 'delisted'; return pd.DataFrame(), True
    self.last_outcome = 'ok'
    return rows(ticker, [d.strftime('%Y-%m-%d') for d in pd.bdate_range(start_date, '2026-09-30')], 11.0), False
YF._download_prices = fake
mk = lambda: YF(tickers_file=W, data_file=P, registry_file=R, manifest_file=M, screen_file=T + '/screen.csv')
quiet = contextlib.redirect_stdout(io.StringIO())
status = lambda: pd.read_csv(R).set_index('ticker')['status'].to_dict()
ok = lambda name, cond: print(('PASS' if cond else 'FAIL'), name)

with quiet: mk().daily_update()
ok('A2 first delisted result does not flag', status()['DEAD'] == 'HEALTHY')
ok('A2 manifest records delisted', pd.read_csv(M).set_index('ticker').at['DEAD', 'outcome'] == 'delisted')
with quiet: mk().daily_update()
ok('A2 second result on the same day does not flag', status()['DEAD'] == 'HEALTHY')
m = pd.read_csv(M, dtype=str); m.loc[m.ticker == 'DEAD', 'last_attempt'] = '2026-10-01 09:00:00'; m.to_csv(M, index=False)
with quiet: y = mk(); y.daily_update()
ok('A2 second result on a later day flags DELISTED', status()['DEAD'] == 'DELISTED' and 'DEAD' not in y.tickers)
df = pd.read_parquet(P); n = df.ticker.value_counts()
ok('update added rows for AAA, kept DEAD and ORPH history', n['AAA'] > 20 and n['DEAD'] == 20 and n['ORPH'] == 20)

h = hashlib.md5(open(P, 'rb').read()).hexdigest()
try: mk()._write_prices(df[df.ticker != 'ORPH']); raised = False
except RuntimeError as e: raised = True; msg = str(e)
ok('A5 write refused when a ticker would vanish', raised and hashlib.md5(open(P, 'rb').read()).hexdigest() == h)
try: mk()._write_prices(df.iloc[1:]); raised = False
except RuntimeError: raised = True
ok('A5 write refused when one row would be lost', raised and hashlib.md5(open(P, 'rb').read()).hexdigest() == h)
print('     message:', msg)

calls.clear()
with quiet: mk().daily_update(force_refresh_all=True)
df2 = pd.read_parquet(P); n2 = df2.ticker.value_counts()
ok('A5 force refresh re-downloads from 2000', ('AAA', '2000-01-01') in calls)
ok('A5 force refresh keeps tickers outside the watchlist', n2.get('ORPH', 0) == 20 and n2.get('DEAD', 0) == 20)
ok('A5 force refresh overwrote AAA with fresh values', (df2[df2.ticker == 'AAA']['close'] == 11.0).all() and not df2.duplicated(['ticker', 'date']).any())
ok('real data files untouched', real == hashlib.md5(open('data/prices/yf-prices.parquet', 'rb').read()).hexdigest() and real_reg == hashlib.md5(open('data/registry/tickers.csv', 'rb').read()).hexdigest())
