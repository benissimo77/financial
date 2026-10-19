"""YF.verify_sample: the rolling comparison of stored history with Yahoo, and its repairs.
Runs against temporary files with mocked downloads; it never touches the real data. Run: python tests/test_yf_verify.py"""
import os, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import sys, os, io, contextlib, shutil, hashlib, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
import pandas as pd
from pipeline.YF import YF
real = hashlib.md5(open('data/prices/yf-prices.csv', 'rb').read()).hexdigest(); real_man = hashlib.md5(open('data/prices/yf-manifest.csv', 'rb').read()).hexdigest()
T = tempfile.mkdtemp() + '/tmpdata4'; shutil.rmtree(T, ignore_errors=True); os.makedirs(T)
P, M, W, R = T + '/prices.csv', T + '/manifest.csv', T + '/watch.csv', T + '/reg.csv'
cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
dates = [d.strftime('%Y-%m-%d') for d in pd.bdate_range('2026-06-01', (pd.Timestamp.today() - pd.Timedelta(days=7)))]
def rows(t, adj=10.0): return pd.DataFrame([[t, d, 10.0, 10.0, 10.0, 10.0, 100.0, adj] for d in dates], columns=cols)
names = [f'T{i:02d}' for i in range(12)]
stored = pd.concat([rows(t) for t in names]); stale = (stored.ticker == 'T03') & (stored.date < '2026-07-01')
stored.loc[stale, 'adj_close'] = 9.9          # T03: old rows carry a stale adjustment
stored = stored[~((stored.ticker == 'T05') & (stored.date == dates[10]))]   # T05: one missing day
stored.to_csv(P, index=False)
pd.DataFrame({'ticker': names}).to_csv(W, index=False)
pd.DataFrame({'ticker': names, 'status': 'HEALTHY', 'last_update': '2026-01-01'}).to_csv(R, index=False)
def fake(self, ticker, start_date="2000-01-01", end_date=None):
    if ticker == 'T01': self.last_outcome = 'transient_error'; return pd.DataFrame(), False
    self.last_outcome = 'ok'; r = rows(ticker); return r[r.date >= start_date], False
YF._download_prices = fake
mk = lambda: YF(tickers_file=W, data_file=P, registry_file=R, manifest_file=M, screen_file=T + '/s.csv')
ok = lambda name, cond: print(('PASS' if cond else 'FAIL'), name)
quiet = lambda: contextlib.redirect_stdout(io.StringIO())
man = lambda: pd.read_csv(M, dtype=str).set_index('ticker')

# Seed a manifest the way an earlier run would have: T00 failed last time.
pd.DataFrame({'ticker': names, 'last_date': dates[-1], 'last_attempt': '2026-10-01 09:00:00', 'outcome': ['unknown_error'] + ['ok'] * 11, 'rows_fetched': '1'}).to_csv(M, index=False)
with quiet(): r1 = mk().verify_sample()
res = dict(r1.values)
ok('sample is 8 tickers, skipping the one whose last scrape failed', len(r1) == 8 and 'T00' not in res and list(res) == names[1:9])
ok('stale adjustment repaired', res['T03'] == 'repaired' and (pd.read_csv(P).query("ticker == 'T03'")['adj_close'] == 10.0).all())
ok('missing day repaired', res['T05'] == 'repaired' and len(pd.read_csv(P).query("ticker == 'T05'")) == len(dates))
ok('download failure reported, not marked verified', res['T01'] == 'failed' and pd.isna(man().at['T01', 'last_verified']))
ok('matching tickers reported as match', all(res[t] == 'match' for t in ['T02', 'T04', 'T06', 'T07', 'T08']))
ok('verified tickers stamped', man()['last_verified'].notna().sum() == 7)
with quiet(): y = mk(); y.daily_update()
m = man()
ok('daily_update keeps last_verified and runs the sample', m.at['T02', 'last_verified'] == pd.Timestamp.today().strftime('%Y-%m-%d') and m['last_verified'].notna().sum() == 11 and pd.isna(m.at['T01', 'last_verified']))
with quiet(): r3 = mk().verify_sample()
ok('next sample starts with the unverified ticker, then cycles', r3['ticker'].iloc[0] == 'T01' and (r3['result'].iloc[1:] == 'match').all())
ok('no rows lost, no duplicates', pd.read_csv(P).ticker.value_counts().eq(len(dates)).all() and not pd.read_csv(P).duplicated(['ticker', 'date']).any())
ok('real data files untouched', real == hashlib.md5(open('data/prices/yf-prices.csv', 'rb').read()).hexdigest() and real_man == hashlib.md5(open('data/prices/yf-manifest.csv', 'rb').read()).hexdigest())
