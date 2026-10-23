"""Edgar.update: outcome classification, skipping, and never replacing a good raw file with a bad download.
Runs against temporary files with mocked downloads; it never touches the real data. Run: python tests/test_edgar_fetch.py"""
import os, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import sys, os, io, json, contextlib, shutil, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
import pandas as pd, requests
from pipeline.Edgar import Edgar
T = tempfile.mkdtemp() + '/tmpedgar'; shutil.rmtree(T, ignore_errors=True); os.makedirs(T)
W = T + '/watch.csv'; pd.DataFrame({'ticker': ['GOOD', 'ETF', 'DOWN', 'JUNK', 'NOCIK']}).to_csv(W, index=False)
R = T + '/registry.csv'; pd.DataFrame({'ticker': ['GOOD', 'ETF', 'DOWN', 'JUNK'], 'cik': [1, 2, 3, 4]}).to_csv(R, index=False)   # NOCIK is not in the registry


class Resp:
    def __init__(self, status, content=b''): self.status_code, self.content = status, content
    def json(self): return json.loads(self.content)
    def raise_for_status(self):
        if self.status_code != 200: raise requests.HTTPError(self.status_code)


def facts(n, filed='2026-08-01'):
    rows = [{'end': '2026-06-30', 'val': i, 'filed': filed, 'form': '10-Q'} for i in range(n)]
    return json.dumps({'cik': 1, 'facts': {'us-gaap': {'Assets': {'units': {'USD': rows}}}}}).encode()


calls = []
mode = {'GOOD': Resp(200, facts(200))}
def fake_get(self, url):
    calls.append(url)
    cik = int(url.split('CIK')[1][:10])
    if cik == 1: return mode['GOOD']
    if cik == 2: return Resp(404)
    if cik == 3: raise requests.ConnectionError('network down')
    return Resp(200, b'<html>not json</html>')
Edgar._get = fake_get
mk = lambda: Edgar(tickers_file=W, data_dir=T + '/EDGAR', registry_file=R)
ok = lambda name, cond: print(('PASS' if cond else 'FAIL'), name)
quiet = lambda: contextlib.redirect_stdout(io.StringIO())
man = lambda: pd.read_csv(T + '/EDGAR/edgar-manifest.csv', dtype=str).set_index('ticker')

with quiet(): r = mk().update()
m = man()
ok('outcomes classified', m['outcome'].to_dict() == {'GOOD': 'ok', 'ETF': 'no_facts', 'DOWN': 'transient_error', 'JUNK': 'invalid_json', 'NOCIK': 'no_cik'})
ok('only the sound download is stored', os.listdir(T + '/EDGAR/raw') == ['GOOD.json'])
ok('manifest records last filing date and size', m.at['GOOD', 'last_filed'] == '2026-08-01' and float(m.at['GOOD', 'size_kb']) > 0)
ok('no temp files left behind', not [f for d, _, fs in os.walk(T) for f in fs if f.endswith('.tmp')])

calls.clear()
with quiet(): r2 = mk().update()
ok('recent ok and no_facts are skipped, failures are retried', sorted(r2['ticker']) == ['DOWN', 'JUNK', 'NOCIK'] and not any('CIK0000000001' in c for c in calls))
ok('company IDs come from the registry: only company records are requested', all('CIK' in c for c in calls))
ok('skipped ticker keeps its manifest row', man().at['GOOD', 'outcome'] == 'ok' and len(man()) == 5)

before = open(T + '/EDGAR/raw/GOOD.json', 'rb').read()
mode['GOOD'] = Resp(200, facts(20))
with quiet(): mk().update(tickers=['GOOD'], force=True)
ok('much smaller download refused, stored file kept', man().at['GOOD', 'outcome'] == 'shrunk' and open(T + '/EDGAR/raw/GOOD.json', 'rb').read() == before)
ok('failed attempt keeps the known last filing date', man().at['GOOD', 'last_filed'] == '2026-08-01')
mode['GOOD'] = Resp(503)
with quiet(): mk().update(tickers=['GOOD'], force=True)
ok('server error leaves stored file untouched', man().at['GOOD', 'outcome'] == 'transient_error' and open(T + '/EDGAR/raw/GOOD.json', 'rb').read() == before)
mode['GOOD'] = Resp(200, facts(260, filed='2026-11-01'))
with quiet(): mk().update(tickers=['GOOD'])
m = man()
ok('previous failure is retried without force, and new data replaces the file', m.at['GOOD', 'outcome'] == 'ok' and m.at['GOOD', 'last_filed'] == '2026-11-01' and open(T + '/EDGAR/raw/GOOD.json', 'rb').read() != before)
ok('manifest still has one row per ticker', len(m) == 5 and m.index.is_unique)

# An identical download leaves the stored file untouched, so the statement builder can skip it; a changed one replaces it.
good = T + '/EDGAR/raw/GOOD.json'
os.utime(good, (1_000_000_000, 1_000_000_000))                 # pretend it was written long ago
with quiet(): mk().update(tickers=['GOOD'], force=True)        # same response as before
ok('an identical download does not rewrite the file', os.path.getmtime(good) == 1_000_000_000 and man().at['GOOD', 'outcome'] == 'ok')
mode['GOOD'] = Resp(200, facts(270, filed='2026-12-01'))
with quiet(): mk().update(tickers=['GOOD'], force=True)
ok('a changed download does', os.path.getmtime(good) > 1_000_000_000 and man().at['GOOD', 'last_filed'] == '2026-12-01')
