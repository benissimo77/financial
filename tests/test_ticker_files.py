"""TickerFiles: safe file names for tickers that Windows reserves (CON, NUL, COM1...), and a full round trip through
the SEC fetcher and statement builder. Uses temporary files only. Run: python tests/test_ticker_files.py"""
import os, sys, tempfile, json, io, contextlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import pandas as pd
from pipeline.TickerFiles import path_for, ticker_from, tickers_in, RESERVED
from pipeline.Edgar import Edgar
from pipeline.EdgarStatements import EdgarStatements

T = tempfile.mkdtemp()
ok = lambda name, cond: print(('PASS' if cond else 'FAIL'), name)

ok('an ordinary ticker is its own file name', os.path.basename(path_for('x', 'AAPL', '.json')) == 'AAPL.json' and os.path.basename(path_for('x', 'BRK-B', '.parquet')) == 'BRK-B.parquet')
ok('a reserved name gets a leading underscore', os.path.basename(path_for('x', 'CON', '.json')) == '_CON.json' and os.path.basename(path_for('x', 'COM1', '.parquet')) == '_COM1.parquet')
ok('every reserved name is covered', len(RESERVED) == 22 and {'CON', 'PRN', 'AUX', 'NUL', 'COM9', 'LPT1'} <= RESERVED)
ok('reading a file name gives back the plain ticker', ticker_from('data/x/_CON.json') == 'CON' and ticker_from('AAPL.parquet') == 'AAPL')
ok('an underscore that is not protecting a reserved name is left alone', ticker_from('_ABC.json') == '_ABC')
ok('every reserved name survives the round trip', all(ticker_from(path_for(T, t, '.json')) == t for t in RESERVED))
for bad in ('A/B', 'X:Y', ''):
    try: path_for(T, bad, '.json'); raised = False
    except ValueError: raised = True
    ok(f'a ticker that cannot be a file name raises: {bad!r}', raised)

# Current Windows 11 accepts reserved names that carry an extension (CON.json); older Windows and some tools do not.
# The safe name works everywhere, which is what matters.
open(path_for(T, 'CON', '.json'), 'w').write('{}')
ok('the safe name can be created and listed', os.path.exists(os.path.join(T, '_CON.json')) and 'CON' in tickers_in(T, '.json'))

# Round trip: the fetcher stores CON, the manifest says CON, the builder reads it back as CON.
D = os.path.join(T, 'edgar'); W = os.path.join(T, 'watch.csv')
pd.DataFrame({'ticker': ['CON', 'AAA']}).to_csv(W, index=False)
facts = {'cik': 1, 'facts': {'us-gaap': {'Revenues': {'units': {'USD': [
    {'start': '2025-01-01', 'end': '2025-03-31', 'val': 100, 'filed': '2025-05-01', 'form': '10-Q'},
    {'start': '2025-04-01', 'end': '2025-06-30', 'val': 110, 'filed': '2025-08-01', 'form': '10-Q'}]}}}}}


class Resp:
    def __init__(self, status, content): self.status_code, self.content = status, content
    def json(self): return json.loads(self.content)
    def raise_for_status(self): pass


def fake_get(self, url):
    if url == Edgar.TICKERS_URL:
        return Resp(200, json.dumps({'0': {'cik_str': 1, 'ticker': 'CON', 'title': 'Con Co'}, '1': {'cik_str': 2, 'ticker': 'AAA', 'title': 'A Co'}}).encode())
    return Resp(200, json.dumps(facts).encode())
Edgar._get = fake_get
with contextlib.redirect_stdout(io.StringIO()):
    Edgar(tickers_file=W, data_dir=D).update()
raw = sorted(os.listdir(os.path.join(D, 'raw')))
manifest = pd.read_csv(os.path.join(D, 'edgar-manifest.csv'))
ok('the fetcher writes _CON.json, and the manifest records CON', raw == ['AAA.json', '_CON.json'] and set(manifest['ticker']) == {'AAA', 'CON'})
ok('listing the folder gives plain tickers', tickers_in(os.path.join(D, 'raw'), '.json') == ['AAA', 'CON'])
tags = os.path.join(T, 'tags.csv'); open(tags, 'w').write('metric,kind,priority,tag\nrevenue,duration,1,Revenues\n')
with contextlib.redirect_stdout(io.StringIO()):
    q, a = EdgarStatements(data_dir=D, tags_file=tags).build()
ok('the statement files list CON, with its figures', sorted(q['ticker'].unique()) == ['AAA', 'CON'] and q[q.ticker == 'CON']['revenue'].tolist() == [100, 110])
ok('a single reserved ticker can be built on request', EdgarStatements(data_dir=D, tags_file=tags).build(tickers=['CON'])[0]['ticker'].unique().tolist() == ['CON'])
