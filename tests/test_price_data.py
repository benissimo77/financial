"""PriceData: loading, long and wide formats, filters, the in/out-of-sample split, and failing loudly on a bad file.
Runs against a small temporary file; it never touches the real data. Run: python tests/test_price_data.py"""
import os, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import pandas as pd
from PriceData import PriceData

T = tempfile.mkdtemp()
ok = lambda name, cond: print(('PASS' if cond else 'FAIL'), name)
dates = [d.strftime('%Y-%m-%d') for d in pd.bdate_range('2026-01-05', periods=10)]
rows = [(t, d, 10.0 + i, 11.0 + i, 9.0 + i, base + i, 1000.0, (base + i) * 0.9)
        for t, base in (('BBB', 200.0), ('AAA', 100.0)) for i, d in enumerate(dates)]   # deliberately unsorted; adj_close = 0.9 * close
good = os.path.join(T, 'prices.parquet')
pd.DataFrame(rows, columns=['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']).to_parquet(good, index=False)

pdata = PriceData(good, tickers_file=None)   # every ticker in the file
ok('tickers listed in order', pdata.get_tickers() == ['AAA', 'BBB'])
ok('dates for one ticker', pdata.get_dates('AAA') == dates)
wide = pdata.get_data_wide()
ok('wide format: dates down, tickers across, sorted', list(wide.columns) == ['AAA', 'BBB'] and isinstance(wide.index, pd.DatetimeIndex) and wide.index.is_monotonic_increasing and len(wide) == 10)
ok('adjusted close is the default price', wide['AAA'].iloc[0] == 90.0 and pdata.get_data_long()['adj_close'].iloc[0] == 90.0)
ok('raw close still available on request', pdata.get_data_wide(price_col='close')['AAA'].iloc[0] == 100.0)
ok('single ticker still returns a table, not a series', list(pdata.get_data_wide(tickers='BBB').columns) == ['BBB'])
ok('ticker list filter', list(pdata.get_data_wide(tickers=['BBB']).columns) == ['BBB'] and set(pdata.get_data_long(tickers=['AAA'])['ticker']) == {'AAA'})
sub = pdata.get_data(tickers='AAA', start=dates[2], end=dates[5])
ok('date filter is inclusive at both ends', len(sub) == 4 and sub['date'].min() == pd.Timestamp(dates[2]) and sub['date'].max() == pd.Timestamp(dates[5]))
ins, outs = pdata.split_inout_sample()
ok('70/30 split is chronological and on adjusted prices', len(ins) == 7 and len(outs) == 3 and ins.index.max() < outs.index.min() and ins['AAA'].iloc[0] == 90.0)

try: PriceData(os.path.join(T, 'missing.parquet'), tickers_file=None); raised = None
except Exception as e: raised = e
ok('missing file raises, naming the file', isinstance(raised, FileNotFoundError) and 'missing.parquet' in str(raised))
bad = os.path.join(T, 'bad.parquet'); pd.DataFrame({'ticker': ['AAA'], 'date': ['2026-01-05'], 'close': [1.0]}).to_parquet(bad, index=False)
try: PriceData(bad, tickers_file=None); raised = None
except Exception as e: raised = e
ok('file without the required columns raises, naming the file and the columns', isinstance(raised, ValueError) and 'bad.parquet' in str(raised) and 'adj_close' in str(raised))

watch = os.path.join(T, 'watch.csv')
with open(watch, 'w') as f:
    f.write('\n'.join(['ticker', '# a comment line', 'bbb', 'ZZZ']) + '\n')
scoped = PriceData(good, tickers_file=watch)
ok('watchlist scope: only watchlist tickers are served', scoped.get_tickers() == ['BBB'] and list(scoped.get_data_wide().columns) == ['BBB'])
ok('watchlist scope: a watchlist ticker with no prices is simply absent', 'ZZZ' not in scoped.get_tickers())
ok('watchlist scope: an off-watchlist ticker cannot be asked for by name', scoped.get_data_wide(tickers=['AAA', 'BBB']).columns.tolist() == ['BBB'])
try: PriceData(good, tickers_file=os.path.join(T, 'no-watchlist.csv')); raised = None
except Exception as e: raised = e
ok('missing watchlist file raises, naming the file', isinstance(raised, FileNotFoundError) and 'no-watchlist.csv' in str(raised))
