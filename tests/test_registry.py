"""Registry: which SEC listings become registry entries, Fama-French sectors, cached company records, and refreshes.
Uses temporary files and a mocked SEC; it never touches the real data. Run: python tests/test_registry.py"""
import os, sys, tempfile, json, io, contextlib, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import pandas as pd
from pipeline.Registry import Registry, sector_for

T = tempfile.mkdtemp()
ok = lambda name, cond: print(('PASS' if cond else 'FAIL'), name)

ok('sectors: Fama-French 12 groups from SIC codes', sector_for(3674) == 'Business Equipment' and sector_for(7372) == 'Business Equipment'
   and sector_for(2834) == 'Healthcare' and sector_for(6022) == 'Finance' and sector_for(1311) == 'Energy' and sector_for(5961) == 'Shops'
   and sector_for(3711) == 'Consumer Durables' and sector_for(2080) == 'Consumer Non-Durables' and sector_for(4911) == 'Utilities'
   and sector_for(4813) == 'Telecom' and sector_for(2860) == 'Chemicals' and sector_for(3560) == 'Manufacturing')
ok('sectors: anything else is Other, including missing codes', sector_for(9999) == 'Other' and sector_for(None) == 'Other' and sector_for('x') == 'Other')

LISTING = [(1, 'Big Tech Inc', 'BIGT', 'Nasdaq'), (2, 'Bank Corp', 'BANK', 'NYSE'), (3, 'Two Class Co', 'TWO', 'NYSE'), (3, 'Two Class Co', 'TWO-B', 'NYSE'),
           (4, 'Pink Sheet Co', 'PINK', 'OTC'), (5, 'Some Fund', 'FUND', 'NYSE'), (6, 'Blank Check Corp', 'SPAC', 'Nasdaq'), (6, 'Blank Check Corp', 'SPAC-U', 'Nasdaq'),
           (6, 'Blank Check Corp', 'SPAC-WT', 'Nasdaq'), (2, 'Bank Corp', 'BANK-PA', 'NYSE'), (7, 'Cboe Listed', 'CBX', 'CBOE'), (8, 'Dotted Co', 'DOT.A', 'NYSE'),
           (9, 'Flaky Co', 'FLAKY', 'Nasdaq'), (11, 'Foreign Chips plc', 'FRGN', 'Nasdaq'), (12, 'Index Trust', 'ETFX', 'NYSE'),
           (13, 'Closed End Fund', 'CEF', 'NYSE'), (14, 'Rights Co', 'CVR-RI', 'Nasdaq'), (15, 'Warrant Co', 'WRN-WTA', 'NYSE'),
           (16, 'Alpha Corp', 'ALPH', 'Nasdaq'), (16, 'Alpha Corp', 'ALPHW', 'Nasdaq'), (16, 'Alpha Corp', 'ALPHU', 'Nasdaq'), (16, 'Alpha Corp', 'ALPHP', 'Nasdaq'),
           (17, 'Gogo Corp', 'GOGO', 'Nasdaq'), (17, 'Gogo Corp', 'GOGOL', 'Nasdaq'), (18, 'Lone Warrant Co', 'LONEW', 'Nasdaq'), (19, 'Zeta Inc', 'ZETAZ', 'Nasdaq')]
RECORDS = {1: (3674, 'Semiconductors & Related Devices', 'operating'), 2: (6022, 'State Commercial Banks', 'operating'),
           3: (2834, 'Pharmaceutical Preparations', 'operating'), 4: (1311, 'Crude Petroleum & Natural Gas', 'operating'),
           5: (6726, 'Unit Inv Trust', 'other'), 6: (6770, 'Blank Checks', 'operating'), 7: (7372, 'Prepackaged Software', 'operating'),
           8: (5961, 'Retail-Catalog & Mail-Order Houses', 'operating'), 9: (3711, 'Motor Vehicles', 'operating'),
           11: (3674, 'Semiconductors & Related Devices', 'other'), 12: (None, None, 'other'), 13: (6726, 'Unit Inv Trust, Closed-End Mgmt', 'other'),
           14: (2834, 'Pharmaceutical Preparations', 'operating'), 15: (1311, 'Crude Petroleum & Natural Gas', 'operating'),
           16: (3674, 'Semiconductors & Related Devices', 'operating'), 17: (7370, 'Services-Computer Programming', 'operating'),
           18: (2834, 'Pharmaceutical Preparations', 'operating'), 19: (6022, 'State Commercial Banks', 'operating')}
calls, down = [], {9}


class Resp:
    def __init__(self, status, payload): self.status_code, self.payload = status, payload
    def json(self): return self.payload
    def raise_for_status(self): pass


def fake_get(self, url):
    calls.append(url)
    if url == Registry.LIST_URL:
        return Resp(200, {'fields': ['cik', 'name', 'ticker', 'exchange'], 'data': [list(r) for r in LISTING]})
    cik = int(url.split('CIK')[1][:10])
    if cik in down:
        return Resp(503, None)
    sic, desc, kind = RECORDS[cik]
    return Resp(200, {'cik': str(cik), 'sic': '' if sic is None else str(sic), 'sicDescription': desc, 'entityType': kind})
Registry._get = fake_get
REG, COMP = os.path.join(T, 'tickers.csv'), os.path.join(T, 'sec-companies.csv')
quiet = lambda: contextlib.redirect_stdout(io.StringIO())
with quiet(): reg = Registry(REG, COMP).update()
tickers = set(reg['ticker'])
ok('kept: operating companies with common shares on NYSE or Nasdaq', {'BIGT', 'BANK', 'TWO', 'TWO-B', 'DOT-A'} <= tickers)
ok('left out: over the counter, CBOE, funds and blank-check companies', not {'PINK', 'CBX', 'FUND', 'SPAC'} & tickers)
ok('left out: warrants, units and preferred shares', not {'SPAC-U', 'SPAC-WT', 'BANK-PA'} & tickers)
ok('a company whose record could not be fetched is left out for now', 'FLAKY' not in tickers)
ok('a foreign company listed in the US is kept, marked as foreign', 'FRGN' in tickers and reg.set_index('ticker').at['FRGN', 'filer'] == 'foreign'
   and reg.set_index('ticker').at['BIGT', 'filer'] == 'domestic')
ok('funds are left out: no industry code, or a fund industry code', not {'ETFX', 'CEF'} & tickers)
ok('rights and lettered warrants are left out', not {'CVR-RI', 'WRN-WTA'} & tickers)
ok('Nasdaq fifth letters: warrants, units and preferreds left out', not {'ALPHW', 'ALPHU', 'ALPHP', 'LONEW'} & tickers and 'ALPH' in tickers)
ok('Nasdaq fifth letters: a share class like GOOGL, and a lone Z ticker, are kept', {'GOGO', 'GOGOL', 'ZETAZ'} <= tickers)
ok('tickers use the dash form for share classes', 'DOT-A' in tickers and 'DOT.A' not in tickers)
row = reg.set_index('ticker').loc['BIGT']
ok('each entry carries company ID, name, exchange, industry and sector', int(row['cik']) == 1 and row['name'] == 'Big Tech Inc' and row['exchange'] == 'Nasdaq'
   and int(row['sic']) == 3674 and row['industry'] == 'Semiconductors & Related Devices' and row['sector'] == 'Business Equipment')
ok('both share classes of a company are listed, sharing its company ID', set(reg.loc[reg['cik'] == 3, 'ticker']) == {'TWO', 'TWO-B'})
ok('the file is written with the expected columns', list(pd.read_csv(REG).columns) == Registry.COLUMNS)

calls.clear()
with quiet(): Registry(REG, COMP).update()
ok('a recent registry is not rebuilt', calls == [])
d = pd.read_csv(REG, dtype=str); d.loc[d['ticker'] == 'BANK', 'status'] = 'DELISTED'; d.to_csv(REG, index=False)
down.clear(); LISTING.append((10, 'New Listing Co', 'NEWL', 'NYSE')); RECORDS[10] = (4911, 'Electric Services', 'operating')
os.utime(REG, (time.time() - 40 * 86400, time.time() - 40 * 86400))       # older than LIST_DAYS
calls.clear()
with quiet(): reg2 = Registry(REG, COMP).update()
fetched = sorted(int(u.split('CIK')[1][:10]) for u in calls if 'CIK' in u)
ok('a refresh fetches records only for companies not yet cached (and those that failed before)', fetched == [9, 10])
ok('new listings arrive and a confirmed DELISTED status survives the refresh', 'NEWL' in set(reg2['ticker']) and 'FLAKY' in set(reg2['ticker'])
   and reg2.set_index('ticker').at['BANK', 'status'] == 'DELISTED' and reg2.set_index('ticker').at['NEWL', 'sector'] == 'Utilities')
LISTING[:] = [r for r in LISTING if r[2] != 'BIGT']
calls.clear()
with quiet(): reg3 = Registry(REG, COMP).update(force=True)
ok('a company that leaves the exchange list leaves the registry', 'BIGT' not in set(reg3['ticker']))
eligible = set(Registry(REG, COMP)._listed()['cik'])
ok('force re-fetches the record of every company with a common-share listing', len([u for u in calls if 'CIK' in u]) == len(eligible))
