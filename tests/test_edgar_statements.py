"""EdgarStatements.build: quarters from year-to-date figures, restatements, tag priority, currencies.
Runs against temporary files with mocked downloads; it never touches the real data. Run: python tests/test_edgar_statements.py"""
import os, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import sys, os, io, json, contextlib, shutil, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
import pandas as pd
from pipeline.EdgarStatements import EdgarStatements
T = tempfile.mkdtemp() + '/tmpstmt'; shutil.rmtree(T, ignore_errors=True); os.makedirs(T + '/EDGAR/raw')
ok = lambda name, cond: print(('PASS' if cond else 'FAIL'), name)


def fact(val, end, filed, start=None, form='10-Q'):
    r = {'end': end, 'val': val, 'filed': filed, 'form': form}
    if start: r['start'] = start
    return r


def save(ticker, concepts):
    facts = {}
    for (tax, tag, unit), rows in concepts.items():
        facts.setdefault(tax, {}).setdefault(tag, {'units': {}})['units'][unit] = rows
    json.dump({'cik': 1, 'facts': facts}, open(f"{T}/EDGAR/raw/{ticker}.json", 'w'))


# AAA: calendar-year US filer, fiscal 2025.
Q = [('2025-01-01', '2025-03-31', '2025-05-01'), ('2025-04-01', '2025-06-30', '2025-08-01'), ('2025-07-01', '2025-09-30', '2025-11-01')]
save('AAA', {
    # Revenue: Q1 under the old tag, then the company switches tag. Q1 is restated a year later. Full year in the 10-K.
    ('us-gaap', 'Revenues', 'USD'): [fact(100, Q[0][1], Q[0][2], Q[0][0]), fact(104, Q[0][1], '2026-05-01', Q[0][0])],
    ('us-gaap', 'RevenueFromContractWithCustomerExcludingAssessedTax', 'USD'): [
        fact(110, Q[1][1], Q[1][2], Q[1][0]), fact(120, Q[2][1], Q[2][2], Q[2][0]), fact(470, '2025-12-31', '2026-02-15', '2025-01-01', '10-K'),
        fact(104, Q[0][1], '2026-05-01', Q[0][0])],
    # Cash flow: only year-to-date figures, as filed.
    ('us-gaap', 'NetCashProvidedByUsedInOperatingActivities', 'USD'): [
        fact(10, '2025-03-31', '2025-05-01', '2025-01-01'), fact(25, '2025-06-30', '2025-08-01', '2025-01-01'),
        fact(45, '2025-09-30', '2025-11-01', '2025-01-01'), fact(70, '2025-12-31', '2026-02-15', '2025-01-01', '10-K')],
    ('us-gaap', 'Assets', 'USD'): [fact(1000, '2025-03-31', '2025-05-01'), fact(1100, '2025-12-31', '2026-02-15')],
    ('dei', 'EntityCommonStockSharesOutstanding', 'shares'): [fact(50, '2025-04-25', '2025-05-01'), fact(52, '2026-02-10', '2026-02-15')],
    ('us-gaap', 'CommonStockSharesOutstanding', 'shares'): [fact(51, '2025-06-30', '2025-08-01'), fact(49, '2025-03-31', '2025-05-01')],
})
# WEEK: 52/53-week filer with a 14-week (98-day) quarter.
save('WEEK', {('us-gaap', 'Revenues', 'USD'): [fact(200, '2025-12-27', '2026-02-01', '2025-09-28'), fact(230, '2026-04-04', '2026-05-10', '2025-12-28')]})
# FRGN: annual-only foreign filer reporting in TWD, with a USD convenience translation of the latest year.
save('FRGN', {
    ('ifrs-full', 'Revenue', 'TWD'): [fact(3000, '2023-12-31', '2024-04-01', '2023-01-01', '20-F'), fact(3300, '2024-12-31', '2025-04-01', '2024-01-01', '20-F'),
                                      fact(3600, '2025-12-31', '2026-04-01', '2025-01-01', '20-F')],
    ('ifrs-full', 'Revenue', 'USD'): [fact(110, '2025-12-31', '2026-04-01', '2025-01-01', '20-F')],
    ('ifrs-full', 'Equity', 'TWD'): [fact(9000, '2025-12-31', '2026-04-01', form='20-F')],
    ('ifrs-full', 'Equity', 'USD'): [fact(275, '2025-12-31', '2026-04-01', form='20-F')],
})
open(f"{T}/EDGAR/raw/BROKEN.json", 'w').write('not json')

es = EdgarStatements(data_dir=T + '/EDGAR')
with contextlib.redirect_stdout(io.StringIO()) as log:
    q, a = es.build()
q['period_end'] = pd.to_datetime(q['period_end']); a['period_end'] = pd.to_datetime(a['period_end'])
A = q[q.ticker == 'AAA'].set_index(q[q.ticker == 'AAA']['period_end'].dt.strftime('%Y-%m-%d'))
ok('four quarters built for AAA', list(A.index) == ['2025-03-31', '2025-06-30', '2025-09-30', '2025-12-31'])
ok('fourth quarter = year minus three quarters (470 - 104 - 110 - 120)', A.at['2025-12-31', 'revenue'] == 136)
ok('restated value wins, tags merged by priority', A['revenue'].tolist() == [104, 110, 120, 136])
ok('filed date is the first filing of the period, not the restatement', str(A.at['2025-03-31', 'filed'])[:10] == '2025-05-01' and str(A.at['2025-12-31', 'filed'])[:10] == '2026-02-15')
ok('cash flow quarters from year-to-date differences', A['cfo'].tolist() == [10, 15, 20, 25])
ok('balance aligned to its exact period end only', A['assets'].fillna(-1).tolist() == [1000, -1, -1, 1100])
ok('cover-page shares aligned to the preceding period end, balance-sheet tag as fallback', A['shares'].fillna(-1).tolist() == [50, 51, -1, 52])
W = q[q.ticker == 'WEEK']
ok('14-week quarter accepted', W['revenue'].tolist() == [200, 230])
F = a[a.ticker == 'FRGN']
ok('annual-only filer has no quarterly rows and three annual rows', q[q.ticker == 'FRGN'].empty and F['revenue'].tolist() == [3000, 3300, 3600])
ok('reporting currency chosen over the convenience translation', set(F['currency']) == {'TWD'} and F['equity'].iloc[-1] == 9000)
ok('annual table for AAA has the full-year figures', a[a.ticker == 'AAA'][['revenue', 'cfo', 'assets', 'shares']].values.tolist() == [[470, 70, 1100, 52]])
ok('unreadable raw file skipped and reported', 'BROKEN' not in set(q.ticker) | set(a.ticker) and 'SKIPPED' in log.getvalue() and 'BROKEN' in log.getvalue())
ok('both files written, no temp files', sorted(f for f in os.listdir(T + '/EDGAR') if f != 'raw') == ['edgar-annual.csv', 'edgar-quarterly.csv'])
rt = pd.read_csv(T + '/EDGAR/edgar-quarterly.csv')
ok('file round-trips with expected columns', list(rt.columns[:4]) == ['ticker', 'period_end', 'filed', 'currency'] and rt['period_end'].iloc[0] == '2025-03-31' and len(rt) == len(q))
os.remove(T + '/EDGAR/edgar-quarterly.csv')
with contextlib.redirect_stdout(io.StringIO()): es.build(tickers=['AAA'])
ok('building selected tickers writes nothing', not os.path.exists(T + '/EDGAR/edgar-quarterly.csv'))
bad = T + '/badtags.csv'; open(bad, 'w').write('metric,kind,priority,tag\nrevenue,yearly,1,Revenues\n')
try: EdgarStatements(data_dir=T + '/EDGAR', tags_file=bad).build(); raised = False
except ValueError: raised = True
ok('invalid kind in the tag file is rejected', raised)

# --- Choosing between competing values (restatements, abandoned tags, proxy statements)
CONTRACT = ('us-gaap', 'RevenueFromContractWithCustomerExcludingAssessedTax', 'USD')
# PRX: a proxy statement filed after the annual report quotes a different net income.
save('PRX', {('us-gaap', 'NetIncomeLoss', 'USD'): [fact(70, '2025-12-31', '2026-02-15', '2025-01-01', '10-K'), fact(65, '2025-12-31', '2026-04-20', '2025-01-01', 'DEF 14A')]})
# RST: fiscal 2022 restated two years later (a business was sold). The nine-month figure was never restated,
# and the old 'Revenues' tag was abandoned, so it still carries the original annual figure.
save('RST', {
    ('us-gaap', 'Revenues', 'USD'): [fact(500, '2022-12-31', '2023-02-15', '2022-01-01', '10-K')],
    CONTRACT: [fact(120, '2022-03-31', '2022-05-01', '2022-01-01'), fact(245, '2022-06-30', '2022-08-01', '2022-01-01'),
               fact(370, '2022-09-30', '2022-11-01', '2022-01-01'), fact(370, '2022-09-30', '2023-11-01', '2022-01-01'),
               fact(500, '2022-12-31', '2023-02-15', '2022-01-01', '10-K'), fact(500, '2022-12-31', '2024-02-15', '2022-01-01', '10-K'),
               fact(450, '2022-12-31', '2025-02-15', '2022-01-01', '10-K'), fact(520, '2023-12-31', '2024-02-15', '2023-01-01', '10-K')]})
# RS2: the same sale, but here the nine-month figure was restated too, in the next year's third-quarter report.
save('RS2', {CONTRACT: [fact(120, '2022-03-31', '2022-05-01', '2022-01-01'), fact(245, '2022-06-30', '2022-08-01', '2022-01-01'),
                        fact(370, '2022-09-30', '2022-11-01', '2022-01-01'), fact(340, '2022-09-30', '2023-11-01', '2022-01-01'),
                        fact(500, '2022-12-31', '2023-02-15', '2022-01-01', '10-K'), fact(450, '2022-12-31', '2024-02-15', '2022-01-01', '10-K')]})
q2, a2 = es.build(tickers=['PRX', 'RST', 'RS2'])
rev = lambda t: q2[q2.ticker == t]['revenue'].tolist()
ok('proxy statement does not override the annual report', a2[a2.ticker == 'PRX']['net_income'].tolist() == [70])
ok('restated figure beats the original left under an abandoned tag', a2[a2.ticker == 'RST']['revenue'].tolist() == [450, 520])
ok('fourth quarter derived on one basis: original year minus original nine months', rev('RST') == [120, 125, 125, 130])
ok('each derived quarter on one basis: Q3 original minus original, Q4 restated minus restated', rev('RS2') == [120, 125, 125, 110])
