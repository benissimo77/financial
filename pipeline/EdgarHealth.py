"""Data-health metrics for the EDGAR statement files. Observe only: never modifies any data file."""
import os
import numpy as np
import pandas as pd


class EdgarHealth:
    RECENT_YEARS = 5            # window of history that is scored
    CORE = ['revenue', 'net_income', 'cfo', 'assets', 'equity']   # a period is complete when all of these are present
    QUARTERLY_DUE_DAYS = 185    # next quarter end (91) plus the longest filing deadline (about 90)
    ANNUAL_DUE_DAYS = 490       # next year end (365) plus the four months a 20-F filer is allowed
    MIN_RECENT_QUARTERS = 3     # this many quarters in the last 18 months makes a ticker a quarterly filer
    SUM_TOLERANCE = 0.01        # four quarters must sum to the annual figure within this share
    JUMP_RATIO = 4              # revenue changing by more than this factor between periods is suspect
    WEIGHTS = dict(late_per_30d=5, late_cap=30, coverage_per_pct=1, coverage_cap=30, gap_per_period=5, gap_cap=20,
                   negative_revenue=10, revenue_jump=5)
    # Bands: green = use as is, amber = needs attention, red = exclude from analysis.
    # Restated years (quarters not summing to the annual figure) are reported but not scored: they are corporate history, not bad data.
    BANDS = (95, 80)            # score >= 95 green, >= 80 amber, otherwise red

    def __init__(self, data_dir='data/edgar', tickers_file='data/watchlist.csv', registry_file='data/registry/tickers.csv'):
        self.quarterly_file = os.path.join(data_dir, 'edgar-quarterly.csv')
        self.annual_file = os.path.join(data_dir, 'edgar-annual.csv')
        self.manifest_file = os.path.join(data_dir, 'edgar-manifest.csv')
        self.tickers_file = tickers_file
        self.registry_file = registry_file

    def _active_tickers(self):
        wl = pd.read_csv(self.tickers_file, comment='#', skip_blank_lines=True)
        return list(dict.fromkeys(wl.iloc[:, 0].dropna().astype(str).str.strip().str.upper()))

    def _load(self):
        q = pd.read_csv(self.quarterly_file, parse_dates=['period_end', 'filed'])
        a = pd.read_csv(self.annual_file, parse_dates=['period_end', 'filed'])
        return q, a

    def _basis(self, tq, ta, today):
        """Quarterly filers are scored on their quarters; everyone else on annual rows, so a foreign annual filer is not penalised."""
        if (tq['period_end'] >= today - pd.Timedelta(days=550)).sum() >= self.MIN_RECENT_QUARTERS:
            return 'quarterly', tq, 91.31, self.QUARTERLY_DUE_DAYS
        if not ta.empty:
            return 'annual', ta, 365.25, self.ANNUAL_DUE_DAYS
        return 'none', ta, None, None

    def _sum_mismatches(self, tq, ta):
        """Number of fiscal years where four quarters exist but do not sum to the annual revenue, net income or operating cash flow."""
        bad = 0
        for year in ta.itertuples():
            # 350 days back takes in this year's first quarter (about 275 days back) but not last year's fourth (364 or more).
            quarters = tq[(tq['period_end'] > year.period_end - pd.Timedelta(days=350)) & (tq['period_end'] <= year.period_end)]
            if len(quarters) != 4:
                continue
            for metric in ('revenue', 'net_income', 'cfo'):
                total, annual = quarters[metric].sum(min_count=4), getattr(year, metric)
                if pd.notna(total) and pd.notna(annual) and abs(total - annual) > self.SUM_TOLERANCE * abs(annual) + 1:
                    bad += 1
                    break
        return bad

    def metrics(self, as_of=None):
        """One row per watchlist ticker (index = ticker). as_of overrides today, for tests."""
        today = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.today().normalize()
        q, a = self._load()
        start = today - pd.DateOffset(years=self.RECENT_YEARS)
        man = pd.read_csv(self.manifest_file, dtype=str).set_index('ticker') if os.path.exists(self.manifest_file) else pd.DataFrame(columns=['outcome'])
        rows = []
        for ticker in self._active_tickers():
            tq = q[(q['ticker'] == ticker) & (q['period_end'] >= start)].sort_values('period_end')
            ta = a[(a['ticker'] == ticker) & (a['period_end'] >= start)].sort_values('period_end')
            basis, tb, step, due = self._basis(tq, ta, today)
            row = dict(ticker=ticker, basis=basis, last_outcome=man['outcome'].get(ticker))
            if basis != 'none':
                ends, revenue = tb['period_end'], tb['revenue']
                expected = round((ends.max() - ends.min()).days / step) + 1
                ratio = revenue / revenue.shift()
                material = (revenue.abs() > 1e6) & (revenue.shift().abs() > 1e6)
                row.update(
                    currency=tb['currency'].iloc[-1], periods=len(tb), expected_periods=expected,
                    first_period=ends.min(), last_period=ends.max(), last_filed=tb['filed'].max(),
                    days_late=max(0, (today - ends.max()).days - due),
                    core_pct=min(100.0, 100 * tb[self.CORE].notna().all(axis=1).sum() / expected),
                    missing_periods=int((ends.diff().dt.days / step).round().sub(1).clip(lower=0).sum()),
                    sum_mismatches=self._sum_mismatches(tq, ta) if basis == 'quarterly' else 0,
                    negative_revenue=int((revenue < 0).sum()),
                    revenue_jumps=int((((ratio > self.JUMP_RATIO) | (ratio < 1 / self.JUMP_RATIO)) & material).sum()),
                    debt_pct=100 * tb['debt_long'].notna().mean(), shares_pct=100 * tb['shares'].notna().mean())   # shown, not scored
            rows.append(row)
        out = pd.DataFrame(rows).set_index('ticker')
        if os.path.exists(self.registry_file):
            reg = pd.read_csv(self.registry_file, dtype=str).dropna(subset=['ticker'])
            reg['key'] = reg['ticker'].str.upper().str.replace('.', '-', regex=False)
            reg = reg.drop_duplicates('key').set_index('key')
            out['sector'], out['industry'] = reg['sector'], reg['zacks_x_ind_desc']
        return out

    def score(self, metrics=None, as_of=None):
        """Add score (0-100), band and issues text. Tickers the SEC holds no statements for (ETFs) get band 'n/a' and no score."""
        m = (self.metrics(as_of) if metrics is None else metrics).copy()
        w = self.WEIGHTS
        penalty = (np.minimum(m['days_late'] / 30 * w['late_per_30d'], w['late_cap'])
                   + np.minimum((100 - m['core_pct']) * w['coverage_per_pct'], w['coverage_cap'])
                   + np.minimum(m['missing_periods'] * w['gap_per_period'], w['gap_cap'])
                   + w['negative_revenue'] * (m['negative_revenue'] > 0) + w['revenue_jump'] * (m['revenue_jumps'] > 0))
        applicable = m['last_outcome'] != 'no_facts'
        m['score'] = (100 - penalty).clip(lower=0).where(m['basis'] != 'none', 0).where(applicable)
        m['band'] = np.select([~applicable, m['score'] >= self.BANDS[0], m['score'] >= self.BANDS[1]], ['n/a', 'green', 'amber'], 'red')
        m['issues'] = m.apply(self._issues, axis=1)
        return m

    def _findings(self, r):
        """List of (issue, suggested action) pairs for one scored row."""
        if r['band'] == 'n/a':
            return []
        if r['basis'] == 'none':
            return [(f"no statements (last fetch: {r['last_outcome']})", 'run Edgar().update() and EdgarStatements().build(); check the manifest outcome')]
        found = []
        if r['days_late'] > 0:
            found.append((f"latest period {r['last_period'].date()}, {int(r['days_late'])}d overdue",
                          'run the EDGAR update; if it stays overdue the company may have stopped filing or changed filer type'))
        if r['core_pct'] < 100:
            found.append((f"core metrics complete in {r['core_pct']:.0f}% of periods", 'a core metric is missing: look for an unmapped tag and add it to edgar-tags.csv'))
        if r['missing_periods'] > 0:
            found.append((f"{int(r['missing_periods'])} missing period(s)", 'gap in the filing history: often a fiscal-year change; accept it if old'))
        if r['sum_mismatches'] > 0:
            found.append((f"{int(r['sum_mismatches'])} restated year(s): quarters do not sum to the annual figure (not scored)",
                          'no action: the annual figure was restated and the quarters were not; prefer annual rows for those years'))
        if r['negative_revenue'] > 0:
            found.append((f"{int(r['negative_revenue'])} period(s) with negative revenue", 'a derived quarter is wrong: check which revenue tags were combined'))
        if r['revenue_jumps'] > 0:
            found.append((f"{int(r['revenue_jumps'])} revenue jump(s) over {self.JUMP_RATIO}x", 'check for an acquisition, or tags with different meanings being combined'))
        if isinstance(r['last_outcome'], str) and r['last_outcome'] != 'ok':
            found.append((f"last fetch: {r['last_outcome']}", 're-run the EDGAR update'))
        return found

    def _issues(self, r):
        return '; '.join(issue for issue, _ in self._findings(r))

    def problems(self, scored=None):
        """Amber and red tickers, worst first, with their issues and suggested actions."""
        s = self.score() if scored is None else scored
        bad = s[s['band'].isin(['amber', 'red'])].sort_values('score')
        cols = ['score', 'band', 'basis', 'sector', 'industry', 'issues', 'action']
        if bad.empty:
            return pd.DataFrame(columns=cols)
        action = bad.apply(lambda r: '; '.join(dict.fromkeys(a for _, a in self._findings(r))), axis=1)
        return bad.assign(action=action)[cols]

    def summary(self, scored=None):
        """Headline numbers for the report."""
        s = self.score() if scored is None else scored
        have = s[s['basis'] != 'none']
        return {
            'watchlist tickers': len(s), 'quarterly filers': int((s['basis'] == 'quarterly').sum()), 'annual only': int((s['basis'] == 'annual').sum()),
            'not applicable (no statements at the SEC)': int((s['band'] == 'n/a').sum()),
            'no statements but expected': int(((s['basis'] == 'none') & (s['band'] != 'n/a')).sum()),
            'green': int((s['band'] == 'green').sum()), 'amber': int((s['band'] == 'amber').sum()), 'red': int((s['band'] == 'red').sum()),
            'mean score': round(float(s['score'].mean()), 1), 'overdue': int((have['days_late'] > 0).sum()),
            'latest period range': f"{have['last_period'].min().date()} to {have['last_period'].max().date()}",
            'reporting currencies': have['currency'].value_counts().to_dict(),
            'tickers with a debt figure in most periods': int((have['debt_pct'] >= 50).sum()),
            'tickers with a restated year (reported, not scored)': int((have['sum_mismatches'] > 0).sum()),
            'last fetch outcomes': s['last_outcome'].value_counts().to_dict(),
        }

    def treemap(self, scored=None):
        """Sector > industry > ticker, coloured by band, drawn by the price report's treemap so the two look alike."""
        from pipeline.DataHealth import DataHealth
        s = (self.score() if scored is None else scored)
        s = s[s['band'] != 'n/a'].assign(last_date=lambda d: d['last_period'], rows=lambda d: d['periods'])
        fig = DataHealth().treemap(s)
        fig.update_layout(title=f"Fundamentals health: {len(s)} tickers, mean score {s['score'].mean():.1f}")
        return fig

    def heatmap(self, scored=None):
        """Ticker x calendar quarter: share of core metrics present. Blank where no period is expected (annual filers between year ends)."""
        import plotly.graph_objects as go
        s = (self.score() if scored is None else scored)
        s = s[s['basis'] != 'none'].copy()
        s['sector'], s['industry'] = s['sector'].fillna('Unclassified'), s['industry'].fillna('Unclassified')
        s = s.reset_index().sort_values(['basis', 'sector', 'industry', 'ticker'], ascending=[False, True, True, True]).set_index('ticker')
        q, a = self._load()
        shift = pd.Timedelta(days=10)   # a fiscal quarter ending in the first days of a month belongs to the calendar quarter just ended
        quarters = pd.period_range(s['first_period'].min() - shift, s['last_period'].max() - shift, freq='Q')
        grid = pd.DataFrame(np.nan, index=s.index, columns=quarters)
        for ticker, r in s.iterrows():
            tb = (q if r['basis'] == 'quarterly' else a)
            tb = tb[(tb['ticker'] == ticker) & (tb['period_end'] >= r['first_period'])]
            if r['basis'] == 'quarterly':
                grid.loc[ticker, pd.period_range(r['first_period'] - shift, r['last_period'] - shift, freq='Q')] = 0.0   # expected; stays 0 if missing
            filled = (100 * tb[self.CORE].notna().mean(axis=1)).groupby((tb['period_end'] - shift).dt.to_period('Q')).max()
            filled = filled[filled.index.isin(quarters)]
            grid.loc[ticker, filled.index] = filled.values
        label = (s['basis'].map({'quarterly': '', 'annual': ' (annual)'}) + ' ' + s['industry']).values
        fig = go.Figure(go.Heatmap(
            z=grid.values, x=[str(c) for c in quarters], y=list(s.index), zmin=0, zmax=100, colorscale='RdYlGn', hoverongaps=False,
            customdata=np.repeat(label[:, None], len(quarters), axis=1), showscale=False,
            hovertemplate='%{y}%{customdata}<br>%{x}: %{z:.0f}% of core metrics<extra></extra>'))
        fig.update_yaxes(autorange='reversed', type='category', tickfont=dict(size=9))
        fig.update_xaxes(tickangle=-90, tickfont=dict(size=9))
        fig.update_layout(title='Core-metric coverage by quarter (blank = not expected, or not reported yet)',
                          height=max(450, 15 * len(s) + 150), margin=dict(t=50, l=5, r=20, b=60))
        return fig
