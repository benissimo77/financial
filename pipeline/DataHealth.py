"""Data-health metrics for the YF price database. Observe only: never modifies any data file."""
import os
import numpy as np
import pandas as pd


class DataHealth:
    LAG_DAYS = 7           # matches YF.LAG_DAYS: the newest data we expect is today minus this
    RECENT_ROWS = 252      # window for zero-volume and flat-close shares
    WEIGHTS = dict(freshness_per_day=2, freshness_cap=40, coverage_per_pct=2, coverage_cap=30,
                   gap_free_days=3, gap_per_day=2, gap_cap=20,
                   bad_ohlc=10, adj_inconsistency=10, zero_volume=5, flat_close=5)
    ZERO_VOLUME_LIMIT = 10  # percent of the recent window
    FLAT_CLOSE_LIMIT = 5    # percent of the recent window
    BANDS = (95, 80)        # score >= 95 green, >= 80 amber, otherwise red

    def __init__(self, data_file='data/prices/yf-prices.csv', manifest_file='data/prices/yf-manifest.csv',
                 tickers_file='data/watchlist.csv', registry_file='data/registry/tickers.csv', calendar_ticker='SPY'):
        self.data_file = data_file
        self.manifest_file = manifest_file
        self.tickers_file = tickers_file
        self.registry_file = registry_file
        self.calendar_ticker = calendar_ticker

    def _active_tickers(self):
        wl = pd.read_csv(self.tickers_file, comment='#', skip_blank_lines=True)
        return list(dict.fromkeys(wl.iloc[:, 0].dropna().astype(str).str.strip().str.upper()))

    @staticmethod
    def _trading_calendar(px, calendar_ticker):
        """SPY's dates are the trading calendar; fall back to dates where at least half the usual tickers have a row."""
        spy = px.loc[px['ticker'] == calendar_ticker, 'date']
        if len(spy) > 250:
            return pd.DatetimeIndex(spy.sort_values().unique())
        cnt = px.groupby('date')['ticker'].nunique()
        return pd.DatetimeIndex(cnt[cnt >= 0.5 * cnt.rolling(250, min_periods=20).median()].index)

    def metrics(self, as_of=None):
        """One row per watchlist ticker (index = ticker). as_of overrides today, for tests."""
        today = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.today().normalize()
        active = self._active_tickers()
        px = pd.read_csv(self.data_file, low_memory=False, parse_dates=['date'])
        cal = self._trading_calendar(px, self.calendar_ticker)
        cutoff = today - pd.Timedelta(days=self.LAG_DAYS)
        # Weekdays after the calendar's last date also count as expected, so a database-wide stall still shows as behind.
        beyond = max(0, int(np.busday_count((cal.max() + pd.Timedelta(days=1)).date(), (cutoff + pd.Timedelta(days=1)).date())))
        exp_idx = cal.searchsorted(cutoff, side='right') + beyond

        px = px[px['ticker'].isin(active)].sort_values(['ticker', 'date']).reset_index(drop=True)
        g = px.groupby('ticker')
        prev_close, prev_adj = g['close'].shift(), g['adj_close'].shift()
        r_raw, r_adj = px['close'] / prev_close - 1, px['adj_close'] / prev_adj - 1
        px['idx'] = cal.searchsorted(px['date'])
        px['in_cal'] = px['date'].isin(cal)
        px['bad_ohlc'] = (px['high'] < px['low']) | (px['close'] > px['high'] * 1.005) | (px['close'] < px['low'] * 0.995) \
            | (px[['open', 'high', 'low', 'close', 'adj_close']] <= 0).any(axis=1)
        px['adj_bad'] = (r_raw.abs() > 0.3) & ((r_raw - r_adj).abs() > 0.05)
        px['big_move'] = r_adj.abs() > 0.5
        px['gap'] = px.groupby('ticker')['idx'].diff() - 1
        recent = px.groupby('ticker').tail(self.RECENT_ROWS)
        flat = (recent['close'].diff() == 0) & (recent['volume'] > 0) & (recent['ticker'] == recent['ticker'].shift())

        out = g['date'].agg(first_date='min', last_date='max', rows='size')
        out['expected_rows'] = cal.searchsorted(out['last_date'], side='right') - cal.searchsorted(out['first_date'], side='left')
        out['present_rows'] = px.groupby('ticker')['in_cal'].sum()
        out['coverage_pct'] = (100 * out['present_rows'] / out['expected_rows']).clip(upper=100)
        out['days_behind'] = np.maximum(exp_idx - cal.searchsorted(out['last_date'], side='right'), 0)
        out['longest_gap'] = px.groupby('ticker')['gap'].max().fillna(0)
        out['bad_ohlc_rows'] = px.groupby('ticker')['bad_ohlc'].sum()
        out['adj_inconsistency'] = px.groupby('ticker')['adj_bad'].sum()
        out['big_moves'] = px.groupby('ticker')['big_move'].sum()
        out['zero_volume_pct'] = 100 * recent.assign(z=recent['volume'] == 0).groupby('ticker')['z'].mean()
        out['flat_close_pct'] = 100 * flat.groupby(recent['ticker']).mean()

        out = out.reindex(active)  # watchlist tickers missing from the database stay as all-NaN rows
        out['in_database'] = out['rows'].notna()
        if os.path.exists(self.manifest_file):
            man = pd.read_csv(self.manifest_file, dtype=str).set_index('ticker')
            out['last_outcome'], out['last_attempt'] = man['outcome'], man['last_attempt']
        if os.path.exists(self.registry_file):
            reg = pd.read_csv(self.registry_file, dtype=str).dropna(subset=['ticker'])
            reg['key'] = reg['ticker'].str.upper().str.replace('.', '-', regex=False)
            reg = reg.drop_duplicates('key').set_index('key')
            out['sector'], out['industry'] = reg['sector'], reg['zacks_x_ind_desc']
        out.index.name = 'ticker'
        return out

    def score(self, metrics=None, as_of=None):
        """Add score (0-100), band and issues text to the metrics table."""
        m = (self.metrics(as_of) if metrics is None else metrics).copy()
        w = self.WEIGHTS
        penalty = (np.minimum(m['days_behind'] * w['freshness_per_day'], w['freshness_cap'])
                   + np.minimum((100 - m['coverage_pct']) * w['coverage_per_pct'], w['coverage_cap'])
                   + np.minimum(np.maximum(m['longest_gap'] - w['gap_free_days'], 0) * w['gap_per_day'], w['gap_cap'])
                   + w['bad_ohlc'] * (m['bad_ohlc_rows'] > 0) + w['adj_inconsistency'] * (m['adj_inconsistency'] > 0)
                   + w['zero_volume'] * (m['zero_volume_pct'] > self.ZERO_VOLUME_LIMIT)
                   + w['flat_close'] * (m['flat_close_pct'] > self.FLAT_CLOSE_LIMIT))
        m['score'] = (100 - penalty).clip(lower=0).where(m['in_database'], 0)
        m['band'] = np.select([m['score'] >= self.BANDS[0], m['score'] >= self.BANDS[1]], ['green', 'amber'], 'red')
        m['issues'] = m.apply(self._issues, axis=1)
        return m

    def _findings(self, r):
        """List of (issue, suggested action) pairs for one scored row."""
        if not r['in_database']:
            return [('not in database', 'run the daily update; if it stays missing the ticker may be dead on Yahoo')]
        found = []
        if r['days_behind'] > 0:
            found.append((f"behind {int(r['days_behind'])}d", 'run the daily update'))
        if r['coverage_pct'] < 100:
            found.append((f"coverage {r['coverage_pct']:.1f}% (longest gap {int(r['longest_gap'])}d)",
                          'gap in history: refresh the ticker from scratch, or accept it if the gap is old'))
        if r['bad_ohlc_rows'] > 0:
            found.append((f"{int(r['bad_ohlc_rows'])} bad OHLC rows", 'inspect the rows (likely bad prints) and re-fetch the ticker'))
        if r['adj_inconsistency'] > 0:
            found.append((f"{int(r['adj_inconsistency'])} raw/adjusted mismatches",
                          'adjustment problem around a corporate action: re-fetch history and compare with Yahoo'))
        if r['zero_volume_pct'] > self.ZERO_VOLUME_LIMIT:
            found.append((f"zero volume {r['zero_volume_pct']:.0f}%", 'illiquid or dead: consider dropping it from the rules'))
        if r['flat_close_pct'] > self.FLAT_CLOSE_LIMIT:
            found.append((f"flat close {r['flat_close_pct']:.0f}%", 'stale prices: consider dropping it from the rules'))
        outcome = r.get('last_outcome')
        if isinstance(outcome, str) and outcome not in ('ok', 'no_new_data'):
            found.append((f"last scrape: {outcome}",
                          're-run the daily update' if outcome == 'transient_error' else 'check the scraper output for this ticker'))
        return found

    def _issues(self, r):
        return '; '.join(issue for issue, _ in self._findings(r))

    def problems(self, scored=None):
        """Amber and red tickers, worst first, with their issues and suggested actions."""
        s = self.score() if scored is None else scored
        bad = s[s['band'] != 'green'].sort_values('score')
        cols = ['score', 'band', 'sector', 'industry', 'issues', 'action']
        if bad.empty:
            return pd.DataFrame(columns=cols)
        action = bad.apply(lambda r: '; '.join(dict.fromkeys(a for _, a in self._findings(r))), axis=1)
        return bad.assign(action=action)[cols]

    def summary(self, scored=None):
        """Headline numbers for the report."""
        s = self.score() if scored is None else scored
        db = s[s['in_database']]
        return {
            'watchlist tickers': len(s), 'in database': int(s['in_database'].sum()),
            'current (0d behind)': int((db['days_behind'] == 0).sum()),
            'behind 1-5d': int(db['days_behind'].between(1, 5).sum()), 'stale (>5d)': int((db['days_behind'] > 5).sum()),
            'green': int((s['band'] == 'green').sum()), 'amber': int((s['band'] == 'amber').sum()), 'red': int((s['band'] == 'red').sum()),
            'mean score': round(float(s['score'].mean()), 1), 'total rows': int(db['rows'].sum()),
            'date range': f"{db['first_date'].min().date()} to {db['last_date'].max().date()}",
            'last scrape outcomes': s['last_outcome'].value_counts().to_dict() if 'last_outcome' in s else {},
        }

    def treemap(self, scored=None):
        """Sector > industry > ticker. Equal-size ticker tiles coloured by quality band; parent tiles show their worst child."""
        import plotly.graph_objects as go
        s = (self.score() if scored is None else scored).copy()
        s['sector'], s['industry'] = s['sector'].fillna('Unclassified'), s['industry'].fillna('Unclassified')
        nodes = []  # (id, label, parent, size, score, hover)

        def add(id_, label, parent, size, score, hover):
            nodes.append((id_, label, parent, size, float(score), hover))

        add('all', 'Watchlist', '', len(s), s['score'].min(), f"{len(s)} tickers<br>worst score {s['score'].min():.0f}")
        for sector, sg in s.groupby('sector'):
            add(sector, sector, 'all', len(sg), sg['score'].min(), f"{sector}<br>{len(sg)} tickers<br>worst score {sg['score'].min():.0f}")
            for industry, ig in sg.groupby('industry'):
                iid = f"{sector}/{industry}"
                add(iid, industry, sector, len(ig), ig['score'].min(), f"{industry}<br>{len(ig)} tickers<br>worst score {ig['score'].min():.0f}")
                for ticker, r in ig.iterrows():
                    last = r['last_date'].date() if pd.notna(r['last_date']) else 'n/a'
                    add(f"{iid}/{ticker}", f"{ticker}<br>{r['score']:.0f}", iid, 1, r['score'],
                        f"{ticker}: score {r['score']:.0f} ({r['band']})<br>{r['issues'] or 'no issues'}<br>last date {last}, {int(r['rows']) if pd.notna(r['rows']) else 0} rows")

        green, amber, red = '#2ca02c', '#ffbf00', '#d62728'
        lo, hi = self.BANDS[1] / 100, self.BANDS[0] / 100
        ids, labels, parents, sizes, scores, hovers = zip(*nodes)
        fig = go.Figure(go.Treemap(
            ids=ids, labels=labels, parents=parents, values=sizes, branchvalues='total',
            marker=dict(colors=scores, cmin=0, cmax=100, colorbar=dict(title='Score'),
                        colorscale=[[0, red], [lo, red], [lo, amber], [hi, amber], [hi, green], [1, green]]),
            text=hovers, hovertemplate='%{text}<extra></extra>', textinfo='label'))
        fig.update_layout(title=f"Price data health: {len(s)} tickers, mean score {s['score'].mean():.1f}", margin=dict(t=50, l=5, r=5, b=5))
        return fig

    def heatmap(self, scored=None, since=None):
        """Ticker x quarter grid: share of expected trading days present. Blank before a ticker's first date. Grouped by industry."""
        import plotly.graph_objects as go
        s = (self.score() if scored is None else scored).copy()
        s = s[s['in_database']]
        s['sector'], s['industry'] = s['sector'].fillna('Unclassified'), s['industry'].fillna('Unclassified')
        s = s.reset_index().sort_values(['sector', 'industry', 'ticker']).set_index('ticker')

        px = pd.read_csv(self.data_file, usecols=['ticker', 'date'], parse_dates=['date'])
        cal = self._trading_calendar(px, self.calendar_ticker)
        if since is not None:
            cal = cal[cal >= pd.Timestamp(since)]
        quarters = cal.to_period('Q')
        per_q = pd.Series(1, index=cal).groupby(quarters).sum()
        px = px[px['ticker'].isin(s.index) & px['date'].isin(cal)]
        present = px.groupby(['ticker', px['date'].dt.to_period('Q')]).size().unstack(fill_value=0).reindex(index=s.index, columns=per_q.index, fill_value=0)

        q_start = np.concatenate([[0], per_q.cumsum().values[:-1]])
        before_first = np.clip(cal.searchsorted(s['first_date'].values)[:, None] - q_start[None, :], 0, per_q.values[None, :])
        expected = per_q.values[None, :] - before_first
        with np.errstate(invalid='ignore', divide='ignore'):
            coverage = np.where(expected > 0, np.minimum(100 * present.values / expected, 100), np.nan)

        fig = go.Figure(go.Heatmap(
            z=coverage, x=[str(q) for q in per_q.index], y=list(s.index), zmin=70, zmax=100, colorscale='RdYlGn',
            customdata=np.repeat(s['industry'].values[:, None], len(per_q), axis=1), hoverongaps=False,
            hovertemplate='%{y} (%{customdata})<br>%{x}: %{z:.1f}% of trading days<extra></extra>',
            showscale=False))
        start = 0
        for industry, n in s.groupby('industry', sort=False).size().items():
            fig.add_hline(y=start - 0.5, line_width=1, line_color='black')
            fig.add_annotation(xref='paper', yref='y', x=1.0, y=start + (n - 1) / 2, text=industry, showarrow=False, xanchor='left', font=dict(size=10))
            start += n
        fig.update_yaxes(autorange='reversed', type='category', tickfont=dict(size=9))
        fig.update_xaxes(tickangle=-90, tickfont=dict(size=9))
        fig.update_layout(title='Trading-day coverage by quarter (blank = before first date)',
                          height=max(450, 15 * len(s) + 150), margin=dict(t=50, l=5, r=300, b=60))
        return fig
