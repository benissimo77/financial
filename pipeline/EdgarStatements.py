"""Builds quarterly and annual statement tables from the raw EDGAR facts, using the tag map in pipeline/edgar-tags.csv.
Reads data/edgar/raw only; the two output files are regenerated in full on every build."""
import json
import os
import pandas as pd


class EdgarStatements:
    QUARTER_DAYS = (80, 100)   # one fiscal quarter, including the 14-week quarter of a 53-week year
    YEAR_DAYS = (350, 380)
    COVER_WINDOW_DAYS = 100    # cover-page values are dated up to this long after the period end
    ABANDONED_DAYS = 200       # a tag whose last value is this far behind the metric's latest is one the company stopped using
    KINDS = {'duration', 'instant', 'cover'}
    REPORT_FORMS = {'10-K', '10-Q', '10-KT', '10-QT', '20-F', '40-F', '6-K', '8-K'}   # and their /A amendments; proxy statements are excluded
    KEYS = ['ticker', 'period_end', 'filed', 'currency']

    def __init__(self, data_dir='data/edgar', tags_file='pipeline/edgar-tags.csv'):
        self.raw_dir = os.path.join(data_dir, 'raw')
        self.quarterly_file = os.path.join(data_dir, 'edgar-quarterly.csv')
        self.annual_file = os.path.join(data_dir, 'edgar-annual.csv')
        self.tags_file = tags_file

    def _load_tags(self):
        tags = pd.read_csv(self.tags_file, comment='#', skip_blank_lines=True)
        for col in ('metric', 'kind', 'tag'):
            tags[col] = tags[col].astype(str).str.strip()
        bad = tags[~tags['kind'].isin(self.KINDS)]
        if not bad.empty:
            raise ValueError(f"Invalid kind in {self.tags_file}: {bad[['metric', 'kind', 'tag']].values.tolist()}. Allowed: {sorted(self.KINDS)}.")
        tags['order'] = tags.groupby('metric', sort=False).ngroup()   # keep the file's metric order for the output columns
        return tags.sort_values(['order', 'priority'], kind='stable')

    def _facts(self, ticker, tags):
        """All reported values for the mapped tags, in the company's reporting currency (plus share counts)."""
        with open(os.path.join(self.raw_dir, f"{ticker}.json"), encoding='utf-8') as f:
            raw = json.load(f)['facts']
        wanted = set(tags['tag'])
        rows = [(tag, unit, r.get('start'), r['end'], r['val'], r['filed'])
                for concepts in raw.values() for tag, node in concepts.items() if tag in wanted
                for unit, entries in node['units'].items() for r in entries
                if str(r.get('form', '')).split('/')[0] in self.REPORT_FORMS]
        facts = pd.DataFrame(rows, columns=['tag', 'unit', 'start', 'end', 'val', 'filed'])
        for col in ('start', 'end', 'filed'):
            facts[col] = pd.to_datetime(facts[col])
        money = facts[facts['unit'] != 'shares']
        revenue = money[money['tag'].isin(tags.loc[tags['metric'] == 'revenue', 'tag'])]
        units = (revenue if not revenue.empty else money)['unit']
        currency = units.mode().iloc[0] if not units.empty else None   # e.g. TWD, not the USD convenience translation
        return facts[facts['unit'].isin([currency, 'shares'])], currency

    def _quarters(self, d):
        """{period end: value} for one duration tag, from every filed version of its values (sorted by filing date).
        Reported quarters first, then year-to-date differences, then year minus three quarters."""
        latest = d.drop_duplicates(['start', 'end'], keep='last')
        days = (latest['end'] - latest['start']).dt.days
        lo, hi = self.QUARTER_DAYS
        q = latest[days.between(lo, hi)].set_index('end')['val'].to_dict()
        for start, g in latest.groupby('start'):
            g = g.sort_values('end')
            for a, b_end in zip(g.itertuples(), g['end'].iloc[1:]):
                if lo <= (b_end - a.end).days <= hi and b_end not in q:
                    # Subtract like from like: use the version of the longer period filed closest to the shorter one,
                    # so a restated year is never set against unrestated year-to-date figures.
                    versions = d[(d['start'] == start) & (d['end'] == b_end)]
                    b = versions.iloc[(versions['filed'] - a.filed).abs().argmin()]
                    q[b_end] = b['val'] - a.val
        for year in latest[days.between(*self.YEAR_DAYS)].itertuples():
            inside = [v for e, v in q.items() if year.start < e < year.end]
            if year.end not in q and len(inside) == 3:
                q[year.end] = year.val - sum(inside)
        return q

    def _merge(self, by_tag):
        """Combine one metric's tags (Series by period end, in priority order) into one Series.
        Tags the company still uses come first; a tag it has stopped using only fills periods the others lack,
        so an original figure left under an abandoned tag cannot beat a later restatement."""
        by_tag = [s.dropna() for s in by_tag if not s.dropna().empty]
        if not by_tag:
            return pd.Series(dtype=float)
        cutoff = max(s.index.max() for s in by_tag) - pd.Timedelta(days=self.ABANDONED_DAYS)
        ordered = [s for s in by_tag if s.index.max() >= cutoff] + [s for s in by_tag if s.index.max() < cutoff]
        merged = pd.concat(ordered)
        return merged[~merged.index.duplicated(keep='first')].sort_index()   # the first tag in the order wins each period

    def _table(self, ticker, tags, facts, currency, annual):
        """One row per period end for one ticker. Durations define the rows; balances are aligned to them."""
        facts = facts.sort_values('filed')   # so that keep='last' below means the most recent filing wins
        durations = facts.dropna(subset=['start'])
        instants = facts[facts['start'].isna()].drop_duplicates(['tag', 'end'], keep='last')
        columns = {}
        for metric, group in tags[tags['kind'] == 'duration'].groupby('metric', sort=False):
            by_tag = []
            for tag in group['tag']:
                d = durations[durations['tag'] == tag]
                if annual:
                    d = d.drop_duplicates(['start', 'end'], keep='last')
                    values = d[(d['end'] - d['start']).dt.days.between(*self.YEAR_DAYS)].set_index('end')['val'].to_dict()
                else:
                    values = self._quarters(d)
                by_tag.append(pd.Series(values, dtype=float))
            columns[metric] = self._merge(by_tag)
        table = pd.DataFrame(columns)
        table.index = pd.DatetimeIndex(table.index)
        table = table.dropna(how='all').sort_index()
        for metric, group in tags[tags['kind'] != 'duration'].groupby('metric', sort=False):
            by_tag = []
            for tag, kind in zip(group['tag'], group['kind']):
                s = instants[instants['tag'] == tag].set_index('end')['val'].sort_index()
                how = dict(method='bfill', tolerance=pd.Timedelta(days=self.COVER_WINDOW_DAYS)) if kind == 'cover' and not s.empty else {}
                by_tag.append(s.reindex(table.index, **how).astype(float))
            table[metric] = self._merge(by_tag).reindex(table.index)
        table.insert(0, 'currency', currency)
        table.insert(0, 'filed', facts.groupby('end')['filed'].min().reindex(table.index))   # when the period was first reported
        table.insert(0, 'ticker', ticker)
        return table.rename_axis('period_end').reset_index()

    @staticmethod
    def _write_csv_atomic(df, path):
        tmp_path = path + '.tmp'
        df.to_csv(tmp_path, index=False, date_format='%Y-%m-%d')
        os.replace(tmp_path, path)

    def build(self, tickers=None):
        """Rebuild both statement files from the raw files. Returns (quarterly, annual). With tickers given, nothing is written."""
        tags = self._load_tags()
        metrics = list(dict.fromkeys(tags['metric']))
        names = sorted(f[:-5] for f in os.listdir(self.raw_dir) if f.endswith('.json')) if tickers is None else list(tickers)
        quarterly, annual = [], []
        for ticker in names:
            try:
                facts, currency = self._facts(ticker, tags)
            except (ValueError, KeyError, OSError) as e:
                print(f"  [SKIPPED] {ticker}: {e!r}")
                continue
            quarterly.append(self._table(ticker, tags, facts, currency, annual=False))
            annual.append(self._table(ticker, tags, facts, currency, annual=True))
        cols = self.KEYS + metrics
        out = [pd.concat(parts, ignore_index=True)[cols].sort_values(['ticker', 'period_end']).reset_index(drop=True) if parts else pd.DataFrame(columns=cols)
               for parts in (quarterly, annual)]
        if tickers is None:
            self._write_csv_atomic(out[0], self.quarterly_file)
            self._write_csv_atomic(out[1], self.annual_file)
            print(f"EDGAR statements: {out[0]['ticker'].nunique()} tickers with quarterly rows ({len(out[0])} rows), {out[1]['ticker'].nunique()} with annual rows ({len(out[1])} rows).")
        return out[0], out[1]
