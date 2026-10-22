"""SEC EDGAR fundamentals: downloads each watchlist company's reported facts as raw JSON and keeps a manifest."""
import json
import os
import time
from datetime import datetime
import pandas as pd
import requests
from pipeline.TickerFiles import path_for


class Edgar:
    USER_AGENT = 'financial-research ben.silburn@gmail.com'  # SEC requires a contact address on every request
    REQUEST_GAP = 0.25      # seconds between requests; SEC allows 10 per second
    REFETCH_DAYS = 7        # a ticker with a settled answer (ok or no_facts) within this many days is skipped
    TICKER_LIST_DAYS = 30   # refresh SEC's ticker-to-CIK list when older than this
    MIN_SIZE_RATIO = 0.5    # refuse a download smaller than this share of the stored file
    TICKERS_URL = 'https://www.sec.gov/files/company_tickers.json'
    FACTS_URL = 'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json'
    MANIFEST_COLUMNS = ['ticker', 'cik', 'last_attempt', 'outcome', 'last_filed', 'size_kb']

    def __init__(self, tickers_file='data/watchlist.csv', data_dir='data/edgar'):
        self.tickers_file = tickers_file
        self.raw_dir = os.path.join(data_dir, 'raw')
        self.manifest_file = os.path.join(data_dir, 'edgar-manifest.csv')
        self.cik_file = os.path.join(data_dir, 'edgar-tickers.csv')
        os.makedirs(self.raw_dir, exist_ok=True)

    @staticmethod
    def _write_atomic(path, content):
        # Write to a temp file, then rename, so a crash mid-write cannot corrupt the existing file.
        tmp_path = path + '.tmp'
        with open(tmp_path, 'wb') as f:
            f.write(content)
        os.replace(tmp_path, path)

    def _get(self, url):
        time.sleep(self.REQUEST_GAP)
        return requests.get(url, headers={'User-Agent': self.USER_AGENT, 'Accept-Encoding': 'gzip, deflate'}, timeout=60)

    def _load_tickers(self):
        df = pd.read_csv(self.tickers_file, comment='#', skip_blank_lines=True)
        return list(dict.fromkeys(df.iloc[:, 0].dropna().astype(str).str.strip().str.upper()))

    def raw_path(self, ticker):
        return path_for(self.raw_dir, ticker, '.json')

    def cik_map(self):
        """Series of CIK by ticker, from SEC's list. SEC uses '-' for share classes (BRK-B), the same as yfinance."""
        stale = not os.path.exists(self.cik_file) or (time.time() - os.path.getmtime(self.cik_file)) > self.TICKER_LIST_DAYS * 86400
        if stale:
            r = self._get(self.TICKERS_URL)
            r.raise_for_status()
            listing = pd.DataFrame(r.json().values()).rename(columns={'cik_str': 'cik', 'title': 'name'})
            listing['ticker'] = listing['ticker'].str.upper()
            self._write_atomic(self.cik_file, listing[['ticker', 'cik', 'name']].to_csv(index=False).encode('utf-8'))
        listing = pd.read_csv(self.cik_file)
        return listing.drop_duplicates('ticker').set_index('ticker')['cik']

    def _fetch(self, ticker, cik):
        """Download one company's facts. Returns (outcome, last_filed); the stored file is only replaced by a sound download."""
        path = self.raw_path(ticker)
        try:
            r = self._get(self.FACTS_URL.format(cik=int(cik)))
        except requests.RequestException as e:
            print(f"  [TRANSIENT] {ticker}: {e}")
            return 'transient_error', None
        if r.status_code == 404:
            return 'no_facts', None   # SEC has the company but no XBRL facts (ETFs, some foreign listings)
        if r.status_code != 200:
            print(f"  [TRANSIENT] {ticker}: HTTP {r.status_code}")
            return 'transient_error', None
        try:
            facts = json.loads(r.content)['facts']
            filed = [row['filed'] for taxonomy in facts.values() for concept in taxonomy.values()
                     for rows in concept['units'].values() for row in rows]
        except (ValueError, KeyError, TypeError) as e:
            print(f"  [INVALID]   {ticker}: {e!r}")
            return 'invalid_json', None
        if os.path.exists(path) and len(r.content) < self.MIN_SIZE_RATIO * os.path.getsize(path):
            print(f"  [SHRUNK]    {ticker}: download is {len(r.content)} bytes, stored file is {os.path.getsize(path)}; kept the stored file.")
            return 'shrunk', None
        if os.path.exists(path):
            with open(path, 'rb') as f:
                unchanged = f.read() == r.content
        else:
            unchanged = False
        if not unchanged:   # an identical download leaves the file untouched, so the statement builder can skip it
            self._write_atomic(path, r.content)
        return 'ok', max(filed) if filed else None

    def update(self, tickers=None, force=False):
        """Fetch raw facts for the watchlist (or the given tickers). Returns this run's manifest rows."""
        tickers = self._load_tickers() if tickers is None else [str(t).strip().upper() for t in tickers]
        ciks = self.cik_map()
        old = pd.read_csv(self.manifest_file, dtype=str).set_index('ticker') if os.path.exists(self.manifest_file) else pd.DataFrame(columns=self.MANIFEST_COLUMNS[1:]).rename_axis('ticker')
        now = datetime.now()
        rows, skipped = [], 0
        for ticker in tickers:
            if ticker not in ciks.index:
                rows.append((ticker, None, now.strftime('%Y-%m-%d %H:%M:%S'), 'no_cik', None, None))
                continue
            # 'no_facts' is a settled answer too (an ETF will not start filing statements), so it waits like a success.
            settled = ticker in old.index and (old.at[ticker, 'outcome'] == 'no_facts'
                                               or (old.at[ticker, 'outcome'] == 'ok' and os.path.exists(self.raw_path(ticker))))
            recent = settled and (now - pd.Timestamp(old.at[ticker, 'last_attempt'])).days < self.REFETCH_DAYS
            if recent and not force:
                skipped += 1
                continue
            outcome, last_filed = self._fetch(ticker, ciks[ticker])
            if last_filed is None and ticker in old.index:
                last_filed = old.at[ticker, 'last_filed']   # a failed attempt keeps what is known about the stored file
            size_kb = round(os.path.getsize(self.raw_path(ticker)) / 1024) if os.path.exists(self.raw_path(ticker)) else None
            rows.append((ticker, int(ciks[ticker]), now.strftime('%Y-%m-%d %H:%M:%S'), outcome, last_filed, size_kb))

        new = pd.DataFrame(rows, columns=self.MANIFEST_COLUMNS)
        merged = pd.concat([old.reset_index()[~old.index.isin(new['ticker'])], new.astype(object)], ignore_index=True).sort_values('ticker')
        self._write_atomic(self.manifest_file, merged[self.MANIFEST_COLUMNS].to_csv(index=False).encode('utf-8'))
        print(f"EDGAR update: {len(new)} attempted {new['outcome'].value_counts().to_dict()}, {skipped} skipped as fetched within {self.REFETCH_DAYS} days.")
        return new
