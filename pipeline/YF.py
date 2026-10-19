"""Yahoo Finance price scraper: incremental downloads, manifest, liquidity screen and single-ticker refresh."""
import os
from datetime import datetime
import pandas as pd
import yfinance as yf


class YF:
    LAG_DAYS = 7       # never request data newer than today minus this, so no partial bars are stored
    OVERLAP_DAYS = 10  # re-fetch this many days before the last stored date so provisional rows are overwritten
    SCREEN_WINDOW_DAYS = 120          # calendar days of history for the liquidity screen (>= 60 trading days)
    SCREEN_MIN_DOLLAR_VOLUME = 10_000_000  # median daily close*volume over the last 60 trading days
    SCREEN_MIN_PRICE = 2.0
    RESCREEN_DAYS = 90
    RETRY_NO_DATA_DAYS = 7  # 'no data' may be transient, so retry sooner than a real pass/fail
    ADJ_DRIFT_TOLERANCE = 1e-4  # relative close/adj_close difference vs Yahoo that signals a dividend or split since the last fetch
    MAX_REFRESH_PER_RUN = 20    # safeguard against a Yahoo glitch triggering many full re-downloads
    VERIFY_PER_RUN = 8          # tickers whose full history is compared with Yahoo on each daily update

    def __init__(self, tickers_file="data/watchlist.csv", data_file="data/prices/yf-prices.csv",
                 registry_file="data/registry/tickers.csv", manifest_file="data/prices/yf-manifest.csv",
                 screen_file="data/prices/yf-screen.csv"):
        self.tickers_file = tickers_file
        self.data_file = data_file
        self.registry_file = registry_file
        self.manifest_file = manifest_file
        self.screen_file = screen_file
        self.tickers = self._load_tickers()

    @staticmethod
    def _mt_symbol(ticker):
        # MacroTrends uses '.' for share classes (BRK.B) where yfinance uses '-' (BRK-B).
        return str(ticker).strip().upper().replace('-', '.')

    @staticmethod
    def _write_csv_atomic(df, path):
        # Write to a temp file, then rename, so a crash mid-write cannot corrupt the existing file.
        tmp_path = path + '.tmp'
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)

    def _write_prices(self, df):
        """Write the prices file, refusing if any ticker would end up with fewer rows than it has on disk."""
        if os.path.exists(self.data_file):
            before = pd.read_csv(self.data_file, usecols=['ticker'])['ticker'].value_counts()
            lost = before - df['ticker'].value_counts().reindex(before.index, fill_value=0)
            lost = lost[lost > 0]
            if not lost.empty:
                raise RuntimeError(f"Write refused: {len(lost)} ticker(s) would lose rows, e.g. {lost.head(5).to_dict()}. {self.data_file} left unchanged.")
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        self._write_csv_atomic(df, self.data_file)

    def _load_tickers(self):
        if os.path.exists(self.tickers_file):
            df = pd.read_csv(self.tickers_file)
            tickers = (df['ticker'] if 'ticker' in df.columns else df.iloc[:, 0]).dropna().astype(str).str.strip().str.upper().tolist()
            tickers = [t for t in tickers if not t.startswith('#')]
            tickers = list(dict.fromkeys(tickers))
        else:
            tickers = ['AAPL', 'MSFT', 'GOOGL']

        # DELISTED is the single exclusion gate for YF — set here when yfinance confirms a delisting.
        if os.path.exists(self.registry_file):
            mt_df = pd.read_csv(self.registry_file, dtype={'status': str})
            delisted = set(mt_df.loc[mt_df['status'] == 'DELISTED', 'ticker'].astype(str).str.strip().str.upper())
            if delisted:
                before = len(tickers)
                tickers = [t for t in tickers if self._mt_symbol(t) not in delisted]
                if len(tickers) < before:
                    print(f"Filtered out {before - len(tickers)} DELISTED tickers.")

        return tickers

    def _flag_delisted(self, tickers):
        """Confirmed delisting: write DELISTED to the ticker registry. Both scrapers skip DELISTED tickers."""
        tickers = sorted(set(t.strip().upper() for t in tickers))
        if not tickers:
            return
        print(f"Confirmed delisted (per yfinance): {tickers}")

        if os.path.exists(self.registry_file):
            mt_df = pd.read_csv(self.registry_file, dtype={'status': str})
            mt_symbols = {self._mt_symbol(t) for t in tickers}
            mask = mt_df['ticker'].astype(str).str.strip().str.upper().isin(mt_symbols)
            if mask.any():
                mt_df.loc[mask, 'status'] = 'DELISTED'
                # StringDtype columns reject datetime objects; write as string explicitly.
                mt_df.loc[mask, 'last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                self._write_csv_atomic(mt_df, self.registry_file)
                print(f"Updated status=DELISTED for {mask.sum()} ticker(s) in {self.registry_file}.")

            not_found = [t for t in tickers if self._mt_symbol(t) not in set(mt_df['ticker'].astype(str).str.strip().str.upper())]
            if not_found:
                print(f"  [WARN] Not in the ticker registry — DELISTED status cannot be persisted: {not_found}")

    def _download_prices(self, ticker, start_date="2000-01-01", end_date=None):
        """Download price data for a single ticker. Returns (DataFrame, is_delisted); self.last_outcome holds the classification."""
        is_delisted = False
        self.last_outcome = 'ok'
        try:
            yf.shared._ERRORS.clear()
            data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)

            # Classify any error yfinance reported — four explicit cases, nothing silently swallowed.
            if ticker in yf.shared._ERRORS:
                msg = str(yf.shared._ERRORS[ticker]).lower()
                if 'delisted' in msg and not any(x in msg for x in ('timed out', 'timeout', 'curl', 'connection')):
                    print(f"  [DELISTED]  {ticker}: {yf.shared._ERRORS[ticker]}")
                    is_delisted = True
                    self.last_outcome = 'delisted'
                elif 'no data found' in msg or 'no price data' in msg:
                    print(f"  [NO DATA]   {ticker}: {yf.shared._ERRORS[ticker]}")
                    self.last_outcome = 'no_data'
                elif any(x in msg for x in ('timed out', 'timeout', 'connection', 'curl')):
                    print(f"  [TRANSIENT] {ticker}: {yf.shared._ERRORS[ticker]}")
                    self.last_outcome = 'transient_error'
                else:
                    print(f"  [UNKNOWN]   {ticker}: {yf.shared._ERRORS[ticker]}")
                    self.last_outcome = 'unknown_error'

            if data.empty:
                if self.last_outcome == 'ok':
                    self.last_outcome = 'no_new_data'
                return pd.DataFrame(), is_delisted

            # Normalize columns — handles both flat and MultiIndex yfinance versions.
            data = data.reset_index()
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.columns = [str(c).lower().replace(' ', '_') for c in data.columns]
            data['ticker'] = ticker

            for col in ['close', 'adj_close', 'open', 'high', 'low', 'volume']:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')

            required = [col for col in ['close', 'adj_close', 'open', 'high', 'low'] if col in data.columns]
            data = data.dropna(subset=required)
            data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')
            return data, is_delisted

        except Exception as e:
            print(f"  [ERROR]     {ticker}: {e}")
            self.last_outcome = 'unknown_error'
            return pd.DataFrame(), is_delisted

    def _save_manifest(self, rows):
        """Merge this run's per-ticker results into the manifest (one row per ticker, latest attempt wins)."""
        new = pd.DataFrame(rows, columns=['ticker', 'last_date', 'last_attempt', 'outcome', 'rows_fetched'])
        if os.path.exists(self.manifest_file):
            old = pd.read_csv(self.manifest_file, dtype=str)
            if 'last_verified' in old.columns:
                new['last_verified'] = new['ticker'].map(old.set_index('ticker')['last_verified'])
            new = pd.concat([old[~old['ticker'].isin(new['ticker'])], new], ignore_index=True)
        os.makedirs(os.path.dirname(self.manifest_file), exist_ok=True)
        self._write_csv_atomic(new.sort_values('ticker'), self.manifest_file)

    def _mark_verified(self, ticker):
        """Record today as the date this ticker's stored history last matched Yahoo."""
        if not os.path.exists(self.manifest_file):
            return
        man = pd.read_csv(self.manifest_file, dtype=str)
        if 'last_verified' not in man.columns:
            man['last_verified'] = None
        man.loc[man['ticker'] == ticker, 'last_verified'] = datetime.now().strftime('%Y-%m-%d')
        self._write_csv_atomic(man, self.manifest_file)

    def verify_sample(self, n=None):
        """Compare the full stored history of the n least recently verified tickers with Yahoo and refresh any that differ.
        Returns DataFrame[ticker, result] with result 'match', 'repaired' or 'failed'."""
        n = self.VERIFY_PER_RUN if n is None else n
        man = pd.read_csv(self.manifest_file, dtype=str).set_index('ticker') if os.path.exists(self.manifest_file) else pd.DataFrame(columns=['outcome'])
        last = man['last_verified'].fillna('') if 'last_verified' in man.columns else pd.Series(dtype=str)
        # Tickers whose last scrape failed are already reported elsewhere and would only block the queue.
        eligible = [t for t in self.tickers if t in man.index and man.at[t, 'outcome'] in ('ok', 'no_new_data')]
        results = []
        for ticker in sorted(eligible, key=lambda t: (last.get(t, ''), t))[:n]:
            report = self.refresh_ticker(ticker)
            if report is not None and report['differs']:
                report = self.refresh_ticker(ticker, apply=True)
                results.append((ticker, 'repaired' if report is not None else 'failed'))
            else:
                results.append((ticker, 'match' if report is not None else 'failed'))
        out = pd.DataFrame(results, columns=['ticker', 'result'])
        print(f"Verified {len(out)} ticker(s) against Yahoo: {out['result'].value_counts().to_dict()}")
        return out

    def _verdict(self, dollar_volume, price):
        """Pass/fail and reason for one measured ticker."""
        reasons = (['low liquidity'] if dollar_volume < self.SCREEN_MIN_DOLLAR_VOLUME else []) + \
                  (['low price'] if price < self.SCREEN_MIN_PRICE else [])
        return not reasons, ' | '.join(reasons)

    def screen(self, candidates, batch_size=50):
        """Liquidity screen for candidates (DataFrame[ticker, matched_by]). Only tickers never screened, or due for a re-screen, are downloaded.
        Returns the screen table for the candidates; the full table is persisted to screen_file."""
        cols = ['ticker', 'screened_on', 'median_dollar_volume', 'last_price', 'passed', 'reason', 'matched_by']
        old = pd.read_csv(self.screen_file, dtype={'ticker': str}) if os.path.exists(self.screen_file) else pd.DataFrame(columns=cols)
        today = pd.Timestamp.today().normalize()

        # Thresholds are applied on read, so changing them needs no re-download.
        measured = old['median_dollar_volume'].notna()
        if measured.any():
            verdicts = [self._verdict(d, p) for d, p in zip(old.loc[measured, 'median_dollar_volume'], old.loc[measured, 'last_price'])]
            old['passed'] = old['passed'].astype(object)
            old.loc[measured, 'passed'] = [v[0] for v in verdicts]
            old.loc[measured, 'reason'] = [v[1] for v in verdicts]

        age = (today - pd.to_datetime(old['screened_on'])).dt.days if len(old) else pd.Series(dtype=float)
        limit = old['reason'].eq('no data').map({True: self.RETRY_NO_DATA_DAYS, False: self.RESCREEN_DAYS}) if len(old) else age
        fresh = set(old.loc[age < limit, 'ticker']) if len(old) else set()
        due = candidates[~candidates['ticker'].isin(fresh)]
        print(f"Screen: {len(candidates)} candidates, {len(due)} due for screening.")

        end = (today - pd.Timedelta(days=self.LAG_DAYS) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')  # exclusive
        start = (today - pd.Timedelta(days=self.LAG_DAYS + self.SCREEN_WINDOW_DAYS)).strftime('%Y-%m-%d')
        results = []
        for i in range(0, len(due), batch_size):
            batch = due.iloc[i:i + batch_size]
            try:
                data = yf.download(batch['ticker'].tolist(), start=start, end=end, auto_adjust=False,
                                   progress=False, group_by='ticker', threads=True)
            except Exception as e:
                print(f"  [ERROR] screen batch {i // batch_size + 1}: {e}")  # not recorded, so retried next run
                continue
            for t, matched_by in zip(batch['ticker'], batch['matched_by']):
                try:
                    px = (data[t] if isinstance(data.columns, pd.MultiIndex) else data)[['Close', 'Volume']].dropna()
                except KeyError:
                    px = pd.DataFrame(columns=['Close', 'Volume'])
                px = px.tail(60)
                if len(px) < 20:
                    results.append((t, today.strftime('%Y-%m-%d'), None, None, False, 'no data', matched_by))
                    continue
                dollar_volume = float((px['Close'] * px['Volume']).median())
                price = float(px['Close'].iloc[-1])
                passed, reason = self._verdict(dollar_volume, price)
                results.append((t, today.strftime('%Y-%m-%d'), round(dollar_volume), round(price, 2), passed, reason, matched_by))

        merged = pd.concat([old[~old['ticker'].isin([r[0] for r in results])], pd.DataFrame(results, columns=cols)], ignore_index=True).astype({'passed': bool})
        os.makedirs(os.path.dirname(self.screen_file), exist_ok=True)
        self._write_csv_atomic(merged[cols].sort_values('ticker'), self.screen_file)
        return merged[merged['ticker'].isin(candidates['ticker'])][cols].sort_values('ticker').reset_index(drop=True)

    def refresh_ticker(self, ticker, apply=False):
        """Re-download one ticker's full history and overwrite its stored rows. Dry run unless apply=True.
        Stored rows Yahoo no longer returns are kept, and nothing is touched if the download fails or is empty."""
        ticker = ticker.strip().upper()
        cutoff_ts = pd.Timestamp.today().normalize() - pd.Timedelta(days=self.LAG_DAYS)
        new, is_delisted = self._download_prices(ticker, start_date="2000-01-01", end_date=(cutoff_ts + pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
        if new.empty or is_delisted:
            print(f"{ticker}: nothing changed ({self.last_outcome}).")
            return None
        existing = pd.read_csv(self.data_file)
        old = existing[existing['ticker'] == ticker]
        both = old.merge(new, on=['ticker', 'date'], how='outer', suffixes=('_old', '_new'), indicator=True)
        same = both[both['_merge'] == 'both']
        report = {'ticker': ticker, 'applied': apply, 'rows_stored': len(old), 'rows_downloaded': len(new),
                  'only_stored (kept)': int((both['_merge'] == 'left_only').sum()), 'only_downloaded (added)': int((both['_merge'] == 'right_only').sum()),
                  'max_close_change': float((same['close_new'] / same['close_old'] - 1).abs().max()),
                  'max_adj_close_change': float((same['adj_close_new'] / same['adj_close_old'] - 1).abs().max())}
        report['differs'] = bool(report['only_downloaded (added)'] > 0 or report['max_close_change'] > self.ADJ_DRIFT_TOLERANCE
                                 or report['max_adj_close_change'] > self.ADJ_DRIFT_TOLERANCE)
        if apply:
            merged = pd.concat([existing, new[existing.columns]], ignore_index=True)
            merged = merged.drop_duplicates(subset=['ticker', 'date'], keep='last').sort_values(['ticker', 'date'])
            self._write_prices(merged)
            self._save_manifest([(ticker, new['date'].max(), datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'ok', len(new))])
        if apply or not report['differs']:
            self._mark_verified(ticker)
        print(report)
        return report

    def _has_drifted(self, existing_df, ticker, fresh):
        """True when Yahoo's current close/adj_close differ from the stored rows on shared dates, i.e. a dividend or split happened since the last fetch."""
        old = existing_df.loc[existing_df['ticker'] == ticker].set_index('date')
        new = fresh.set_index('date')
        shared = old.index.intersection(new.index)
        if shared.empty:
            return False
        drift = max((new.loc[shared, col] / old.loc[shared, col] - 1).abs().max() for col in ('close', 'adj_close'))
        return drift > self.ADJ_DRIFT_TOLERANCE

    def daily_update(self, force_refresh_all=False):
        """Incremental update: each ticker re-fetches from OVERLAP_DAYS before its last stored date, up to today minus LAG_DAYS.
        A ticker whose overlap rows no longer match Yahoo (dividend/split adjustment) gets a full refresh_ticker afterwards."""
        existing_df = pd.read_csv(self.data_file) if os.path.exists(self.data_file) else pd.DataFrame()
        if not existing_df.empty:
            existing_df = existing_df.dropna(subset=['adj_close'])
            existing_df['date'] = pd.to_datetime(existing_df['date']).dt.strftime('%Y-%m-%d')
            latest_dates = existing_df.groupby('ticker')['date'].max().to_dict()
        else:
            latest_dates = {}
        if force_refresh_all:
            latest_dates = {}  # re-download every ticker from the start; stored rows are overwritten, never removed

        new_data_list = []
        confirmed_delisted = []
        drifted = []
        manifest_rows = []
        cutoff_ts = pd.Timestamp.today().normalize() - pd.Timedelta(days=self.LAG_DAYS)
        cutoff = cutoff_ts.strftime('%Y-%m-%d')
        end_date = (cutoff_ts + pd.Timedelta(days=1)).strftime('%Y-%m-%d')  # yfinance end is exclusive
        up_to_date = pd.offsets.BDay().rollback(cutoff_ts).strftime('%Y-%m-%d')  # last weekday on or before cutoff
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        previous = pd.read_csv(self.manifest_file, dtype=str).set_index('ticker') if os.path.exists(self.manifest_file) else pd.DataFrame(columns=['outcome', 'last_attempt'])

        for ticker in self.tickers:
            last_date = latest_dates.get(ticker)
            if last_date is not None and last_date >= up_to_date:
                manifest_rows.append((ticker, last_date, now, 'no_new_data', 0))
                continue
            start_date = "2000-01-01" if last_date is None else (pd.to_datetime(last_date) - pd.Timedelta(days=self.OVERLAP_DAYS)).strftime('%Y-%m-%d')

            print(f"  Fetching {ticker} {start_date} to {cutoff}...")
            ticker_df, is_delisted = self._download_prices(ticker, start_date=start_date, end_date=end_date)
            stored_last = ticker_df['date'].max() if not ticker_df.empty else (last_date or '')
            manifest_rows.append((ticker, stored_last, now, self.last_outcome, len(ticker_df)))

            if is_delisted:
                # DELISTED is permanent, so it needs a second 'delisted' result on a later day.
                seen_before = ticker in previous.index and previous.at[ticker, 'outcome'] == 'delisted' and str(previous.at[ticker, 'last_attempt'])[:10] < now[:10]
                if seen_before:
                    confirmed_delisted.append(ticker)
                else:
                    print(f"  {ticker}: first 'delisted' result, not flagged yet; a second one on a later day confirms it.")
            elif not ticker_df.empty:
                new_data_list.append(ticker_df)
                if last_date is not None and self._has_drifted(existing_df, ticker, ticker_df):
                    drifted.append(ticker)

        if new_data_list:
            final_df = pd.concat([existing_df] + new_data_list, ignore_index=True)
            final_df = final_df.drop_duplicates(subset=['ticker', 'date'], keep='last').sort_values(['ticker', 'date'])
            self._write_prices(final_df)
            print(f"\nUpdate complete. Total rows: {len(final_df)}")
        else:
            print("No new data fetched.")

        self._save_manifest(manifest_rows)

        for ticker in drifted[:self.MAX_REFRESH_PER_RUN]:
            print(f"  Adjustment drift detected for {ticker}: refreshing full history.")
            self.refresh_ticker(ticker, apply=True)
        if len(drifted) > self.MAX_REFRESH_PER_RUN:
            print(f"  Drift pending for {len(drifted) - self.MAX_REFRESH_PER_RUN} more ticker(s): {drifted[self.MAX_REFRESH_PER_RUN:]}")

        self.verify_sample()

        if confirmed_delisted:
            self._flag_delisted(confirmed_delisted)
            self.tickers = self._load_tickers()

    def clean_database(self):
        """Purge rows with missing prices and duplicates. DELISTED tickers' history is kept (they are only excluded from fetching)."""
        if not os.path.exists(self.data_file): return
        df = pd.read_csv(self.data_file)
        initial = len(df)

        for col in ['close', 'adj_close', 'open', 'high', 'low']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['close', 'adj_close', 'open', 'high', 'low'])

        df = df.drop_duplicates(subset=['ticker', 'date'])
        self._write_csv_atomic(df, self.data_file)
        print(f"Clean complete. Removed {initial - len(df)} rows.")

