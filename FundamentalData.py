"""FundamentalData: loader and accessor for the EDGAR statement files. Read-only; the counterpart of PriceData."""
import os
import pandas as pd
from typing import Optional, List, Union
from pipeline.EdgarHealth import EdgarHealth


class FundamentalData:
    KEYS = ['ticker', 'period_end', 'filed', 'currency']
    PERIODS = ('quarterly', 'annual', 'ttm')
    BALANCES = {'assets', 'equity', 'cash', 'debt_long', 'debt_short', 'shares'}   # a trailing-twelve-month view takes the latest, not a sum
    TTM_SPAN_DAYS = (255, 290)   # four consecutive quarters end this far apart, first to last; anything else has a gap or a duplicate
    STALE_DAYS = {'quarterly': 200, 'ttm': 200, 'annual': 500}   # how long a filed value stays usable in a daily table

    def __init__(self, data_dir: str = 'data/edgar', currency: Optional[str] = 'USD', include_red: bool = False, health: Optional[EdgarHealth] = None):
        """Scope is the watchlist. By default red tickers and non-USD reporters are left out; currency=None keeps every currency.
        Health bands come from the scores saved by the last update, unless a health object is passed in."""
        self.currency = currency
        self.include_red = include_red
        self.health = health           # None: use the scores saved by the last update
        self.data_dir = data_dir
        self.quarterly = self._load(os.path.join(data_dir, 'edgar-quarterly.csv'))
        self.annual = self._load(os.path.join(data_dir, 'edgar-annual.csv'))
        self._filers = None

    def _load(self, path: str) -> pd.DataFrame:
        # A missing or malformed file raises here, naming the file.
        df = pd.read_csv(path, parse_dates=['period_end', 'filed'])
        missing = [col for col in self.KEYS if col not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing column(s): {missing}")
        return df.sort_values(['ticker', 'period_end']).reset_index(drop=True)

    def get_metrics(self) -> List[str]:
        return [col for col in self.quarterly.columns if col not in self.KEYS]

    def filers(self) -> pd.DataFrame:
        """One row per watchlist ticker: basis ('quarterly', 'annual' or 'none'), health band, currency, and whether this loader includes it."""
        if self._filers is None:
            scored = self.health.score() if self.health is not None else EdgarHealth(data_dir=self.data_dir).saved()
            f = scored[['basis', 'band']].copy()
            f['currency'] = scored['currency'] if 'currency' in scored.columns else None
            f['included'] = (f['basis'] != 'none') & (self.include_red | (f['band'] != 'red')) \
                & ((f['currency'] == self.currency) if self.currency is not None else True)
            self._filers = f
        return self._filers

    def get_tickers(self, period: str = 'quarterly') -> List[str]:
        """Included tickers for a period type. Quarterly and trailing-twelve-month views hold quarterly filers only; annual holds everyone."""
        if period not in self.PERIODS:
            raise ValueError(f"Unknown period '{period}'. Allowed: {self.PERIODS}.")
        f = self.filers()
        f = f[f['included']]
        return sorted(f.index if period == 'annual' else f.index[f['basis'] == 'quarterly'])

    def _ttm(self, q: pd.DataFrame) -> pd.DataFrame:
        """Trailing twelve months on quarterly rows: sum of four consecutive quarters for flows, the latest value for balances."""
        flows = [m for m in self.get_metrics() if m not in self.BALANCES]
        out = q.copy()
        out[flows] = q.groupby('ticker')[flows].rolling(4, min_periods=4).sum().reset_index(level=0, drop=True)
        span = (q['period_end'] - q.groupby('ticker')['period_end'].shift(3)).dt.days
        out.loc[~span.between(*self.TTM_SPAN_DAYS), flows] = float('nan')
        return out

    def get_data_long(self, period: str = 'quarterly', tickers: Optional[Union[str, List[str]]] = None,
                      as_of: Optional[Union[str, pd.Timestamp]] = None) -> pd.DataFrame:
        """
        Rows of ticker, period_end, filed, currency and every metric, for the included tickers.
        period: 'quarterly', 'annual' or 'ttm'.
        as_of: keep only periods first filed on or before this date (point-in-time). Values are as most recently reported.
        tickers: str or list to narrow further; a ticker this loader excludes stays excluded.
        """
        keep = self.get_tickers(period)
        if tickers is not None:
            wanted = [tickers] if isinstance(tickers, str) else list(tickers)
            keep = [t for t in keep if t in wanted]
        df = self.annual if period == 'annual' else self.quarterly
        df = df[df['ticker'].isin(keep)]
        if period == 'ttm':
            df = self._ttm(df)
        if as_of is not None:
            df = df[df['filed'] <= pd.to_datetime(as_of)]
        return df.reset_index(drop=True)

    def get_data_wide(self, metric: str, period: str = 'quarterly', tickers: Optional[Union[str, List[str]]] = None,
                      as_of: Optional[Union[str, pd.Timestamp]] = None) -> pd.DataFrame:
        """
        One metric in wide format: index=period_end, columns=tickers.
        Companies with different fiscal calendars sit on different rows, so expect gaps; see get_data_long for the arguments.
        """
        if metric not in self.get_metrics():
            raise ValueError(f"Unknown metric '{metric}'. Available: {self.get_metrics()}.")
        df = self.get_data_long(period, tickers, as_of)
        return df.pivot(index='period_end', columns='ticker', values=metric).sort_index()

    def to_daily(self, table: pd.DataFrame, dates: pd.DatetimeIndex, period: str = 'quarterly') -> pd.DataFrame:
        """
        Convert a period-end table (from get_data_wide, or an indicator built on it) to the given dates:
        each date holds the value of the latest period that had been filed by then, so it lines up with prices without look-ahead.
        A value is carried forward for at most STALE_DAYS, so a company that stops filing drops out instead of staying frozen.
        """
        rows = self.annual if period == 'annual' else self.quarterly
        filed = rows[rows['ticker'].isin(table.columns)].pivot(index='period_end', columns='ticker', values='filed')
        limit = pd.Timedelta(days=self.STALE_DAYS[period])
        out = {}
        for ticker in table.columns:
            s = table[ticker].dropna()
            known = pd.DataFrame({'value': s, 'filed': filed[ticker].reindex(s.index) if ticker in filed.columns else pd.NaT}).dropna().sort_index()
            # Drop a period that only became known after a later one did (old periods that first appear as comparatives).
            later = known['filed'][::-1].cummin()[::-1].shift(-1)
            known = known[later.isna() | (known['filed'] < later)]
            out[ticker] = pd.Series(known['value'].values, index=pd.DatetimeIndex(known['filed'])).reindex(dates, method='ffill', tolerance=limit)
        return pd.DataFrame(out, index=dates)

    def info(self):
        f = self.filers()
        print(f"Included: {int(f['included'].sum())} of {len(f)} watchlist tickers ({len(self.get_tickers('quarterly'))} quarterly filers)")
        print(f"Metrics: {self.get_metrics()}")
        print(f"Period range: {self.quarterly['period_end'].min().date()} to {self.quarterly['period_end'].max().date()}")
