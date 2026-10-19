"""FundamentalIndicators: ratios and growth rates built on FundamentalData; the counterpart of PriceIndicators.

Every indicator follows the project's standard pattern: one function, one metric, one wide table (dates down, tickers across),
for all included tickers unless narrowed. A ticker with a missing or unusable input gets a blank; nothing raises for one company.
Arguments shared by every indicator:
    period  'ttm' (trailing twelve months, quarterly filers) or 'annual' (everyone).
    tickers str or list to narrow; default all.
    as_of   only use figures filed on or before this date.
    daily   False: indexed by period end, as reported. True: indexed by trading day, each day holding the latest value filed by then.
"""
import pandas as pd
from typing import Optional, List, Union
from FundamentalData import FundamentalData
from PriceData import PriceData

Tickers = Optional[Union[str, List[str]]]
AsOf = Optional[Union[str, pd.Timestamp]]


class FundamentalIndicators:
    INDICATORS = ['revenue_growth', 'operating_margin', 'net_margin', 'fcf_margin', 'roic', 'fcf_yield']
    DEFAULT_TAX_RATE = 0.21       # used for ROIC when the reported rate is missing
    TAX_RATE_RANGE = (0.0, 0.35)  # reported rates outside this are clipped (loss years, one-off tax items)
    YEAR_AGO_TOLERANCE_DAYS = 15  # a 52/53-week year moves the period end by a few days

    def __init__(self, fundamentals: Optional[FundamentalData] = None, prices: Optional[PriceData] = None):
        self.fundamentals = fundamentals if fundamentals is not None else FundamentalData()
        self._prices = prices   # loaded on first use: only the daily form and fcf_yield need prices

    @property
    def prices(self) -> PriceData:
        if self._prices is None:
            self._prices = PriceData()
        return self._prices

    def _get(self, metric: str, period: str, tickers: Tickers, as_of: AsOf) -> pd.DataFrame:
        if period not in ('ttm', 'annual'):
            raise ValueError(f"Unknown period '{period}'. Indicators use 'ttm' or 'annual'.")
        return self.fundamentals.get_data_wide(metric, period, tickers, as_of)

    def _finish(self, table: pd.DataFrame, period: str, as_of: AsOf, daily: bool) -> pd.DataFrame:
        """Blank out infinities and, for the daily form, move from period ends to trading days."""
        table = table.where(table.abs() != float('inf'))
        if not daily:
            return table
        dates = self.prices.get_data_wide('close').index
        if as_of is not None:
            dates = dates[dates <= pd.to_datetime(as_of)]
        return self.fundamentals.to_daily(table, dates, period)

    def _year_ago(self, table: pd.DataFrame) -> pd.DataFrame:
        """Each ticker's value one year before each period end, matched by date so a missing quarter cannot shift the comparison."""
        tolerance = pd.Timedelta(days=self.YEAR_AGO_TOLERANCE_DAYS)
        out = {}
        for ticker in table.columns:
            s = table[ticker].dropna()
            prior = s.reindex(s.index - pd.DateOffset(years=1), method='nearest', tolerance=tolerance) if len(s) else s
            out[ticker] = pd.Series(prior.values, index=s.index)
        return pd.DataFrame(out).reindex(index=table.index, columns=table.columns)

    def _fcf(self, period: str, tickers: Tickers, as_of: AsOf) -> pd.DataFrame:
        """Free cash flow: operating cash flow minus capex. A company with no capex figure is treated as having none."""
        return self._get('cfo', period, tickers, as_of) - self._get('capex', period, tickers, as_of).fillna(0)

    def _over_revenue(self, numerator: pd.DataFrame, period: str, tickers: Tickers, as_of: AsOf) -> pd.DataFrame:
        revenue = self._get('revenue', period, tickers, as_of)
        return numerator / revenue.where(revenue > 0)

    def revenue_growth(self, period: str = 'ttm', tickers: Tickers = None, as_of: AsOf = None, daily: bool = False) -> pd.DataFrame:
        """Revenue against the same figure one year earlier, as a fraction (0.10 = 10% growth)."""
        revenue = self._get('revenue', period, tickers, as_of)
        prior = self._year_ago(revenue)
        return self._finish(revenue / prior.where(prior > 0) - 1, period, as_of, daily)

    def operating_margin(self, period: str = 'ttm', tickers: Tickers = None, as_of: AsOf = None, daily: bool = False) -> pd.DataFrame:
        """Operating income over revenue."""
        return self._finish(self._over_revenue(self._get('op_income', period, tickers, as_of), period, tickers, as_of), period, as_of, daily)

    def net_margin(self, period: str = 'ttm', tickers: Tickers = None, as_of: AsOf = None, daily: bool = False) -> pd.DataFrame:
        """Net income over revenue."""
        return self._finish(self._over_revenue(self._get('net_income', period, tickers, as_of), period, tickers, as_of), period, as_of, daily)

    def fcf_margin(self, period: str = 'ttm', tickers: Tickers = None, as_of: AsOf = None, daily: bool = False) -> pd.DataFrame:
        """Free cash flow (operating cash flow minus capex) over revenue."""
        return self._finish(self._over_revenue(self._fcf(period, tickers, as_of), period, tickers, as_of), period, as_of, daily)

    def roic(self, period: str = 'ttm', tickers: Tickers = None, as_of: AsOf = None, daily: bool = False) -> pd.DataFrame:
        """
        Return on invested capital: after-tax operating income over (equity + debt - cash) at the period end.
        A missing debt figure counts as zero, which overstates ROIC for an indebted company whose debt tag is not mapped.
        Blank when invested capital is zero or negative.
        """
        get = lambda metric: self._get(metric, period, tickers, as_of)
        rate = (get('tax') / get('pretax_income')).clip(*self.TAX_RATE_RANGE).fillna(self.DEFAULT_TAX_RATE)
        invested = get('equity') + get('debt_long').fillna(0) + get('debt_short').fillna(0) - get('cash').fillna(0)
        return self._finish(get('op_income') * (1 - rate) / invested.where(invested > 0), period, as_of, daily)

    def fcf_yield(self, period: str = 'ttm', tickers: Tickers = None, as_of: AsOf = None, daily: bool = False) -> pd.DataFrame:
        """
        Free cash flow over market value (raw close times shares outstanding).
        As reported: the price on the period end. Daily: that day's price against the latest filed free cash flow and share count.
        """
        fcf, shares = self._fcf(period, tickers, as_of), self._get('shares', period, tickers, as_of)
        close = self.prices.get_data_wide('close')
        common = [t for t in fcf.columns if t in close.columns]
        if daily:
            fcf, shares = self._finish(fcf[common], period, as_of, True), self._finish(shares[common], period, as_of, True)
            value = close.reindex(index=fcf.index, columns=common) * shares
        else:
            value = close[common].reindex(close.index.union(fcf.index)).ffill().reindex(fcf.index) * shares[common]
        return self._finish((fcf[common] / value.where(value > 0)).reindex(columns=fcf.columns), period, as_of, False)

    def snapshot(self, indicators: Optional[List[str]] = None, tickers: Tickers = None, as_of: AsOf = None) -> pd.DataFrame:
        """
        One row per ticker, one column per indicator: each company's latest known value on the date (default today).
        Defaults to every indicator and every included ticker. Quarterly filers use trailing-twelve-month figures and
        annual-only filers annual figures; the 'basis' column says which. A value older than the loader's STALE_DAYS is blank.
        """
        names = self.INDICATORS if indicators is None else list(indicators)
        unknown = [n for n in names if n not in self.INDICATORS]
        if unknown:
            raise ValueError(f"Unknown indicator(s) {unknown}. Available: {self.INDICATORS}.")
        today = pd.to_datetime(as_of) if as_of is not None else pd.Timestamp.today().normalize()
        quarterly = set(self.fundamentals.get_tickers('ttm'))
        parts = []
        for period, basis in (('ttm', 'quarterly'), ('annual', 'annual')):
            keep = lambda table: table[[t for t in table.columns if (t in quarterly) == (basis == 'quarterly')]]
            revenue = keep(self._get('revenue', period, tickers, as_of))
            if revenue.shape[1] == 0:
                continue
            part = pd.DataFrame({'basis': basis, 'period_end': revenue.apply(lambda s: s.last_valid_index())})
            cutoff = today - pd.Timedelta(days=self.fundamentals.STALE_DAYS[period])
            for name in names:
                table = keep(getattr(self, name)(period=period, tickers=tickers, as_of=as_of))
                recent = table[table.index >= cutoff]
                part[name] = recent.ffill().iloc[-1] if len(recent) else float('nan')
            parts.append(part)
        if not parts:
            return pd.DataFrame(columns=['basis', 'period_end'] + names).rename_axis('ticker')
        return pd.concat(parts).rename_axis('ticker').sort_index()[['basis', 'period_end'] + names]
