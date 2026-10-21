"""What a strategy author needs: the data handed to a strategy, and three helpers for saying what to hold.

A strategy is any function that takes `data` (plus optional keyword arguments) and returns what it wants to hold:

    def momentum(data, n=10):
        return top(data.prices.shift(21) / data.prices.shift(252), n=n)

Every table a strategy sees or returns follows the project's standard pattern: trading days down, tickers across.
A blank means "no opinion", and the stock is not held. The backtest lines every table up with its own dates and tickers.
"""
from typing import Optional
import pandas as pd
from PriceData import PriceData
from PriceIndicators import PriceIndicators


class Target:
    """What a strategy asked for, turned into weights only once the backtest knows which tickers are eligible each day."""
    RULES = ('hold', 'top', 'weights')

    def __init__(self, table: pd.DataFrame, rule: str, n: Optional[int] = None):
        if rule not in self.RULES:
            raise ValueError(f"Unknown rule '{rule}'. Allowed: {self.RULES}.")
        if rule == 'top' and (n is None or n < 1):
            raise ValueError("top() needs n of at least 1.")
        self.table, self.rule, self.n = table, rule, n

    def to_weights(self, eligible: pd.DataFrame) -> pd.DataFrame:
        """Weights by date and ticker. Ineligible tickers get zero, and equal weights are shared only among eligible ones."""
        table = self.table.reindex(index=eligible.index, columns=eligible.columns)
        if self.rule == 'weights':
            numbers = table.apply(pd.to_numeric, errors='coerce')
            return numbers.where(numbers.abs() != float('inf')).fillna(0.0).where(eligible, 0.0)
        if self.rule == 'hold':
            chosen = table.fillna(False).astype(bool) & eligible
        else:
            score = table.apply(pd.to_numeric, errors='coerce')
            score = score.where(eligible & (score.abs() != float('inf')))
            chosen = score.rank(axis=1, ascending=False, method='first') <= self.n
        counts = chosen.sum(axis=1)
        return chosen.astype(float).div(counts.where(counts > 0), axis=0).fillna(0.0)


def hold(signal: pd.DataFrame) -> Target:
    """Own every eligible ticker where `signal` is true, in equal amounts. Nothing true on a date means all cash."""
    return Target(signal, 'hold')


def top(score: pd.DataFrame, n: int) -> Target:
    """Own the `n` eligible tickers with the highest score on each date, in equal amounts. Fewer than n with a score: own those."""
    return Target(score, 'top', n)


def weights(table: pd.DataFrame) -> Target:
    """Use the weights exactly as given (negative for a short). What is not invested is cash; ineligible tickers get zero."""
    return Target(table, 'weights')


class StrategyData:
    """The data a strategy receives. Every table covers the full price history, so indicators can warm up before the test starts."""

    def __init__(self, prices: PriceData, tickers: list, universe: pd.DataFrame, benchmark: Optional[str] = 'SPY'):
        self._source = prices
        self.tickers = list(tickers)
        self.universe = universe      # true where a ticker is eligible to be held that day
        self.price_ind = PriceIndicators()
        self._benchmark = benchmark
        self._cache = {}

    def _wide(self, column: str) -> pd.DataFrame:
        if column not in self._cache:
            table = self._source.get_data_wide(price_col=column, tickers=self.tickers)
            self._cache[column] = table.reindex(index=self.universe.index, columns=self.tickers)
        return self._cache[column]

    @property
    def prices(self) -> pd.DataFrame:
        """Adjusted closing prices (dividends and splits included). Use for returns, trends and momentum."""
        return self._wide('adj_close')

    @property
    def close(self) -> pd.DataFrame:
        """Raw closing prices. Use only where the actual traded price matters, such as market value."""
        return self._wide('close')

    @property
    def volume(self) -> pd.DataFrame:
        return self._wide('volume')

    @property
    def benchmark(self) -> pd.Series:
        """The benchmark's adjusted price (SPY by default), for market-wide filters such as 'only invest in an uptrend'."""
        if 'benchmark' not in self._cache:
            wide = self._source.get_data_wide(tickers=self._benchmark) if self._benchmark in self._source.get_tickers() else pd.DataFrame({self._benchmark: float('nan')}, index=self.universe.index)
            self._cache['benchmark'] = wide[self._benchmark].reindex(self.universe.index)
        return self._cache['benchmark']

    @property
    def fund_ind(self):
        """FundamentalIndicators. Use daily=True so each day only sees figures already filed."""
        if 'fund_ind' not in self._cache:
            from FundamentalIndicators import FundamentalIndicators   # loaded only by strategies that use fundamentals
            self._cache['fund_ind'] = FundamentalIndicators(prices=self._source)
        return self._cache['fund_ind']

    def fundamental(self, metric: str, period: str = 'ttm') -> pd.DataFrame:
        """Any statement figure (e.g. 'revenue', 'cash'), daily as known: each day holds the latest value filed by then."""
        fundamentals = self.fund_ind.fundamentals
        table = fundamentals.get_data_wide(metric, period)
        return fundamentals.to_daily(table, self.universe.index, period).reindex(columns=self.tickers)
