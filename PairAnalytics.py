import pandas as pd
from statsmodels.tsa.stattools import coint
from typing import Optional, Tuple

class PairAnalytics:

    """
    Provides pairwise analytics for two tickers, leveraging wide-format DataFrames.
    Note: methods take a wide-format DataFrame (index=date, columns=tickers) with exactly 2 columns.
    The first column is the ticker held; the second is the one it is hedged with.
    """
    def __init__(self):
        """
        No arguments needed now. All methods accept DataFrames directly.
        """
        pass

    @staticmethod
    def _check_pair(prices: pd.DataFrame, what: str):
        if prices.shape[1] != 2:
            raise ValueError(f"Exactly two columns (tickers) are required for {what}.")

    def cointegration_test(self, prices: pd.DataFrame) -> Tuple[float, float]:
        """
        Perform cointegration test between two tickers in the provided wide DataFrame.
        Input: wide-format DataFrame (index=date, columns=tickers) with exactly 2 columns.
        Returns: (score, p-value).
        """
        self._check_pair(prices, "pairwise cointegration test")

        s1 = prices.iloc[:, 0].dropna()
        s2 = prices.iloc[:, 1].dropna()

        # Align on common dates (already aligned if they come from get_data_wide, but dropna might shift them)
        common = s1.index.intersection(s2.index)
        s1c, s2c = s1.loc[common], s2.loc[common]

        if len(s1c) < 50:
            raise ValueError('Not enough overlapping data for cointegration test.')

        score, pvalue, _ = coint(s1c, s2c)
        return score, pvalue

    def hedge_ratio(self, prices: pd.DataFrame, window: Optional[int] = None) -> pd.Series:
        """
        Units of the second ticker that offset one unit of the first: the slope of the first price on the second.
        window=None: one ratio estimated over the whole history. It uses future prices, so it is for exploring a pair, not for a backtest.
        window=n: a rolling estimate from the n days up to and including each date, which a backtest can use.
        Returns a Series indexed by date.
        """
        self._check_pair(prices, "a hedge ratio")
        first, second = prices.iloc[:, 0], prices.iloc[:, 1]
        if window is None:
            both = prices.dropna()
            return pd.Series(both.iloc[:, 0].cov(both.iloc[:, 1]) / both.iloc[:, 1].var(), index=prices.index)
        return first.rolling(window).cov(second) / second.rolling(window).var()

    def spread(self, prices: pd.DataFrame, hedge_window: Optional[int] = None) -> pd.Series:
        """
        The first price minus the hedge ratio times the second, so the two tickers are compared on the same scale.
        hedge_window is passed to hedge_ratio (None = whole history, n = rolling).
        Returns a Series indexed by date.
        """
        self._check_pair(prices, "pairwise spread")
        return prices.iloc[:, 0] - self.hedge_ratio(prices, hedge_window) * prices.iloc[:, 1]

    def zscore(self, prices: pd.DataFrame, window: int = 30, hedge_window: Optional[int] = None) -> pd.Series:
        """
        Calculate the rolling z-score of the spread between two tickers.
        Input: wide-format DataFrame (index=date, columns=tickers).
        Returns a Series indexed by date.
        """
        spr = self.spread(prices, hedge_window)
        rolling_mean = spr.rolling(window=window).mean()
        rolling_std = spr.rolling(window=window).std()
        z = (spr - rolling_mean) / rolling_std
        return z

    def generate_signals(
        self,
        prices: pd.DataFrame,
        window: int = 30,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        hedge_window: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate buy/sell signals based on z-score divergence for two tickers.
        signal: 1 = long spread (long ticker1, short ticker2), -1 = short spread, 0 = no position.
        A position opens only when the z-score passes an entry level, closes when it comes back inside the exit band,
        and then stays closed until the next entry. A missing z-score holds the current position.
        Returns a DataFrame with columns: ['zscore', 'signal']
        """
        z = self.zscore(prices, window, hedge_window)
        signals, position = [], 0
        for value in z:
            if value > entry_z:
                position = -1   # short spread
            elif value < -entry_z:
                position = 1    # long spread
            elif abs(value) < exit_z:
                position = 0
            signals.append(position)
        return pd.DataFrame({'zscore': z, 'signal': pd.Series(signals, index=z.index, dtype=int)})
