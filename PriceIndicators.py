from PriceData import PriceData
import pandas as pd
from typing import Optional

class PriceIndicators:
    """
    Computes technical indicators natively using wide-format DataFrames.
    Expects DataFrames where index=date and columns=tickers.
    """
    def __init__(self):
        pass

    def sma(self, prices: pd.DataFrame, window: int = 50) -> pd.DataFrame:
        """
        Calculate simple moving average.
        Input: wide-format DataFrame (index=date, columns=tickers)
        Returns: wide-format DataFrame
        """
        return prices.rolling(window=window).mean()

    def ema(self, prices: pd.DataFrame, window: int = 50) -> pd.DataFrame:
        """
        Calculate exponential moving average.
        Input: wide-format DataFrame (index=date, columns=tickers)
        Returns: wide-format DataFrame
        """
        return prices.ewm(span=window, adjust=False).mean()

    def daily_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily returns.
        Input: wide-format DataFrame (index=date, columns=tickers)
        Returns: wide-format DataFrame
        """
        return prices.pct_change()

    def cumulative_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate cumulative returns.
        Input: wide-format DataFrame (index=date, columns=tickers)
        Returns: wide-format DataFrame
        """
        returns = prices.pct_change().fillna(0)
        return (1 + returns).cumprod() - 1

    def rolling_volatility(self, prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Calculate rolling volatility.
        Input: wide-format DataFrame (index=date, columns=tickers)
        Returns: wide-format DataFrame
        """
        return prices.pct_change().rolling(window=window).std()

    @staticmethod
    def _wilder_average(changes: pd.Series, window: int) -> pd.Series:
        """Wilder's smoothing: the mean of the first `window` values, then (previous x (window - 1) + new) / window."""
        values = changes.dropna()
        if len(values) < window:
            return pd.Series(float('nan'), index=changes.index)
        seeded = pd.concat([pd.Series([values.iloc[:window].mean()], index=[values.index[window - 1]]), values.iloc[window:]])
        return seeded.ewm(alpha=1 / window, adjust=False).mean().reindex(changes.index)

    def rsi(self, prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """
        Calculate Relative Strength Index (RSI), the standard Wilder version, so the usual 30/70 levels apply.
        Each ticker warms up over its own first `window` price changes. Blank while the price has not moved at all.
        Input: wide-format DataFrame (index=date, columns=tickers)
        Returns: wide-format DataFrame
        """
        delta = prices.diff()
        gain = delta.clip(lower=0).apply(self._wilder_average, window=window)
        loss = (-delta.clip(upper=0)).apply(self._wilder_average, window=window)
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def momentum(self, prices: pd.DataFrame, window: int = 252) -> pd.DataFrame:
        """
        Calculate Momentum (Rate of Change).
        Input: wide-format DataFrame (index=date, columns=tickers)
        Returns: wide-format DataFrame
        """
        return prices.pct_change(periods=window)

    def correlation(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the correlation matrix.
        Input: wide-format DataFrame (index=date, columns=tickers)
        Returns: Square DataFrame correlation matrix
        """
        return prices.corr()

