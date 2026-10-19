# PriceData: Robust loader and accessor for price data
import pandas as pd
from typing import Optional, List, Union

class PriceData:
        
    def split_inout_sample(self, price_col: str = 'adj_close', tickers: Optional[Union[str, List[str]]] = None, frac: float = 0.7):
        """
        Splits the data chronologically into in-sample and out-of-sample sets (default 70/30 split).
        Returns two DataFrames in wide format (index=date, columns=tickers, values=price_col), matching get_data_wide.
        """
        wide = self.get_data_wide(price_col=price_col, tickers=tickers)
        n = len(wide)
        if n == 0:
            return wide.copy(), wide.copy()
        split_idx = int(n * frac)
        in_sample = wide.iloc[:split_idx].copy()
        out_sample = wide.iloc[split_idx:].copy()
        return in_sample, out_sample

    def __init__(self, csv_file: str = "data/prices/yf-prices.csv", tickers_file: Optional[str] = "data/watchlist.csv"):
        """Scope is the watchlist, like FundamentalData. tickers_file=None serves every ticker in the price file,
        including ones that have left the watchlist and are no longer updated."""
        self.csv_file = csv_file
        self.df = self._load_data()
        if tickers_file is not None:
            watch = pd.read_csv(tickers_file, comment='#', skip_blank_lines=True)
            self.df = self.df[self.df['ticker'].isin(watch.iloc[:, 0].dropna().astype(str).str.strip().str.upper())]
        self.df['date'] = pd.to_datetime(self.df['date'], format='%Y-%m-%d')
        self.df = self.df.sort_values(['ticker', 'date'])

    def _load_data(self) -> pd.DataFrame:
        # A missing or malformed file raises here, rather than returning an empty table that fails confusingly later.
        df = pd.read_csv(self.csv_file)
        missing = [col for col in ['ticker', 'date', 'close', 'adj_close', 'volume'] if col not in df.columns]
        if missing:
            raise ValueError(f"{self.csv_file} is missing column(s): {missing}")
        return df

    def get_tickers(self) -> List[str]:
        return sorted(self.df['ticker'].unique())

    def get_dates(self, tickers: Optional[Union[str, List[str]]] = None) -> List[str]:
        """
        Returns a sorted list of date strings ('YYYY-MM-DD') for the given tickers (str or list) or all tickers.
        tickers: str, list of str, or None (all)
        """
        df = self.df
        if tickers is not None:
            if isinstance(tickers, str):
                df = df[df['ticker'] == tickers]
            else:
                df = df[df['ticker'].isin(tickers)]
        dates = df['date'].dt.strftime('%Y-%m-%d').unique()
        return sorted(dates)

    def get_data_long(self, price_col: str = 'adj_close', tickers: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """
        Returns data in long format: columns = ['date', 'ticker', price_col]
        price_col defaults to 'adj_close' (adjusted for dividends and splits); pass 'close' for the raw price.
        Optionally filter by tickers.
        """
        df = self.df[['date', 'ticker', price_col]].copy()
        if tickers is not None:
            if isinstance(tickers, str):
                df = df[df['ticker'] == tickers]
            else:
                df = df[df['ticker'].isin(tickers)]
        return df.sort_values(['date', 'ticker']).reset_index(drop=True)

    def get_data_wide(self, price_col: str = 'adj_close', tickers: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """
        Returns data in wide format: index=date, columns=tickers, values=price_col
        price_col defaults to 'adj_close' (adjusted for dividends and splits); pass 'close' for the raw price.
        Optionally filter by tickers.
        """
        df = self.get_data_long(price_col, tickers)
        wide = df.pivot(index='date', columns='ticker', values=price_col)
        wide.index = pd.to_datetime(wide.index)
        # Always return a DataFrame, even for a single ticker
        wide = wide.sort_index()
        if tickers is not None and isinstance(tickers, str):
            return wide[[tickers]]
        return wide

    def get_data(self, tickers: Optional[Union[str, List[str]]] = None, start: Optional[Union[str, pd.Timestamp]] = None, end: Optional[Union[str, pd.Timestamp]] = None) -> pd.DataFrame:
        """
        Returns a DataFrame filtered by tickers (str, list, or None), start, and end dates.
        tickers: str, list of str, or None (all)
        """
        df = self.df.copy()
        if tickers is not None:
            if isinstance(tickers, str):
                df = df[df['ticker'] == tickers]
            else:
                df = df[df['ticker'].isin(tickers)]
        if start:
            df = df[df['date'] >= pd.to_datetime(start)]
        if end:
            df = df[df['date'] <= pd.to_datetime(end)]
        return df.reset_index(drop=True)

    def info(self):
        print(f"Tickers: {self.get_tickers()}")
        print(f"Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        print(f"Total rows: {len(self.df)}")
