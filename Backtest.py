import pandas as pd
import numpy as np

class Backtest:
    """
    Generic backtesting engine for portfolio strategies.
    Accepts any price DataFrame and a strategy function that returns weights/signals.
    Supports trading costs, slippage, and detailed logging for validation/debugging.
    """
    def __init__(self, prices, initial_capital=1, trading_cost=0.0, slippage=0.0, log_trades=False):
        self.prices = prices.copy()
        self.initial_capital = initial_capital
        self.trading_cost = trading_cost  # as a fraction (e.g., 0.001 for 0.1% per trade)
        self.slippage = slippage          # as a fraction (e.g., 0.001 for 0.1% per trade)
        self.log_trades = log_trades
        self.trade_log = []

    def run(self, strategy_fn, strategy_kwargs=None):
        """
        Run the backtest using the supplied strategy function.
        strategy_fn(prices, **strategy_kwargs) -> DataFrame of weights (indexed by date, columns = tickers)
        Returns a dict with a single DataFrame ('timeseries') containing all time-series outputs, and (optionally) a trade log.
        """
        if strategy_kwargs is None:
            strategy_kwargs = {}
        prices = self.prices
        returns = prices.pct_change().fillna(0)
        weights = strategy_fn(prices, **strategy_kwargs)
        weights = weights.shift().fillna(0)  # Use yesterday's weights for today's returns
        # Calculate portfolio returns
        port_ret = (weights * returns).sum(axis=1)
        # Trading costs: sum of absolute weight changes * trading_cost
        weight_diff = weights.diff().abs().sum(axis=1)
        cost = weight_diff * self.trading_cost
        # Slippage: apply to gross returns
        slip = np.abs(port_ret) * self.slippage
        net_ret = port_ret - cost - slip
        equity = (net_ret + 1).cumprod() * self.initial_capital
        # Build main timeseries DataFrame
        timeseries = pd.DataFrame({
            'portfolio_value': equity,
            'return_net': net_ret,
            'return_gross': port_ret,
            'cost': cost,
            'slippage': slip
        }, index=prices.index)
        # Optional: log trades/portfolio for debugging
        trade_log = None
        if self.log_trades:
            trade_log = []
            for date in weights.index:
                trade_log.append({
                    'date': date,
                    'weights': weights.loc[date].to_dict(),
                    'returns': returns.loc[date].to_dict(),
                    'portfolio_return': port_ret.loc[date],
                    'cost': cost.loc[date],
                    'slippage': slip.loc[date],
                    'net_return': net_ret.loc[date],
                    'equity': equity.loc[date]
                })
        return {
            'timeseries': timeseries,
            'weights': weights,
            'trade_log': trade_log
        }
