"""Backtest: runs a strategy over a point-in-time universe and records what a real portfolio would have done.

    from Backtest import Backtest
    result = Backtest(momentum, n=15).run()

How it trades:
- A strategy's targets are read on each rebalance date and traded at the next day's close (one day of delay, so no look-ahead).
- Between rebalances holdings drift with their prices; nothing is rebalanced for free.
- Each trade costs COST times the value traded.
- A missing price day keeps its return: the value catches up when trading resumes.
- A holding whose prices stop is sold at its last price and the money held as cash. Cash earns nothing.
"""
from typing import Callable, Optional, Union, List
import numpy as np
import pandas as pd
from PriceData import PriceData
from Strategy import Target, StrategyData, hold, weights as as_weights


class BacktestResult:
    """Daily record of a run. Every table follows the standard pattern: trading days down, tickers across."""
    TRADING_DAYS = 252

    def __init__(self, name: str, equity: pd.Series, holdings: pd.DataFrame, turnover: pd.Series, costs: pd.Series, settings: dict,
                 contribution: Optional[pd.DataFrame] = None):
        self.name = name
        self.equity = equity          # portfolio value, starting at 1
        self.returns = equity.pct_change().fillna(0.0)
        self.holdings = holdings      # weight of each ticker at each day's close, after any trade
        self.turnover = turnover      # value traded as a share of the portfolio (buys plus sells), on trading days
        self.costs = costs            # trading costs as a share of the portfolio
        self.settings = settings
        self.baselines = {}           # name -> BacktestResult; filled by Backtest.run() for the strategy's result only
        self.contribution = contribution   # each ticker's share of each day's return (weight held into the day x its return)

    @property
    def holdings_count(self) -> pd.Series:
        return (self.holdings.abs() > 1e-12).sum(axis=1)

    def _measures(self) -> dict:
        e, r = self.equity, self.returns
        years = (e.index[-1] - e.index[0]).days / 365.25
        underwater = e < e.cummax()
        spell = underwater.ne(underwater.shift()).cumsum()[underwater]   # numbers each unbroken stretch below a previous peak
        longest = max(((g.index[-1] - g.index[0]).days + 1 for _, g in e[underwater].groupby(spell)), default=0)
        invested = self.holdings_count[self.turnover.gt(0).cumsum() > 0]   # from the first trade on
        vol = r.std() * np.sqrt(self.TRADING_DAYS)
        return {'total_return': e.iloc[-1] / e.iloc[0] - 1, 'annual_return': (e.iloc[-1] / e.iloc[0]) ** (1 / years) - 1,
                'volatility': vol, 'sharpe': r.mean() * self.TRADING_DAYS / vol if vol > 0 else np.nan,
                'worst_drawdown': (e / e.cummax() - 1).min(), 'longest_drawdown_days': longest,
                'turnover_per_year': self.turnover.sum() / years, 'cost_per_year': self.costs.sum() / years,
                'avg_holdings': invested.mean() if len(invested) else 0.0}

    def metrics(self) -> pd.DataFrame:
        """A snapshot: one row for the strategy and one per baseline, one column per measure. Returns and costs are fractions
        (0.12 = 12%); annual figures are per calendar year; volatility and Sharpe are annualised from daily returns, with cash at 0%."""
        runs = [self] + list(self.baselines.values())
        return pd.DataFrame([r._measures() for r in runs], index=pd.Index([r.name for r in runs], name='run'))

    # ---- tear sheet
    EQUAL_WEIGHT = 'Equal-weight universe'
    COLOURS = ['#1f77b4', '#7f7f7f', '#bcbd22']   # strategy, equal weight, benchmark

    def _runs(self) -> list:
        return [self] + list(self.baselines.values())

    def calendar_returns(self) -> pd.DataFrame:
        """Return in each calendar year (a part year at either end), one column per run, as fractions."""
        return pd.DataFrame({r.name: r.equity.groupby(r.equity.index.year).last() / r.equity.groupby(r.equity.index.year).last().shift().fillna(r.equity.iloc[0]) - 1
                             for r in self._runs()}).rename_axis('year')

    def contributions(self) -> pd.Series:
        """Each ticker's total contribution to the strategy's return (sum of its daily contributions), largest first.
        Simple addition, not compounded, so the figures show relative importance rather than adding up to the total return."""
        if self.contribution is None:
            return pd.Series(dtype=float)
        total = self.contribution.sum()
        return total[total != 0].sort_values(ascending=False)

    def caveats(self) -> list:
        """Plain-language warnings that apply to this run."""
        st = self.settings
        years = (st['end'] - st['start']).days / 365.25
        notes = [f"Period {st['start'].date()} to {st['end'].date()} ({years:.1f} years), rebalanced {st['rebalance']}, trading costs {100 * st['cost']:.2f}% of each trade, cash earns nothing."]
        if st.get('params'):
            notes.append("Strategy settings: " + ', '.join(f"{k}={v}" for k, v in st['params'].items()) + ". One setting proves little: check that nearby values give similar results.")
        notes.append(f"Universe: {st['tickers']} tickers from today's watchlist, {st.get('eligible_start', '?')} of them eligible at the start and {st.get('eligible_end', '?')} at the end. "
                     "Companies that failed or were delisted are missing, which flatters every strategy and the equal-weight baseline alike: judge the strategy against the equal-weight baseline, not the market.")
        c = self.contributions()
        gains = c[c > 0]
        if len(gains) and gains.iloc[0] > 0.25 * gains.sum():
            notes.append(f"{gains.index[0]} made {100 * gains.iloc[0] / gains.sum():.0f}% of all the gains. Check that the result survives without it.")
        m = self._measures()
        if 0 < m['avg_holdings'] < 5:
            notes.append(f"On average only {m['avg_holdings']:.1f} stocks were held, so the result depends heavily on a few names.")
        return notes

    def figure(self):
        """Growth of $1 (log scale), drawdowns and rolling one-year return for the strategy and its baselines, as one Plotly figure."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.5, 0.25, 0.25],
                            subplot_titles=('Growth of $1', 'Drawdown from previous peak', 'Rolling one-year return'))
        for r, colour in zip(self._runs(), self.COLOURS):
            style = dict(color=colour, width=2 if r is self else 1.2, dash='dot' if r.name not in (self.name, self.EQUAL_WEIGHT) else None)
            fig.add_trace(go.Scatter(x=r.equity.index, y=r.equity, name=r.name, legendgroup=r.name, line=style), row=1, col=1)
            fig.add_trace(go.Scatter(x=r.equity.index, y=r.equity / r.equity.cummax() - 1, name=r.name, legendgroup=r.name, showlegend=False, line=style), row=2, col=1)
            fig.add_trace(go.Scatter(x=r.equity.index, y=r.equity.pct_change(self.TRADING_DAYS), name=r.name, legendgroup=r.name, showlegend=False, line=style), row=3, col=1)
        fig.update_yaxes(type='log', row=1, col=1)
        fig.update_yaxes(tickformat='.0%', row=2, col=1)
        fig.update_yaxes(tickformat='.0%', row=3, col=1)
        fig.update_layout(height=850, title=f"{self.name}: backtest {self.settings['start'].date()} to {self.settings['end'].date()}",
                          hovermode='x unified', legend=dict(orientation='h', y=1.06, x=0), margin=dict(t=110, l=60, r=20, b=40))
        return fig

    def tear_sheet(self, top_n: int = 10):
        """Show the full report: summary table, charts, calendar years, biggest contributors and detractors, and caveats."""
        try:
            from IPython import get_ipython
            from IPython.display import display, Markdown
            in_notebook = get_ipython() is not None
        except ImportError:
            in_notebook = False
        if not in_notebook:   # a plain script: print full tables and plain headings
            display = lambda obj: print(obj.to_string() if hasattr(obj, 'to_string') else obj, end='\n\n')
            Markdown = lambda text: text
        pct = lambda t: t.map(lambda v: f"{100 * v:.1f}%" if pd.notna(v) else '')
        m = self.metrics()
        table = pd.concat([pct(m[['total_return', 'annual_return', 'volatility', 'worst_drawdown', 'turnover_per_year', 'cost_per_year']]),
                           m[['sharpe']].round(2), m[['longest_drawdown_days']].astype(int), m[['avg_holdings']].round(1)], axis=1)
        table.columns = ['Total return', 'Annual return', 'Volatility', 'Worst drawdown', 'Turnover / year', 'Costs / year', 'Sharpe', 'Longest drawdown (days)', 'Avg holdings']
        display(Markdown(f"## {self.name}"))
        display(table)
        self.figure().show()
        years = self.calendar_returns()
        if self.EQUAL_WEIGHT in years:
            years[f'{self.name} minus equal weight'] = years[self.name] - years[self.EQUAL_WEIGHT]
        display(Markdown('### Calendar-year returns'))
        display(pct(years))
        c = self.contributions()
        display(Markdown(f'### Biggest contributors and detractors (share of daily returns, summed)'))
        best, worst = c[c > 0].head(top_n), c[c < 0].tail(top_n)[::-1]
        display(pd.DataFrame({'Contributors': best.index, 'Added': pct(best).values}).join(
            pd.DataFrame({'Detractors': worst.index, 'Lost': pct(worst).values}), how='outer').fillna(''))
        display(Markdown('### Caveats\n' + '\n'.join(f'- {note}' for note in self.caveats())))


class Backtest:
    START = '2010-01-01'
    COST = 0.001                  # per unit of value traded: commission, spread and slippage for liquid stocks
    REBALANCE = {'daily': None, 'weekly': 'W', 'monthly': 'M', 'quarterly': 'Q'}
    BENCHMARK = 'SPY'             # a fund, not a company: excluded from the universe, available as data.benchmark
    LIQUIDITY_DAYS = 60           # point-in-time liquidity screen, measured over the days before each date
    LIQUIDITY_MIN_DAYS = 40       # trading days with data needed inside that window, so a short gap does not exclude a ticker
    MIN_DOLLAR_VOLUME = 10_000_000
    MIN_PRICE = 2.0

    def __init__(self, strategy: Callable, tickers: Optional[Union[str, List[str]]] = None, start: str = START, end: Optional[str] = None,
                 rebalance: str = 'monthly', cost: float = COST, prices: Optional[PriceData] = None, health=None, **params):
        """
        strategy  function(data, **params) returning hold(...), top(...), weights(...) or a plain table of weights.
        tickers   narrow the universe (default: every watchlist ticker with prices, except the benchmark).
        params    passed to the strategy, e.g. Backtest(momentum, n=15).
        prices, health  data sources; defaults are the project's PriceData and DataHealth (used to leave out red tickers).
        """
        if rebalance not in self.REBALANCE:
            raise ValueError(f"Unknown rebalance '{rebalance}'. Allowed: {list(self.REBALANCE)}.")
        self.strategy, self.params = strategy, params
        self.start, self.end, self.rebalance, self.cost = pd.Timestamp(start), (pd.Timestamp(end) if end else None), rebalance, cost
        self.prices = prices if prices is not None else PriceData()
        self.health = health
        available = [t for t in self.prices.get_tickers() if t != self.BENCHMARK]
        wanted = available if tickers is None else ([tickers] if isinstance(tickers, str) else list(tickers))
        self.tickers = [t for t in wanted if t in available]

    def _red_tickers(self) -> set:
        """Tickers red in the price health report: from the scores saved by the last update, unless a health object was passed in."""
        if self.health is None:
            from pipeline.DataHealth import DataHealth
            scored = DataHealth(data_file=self.prices.data_file).saved()
        else:
            scored = self.health.score()
        return set(scored.index[scored['band'] == 'red'])

    def universe(self) -> pd.DataFrame:
        """True where a ticker may be held that day: it has a price, is not red in the price health report, and passed the
        liquidity screen as it stood then (median dollar volume and last price over the preceding days only)."""
        adj = self.prices.get_data_wide(tickers=self.tickers)
        close = self.prices.get_data_wide(price_col='close', tickers=self.tickers).reindex_like(adj)
        volume = self.prices.get_data_wide(price_col='volume', tickers=self.tickers).reindex_like(adj)
        dollar_volume = (close * volume).rolling(self.LIQUIDITY_DAYS, min_periods=self.LIQUIDITY_MIN_DAYS).median().shift(1)
        liquid = (dollar_volume >= self.MIN_DOLLAR_VOLUME) & (close.ffill().shift(1) >= self.MIN_PRICE)
        alive = adj.notna() | (adj.ffill().notna() & adj.bfill().notna())   # trading, or inside a gap between prices
        eligible = liquid & alive
        red = self._red_tickers()
        return eligible.loc[:, [t for t in eligible.columns if t not in red]]

    def _decision_dates(self, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """The first trading day of the test, then the last trading day of each period. Targets are read on these dates."""
        freq = self.REBALANCE[self.rebalance]
        if freq is None:
            return dates
        last_of_period = pd.Series(dates, index=dates).groupby(dates.to_period(freq)).max()
        return pd.DatetimeIndex([dates[0]]).union(pd.DatetimeIndex(last_of_period.values))

    def _simulate(self, name: str, wanted: pd.DataFrame, adj: pd.DataFrame, dates: pd.DatetimeIndex, settings: dict) -> BacktestResult:
        """Trade the wanted weights on the schedule, day by day, and record the portfolio."""
        returns = adj.ffill().pct_change().reindex(dates).fillna(0.0).to_numpy()   # a gap's move lands on the day trading resumes
        last_price = adj.apply(lambda s: s.last_valid_index())
        dead_after = np.array([pd.Timestamp(d) if pd.notna(d) else pd.Timestamp.min for d in last_price.values])
        trade_on = {dates[dates.get_loc(d) + 1]: d for d in self._decision_dates(dates) if dates.get_loc(d) + 1 < len(dates)}
        wanted = wanted.reindex(columns=adj.columns).fillna(0.0)
        held, cash = np.zeros(len(adj.columns)), 1.0      # value in each ticker, and cash; the portfolio starts at 1 in cash
        equity, turnover, costs, book, contrib = [], [], [], [], []
        for i, day in enumerate(dates):
            opening = held.sum() + cash
            contrib.append(held / opening * returns[i] if opening > 0 else held * 0)
            held = held * (1 + returns[i])                # today's price moves on yesterday's holdings
            stopped = (dead_after < day) & (held != 0)    # prices have ended: sold at the last price
            if stopped.any():
                cash += held[stopped].sum(); held[stopped] = 0.0
            value = held.sum() + cash
            traded = cost = 0.0
            if day in trade_on:
                w = wanted.loc[trade_on[day]].to_numpy()
                w = np.where(dead_after < day, 0.0, w)
                traded = np.abs(w * value - held).sum() / value if value > 0 else 0.0
                cost = traded * self.cost
                value *= (1 - cost)
                held, cash = w * value, value * (1 - w.sum())
            equity.append(value); turnover.append(traded); costs.append(cost)
            book.append(held / value if value > 0 else held * 0)
        return BacktestResult(name, pd.Series(equity, index=dates), pd.DataFrame(book, index=dates, columns=adj.columns),
                              pd.Series(turnover, index=dates), pd.Series(costs, index=dates), settings,
                              pd.DataFrame(contrib, index=dates, columns=adj.columns))

    def run(self) -> BacktestResult:
        """Run the strategy and its two baselines: the equal-weight universe (same schedule and costs) and the benchmark bought and held."""
        eligible = self.universe()
        data = StrategyData(self.prices, list(eligible.columns), eligible, self.BENCHMARK)
        target = self.strategy(data, **self.params)
        if not isinstance(target, Target):
            target = as_weights(target)   # a plain table is taken as weights
        adj = data.prices
        dates = adj.index[(adj.index >= self.start) & ((adj.index <= self.end) if self.end is not None else True)]
        if len(dates) < 2:
            raise ValueError("Fewer than two trading days between start and end.")
        counts = eligible.reindex(dates).sum(axis=1)
        settings = dict(start=dates[0], end=dates[-1], rebalance=self.rebalance, cost=self.cost, tickers=len(eligible.columns), params=self.params,
                        eligible_start=int(counts.iloc[0]), eligible_end=int(counts.iloc[-1]))

        result = self._simulate(getattr(self.strategy, '__name__', 'strategy'), target.to_weights(eligible), adj, dates, settings)
        result.baselines['Equal-weight universe'] = self._simulate('Equal-weight universe', hold(eligible).to_weights(eligible), adj, dates, settings)
        bench = data.benchmark.reindex(dates).ffill()
        first_trade = dates[1]   # bought with the same one-day delay and cost as the strategy's first trade, then held
        if pd.notna(bench.loc[first_trade]):
            bought = pd.Series(dates >= first_trade, index=dates)
            equity = (bench / bench.loc[first_trade] * (1 - self.cost)).where(bought, 1.0)
            traded = pd.Series(0.0, index=dates); traded.loc[first_trade] = 1.0
            result.baselines[self.BENCHMARK] = BacktestResult(self.BENCHMARK, equity, bought.astype(float).to_frame(self.BENCHMARK),
                                                              traded, traded * self.cost, settings)
        return result
