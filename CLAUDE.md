# CLAUDE.md

Personal research workspace for equity analysis. A pipeline builds a local database of Yahoo Finance prices and SEC EDGAR company accounts and scores its health; analysis modules load and transform that data; research happens in notebooks. [README.md](README.md) describes the project for a human reader.

## Working rules

[.instructions.md](.instructions.md) is binding. Read it before changing anything. In short:

- **Small steps.** More than about 30 lines in one change needs a stated reason first.
- **Spec first.** For architectural changes, outline inputs, outputs and state changes and wait for explicit approval before writing code.
- **A question is not an instruction.** Answer "what if?" and clarifying questions conceptually. Never modify code in response.
- **No unsolicited refactoring.** Propose a fix or optimisation you spot; do not slip it in alongside the requested change.
- **Test before moving on.** Run the tests, and the affected notebook cell on real data, before the next step.
- **Protect the data.** No batch deletion or `dropna` on a data file without first inspecting the impact. Dry run, then apply.
- **Replace, don't accumulate.** When something is improved, remove the old behaviour. No compatibility switches.
- **Don't engineer around individual tickers.** With thousands of tickers available, a company that mis-reports or needs special handling is tolerated or dropped, never fixed. Analysis code must not break on one. Fix a data problem only when it is systematic.
- **Build bottom-up.** Data, then validation, then utilities, then the backtest harness. Do not start a higher layer until the one below is robust. The quality bar is "robust, good enough to see broad trends", not perfection.

## Environment

- Windows, Python 3.14, packages in the **global** interpreter (pandas 3, numpy 2, pyarrow, yfinance, plotly, scipy, statsmodels, requests, ipykernel) (`C:\Users\bensi\AppData\Local\Programs\Python\Python314`). No virtual environment by decision: if a `.venv/` appears (VS Code offers one), delete it, or VS Code picks it as the notebook kernel and nothing imports.
- Tests: `python tests/run_all.py`. Plain scripts, no pytest; they use temporary files and mocked downloads and never touch real data. Every module is covered. Run them before and after any change.
- **The working directory is always the project root**, because every module reads `data/...` relative to it. Notebooks in `pipeline/` and `research/` start with a setup cell that finds the root (the folder containing `pipeline/watchlist-rules.csv`), changes to it and puts it on `sys.path`.

## Layout

```
pipeline/   YF, Edgar, EdgarStatements, Watchlist, DataHealth, EdgarHealth, TickerFiles
            Data Update.ipynb, Data Health.ipynb
            watchlist-rules.csv, edgar-tags.csv   (the only hand-edited inputs)
data/       downloaded and generated files only; never mix code in
root        PriceData, PriceIndicators, PairAnalytics, FundamentalData, FundamentalIndicators, Strategy, Backtest
research/   one notebook per study, plus _template.ipynb
tests/
```

- **Imports:** analysis modules as `from PriceData import PriceData`; pipeline modules as `from pipeline.YF import YF`. Dependencies run one way: analysis may import pipeline (`FundamentalData` uses `EdgarHealth` for bands), never the reverse.
- Code stays flat within each folder; revisit packages only at about twenty modules.
- **Git:** `data/` is ignored except `data/registry/tickers.csv`, which cannot be re-downloaded.

## Data flow

```
pipeline/watchlist-rules.csv + data/registry/tickers.csv
   └─> Watchlist.resolve() ─> YF.screen() ─> data/watchlist.csv
          ├─> YF.daily_update() ─> data/prices/yf-prices.parquet ─> DataHealth; PriceData ─> PriceIndicators, PairAnalytics
          └─> Edgar.update() ─> data/edgar/raw/*.json
                 └─> EdgarStatements.build() (+ pipeline/edgar-tags.csv) ─> data/edgar/edgar-quarterly.csv, edgar-annual.csv
                        ─> EdgarHealth; FundamentalData ─> FundamentalIndicators

Strategy (hold / top / weights, StrategyData) ─> Backtest ─> BacktestResult
```

`YF` writes one thing back to the registry: the permanent `DELISTED` status.

| File | Contents |
| :--- | :--- |
| `pipeline/watchlist-rules.csv` | `action,level,value,list`. `include` at sector, industry or ticker level; `exclude` at ticker level only; `#` comments. The optional `list` column names a sub-list; a ticker gets one line per list. New ticker groups are always sub-lists here, never separate ticker files (a separate file can name tickers nobody fetches). |
| `pipeline/edgar-tags.csv` | `metric,kind,priority,tag`: which SEC tags supply each metric. `kind` is `duration`, `instant` or `cover`. Rebuild statements after editing; no download needed. |
| `data/registry/tickers.csv` | About 7,900 companies: `ticker, zacks_x_ind_desc` (industry), `comp_name`, `comp_name_2, sector, last_update, status`. Originally from MacroTrends; no refresh path yet. |
| `data/watchlist.csv` | Generated by section 1 of `Data Update.ipynb`. Scope for both fetchers, both health reports and both loaders. |
| `data/prices/yf-prices.parquet` | One Parquet file (switched from CSV on 2026-10-21: about 5x smaller, about 20x faster to read, exact values): `ticker,date,open,high,low,close,volume,adj_close`, `date` stored as `YYYY-MM-DD` text, sorted by ticker then date. Read it with `pd.read_parquet`, or through `PriceData`. The old CSV is kept as `yf-prices.backup-20261021.csv` until the owner deletes it. Tickers no longer on the watchlist (failed the liquidity screen) are kept but not updated. |
| `data/prices/yf-manifest.csv` | Per ticker: `last_date, last_attempt, outcome, rows_fetched, last_verified`. |
| `data/prices/yf-screen.csv` | Liquidity screen results per candidate. |
| `data/edgar/raw/*.json` | SEC company facts as downloaded, one per ticker (about 2.5 MB). Never edited. |
| `data/edgar/edgar-manifest.csv`, `edgar-tickers.csv` | Fetch outcomes per ticker; SEC's ticker-to-CIK list (refreshed every 30 days). |
| `data/edgar/edgar-quarterly.csv`, `edgar-annual.csv` | Generated in full by `EdgarStatements.build()`: `ticker, period_end, filed, currency` then `revenue, op_income, pretax_income, tax, net_income, cfo, capex, assets, equity, cash, debt_long, debt_short, shares`, in whole currency units. Built from every raw file, including tickers since removed from the watchlist. |
| `data/macrotrends/` | Retired MacroTrends data (statements for about 6,000 companies, old prices). Nothing reads it. The owner keeps it as a fallback until EDGAR coverage grows: do not delete it, never commit it. |

## The standard calculation pattern (agreed for all layers)

- One function computes one metric and returns one wide table: dates down, tickers across. No function returns a bundle of metrics, and there is no helper returning a dictionary of tables.
- Only two shapes exist: the time series (dates × tickers) and the snapshot (tickers × metrics at one date). A snapshot defaults to every metric and every included ticker.
- Every function takes the same optional narrowing arguments (`tickers`, `as_of`, `period` where relevant) and defaults to all tickers. Narrow to a group with `tickers=Watchlist().members('holdings')`.
- A missing or unusable input gives a blank (NaN) for that ticker; one company never raises or breaks a calculation.
- Fundamentals come in two forms: as reported (indexed by period end) and daily as known (indexed by trading day, each day holding the latest value filed by then). The daily form is how fundamentals and prices combine; point-in-time correctness lives there, not in notebooks.
- No general plotting class (one was deleted): a wide table charts itself with `pd.options.plotting.backend = 'plotly'` and `table.plot()`. A job-specific chart lives with its job.

## Health bands (both reports)

Green = use as is; amber = needs attention; red = exclude from analysis. Section 2 of `Data Update.ipynb` saves both sets of scores (`save()`); `FundamentalData` and the backtest's red-ticker filter read the saved files (`saved()`), so they start instantly and need the update to have run once. Passing a health object explicitly (as the tests do) computes live instead. The Data Health notebook always computes live. Score only what should push a ticker towards attention or exclusion; things merely worth knowing (a missing debt figure, a restated year) are reported in `issues` and `summary` with no penalty. Loaders leave red tickers out by default.

## Conventions that are easy to get wrong

- **Ticker symbols:** the registry uses `.` for share classes (`BRK.B`); Yahoo, the watchlist and the price file use `-` (`BRK-B`).
- **One-file-per-ticker data** (today `data/edgar/raw/`) gets its file names only from `pipeline/TickerFiles.py`: `path_for` when building a path, `tickers_in`/`ticker_from` when listing. Names Windows reserves (`CON`, `NUL`, `COM1`...) are stored with a leading underscore (`_CON.json`) and read back as the plain ticker. Current Windows 11 accepts those names with an extension, older Windows and some tools do not; everything outside that module works with plain tickers.
- **Prices lag by 7 days on purpose** (`YF.LAG_DAYS`, matched by `DataHealth.LAG_DAYS`) so a part-finished trading day is never stored.
- **`PriceData` defaults to `adj_close`.** Use `price_col='close'` only where the raw price is needed, such as market value (price × unadjusted share count).
- **Data files are written atomically** (temp file, then rename). Prices go through `YF._write_prices`, which refuses a write that would leave any ticker with fewer rows than on disk. A Parquet file cannot be edited in place, so every change rewrites the whole price file: `YF.daily_update()` therefore gathers new rows, drift repairs and verification repairs in memory and saves once at the end (about 6 seconds at 3,000 tickers). Keep it that way: never save inside a per-ticker loop. `DELISTED` is only written after two `delisted` results on different days.
- **`DataHealth` cannot see stale adjusted prices or a short history** (a ticker that rejoined the watchlist with a few old rows gets only an incremental fetch). Both are caught by `YF.verify_sample`, which compares the full history of 8 tickers with Yahoo on every update; repair one ticker at once with `refresh_ticker(t, apply=True)`.
- **EDGAR values are as most recently reported** (restatements win), so each series is internally consistent; `filed` is when the period was first published, which is what point-in-time work lags on. A `filed` more than a year after `period_end` means the period appeared only as a comparative.
- **How `EdgarStatements` chooses values:** only financial reports count (proxy statements are ignored); within a tag the latest filing wins; between tags the priority decides, except that a tag the company stopped using (more than 200 days behind) only fills gaps. A quarter derived from year-to-date figures pairs each figure with the version filed closest in time to the other. "Newest filing wins across tags" was tried and rejected: metrics flip between tags with different meanings.
- **Restated years:** after selling a business a company restates the year but not the earlier quarters, so quarters no longer sum to the annual figure. This is reported, not scored, and cannot be reconciled from SEC data.
- **Foreign filers** (20-F, 40-F) have annual rows only, and some report in another currency (`currency` column; BABA reports in CNY). `FundamentalData` leaves out non-USD reporters by default. A few foreign filers are stale at the SEC itself (DOX, GFS, INFY), not because of mapping.
- **`cash` is cash and equivalents only**; ROIC counts a missing debt figure as zero (debt tags are the weakest mapping), and `fcf_yield` is against equity value, so indebted companies show very high yields.
- **Companies with more than one share class** (MA, META) have no usable share count from the SEC API, so `fcf_yield` is blank for them; treat GOOGL's with suspicion.
- **SEC requests** must carry a contact email in the User-Agent (`Edgar.USER_AGENT`). The owner approved `ben.silburn@gmail.com` for this and nothing else.

## Modules

| Module | Role |
| :--- | :--- |
| [pipeline/YF.py](pipeline/YF.py) | Yahoo prices. `daily_update()` (incremental with a 10-day overlap; full refresh after dividend or split drift, at most 20 per run; the verification sample; one save at the end), `verify_sample()` on its own, `refresh_ticker(t, apply=False)` (saves at once: a single manual repair), `screen()` (liquidity: median daily dollar volume at least $10M, price at least $2), `clean_database()`. |
| [pipeline/Edgar.py](pipeline/Edgar.py) | SEC fetcher. `update()` downloads raw company facts for the watchlist, skips tickers with a settled answer in the last 7 days, and only replaces a stored file with a sound download that differs from it (the SEC returns byte-identical files when nothing changed, so an unchanged file keeps its timestamp and the statement builder skips it). |
| [pipeline/EdgarStatements.py](pipeline/EdgarStatements.py) | `build()` brings both statement files up to date, re-parsing only tickers whose raw file changed (about 0.8 s each) and keeping the others' rows; the result equals a full rebuild. Every ticker is re-parsed after an edit to `edgar-tags.csv`, when `data/edgar/edgar-build.csv` (what each ticker was built from: file timestamp, size, tag-map hash) is missing, or with `build(full=True)`. `last_rebuilt` lists what the last build re-parsed. `build(tickers=[...])` returns tables without writing. |
| [pipeline/TickerFiles.py](pipeline/TickerFiles.py) | `path_for(folder, ticker, ext)`, `ticker_from(path)`, `tickers_in(folder, ext)`: the only conversion between tickers and file names. |
| [pipeline/Watchlist.py](pipeline/Watchlist.py) | `resolve()` expands the rules against the registry; `members(name)` and `lists()` serve sub-lists, limited to tickers on the generated watchlist. |
| [pipeline/DataHealth.py](pipeline/DataHealth.py) | Price health: freshness, coverage, gaps, bad bars, adjustment mismatches, zero volume, flat closes. `score`, `summary`, `problems`, `treemap`, `heatmap(by=None)` (one row per ticker up to 150 tickers, per industry above that), `save()` / `saved()` (`data/prices/price-health.csv`). SPY's dates are the trading calendar. |
| [pipeline/EdgarHealth.py](pipeline/EdgarHealth.py) | Fundamentals health over the last 5 years: lateness, core-metric coverage (`revenue, net_income, cfo, assets, equity`), missing periods, negative revenue, revenue jumps. Quarterly filers scored on quarters, annual filers on annual rows; tickers with no SEC statements are `n/a`. Same methods as `DataHealth`, saving to `data/edgar/edgar-health.csv`. |
| [PriceData.py](PriceData.py) | Price loader: `get_data_wide`, `get_data_long`, `get_data`, `split_inout_sample`. Watchlist scope by default; `tickers_file=None` for everything in the file. |
| [PriceIndicators.py](PriceIndicators.py) | `sma`, `ema`, `daily_returns`, `cumulative_returns`, `rolling_volatility` (daily, not annualised), `rsi` (standard Wilder), `momentum`, `correlation`. |
| [PairAnalytics.py](PairAnalytics.py) | Two-column tables (ticker held, hedge): `cointegration_test`, `hedge_ratio`, `spread`, `zscore`, `generate_signals`. The default whole-history hedge ratio uses future prices and is for exploring only; pass `window`/`hedge_window` in a backtest. A pairs backtest must also size the two legs by the ratio. |
| [FundamentalData.py](FundamentalData.py) | Statement loader: `get_data_wide(metric, period)`, `get_data_long(period)` with `period` = `quarterly`, `annual` or `ttm`; `as_of=` for point-in-time; `to_daily()`; `filers()`. Excludes red tickers and non-USD reporters by default (`include_red=True`, `currency=None`); quarterly and ttm views hold quarterly filers only. Wide tables are indexed by actual period end, so different fiscal calendars leave gaps. |
| [FundamentalIndicators.py](FundamentalIndicators.py) | `revenue_growth`, `operating_margin`, `net_margin`, `fcf_margin`, `roic`, `fcf_yield`, each with `period` (`ttm` or `annual`), `tickers`, `as_of`, `daily`; `snapshot()`; `INDICATORS`. The reference implementation of the standard pattern. |
| [Strategy.py](Strategy.py) | What a strategy author imports. A strategy is `function(data, **params)` returning `hold(signal)` (equal weights where true), `top(score, n)` (the n highest scores, equal weights) or `weights(table)` (exact, negative = short); a plain table is taken as weights. `data` (`StrategyData`) offers `prices` (adjusted), `close`, `volume`, `benchmark` (SPY), `universe`, `price_ind`, `fund_ind` (use `daily=True`) and `fundamental(metric, period)`, all on the trading-day calendar over the full history so indicators can warm up. Equal weights are shared only among tickers eligible that day. |
| [Backtest.py](Backtest.py) | `Backtest(strategy, tickers=None, start='2010-01-01', end=None, rebalance='monthly', cost=0.001, **params).run()` returns a `BacktestResult` (`equity` from 1, `returns`, `holdings`, `turnover`, `costs`, `holdings_count`, `settings`). Targets are read on rebalance dates (first test day, then each period end) and traded at the next day's close; holdings drift between rebalances; cost = rate × value traded; a price gap keeps its return; a holding whose prices end is sold at its last price into cash (0% interest). `universe()`: watchlist tickers except SPY and red ones, eligible on a day if they have a price and passed the liquidity screen as of then (median dollar volume ≥ $10M and last close ≥ $2 over the preceding 60 days, at least 40 of them with data). Every run also carries `result.baselines`: 'Equal-weight universe' (every eligible ticker equally, same schedule and costs) and 'SPY' (bought at the strategy's first trade with the same cost, then held; omitted if SPY has no prices). `result.metrics()` is a snapshot, one row per run (strategy first): `total_return, annual_return, volatility, sharpe, worst_drawdown, longest_drawdown_days, turnover_per_year, cost_per_year, avg_holdings`, as fractions, per calendar year, Sharpe with cash at 0%. `result.tear_sheet()` shows the formatted metrics, `figure()` (growth of $1 on a log scale, drawdown, rolling one-year return), `calendar_returns()` with a strategy-minus-equal-weight column, `contributions()` (each ticker's summed daily contributions: relative importance, not additive to the total) and `caveats()` (period, costs, settings, the hindsight in the universe, a dominant single stock, few holdings). In a plain script it prints text instead of notebook output. Image export of the figure would need the `kaleido` package, which is not installed. |

## Notebooks

- [pipeline/Data Update.ipynb](pipeline/Data%20Update.ipynb): section 1 builds the watchlist from the rules; section 2 updates prices and fundamentals and prints both health summaries; section 3 holds optional maintenance commands.
- [pipeline/Data Health.ipynb](pipeline/Data%20Health.ipynb): the full price and fundamentals health reports with charts. Read-only.
- **Research:** one study per notebook, holding idea, code, results and conclusion (a separate research log was rejected as duplication). New studies start from [research/_template.ipynb](research/_template.ipynb): a Summary table written last, a setup cell, then Universe and data, Method, Results, Checks, Conclusion. The template is a deliberate MVP to evolve through real use; do not elaborate it or add tooling (such as an index) before there is a need.
- **Frozen studies:** `Simple-Momentum`, `Regime_Momentum_Strategy`, `MomentumAnalysis`, `VolatilityOptimization`, `Quality_Stock_Selection` predate the current modules and do not run; never update their code. Each has a Summary cell whose verdict is a draft awaiting the owner's confirmation. A study worth revisiting is redone from the template. Some are 1 to 2 MB of stored charts: extract cell sources with a short `json` script instead of reading them whole.

## Research cautions the owner cares about

Validate out of sample; include realistic costs; test cointegration, not correlation, for pairs; check parameter sensitivity; check that the result survives removing the best few trades; and remember the watchlist is itself a survivorship-biased universe chosen with hindsight. The template's Checks section asks these questions.
