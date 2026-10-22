# financial

A personal workspace for researching listed companies. It keeps a local database of daily prices and quarterly company accounts, checks that data for problems, and provides building blocks for analysing it in Jupyter notebooks.

## What it holds

- **Prices:** daily prices from Yahoo Finance, from 2000 where available, adjusted for dividends and splits, stored in one compact Parquet file.
- **Company accounts:** quarterly and annual figures (revenue, profit, cash flow, balance sheet, share count) from the SEC's EDGAR service, with the date each figure was first published.
- **Which companies:** a watchlist built from rules in `pipeline/watchlist-rules.csv`: whole industries, individual tickers, and named groups such as `holdings`. Candidates must pass a liquidity screen.

## Layout

```
pipeline/     fetching, building and checking the data
  Data Update.ipynb     run this to bring the data up to date
  Data Health.ipynb     run this to see what state the data is in
  watchlist-rules.csv   what to track (edit by hand)
  edgar-tags.csv        how SEC figures map to metrics (edit by hand)
data/         the database: downloaded and generated files (not in git, except the ticker registry)
*.py          analysis building blocks: loaders and indicators for prices and company accounts
research/     one notebook per study, starting from research/_template.ipynb
tests/        automated checks
```

## Using it

1. **Update the data:** open `pipeline/Data Update.ipynb`, run the setup cell, then section 2. It fetches new prices and filings and ends with a health summary. After editing `watchlist-rules.csv`, run section 1 first.
2. **Check the data:** `pipeline/Data Health.ipynb` scores every company green (use as is), amber (needs attention) or red (left out of analysis), with charts.
3. **Analyse:** copy `research/_template.ipynb`, rename it, and follow its sections. In a notebook:

```python
from pipeline.Watchlist import Watchlist
from PriceData import PriceData
from FundamentalIndicators import FundamentalIndicators

holdings = Watchlist().members('holdings')
PriceData().get_data_wide(tickers=holdings).plot()          # prices, one column per company
FundamentalIndicators().snapshot(tickers=holdings)          # growth, margins, ROIC, cash flow yield
```

Every calculation returns a table with dates down the side and companies across, so it runs over the whole watchlist at once and can be charted with `.plot()`.

## Requirements

Python 3.14 with pandas, numpy, pyarrow, yfinance, plotly, scipy, statsmodels, requests and ipykernel. Fetching SEC data requires a contact email in the request header, set in `pipeline/Edgar.py`.

## Rebuilding the data

`data/` is not in git. Running `pipeline/Data Update.ipynb` re-downloads full price histories (a few minutes per hundred companies) and SEC filings (about 2.5 MB per company). The one exception is `data/registry/tickers.csv`, the list of companies with their sector, industry and status, which cannot be downloaded again and is therefore kept in git.

## Tests

```
python tests/run_all.py
```

The tests use temporary files and simulated downloads; they never touch the real data.
