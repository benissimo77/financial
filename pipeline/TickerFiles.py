"""The one place where a ticker becomes a file name and back, for data stored one file per ticker.

Windows refuses some file names whatever the extension (CON, PRN, AUX, NUL, COM1-9, LPT1-9), and some of them are real
tickers. They are stored with a leading underscore (CON -> _CON.json). Call these functions only when building a path or
listing a folder: everywhere else, code works with the plain ticker.
"""
import os

RESERVED = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}
PREFIX = '_'
FORBIDDEN = set('\\/:*?"<>|')


def path_for(folder: str, ticker: str, extension: str) -> str:
    """The file that holds this ticker's data, e.g. path_for('data/edgar/raw', 'CON', '.json') -> data/edgar/raw/_CON.json."""
    bad = FORBIDDEN & set(ticker)
    if bad or not ticker:
        raise ValueError(f"Ticker {ticker!r} cannot be a file name (characters {sorted(bad)}).")
    stem = PREFIX + ticker if ticker.upper() in RESERVED else ticker
    return os.path.join(folder, stem + extension)


def ticker_from(path: str) -> str:
    """The ticker a file holds: the reverse of path_for."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem[len(PREFIX):] if stem.startswith(PREFIX) and stem[len(PREFIX):].upper() in RESERVED else stem


def tickers_in(folder: str, extension: str) -> list:
    """Sorted tickers that have a file with this extension in the folder."""
    return sorted(ticker_from(f) for f in os.listdir(folder) if f.endswith(extension)) if os.path.isdir(folder) else []
