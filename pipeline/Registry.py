"""The ticker registry: every operating company with common shares on NYSE or Nasdaq, built from the SEC's own data.

Sources: the SEC's exchange ticker list (ticker, company ID, name, exchange; refreshed every LIST_DAYS) and each company's
SEC record (industry code and entity type; fetched once per company and cached). Funds, over-the-counter listings,
blank-check companies, and warrants, units, rights and preferred shares are left out. Foreign companies listed in the US
are kept (filer = foreign): the SEC gives them the same entity type as funds, so funds are recognised by their industry
code instead (none at all, or a fund code). Sectors are the Fama-French 12
groups, derived from the SEC industry (SIC) code.
"""
import os
import re
import time
from datetime import datetime
import pandas as pd
import requests
from pipeline.Edgar import Edgar


# Fama-French 12 industries: (sector, [(first SIC, last SIC), ...]). Any SIC code not listed is 'Other'.
FAMA_FRENCH_12 = [
    ('Consumer Non-Durables', [(100, 999), (2000, 2399), (2700, 2749), (2770, 2799), (3100, 3199), (3940, 3989)]),
    ('Consumer Durables', [(2500, 2519), (2590, 2599), (3630, 3659), (3710, 3711), (3714, 3714), (3716, 3716), (3750, 3751), (3792, 3792),
                           (3900, 3939), (3990, 3999)]),
    ('Manufacturing', [(2520, 2589), (2600, 2699), (2750, 2769), (3000, 3099), (3200, 3569), (3580, 3629), (3700, 3709), (3712, 3713),
                       (3715, 3715), (3717, 3749), (3752, 3791), (3793, 3799), (3830, 3839), (3860, 3899)]),
    ('Energy', [(1200, 1399), (2900, 2999)]),
    ('Chemicals', [(2800, 2829), (2840, 2899)]),
    ('Business Equipment', [(3570, 3579), (3660, 3692), (3694, 3699), (3810, 3829), (7370, 7379)]),
    ('Telecom', [(4800, 4899)]),
    ('Utilities', [(4900, 4949)]),
    ('Shops', [(5000, 5999), (7200, 7299), (7600, 7699)]),
    ('Healthcare', [(2830, 2839), (3693, 3693), (3840, 3859), (8000, 8099)]),
    ('Finance', [(6000, 6999)]),
]


def sector_for(sic) -> str:
    """The Fama-French 12 sector for a SIC code ('Other' if it falls in none of the eleven named groups)."""
    try:
        code = int(sic)
    except (TypeError, ValueError):
        return 'Other'
    for sector, ranges in FAMA_FRENCH_12:
        if any(lo <= code <= hi for lo, hi in ranges):
            return sector
    return 'Other'


class Registry:
    LIST_URL = 'https://www.sec.gov/files/company_tickers_exchange.json'
    RECORD_URL = 'https://data.sec.gov/submissions/CIK{cik:010d}.json'
    EXCHANGES = {'NYSE', 'Nasdaq'}
    EXCLUDED_SIC = {6770, 6221, 6722, 6726}   # blank checks (SPACs); commodity trusts; open-end funds; closed-end funds and unit trusts
    NOT_COMMON = re.compile(r'-(?:W|WS|WT[A-Z]?|U|UN|R|RI|RT|P[A-Z]?)$')   # warrants, units, rights and preferred shares
    # Nasdaq marks other securities with a fifth letter: W warrant, U unit, R right (always dropped); Z, P, O, N, M other
    # securities and preferreds (dropped when the company's four-letter ticker is listed too). Share classes such as GOOGL stay.
    FIFTH_ALWAYS, FIFTH_WITH_BASE = set('WUR'), set('ZPONM')
    LIST_DAYS = 30                # refresh the exchange list when the registry is older than this
    SAVE_EVERY = 250              # save fetched company records this often, so an interrupted first build keeps its progress
    COLUMNS = ['ticker', 'cik', 'name', 'exchange', 'filer', 'sic', 'industry', 'sector', 'status', 'last_update']

    def __init__(self, registry_file='data/registry/tickers.csv', companies_file='data/registry/sec-companies.csv'):
        self.registry_file = registry_file
        self.companies_file = companies_file   # one row per company: its SEC record, fetched once

    def _get(self, url):
        time.sleep(Edgar.REQUEST_GAP)
        return requests.get(url, headers={'User-Agent': Edgar.USER_AGENT, 'Accept-Encoding': 'gzip, deflate'}, timeout=60)

    @staticmethod
    def _write_csv_atomic(df, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        tmp_path = path + '.tmp'
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)

    def _listed(self) -> pd.DataFrame:
        """Common shares on NYSE or Nasdaq, from the SEC's exchange ticker list."""
        r = self._get(self.LIST_URL)
        r.raise_for_status()
        j = r.json()
        df = pd.DataFrame(j['data'], columns=j['fields'])
        df['ticker'] = df['ticker'].astype(str).str.strip().str.upper().str.replace('.', '-', regex=False)
        df = df[df['exchange'].isin(self.EXCHANGES) & ~df['ticker'].str.contains(self.NOT_COMMON)].drop_duplicates('ticker')
        t = df['ticker']
        five = (t.str.len() == 5) & ~t.str.contains('-')
        base_listed = pd.Series(list(zip(df['cik'], t.str[:4])), index=df.index).isin(set(zip(df['cik'], t)))
        other = five & (t.str[-1].isin(self.FIFTH_ALWAYS) | (t.str[-1].isin(self.FIFTH_WITH_BASE) & base_listed))
        return df[~other]

    def _companies(self, ciks, force=False) -> pd.DataFrame:
        """SEC records (industry code and entity type) for these companies: cached ones reused, the rest fetched."""
        cols = ['cik', 'sic', 'industry', 'entity_type', 'fetched_on']
        known = pd.read_csv(self.companies_file, dtype={'industry': str, 'entity_type': str}) if os.path.exists(self.companies_file) and not force else pd.DataFrame(columns=cols)
        todo = [c for c in ciks if c not in set(known['cik'])]
        if todo:
            print(f"Fetching SEC records for {len(todo)} companies (about {len(todo) * 0.5 / 60:.0f} minutes)...")
        new, failed = [], 0
        for i, cik in enumerate(todo, 1):
            try:
                r = self._get(self.RECORD_URL.format(cik=int(cik)))
                rec = r.json() if r.status_code == 200 else None
            except (requests.RequestException, ValueError):
                rec = None
            if rec is None:
                failed += 1   # not cached, so it is tried again next time
            else:
                new.append((int(cik), pd.to_numeric(rec.get('sic'), errors='coerce'), rec.get('sicDescription'), rec.get('entityType'),
                            datetime.now().strftime('%Y-%m-%d')))
            if new and (i % self.SAVE_EVERY == 0 or i == len(todo)):
                known = pd.concat([known, pd.DataFrame(new, columns=cols)], ignore_index=True)
                self._write_csv_atomic(known, self.companies_file)
                new = []
                print(f"  {i} of {len(todo)} fetched")
        if failed:
            print(f"  {failed} record(s) could not be fetched; they will be retried on the next update.")
        return known

    def update(self, force=False) -> pd.DataFrame:
        """Rebuild the registry when it is older than LIST_DAYS (or with force=True; force also re-fetches every company record).
        Returns the registry. A ticker that Yahoo has confirmed DELISTED keeps that status."""
        if os.path.exists(self.registry_file) and not force and (time.time() - os.path.getmtime(self.registry_file)) < self.LIST_DAYS * 86400:
            return pd.read_csv(self.registry_file, dtype=str)
        listed = self._listed()
        companies = self._companies(list(dict.fromkeys(listed['cik'])), force=force)
        reg = listed.merge(companies, on='cik', how='inner')
        reg = reg[reg['entity_type'].isin(['operating', 'other']) & reg['sic'].notna() & ~reg['sic'].isin(self.EXCLUDED_SIC)].copy()
        reg['filer'] = reg['entity_type'].map({'operating': 'domestic', 'other': 'foreign'})
        reg['sic'] = reg['sic'].astype(int)
        reg['sector'] = reg['sic'].map(sector_for)
        reg['status'] = ''
        if os.path.exists(self.registry_file):
            old = pd.read_csv(self.registry_file, dtype=str)
            if 'status' in old.columns and 'ticker' in old.columns:
                delisted = set(old.loc[old['status'] == 'DELISTED', 'ticker'])
                reg.loc[reg['ticker'].isin(delisted), 'status'] = 'DELISTED'
        reg['last_update'] = datetime.now().strftime('%Y-%m-%d')
        reg = reg[self.COLUMNS].sort_values('ticker').reset_index(drop=True)
        self._write_csv_atomic(reg, self.registry_file)
        print(f"Registry: {len(reg)} tickers from {reg['cik'].nunique()} companies; by sector: {reg['sector'].value_counts().to_dict()}")
        return reg
