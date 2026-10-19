"""Rule-based watchlist: expands include/exclude rules against the ticker registry, and serves named sub-lists."""
import difflib
import pandas as pd


class Watchlist:
    UNUSABLE_STATUSES = {'DELISTED', 'INVALID'}
    # include works at any level; exclude is ticker-only by design.
    ALLOWED = {('include', 'sector'), ('include', 'industry'), ('include', 'ticker'), ('exclude', 'ticker')}
    COLUMN = {'sector': 'sector', 'industry': 'zacks_x_ind_desc', 'ticker': 'symbol'}

    def __init__(self, rules_file="pipeline/watchlist-rules.csv", registry_file="data/registry/tickers.csv", tickers_file="data/watchlist.csv"):
        self.rules_file = rules_file
        self.registry_file = registry_file
        self.tickers_file = tickers_file   # the generated watchlist: what the rules produced, less liquidity-screen failures

    @staticmethod
    def _yf_symbol(ticker):
        # Registry uses '.' for share classes (BRK.B); yfinance uses '-'.
        return str(ticker).strip().upper().replace('.', '-')

    def _load_rules(self):
        rules = pd.read_csv(self.rules_file, comment='#', skip_blank_lines=True, dtype=str).fillna('')
        if 'list' not in rules.columns:
            rules['list'] = ''   # the fourth column is optional: a rule with no list name only adds to the watchlist
        for col in ('action', 'level', 'value', 'list'):
            rules[col] = rules[col].str.strip()
        rules['action'] = rules['action'].str.lower()
        rules['level'] = rules['level'].str.lower()
        for r in rules.itertuples():
            if (r.action, r.level) not in self.ALLOWED:
                raise ValueError(f"Invalid rule '{r.action},{r.level},{r.value}'. Allowed: include sector/industry/ticker, exclude ticker.")
        return rules

    def _matches(self):
        """Every (ticker, rule) match of an include rule, as DataFrame[ticker, matched_by, list], with excluded tickers removed."""
        reg = pd.read_csv(self.registry_file, dtype=str).dropna(subset=['ticker'])
        reg['symbol'] = reg['ticker'].map(self._yf_symbol)
        known = {lvl: set(reg[col].dropna()) for lvl, col in self.COLUMN.items()}
        usable = reg[~reg['status'].isin(self.UNUSABLE_STATUSES)]
        rules = self._load_rules()

        picked = []
        for value, level, name in rules.loc[rules['action'] == 'include', ['value', 'level', 'list']].itertuples(index=False):
            raw, value = value, self._yf_symbol(value) if level == 'ticker' else value
            if value not in known[level]:
                if level == 'ticker':
                    print(f"[WARN] include ticker {value} is not in the registry; skipped.")
                    continue
                hint = difflib.get_close_matches(value, known[level], n=3)
                raise ValueError(f"Unknown {level} '{value}'. Did you mean: {hint}?")
            hits = usable[usable[self.COLUMN[level]] == value]
            picked.append(pd.DataFrame({'ticker': hits['symbol'], 'matched_by': f"{level}:{raw}", 'list': name}))

        out = pd.concat(picked, ignore_index=True) if picked else pd.DataFrame(columns=['ticker', 'matched_by', 'list'])
        drop = {self._yf_symbol(v) for v in rules.loc[rules['action'] == 'exclude', 'value']}
        return out[~out['ticker'].isin(drop)]

    def resolve(self):
        """Return DataFrame[ticker, matched_by] of yfinance symbols selected by the rules (the first rule that matched each)."""
        out = self._matches().drop_duplicates('ticker')
        return out[['ticker', 'matched_by']].sort_values('ticker').reset_index(drop=True)

    def _listed(self):
        """Matches that carry a list name, limited to tickers on the generated watchlist, so every member has data."""
        current = pd.read_csv(self.tickers_file, comment='#', skip_blank_lines=True).iloc[:, 0].dropna().astype(str).str.strip().str.upper()
        m = self._matches()
        return m[(m['list'] != '') & m['ticker'].isin(set(current))].drop_duplicates(['list', 'ticker'])

    def lists(self):
        """Series of member counts by list name."""
        return self._listed().groupby('list').size().rename('members')

    def members(self, name):
        """Sorted tickers in the named sub-list. A ticker can be in several lists, one per rule that names it."""
        listed = self._listed()
        if name not in set(listed['list']):
            raise ValueError(f"Unknown list '{name}'. Available: {sorted(set(listed['list']))}.")
        return sorted(listed.loc[listed['list'] == name, 'ticker'])
