"""pricing.py — Electricity price profiles for the P2P market analysis.

Data source
───────────
energy-charts.info (Fraunhofer ISE / EPEX SPOT) — free, no API key.
Bidding zone DE-LU (Germany / Luxembourg).
Covers 2015 onwards.

Pricing model
─────────────
  spot       EPEX day-ahead wholesale [EUR/MWh → EUR/kWh]
  buy        spot + retail_markup     [EUR/kWh]  household tariff
  sell       feed_in_tariff           [EUR/kWh]  fixed per year (EEG)
  p2p        configurable; default = midpoint(buy, sell)

Retail markup (above spot, year-specific)
─────────────────────────────────────────
The fixed component approximates German household costs:
  Grid fees (Netzentgelte)   ~7.0 ct/kWh
  EEG levy                    year-specific (6.35 → 0.0 from 2023)
  Other levies (KWK, §19…)   ~1.5 ct/kWh
  Stromsteuer                 2.05 ct/kWh
  Konzessionsabgabe           ~1.7 ct/kWh
  Supplier margin             ~2.5 ct/kWh
  VAT (19%) on full bill
Source: Bundesnetzagentur / BDEW Strompreisanalyse (annual editions).

Feed-in tariffs (EEG, ≤ 10 kWp solar, January rate)
──────────────────────────────────────────────────────
  2016: 12.31 ct/kWh   Source: EEG 2014, degression schedule
  2020:  9.02 ct/kWh   Source: EEG 2017 / BNetzA announcement Dec 2019
  2024:  8.11 ct/kWh   Source: EEG 2023 / BNetzA announcement Jan 2024

P2P price modes
───────────────
  'midpoint'    (buy + sell) / 2   — literature standard (Behrunani et al. 2021)
  'spot'        raw day-ahead price — energy-market consistent
  'fixed_ratio' buy × 0.60         — simple proportional rule

Cache
─────
  prices/{year}_DE-LU.csv   — one file per year, downloaded once.

Usage
─────
    from pricing import get_day_prices, PRICE_YEAR
    buy, sell, p2p = get_day_prices('2016-01-15', resolution_h=0.5)
"""

import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent / 'prices'

# ── Active price year ─────────────────────────────────────────────────────────
# Change this to switch the entire analysis to a different price environment.
PRICE_YEAR = 2016


# ── Year-specific parameters ──────────────────────────────────────────────────
# retail_markup : fixed component added to spot price [EUR/kWh]
# feed_in       : solar feed-in tariff ≤ 10 kWp, January rate [EUR/kWh]

_YEAR_PARAMS = {
    2016: dict(retail_markup=0.218, feed_in=0.1231),
    2020: dict(retail_markup=0.231, feed_in=0.0902),
    2024: dict(retail_markup=0.245, feed_in=0.0811),
}


# ── Download and cache ────────────────────────────────────────────────────────

def download_year(year):
    """Download full-year hourly day-ahead prices from energy-charts.info.

    Saves to prices/{year}_DE-LU.csv (downloaded once, reused thereafter).
    Returns a DataFrame indexed by local Berlin timestamps with column
    'spot_eur_mwh'.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    bzn  = 'DE-AT-LU' if year < 2019 else 'DE-LU'
    path = CACHE_DIR / f'{year}_{bzn}.csv'

    if path.exists():
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert('Europe/Berlin')
        print(f"  Prices {year}: loaded from cache")
        return df

    print(f"  Downloading day-ahead prices {year} from energy-charts.info ...")

    # Germany and Austria shared the DE-AT-LU bidding zone until 30 Sep 2018.
    # From 1 Oct 2018 Germany/Luxembourg became DE-LU.
    bzn = 'DE-AT-LU' if year < 2019 else 'DE-LU'

    # API accepts ISO 8601 date strings (YYYY-MM-DD) as start/end.
    url = (f'https://api.energy-charts.info/price'
           f'?bzn={bzn}&start={year}-01-01&end={year}-12-31')

    with urllib.request.urlopen(url, timeout=30) as r:
        raw = json.loads(r.read().decode())

    ts    = pd.to_datetime(raw['unix_seconds'], unit='s', utc=True)
    price = np.array(raw['price'], dtype=float)   # EUR/MWh; NaN for missing

    df = pd.DataFrame({'spot_eur_mwh': price}, index=ts)
    df.index = df.index.tz_convert('Europe/Berlin')

    # API may return 15-min data — resample to hourly so get_day_prices
    # always works with 24 values per day regardless of source resolution.
    if len(df) > 366 * 24:
        df = df.resample('h').mean()

    df.to_csv(path)
    print(f"  Prices {year}: {len(df)} hourly values saved → {path.name}")
    return df


# ── Per-day price arrays ──────────────────────────────────────────────────────

def get_day_prices(date_str, resolution_h=0.5, year=None, p2p_mode='midpoint'):
    """Return buy, sell, P2P price arrays for one calendar day.

    Parameters
    ----------
    date_str     : str    'YYYY-MM-DD' — must match the SimBench profile year
    resolution_h : float  0.25 | 0.5 | 1.0 — must match scenario resolution
    year         : int    price year; defaults to PRICE_YEAR
    p2p_mode     : str    how the P2P price is set:
                          'midpoint'    (buy + sell) / 2      (default)
                          'spot'        day-ahead spot price
                          'fixed_ratio' buy × 0.60

    Returns
    -------
    buy, sell, p2p : list[float] each of length T = 24 / resolution_h
    """
    if year is None:
        year = PRICE_YEAR
    if year not in _YEAR_PARAMS:
        raise ValueError(f"Year {year} not configured. "
                         f"Available: {sorted(_YEAR_PARAMS)}")
    if resolution_h not in (0.25, 0.5, 1.0):
        raise ValueError("resolution_h must be 0.25, 0.5 or 1.0")

    params = _YEAR_PARAMS[year]
    df     = download_year(year)

    # Slice the target date in Berlin local time.
    # Use strftime comparison rather than .loc with a plain string — the latter
    # breaks when the index is tz-aware (as it is after tz_convert('Europe/Berlin')).
    target = pd.Timestamp(date_str).date()
    mask   = pd.to_datetime(df.index).tz_localize(None).normalize() == pd.Timestamp(date_str)
    day  = df.loc[mask, 'spot_eur_mwh'].values   # up to 24 hourly values

    if len(day) == 0:
        raise ValueError(f"No price data for {date_str} (year {year}). "
                         f"Check cache file in prices/.")

    # Fill any gaps (NaN or missing hours) by forward/backward fill
    spot_h = pd.Series(day).reindex(range(24)).ffill().bfill().to_numpy(float)

    # Retail buy price:  spot (clipped at 0 for household tariff) + fixed markup
    # Negative spot prices do not reduce the household bill under standard tariffs.
    buy_h = np.maximum(spot_h, 0.0) / 1000.0 + params['retail_markup']

    # Sell price: fixed feed-in tariff (same every hour)
    sell_h = np.full(24, params['feed_in'])

    # P2P price
    if p2p_mode == 'midpoint':
        p2p_h = (buy_h + sell_h) / 2.0
    elif p2p_mode == 'spot':
        p2p_h = np.maximum(spot_h, 0.0) / 1000.0
    elif p2p_mode == 'fixed_ratio':
        p2p_h = buy_h * 0.60
    else:
        raise ValueError(f"Unknown p2p_mode '{p2p_mode}'. "
                         f"Choose 'midpoint', 'spot', or 'fixed_ratio'.")

    # Enforce sell < p2p < buy at every hour
    p2p_h = np.clip(p2p_h, sell_h + 1e-4, buy_h - 1e-4)

    # Upsample from hourly to target resolution
    reps  = int(round(1.0 / resolution_h))   # 4 for 15-min, 2 for 30-min, 1 for hourly
    buy   = np.repeat(buy_h,  reps).tolist()
    sell  = np.repeat(sell_h, reps).tolist()
    p2p   = np.repeat(p2p_h,  reps).tolist()

    return buy, sell, p2p


# ── Diagnostic ────────────────────────────────────────────────────────────────

def price_summary(date_str, resolution_h=0.5, year=None, p2p_mode='midpoint'):
    """Print a one-line summary of the price profile for a given day."""
    buy, sell, p2p = get_day_prices(date_str, resolution_h, year, p2p_mode)
    b = np.array(buy); s = np.array(sell); p = np.array(p2p)
    yr = year or PRICE_YEAR
    print(f"  {date_str} (year={yr}, mode={p2p_mode}): "
          f"buy=[{b.min():.3f},{b.max():.3f}]  "
          f"p2p=[{p.min():.3f},{p.max():.3f}]  "
          f"sell={s.mean():.3f}  EUR/kWh")


if __name__ == '__main__':
    # Quick test: show price profiles for the 15th of each month in 2016
    print(f"Price year: {PRICE_YEAR}")
    for m in range(1, 13):
        price_summary(f'2016-{m:02d}-15')