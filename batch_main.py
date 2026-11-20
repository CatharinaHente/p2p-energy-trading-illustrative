"""batch_main.py — Batch runner for the P2P market analysis.

Runs centralized and distributed solvers for every date, computes fairness
metrics (from metrics.py), and saves results to the structured result library.

Result layout:
    structures/{STRUCTURE}/results/{date}/
        no_trade.json   — no-trade baseline (cached)
        centralized.json
        distributed.json
        fairness.json   — all metrics: TW, NBS, MM, JI, Var on J_i and d_i

Usage: python batch_main.py
"""
from pathlib import Path
import json, time
import numpy as np

from scenarios     import make_scenario_date, make_scenario_horizon
from algo_central  import centralized
from algo_ADMM     import distributed
from metrics       import no_trade_costs, compute_metrics
from network       import evaluate_feeder
from config import (
    NETWORK_TOPOLOGY,
    PROSUMER_SCENARIO,
    TRADING_TOPOLOGY,
)

ROOT = Path(__file__).resolve().parent


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

STRUCTURE = (
    f'{NETWORK_TOPOLOGY}_{PROSUMER_SCENARIO}_{TRADING_TOPOLOGY}'
)

# 7th, 14th, 21st of each month — 3 days per month, 36 total
DATES = [
    f'2016-{m:02d}-{d:02d}'
    for m in range(1, 13)
    for d in [7, 14, 21]
]

CASES = ['centralized', 'distributed']

SETTINGS = dict(
    resolution_h   = 0.5,
    seed           = 42,
    admm_rho       = 10.0,
    admm_tolerance = 1e-5,
)


# ══════════════════════════════════════════════════════════════════════════════
# Path helpers
# ══════════════════════════════════════════════════════════════════════════════

def _dir(date):
    return ROOT / 'structures' / STRUCTURE / 'results' / date

def _path(date, name):
    return _dir(date) / f'{name}.json'

def _exists(date, name):
    return _path(date, name).exists()

def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))

def _read(path):
    return json.loads(path.read_text())


# ══════════════════════════════════════════════════════════════════════════════
# Serialisation
# ══════════════════════════════════════════════════════════════════════════════

def _serialise(result, date, case):
    s = {}
    for k, v in result.items():
        if k == 'trades':
            s['trades'] = {f'{i}->{j}@{t}': float(v)
                           for (i, j, t), v in result['trades'].items()}
        elif k == 'network':
            s['network'] = result['network']
        elif isinstance(v, np.ndarray):
            s[k] = v.tolist()
        else:
            s[k] = v
    s['_date']      = date
    s['_case']      = case
    s['_structure'] = STRUCTURE
    s['_settings']  = SETTINGS
    return s


# ══════════════════════════════════════════════════════════════════════════════
# Network evaluation
# ══════════════════════════════════════════════════════════════════════════════

def _network(result, data):
    T, N  = len(data['buy_price']), len(data['demand'])
    zeros = np.zeros((N, T))
    p_inj = (np.array(data['generation']) - np.array(data['demand'])
             - result['charge']                        + result['discharge']
             - result.get('ev_charge',  zeros)         + result.get('ev_discharge', zeros)
             - result.get('hp_power',   zeros))
    return evaluate_feeder(
        p_inj, data['agent_bus'], T, data['line_limit_mva'],
        q_injections_kvar=result.get('q_injection'))


# ══════════════════════════════════════════════════════════════════════════════
# Per-date runner
# ══════════════════════════════════════════════════════════════════════════════

def run_date(date):
    cases_needed    = [c for c in CASES if not _exists(date, c)]
    fairness_needed = not _exists(date, 'fairness')

    if not cases_needed and not fairness_needed:
        print(f"  {date}: complete — skipping")
        return

    # ── Scenario ──────────────────────────────────────────────────────────────
    data = make_scenario_horizon(
        date,
        seed=SETTINGS['seed'],
        resolution_h=SETTINGS['resolution_h'],
        lookahead_h=24
    )

    # ── No-trade baseline (cached) ────────────────────────────────────────────
    if _exists(date, 'no_trade'):
        J_nt = np.array(_read(_path(date, 'no_trade'))['J_nt'])
        print(f"  {date}: no-trade loaded from cache")
    else:
        print(f"  {date}: computing no-trade baseline ...")
        J_nt = no_trade_costs(data)
        _write(_path(date, 'no_trade'), {'J_nt': J_nt.tolist(), 'date': date})

    results = {}

    # ── Solver cases ──────────────────────────────────────────────────────────
    for case in CASES:
        if _exists(date, case):
            print(f"  {date}/{case}: exists — skipping")
            # Load costs for metrics even if we don't re-solve
            if case not in results:
                raw = _read(_path(date, case))
                results[case] = {'costs': np.array(raw['costs'])}
            continue

        print(f"  {date}/{case}: running ...")
        t0 = time.time()

        if case == 'centralized':
            result = centralized(data)
        elif case == 'distributed':
            result = distributed(data,
                                 rho=SETTINGS['admm_rho'],
                                 tolerance=SETTINGS['admm_tolerance'])
        else:
            raise ValueError(f"Unknown case: {case}")

        result['network'] = _network(result, data)
        elapsed = time.time() - t0

        m_all  = compute_metrics(result, J_nt)
        d_so   = m_all['d']
        jm     = m_all['J_metrics']
        dm     = m_all['d_metrics']

        print(f"  {date}/{case}:  total={result['costs'].sum():.3f} EUR  "
              f"TW={jm['TW']:.4f}  NBS={jm['NBS']:.4f}  "
              f"MM(d)={dm['MM']:+.4f}  ({elapsed:.1f}s)")

        _write(_path(date, case), _serialise(result, date, case))
        results[case] = result

    # ── Fairness metrics (always from centralized) ────────────────────────────
    if fairness_needed:
        if 'centralized' not in results:
            raw = _read(_path(date, 'centralized'))
            r_c = {'costs': np.array(raw['costs'])}
        else:
            r_c = results['centralized']

        m_all = compute_metrics(r_c, J_nt)
        d     = m_all['d']
        jm    = m_all['J_metrics']
        dm    = m_all['d_metrics']

        fairness = {
            'date':      date,
            'structure': STRUCTURE,
            'settings':  SETTINGS,
            'J_nt':      J_nt.tolist(),
            'J_so':      m_all['J'].tolist(),
            'd':         d.tolist(),
            'J_metrics': {k: float(v) for k, v in jm.items()},
            'd_metrics': {k: float(v) for k, v in dm.items()},
        }
        _write(_path(date, 'fairness'), fairness)
        print(f"  {date}/fairness: "
              f"TW={jm['TW']:.4f}  NBS={jm['NBS']:.4f}  "
              f"MM(J)={jm['MM']:.4f}  MM(d)={dm['MM']:+.4f}  "
              f"JI(J)={jm['JI']:.4f}  JI(d)={dm['JI']:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════

def summarise():
    rows = []
    for date in DATES:
        if not _exists(date, 'fairness'):
            continue
        f = _read(_path(date, 'fairness'))
        rows.append({'date': date, **f['J_metrics'],
                     **{f'd_{k}': v for k, v in f['d_metrics'].items()}})

    if not rows:
        print("\nNo fairness results found yet.")
        return

    import pandas as pd
    df = pd.DataFrame(rows).set_index('date')
    print(f"\n{'═'*80}")
    print(f"Summary — structure: {STRUCTURE}  ({len(df)}/{len(DATES)} dates)")
    print(f"{'─'*80}")
    print(df[['TW','NBS','MM','JI','Var','d_MM','d_JI','d_Var']].round(4).to_string())
    print(f"{'─'*80}")
    print("Means:")
    print(df[['TW','NBS','MM','JI','Var','d_MM','d_JI','d_Var']].mean().round(4).to_string())
    print(f"{'═'*80}\n")

    agg_path = ROOT / 'structures' / STRUCTURE / 'results' / 'summary.json'
    _write(agg_path, {'structure': STRUCTURE,
                      'dates': list(df.index),
                      'rows':  rows})
    print(f"Aggregate saved → {agg_path.relative_to(ROOT)}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"Structure : {STRUCTURE}")
    print(f"Dates     : {DATES[0]} … {DATES[-1]}  ({len(DATES)} dates)")
    print(f"Cases     : {CASES}")
    print(f"Settings  : {SETTINGS}")
    print()

    for date in DATES:
        try:
            run_date(date)
        except RuntimeError as e:
            print(f"\n  !! {date}: skipped — {e}")
            _write(_path(date, 'infeasible'), {'date': date, 'error': str(e)})

    summarise()