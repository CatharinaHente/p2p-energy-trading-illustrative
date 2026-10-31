"""run.py — Batch runner for the P2P market analysis.

Edit the CONFIGURATION block at the top.  Everything else is automatic.

Result layout
─────────────
    structures/
    └── {STRUCTURE}/
        └── results/
            └── {date}/
                ├── no_trade.json   — no-trade baseline (cached; never recomputed)
                ├── centralized.json
                ├── distributed.json
                └── fairness.json   — PoO, PoF, all metric values

Skip logic
──────────
Each file is written atomically on completion.  If a file already exists the
corresponding step is skipped.  Interrupt and restart at any time; only the
missing pieces will be computed.

Usage
─────
    python run.py
"""
from pathlib import Path
import json, time
import numpy as np

from scenarios import make_scenario_date
from algo_central import centralized
from algo_ADMM import distributed
from metrics import (no_trade_costs, savings,
                     sw, tw, mm, jains, var_savings,
                     fair_optimal_SW, fair_optimal_MM,
                     poo, pof)
from network import evaluate_feeder

ROOT = Path(__file__).resolve().parent


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit here to change what runs
# ══════════════════════════════════════════════════════════════════════════════

STRUCTURE = 'base'          # must match a folder under structures/

# Calendar dates to analyse (SimBench reference year is 2016)
DATES = [
    '2016-01-15', '2016-02-15', '2016-03-15',
    '2016-04-15', '2016-05-15', '2016-06-15',
    '2016-07-15', '2016-08-15', '2016-09-15',
    '2016-10-14', '2016-11-15', '2016-12-15',
]

# Which solver cases to run per date; uncomment network cases if desired
CASES = [
    'centralized',
    'distributed',
    #'centralized_network',
    #'distributed_network',
]

# Solver and scenario settings
SETTINGS = dict(
    resolution_h   = 0.5,    # 30-min steps → T=48
    seed           = 42,
    admm_rho       = 10.0,
    admm_tolerance = 1e-5,
)

# Set True to compute PoO and PoF after each centralized solve.
# Adds two extra Gurobi solves (SW-fair and MM-fair) per date.
COMPUTE_FAIRNESS = False


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
    """Convert a result dict to a JSON-serialisable form."""
    s = {}
    for k, v in result.items():
        if k == 'trades':
            s['trades'] = {f'{i}->{j}@{t}': float(v)
                           for (i, j, t), v in result['trades'].items()}
        elif k == 'network':
            s['network'] = result['network']   # already JSON-safe from evaluate_feeder
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
    """Run all missing cases for one date. Skip steps that already exist."""
    cases_needed    = [c for c in CASES if not _exists(date, c)]
    fairness_needed = COMPUTE_FAIRNESS and not _exists(date, 'fairness')

    if not cases_needed and not fairness_needed:
        print(f"  {date}: complete — skipping")
        return

    # ── Scenario ──────────────────────────────────────────────────────────────
    data = make_scenario_date(date,
                              seed=SETTINGS['seed'],
                              resolution_h=SETTINGS['resolution_h'])

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

        d    = savings(result, J_nt)
        J_so = np.array(result['costs'])
        print(f"  {date}/{case}: total={sum(result['costs']):.3f} EUR  "
              f"TW={tw(J_nt, J_so):.4f}  MM={mm(d):.4f}  SW={sw(d):.4f}  "
              f"({elapsed:.1f}s)")

        _write(_path(date, case), _serialise(result, date, case))
        results[case] = result

    # ── Fairness metrics ──────────────────────────────────────────────────────
    if fairness_needed:
        # Load centralized result from file if not in memory
        if 'centralized' not in results:
            if not _exists(date, 'centralized'):
                print(f"  {date}/fairness: centralized result missing — skipping")
                return
            raw = _read(_path(date, 'centralized'))
            r_c = {'costs': np.array(raw['costs'])}
        else:
            r_c = results['centralized']

        J_so = np.array(r_c['costs'])
        d_so = savings(r_c, J_nt)

        print(f"  {date}/fairness: computing SW-fair and MM-fair ...")
        t0 = time.time()
        r_sw = fair_optimal_SW(data, J_nt)
        r_mm = fair_optimal_MM(data, J_nt)
        d_sw = savings(r_sw, J_nt)
        d_mm = savings(r_mm, J_nt)
        elapsed = time.time() - t0

        poo_sw = poo(sw, d_so, d_sw)
        poo_mm = poo(mm, d_so, d_mm)
        pof_sw = pof(r_c, r_sw, J_nt)
        pof_mm = pof(r_c, r_mm, J_nt)

        fairness = {
            'date':       date,
            'structure':  STRUCTURE,
            'settings':   SETTINGS,
            'J_nt':       J_nt.tolist(),
            'J_so':       J_so.tolist(),
            'd_so':       d_so.tolist(),
            'TW':         float(tw(J_nt, J_so)),
            'SW':         float(sw(d_so)),
            'MM':         float(mm(d_so)),
            'JI':         float(jains(d_so)),
            'Var':        float(var_savings(d_so)),
            'J_sw_fair':  r_sw['costs'].tolist(),
            'J_mm_fair':  r_mm['costs'].tolist(),
            'd_sw_fair':  d_sw.tolist(),
            'd_mm_fair':  d_mm.tolist(),
            'PoO_SW':     float(poo_sw),
            'PoO_MM':     float(poo_mm),
            'PoF_SW':     float(pof_sw),
            'PoF_MM':     float(pof_mm),
        }
        _write(_path(date, 'fairness'), fairness)
        print(f"  {date}/fairness: "
              f"PoO_SW={poo_sw:.4f}  PoO_MM={poo_mm:.4f}  "
              f"PoF_SW={pof_sw:.4f}  PoF_MM={pof_mm:.4f}  "
              f"({elapsed:.1f}s)")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════

def summarise():
    """Print a table of all completed fairness results for STRUCTURE."""
    from datetime import datetime
    rows = []
    for date in DATES:
        if not _exists(date, 'fairness'):
            continue
        f = _read(_path(date, 'fairness'))
        rows.append(f)

    if not rows:
        print("\nNo fairness results found yet.")
        return

    print(f"\n{'═'*80}")
    print(f"Fairness summary — structure: {STRUCTURE}  ({len(rows)}/{len(DATES)} dates)")
    print(f"{'─'*80}")
    hdr = (f"{'Date':<12}  {'TW':>6}  {'SW':>6}  {'MM':>7}  {'JI':>6}  "
           f"{'PoO_SW':>8}  {'PoO_MM':>8}  {'PoF_SW':>8}  {'PoF_MM':>8}")
    print(hdr)
    print(f"{'─'*80}")
    for f in rows:
        print(f"{f['date']:<12}  {f['TW']:>6.4f}  {f['SW']:>6.4f}  "
              f"{f['MM']:>7.4f}  {f['JI']:>6.4f}  "
              f"{f['PoO_SW']:>8.4f}  {f['PoO_MM']:>8.4f}  "
              f"{f['PoF_SW']:>8.4f}  {f['PoF_MM']:>8.4f}")
    print(f"{'─'*80}")

    # Monthly averages
    metrics = ['TW', 'SW', 'MM', 'JI', 'PoO_SW', 'PoO_MM', 'PoF_SW', 'PoF_MM']
    avgs = {k: np.mean([f[k] for f in rows]) for k in metrics}
    print(f"{'mean':<12}  {avgs['TW']:>6.4f}  {avgs['SW']:>6.4f}  "
          f"{avgs['MM']:>7.4f}  {avgs['JI']:>6.4f}  "
          f"{avgs['PoO_SW']:>8.4f}  {avgs['PoO_MM']:>8.4f}  "
          f"{avgs['PoF_SW']:>8.4f}  {avgs['PoF_MM']:>8.4f}")
    print(f"{'═'*80}\n")

    # Save aggregate to file
    agg_path = ROOT / 'structures' / STRUCTURE / 'results' / 'summary.json'
    _write(agg_path, {
        'structure': STRUCTURE,
        'dates': [f['date'] for f in rows],
        'rows':  rows,
        'means': avgs,
    })
    print(f"Aggregate saved → {agg_path.relative_to(ROOT)}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"Structure : {STRUCTURE}")
    print(f"Dates     : {DATES[0]} … {DATES[-1]}  ({len(DATES)} days)")
    print(f"Cases     : {CASES}")
    print(f"Fairness  : {COMPUTE_FAIRNESS}")
    print(f"Settings  : {SETTINGS}")
    print()

    for date in DATES:
        run_date(date)

    summarise()