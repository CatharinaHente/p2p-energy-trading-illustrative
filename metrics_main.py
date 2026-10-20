"""
--> loads saved results and computes fairness metrics
--> run after main.py has produced results in results/centralized/summary.json!!!
"""
import json
import numpy as np
from pathlib import Path
from scenarios import make_scenario
from metrics import no_trade_costs, savings, print_metrics

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# ── Configuration ─────────────────────────────────────────────────────────────
SEASON = 'winter'   # must match the season used in main.py

# ── Load scenario data ────────────────────────────────────────────────────────
print(f"Loading {SEASON} scenario data...")
data = make_scenario(SEASON, seed=42)

# ── Load saved result ─────────────────────────────────────────────────────────
def load_result(name):
    """Load a solver result from results/<name>/summary.json."""
    p = ROOT / 'results' / name / 'summary.json'
    if not p.exists():
        print(f"  ✗ No saved result for '{name}' — run main.py first.")
        return None
    raw = json.loads(p.read_text())
    # Reconstruct (i, j, t) trade keys from "i->j@t" strings
    trades = {}
    for k, v in raw['trades'].items():
        ij, t = k.split('@')
        i, j  = ij.split('->')
        trades[(int(i), int(j), int(t))] = v
    raw['trades'] = trades
    raw['costs']  = np.array(raw['costs'])
    print(f"  ✓ Loaded '{name}'  total={sum(raw['costs']):.4f} EUR")
    return raw

# ── No-trade baseline ─────────────────────────────────────────────────────────
print("\nComputing no-trade baseline...")
J_nt = no_trade_costs(data)

# ── Metrics for each available result ────────────────────────────────────────
cases = ['centralized', 'distributed', 'centralized_network']

for case in cases:
    result = load_result(case)
    if result is None:
        continue
    d = savings(result, J_nt)
    print_metrics(d, J_nt, label=case)