# ── Imports ───────────────────────────────────────────────────────────────────
from pathlib import Path
import json
import numpy as np
from algo import centralized, distributed
from scenarios import make_scenario, prosumer_summary
from network import evaluate_feeder
from plots import (plot_for, plot_prices, plot_trades,
                   plot_network, plot_balance, plot_devices)
from metrics import no_trade_costs, savings, print_metrics

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# ── Configuration ─────────────────────────────────────────────────────────────
SEASON = 'winter'       # 'winter' | 'spring' | 'summer' | 'autumn'
INTERACTIVE = True      # True → ask before each solver run
                        # False → run all cases listed below automatically
CASES = [
    'centralized',
    'distributed',
    'centralized_network',
    'distributed_network',
]

# ── Data ──────────────────────────────────────────────────────────────────────
data    = make_scenario(SEASON, seed=42)
results = {}

print("\nProsumer configuration:")
print(prosumer_summary().to_string())
print("\nTrading graph:", {
    i: sorted(j if i==a else a for a,j in data["edges"] if i in (a,j))
    for i in range(len(data["demand"]))})

plot_for(data, ROOT)
plot_prices(SEASON, data, ROOT)


# ── Helpers ───────────────────────────────────────────────────────────────────

def show(name, result):
    results[name] = result
    N = len(data["demand"])
    print(f"\n{name}  iter={result['iterations']}  "
          f"total={sum(result['costs']):.4f} EUR")
    for i, cost in enumerate(result["costs"]):
        print(f"  P{i+1} ({data['prosumer_names'][i]}): {cost:.4f} EUR")
        if data['ev_capacity'][i] > 0:
            ev_soc = result.get('ev_soc')
            if ev_soc is not None:
                print(f"    ev_soc: [{ev_soc[i,0]:.1f} → {ev_soc[i,-1]:.1f}] kWh")
        if data['hp_max_power'][i] > 0:
            temp = result.get('temp')
            if temp is not None:
                print(f"    temp:   [{temp[i,0]:.2f} → {temp[i,-1]:.2f}] °C")
    out = ROOT / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    serial = {k: v.tolist() if isinstance(v, np.ndarray) else v
              for k, v in result.items() if k != "trades"}
    serial["trades"] = {f"{i}->{j}@{t}": v
                        for (i,j,t),v in result["trades"].items()}
    (out / "summary.json").write_text(json.dumps(serial, indent=2))


def check_network(name, result):
    T, N  = len(data["buy_price"]), len(data["demand"])
    zeros = np.zeros((N, T))
    p_inj = (np.array(data["generation"]) - np.array(data["demand"])
             - result["charge"]                        + result["discharge"]
             - result.get("ev_charge",  zeros)         + result.get("ev_discharge", zeros)
             - result.get("hp_power",   zeros))
    result["network"] = evaluate_feeder(
        p_inj, data["agent_bus"], T, data["line_limit_mva"],
        q_injections_kvar=result.get("q_injection"))
    net = result["network"]
    print(f"{name}: V_viol={net['any_voltage_violation']}  "
          f"cong={net['any_congestion']}")


def _run(name, result):
    check_network(name, result)
    show(name, result)
    plot_trades(name, result, data, ROOT)
    plot_network(name, result, data, ROOT)
    plot_balance(name, result, data, ROOT)
    plot_devices(name, result, data, ROOT)


def _ask(name):
    """Return True if the user wants to run this case."""
    if not INTERACTIVE:
        return True
    ans = input(f"\nRun {name}? [Y/n]: ").strip().lower()
    return ans not in ('n', 'no')


# ── Solver cases ──────────────────────────────────────────────────────────────
SOLVERS = {
    'centralized':        lambda: centralized(data),
    'distributed':        lambda: distributed(data),
    'centralized_network':lambda: centralized(data, with_network=True),
    'distributed_network':lambda: distributed(data, with_network=True,
                                               rho=2.0, tolerance=1e-3, max_iterations = 5000),
}

for case in CASES:
    if _ask(case):
        print(f"\n{'─'*50}")
        print(f"Running: {case}")
        print(f"{'─'*50}")
        _run(case, SOLVERS[case]())
    else:
        print(f"  Skipping {case}")

# ── Verification ──────────────────────────────────────────────────────────────

if not results:
    print("\nNo cases run — skipping verification.")
else:
    print(f"\n{'─'*50}")
    print("Verification")
    print(f"{'─'*50}")
    T, N  = len(data['buy_price']), len(data['demand'])
    zeros = np.zeros((N, T))

    for name, result in results.items():
        # Energy balance
        for i in range(N):
            for t in range(T):
                sale = sum(v for (a,b,h),v in result['trades'].items()
                           if a==i and h==t)
                bal  = (data['generation'][i][t] + result['buy'][i,t]
                        + result['discharge'][i,t]
                        + result.get('ev_discharge', zeros)[i,t]
                        - data['demand'][i][t]  - result['sell'][i,t]
                        - result['charge'][i,t]
                        - result.get('ev_charge', zeros)[i,t]
                        - result.get('hp_power',  zeros)[i,t] - sale)
                assert abs(bal) < 1e-5, \
                    f"{name} P{i+1} t={t}: balance={bal:.3g}"

        # BESS SOC
        for i in range(N):
            exp = (result['soc'][i,:-1]
                   + data['dt'] * (data['eta_charge'] * result['charge'][i]
                                   - result['discharge'][i] / data['eta_discharge']))
            assert np.allclose(result['soc'][i,1:], exp, atol=1e-6)
            assert abs(result['soc'][i,0]  - data['initial_soc'][i]) < 1e-6
            assert abs(result['soc'][i,-1] - data['initial_soc'][i]) < 1e-6

        # EV SOC
        for i in range(N):
            if data['ev_capacity'][i] > 0 and result.get('ev_soc') is not None:
                cap_eff = data['ev_capacity'][i] * data['ev_cap_factor'][i]
                drv     = np.array(data['ev_driving_load'][i])
                exp_ev  = (result['ev_soc'][i,:-1]
                           + data['dt'] * (
                               data['ev_eta_charge'] * result['ev_charge'][i]
                               - result['ev_discharge'][i] / data['ev_eta_discharge'])
                           - drv)
                assert np.allclose(result['ev_soc'][i,1:], exp_ev, atol=1e-6)
                assert result['ev_soc'][i].min() >= \
                    (1. - data['ev_dod_max']) * cap_eff - 1e-6

        # Reciprocity
        for i,j in data['edges']:
            for t in range(T):
                assert abs(result['trades'][i,j,t] + result['trades'][j,i,t]) < 1e-5

        # Network
        if result['network']:
            net  = result['network']
            vmin = min(net['voltage_pu'].values())
            vmax = max(net['voltage_pu'].values())
            print(f"  {name}: V=[{vmin:.4f},{vmax:.4f}]  "
                  f"viol={net['any_voltage_violation']}  "
                  f"cong={net['any_congestion']}")
            if name.endswith('_network'):
                assert not net['any_voltage_violation']
                assert not net['any_congestion']

        print(f"  {name}: {sum(result['costs']):.4f} EUR  ✓")

    # Cross-solver total cost check
    if 'centralized' in results and 'distributed' in results:
        diff = abs(sum(results['centralized']['costs']) -
                   sum(results['distributed']['costs']))
        assert diff < 1e-4, f"Centralized/distributed cost mismatch: {diff:.4g}"
        print("  Centralized == Distributed (total cost)  ✓")

    if 'centralized_network' in results and 'distributed_network' in results:
        diff = abs(sum(results['centralized_network']['costs']) -
                   sum(results['distributed_network']['costs']))
        assert diff < 1e-4, f"Network cost mismatch: {diff:.4g}"
        print("  Centralized_network == Distributed_network (total cost)  ✓")

    print("\nAll checks passed.")
