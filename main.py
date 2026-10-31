"""main.py — P2P electricity market optimisation (interactive single-run).

Cases: centralized | distributed | centralized_network | distributed_network
Set INTERACTIVE=False to run all without prompts.

Results are saved to:
    structures/{STRUCTURE}/results/{DATE}/{case}.json
consistent with batch_main.py so both tools share the same result library.
"""
from pathlib import Path
import json
import numpy as np
from algo_central import centralized
from algo_ADMM import distributed
from scenarios import make_scenario_date, prosumer_summary
from network import evaluate_feeder
from plots import (plot_for, plot_prices, plot_trades,
                   plot_network, plot_balance, plot_devices)
from metrics import no_trade_costs, savings, sw, tw, mm, jains, var_savings

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# ── Configuration ─────────────────────────────────────────────────────────────
STRUCTURE    = 'base'
DATE         = '2016-10-16'  # any YYYY-MM-DD in 2016; season inferred from month

RESOLUTION_H = 0.5        # 30-min steps → T=48
SEED         = 42

ADMM_RHO       = 10.0    # initial ρ (adaptive scaling in algo_ADMM.py)
ADMM_TOLERANCE = 1e-5    # convergence: max(||primal||, ||dual||)

INTERACTIVE = True        # True → prompt before each step; False → run everything
DEBUG       = True        # True → print scenario data checks and device trajectories

# Result directory — shared with batch_main.py
RESULT_DIR = ROOT / 'structures' / STRUCTURE / 'results' / DATE


def _ask(label):
    if not INTERACTIVE:
        return True
    ans = input(f"\nRun {label}? [Y/n]: ").strip().lower()
    return ans not in ('n', 'no')


# ── Debug helpers ─────────────────────────────────────────────────────────────

def _debug_scenario(data):
    """Print per-prosumer data ranges and flag likely infeasibility causes."""
    N  = len(data['demand'])
    T  = len(data['buy_price'])
    dt = data['dt']
    print(f"\n{'─'*50}")
    print(f"Scenario debug  {data.get('date', data.get('season', ''))}  "
          f"T={T}  dt={dt}h")
    print(f"{'─'*50}")
    for i in range(N):
        name = data['prosumer_names'][i]
        dem  = np.array(data['demand'][i])
        gen  = np.array(data['generation'][i])
        print(f"\nP{i+1} ({name})")
        print(f"  demand:     min={dem.min():.2f}  max={dem.max():.2f}  "
              f"mean={dem.mean():.2f}  total={dem.sum()*dt:.2f} kWh")
        print(f"  generation: min={gen.min():.2f}  max={gen.max():.2f}  "
              f"total={gen.sum()*dt:.2f} kWh")

        # BESS
        cap = data['battery_capacity'][i]
        if cap > 0:
            e_min = data['soc_min_fraction'] * cap
            e_max = data['soc_max_fraction'] * cap
            e0    = data['initial_soc'][i]
            print(f"  BESS:       {cap:.1f} kWh / {data['battery_power'][i]:.1f} kW  "
                  f"SOC [{e_min:.1f}, {e_max:.1f}]  init={e0:.1f} kWh")

        # HP — check if base profile ever exceeds rated power
        hp_max = data['hp_max_power'][i]
        if hp_max > 0:
            base = np.array(data['hp_base_power'][i])
            q    = np.array(data['hp_q_heat'][i])
            over = (base > hp_max + 1e-6).sum()
            print(f"  HP:         rated={hp_max:.1f} kW  "
                  f"base max={base.max():.2f} kW  "
                  f"q max={q.max():.4f} °C/h  "
                  f"T init={data['hp_T_init'][i]:.1f} °C  "
                  f"bounds=[{data['hp_T_min']:.0f}, {data['hp_T_max']:.0f}] °C")
            if over > 0:
                print(f"  ⚠  HP base power exceeds rated at {over}/{T} timesteps "
                      f"— temperature may be uncontrollable (infeasibility risk)")

        # EV
        ev_cap = data['ev_capacity'][i]
        if ev_cap > 0:
            avail  = np.array(data['ev_availability'][i])
            drive  = np.array(data['ev_driving_load'][i])
            cap_eff = ev_cap * data['ev_cap_factor'][i]
            dod_floor = (1 - data['ev_dod_max']) * cap_eff
            dep   = data['ev_soc_min_departure'][i] * cap_eff
            total_drive = drive.sum()
            home_steps  = avail.sum()
            max_charge  = home_steps * data['ev_power'][i] * dt * data['ev_eta_charge']
            print(f"  EV:         {ev_cap:.0f} kWh nom  eff={cap_eff:.1f} kWh  "
                  f"DoD floor={dod_floor:.1f} kWh  dep SOC={dep:.1f} kWh")
            print(f"             driving={total_drive:.2f} kWh  "
                  f"home={home_steps}/{T} steps  "
                  f"max charge={max_charge:.2f} kWh")
            if total_drive > max_charge + 1e-3:
                print(f"  ⚠  EV driving load ({total_drive:.2f} kWh) "
                      f"exceeds maximum chargeable ({max_charge:.2f} kWh) "
                      f"— SOC cyclic constraint may be infeasible")

    # Price summary
    bp = np.array(data['buy_price'])
    sp = np.array(data['sell_price'])
    pp = np.array(data['p2p_price'])
    print(f"\n  Prices:  buy [{bp.min():.2f}, {bp.max():.2f}]  "
          f"sell {sp.mean():.2f}  p2p [{pp.min():.2f}, {pp.max():.2f}] EUR/kWh")
    print(f"{'─'*50}")


def _debug_result(name, result, data):
    """Print device trajectories and network summary after a solve."""
    N  = len(data['demand'])
    T  = len(data['buy_price'])
    dt = data['dt']
    print(f"\n  [{name}] device summary")
    for i in range(N):
        pname = data['prosumer_names'][i]
        buy   = result['buy'][i]
        sell  = result['sell'][i]
        ch    = result['charge'][i]
        di    = result['discharge'][i]
        soc   = result['soc'][i]
        print(f"  P{i+1} ({pname})")
        print(f"    grid:  buy={buy.sum()*dt:.2f} kWh  sell={sell.sum()*dt:.2f} kWh")
        print(f"    BESS:  charge={ch.sum()*dt:.2f}  discharge={di.sum()*dt:.2f} kWh  "
              f"SOC [{soc.min():.2f}, {soc.max():.2f}] kWh")

        if data['ev_capacity'][i] > 0 and result.get('ev_soc') is not None:
            ec  = result['ev_charge'][i]
            ed  = result['ev_discharge'][i]
            es  = result['ev_soc'][i]
            print(f"    EV:    charge={ec.sum()*dt:.2f}  discharge={ed.sum()*dt:.2f} kWh  "
                  f"SOC [{es.min():.2f}, {es.max():.2f}] kWh")

        if data['hp_max_power'][i] > 0 and result.get('temp') is not None:
            hp  = result['hp_power'][i]
            tmp = result['temp'][i]
            print(f"    HP:    energy={hp.sum()*dt:.2f} kWh  "
                  f"temp [{tmp.min():.2f}, {tmp.max():.2f}] °C")

        peers = [b if a==i else a for a,b in data['edges'] if i in (a,b)]
        net   = sum(result['trades'].get((i,j,t),0.)
                    for j in peers for t in range(T))
        print(f"    P2P:   net export={net*dt:.2f} kWh")

    if result.get('network'):
        net  = result['network']
        vmin = min(float(v) for v in net['voltage_pu'].values())
        vmax = max(float(v) for v in net['voltage_pu'].values())
        smax = max(float(v) for v in net['apparent_mva'].values())
        print(f"  network: V=[{vmin:.4f}, {vmax:.4f}] pu  "
              f"S_max={smax:.3f} MVA  "
              f"V_viol={net['any_voltage_violation']}  "
              f"cong={net['any_congestion']}")


# ── Scenario ──────────────────────────────────────────────────────────────────
data   = make_scenario_date(DATE, seed=SEED, resolution_h=RESOLUTION_H)
SEASON = data['season']   # derived from month — used for price plots

print("\nProsumer configuration:")
print(prosumer_summary().to_string())
print("\nTrading graph:", {
    i: sorted(j if i == a else a for a, j in data["edges"] if i in (a, j))
    for i in range(len(data["demand"]))})

if DEBUG:
    _debug_scenario(data)

if _ask('scenario plots (FOR + prices)'):
    plot_for(data, RESULT_DIR)
    plot_prices(SEASON, data, RESULT_DIR)

# ── No-trade baseline ─────────────────────────────────────────────────────────
# Load from cache if batch_main.py already computed it for this date
_nt_path = RESULT_DIR / 'no_trade.json'
if _nt_path.exists():
    J_nt = np.array(json.loads(_nt_path.read_text())['J_nt'])
    print("No-trade baseline: loaded from cache")
else:
    J_nt = no_trade_costs(data)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    _nt_path.write_text(json.dumps({'J_nt': J_nt.tolist(), 'date': DATE}))

results = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save(name, result):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    serial = {k: v.tolist() if isinstance(v, np.ndarray) else v
              for k, v in result.items() if k != "trades"}
    serial["trades"] = {f"{i}->{j}@{t}": v
                        for (i, j, t), v in result["trades"].items()}
    serial['_date']      = DATE
    serial['_case']      = name
    serial['_structure'] = STRUCTURE
    (RESULT_DIR / f'{name}.json').write_text(json.dumps(serial, indent=2))


def _print_result(name, result):
    results[name] = result
    print(f"\n{name}  total={sum(result['costs']):.4f} EUR")
    for i, cost in enumerate(result["costs"]):
        line = f"  P{i+1} ({data['prosumer_names'][i]}): {cost:.4f} EUR"
        if data['ev_capacity'][i] > 0 and result.get('ev_soc') is not None:
            ev = result['ev_soc']
            line += f"  ev_soc=[{ev[i, 0]:.1f}->{ev[i, -1]:.1f}] kWh"
        if data['hp_max_power'][i] > 0 and result.get('temp') is not None:
            tmp = result['temp']
            line += f"  temp=[{tmp[i, 0]:.2f}->{tmp[i, -1]:.2f}] C"
        print(line)


def _check_network(name, result):
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
    print(f"  network: V_viol={net['any_voltage_violation']}"
          f"  cong={net['any_congestion']}")


def _run(name, result):
    _check_network(name, result)
    _print_result(name, result)
    if DEBUG:
        _debug_result(name, result, data)
    _save(name, result)
    plot_trades(name, result, data, RESULT_DIR)
    plot_network(name, result, data, RESULT_DIR)
    plot_balance(name, result, data, RESULT_DIR)
    plot_devices(name, result, data, RESULT_DIR)


# ── Solver runs ───────────────────────────────────────────────────────────────

if _ask('centralized'):
    print(f"\n{'─'*50}\nRunning: centralized")
    _run('centralized', centralized(data))

if _ask('distributed'):
    print(f"\n{'─'*50}\nRunning: distributed")
    _run('distributed', distributed(data, rho=ADMM_RHO, tolerance=ADMM_TOLERANCE))

if _ask('centralized_network'):
    print(f"\n{'─'*50}\nRunning: centralized_network")
    _run('centralized_network', centralized(data, with_network=True))

if _ask('distributed_network'):
    print(f"\n{'─'*50}\nRunning: distributed_network")
    _run('distributed_network',
         distributed(data, rho=ADMM_RHO, tolerance=ADMM_TOLERANCE, with_network=True))


# ── Fairness metrics ──────────────────────────────────────────────────────────

print(f"\n{'─'*50}\nFairness metrics")
for name, result in results.items():
    d    = savings(result, J_nt)
    J_so = np.array(result['costs'])
    print(f"\n{name}:")
    print(f"  TW={tw(J_nt, J_so):.4f}  SW={sw(d):.4f}  MM={mm(d):.4f}"
          f"  JI={jains(d):.4f}  Var={var_savings(d):.6f}")
    for i, di in enumerate(d):
        print(f"  P{i+1} ({data['prosumer_names'][i]}): d={di:+.4f}")


# ── Verification ──────────────────────────────────────────────────────────────

print(f"\n{'─'*50}\nVerification")
T, N  = len(data['buy_price']), len(data['demand'])
zeros = np.zeros((N, T))

for name, result in results.items():

    for i in range(N):
        for t in range(T):
            sale = sum(v for (a, b, h), v in result['trades'].items()
                       if a == i and h == t)
            bal  = (data['generation'][i][t] + result['buy'][i, t]
                    + result['discharge'][i, t]
                    + result.get('ev_discharge', zeros)[i, t]
                    - data['demand'][i][t]   - result['sell'][i, t]
                    - result['charge'][i, t]
                    - result.get('ev_charge', zeros)[i, t]
                    - result.get('hp_power',  zeros)[i, t] - sale)
            assert abs(bal) < 1e-4, f"{name} P{i+1} t={t}: balance={bal:.3g}"

    for i in range(N):
        exp = (result['soc'][i, :-1]
               + data['dt'] * (data['eta_charge'] * result['charge'][i]
                               - result['discharge'][i] / data['eta_discharge']))
        assert np.allclose(result['soc'][i, 1:], exp, atol=1e-6)
        assert abs(result['soc'][i, 0]  - data['initial_soc'][i]) < 1e-6
        assert abs(result['soc'][i, -1] - data['initial_soc'][i]) < 1e-6

    for i in range(N):
        if data['ev_capacity'][i] > 0 and result.get('ev_soc') is not None:
            cap_eff = data['ev_capacity'][i] * data['ev_cap_factor'][i]
            drv     = np.array(data['ev_driving_load'][i])
            exp_ev  = (result['ev_soc'][i, :-1]
                       + data['dt'] * (
                           data['ev_eta_charge'] * result['ev_charge'][i]
                           - result['ev_discharge'][i] / data['ev_eta_discharge'])
                       - drv)
            assert np.allclose(result['ev_soc'][i, 1:], exp_ev, atol=1e-6)
            assert result['ev_soc'][i].min() >= (1. - data['ev_dod_max']) * cap_eff - 1e-6

    for i, j in data['edges']:
        for t in range(T):
            assert abs(result['trades'][i, j, t] + result['trades'][j, i, t]) < 1e-4

    if result.get('network'):
        net  = result['network']
        vmin = min(net['voltage_pu'].values())
        vmax = max(net['voltage_pu'].values())
        print(f"  {name}: V=[{vmin:.4f},{vmax:.4f}]"
              f"  viol={net['any_voltage_violation']}"
              f"  cong={net['any_congestion']}")
        if name.endswith('_network'):
            assert not net['any_voltage_violation']
            assert not net['any_congestion']

    print(f"  {name}: {sum(result['costs']):.4f} EUR  ok")

if 'centralized' in results and 'distributed' in results:
    diff = abs(sum(results['centralized']['costs']) - sum(results['distributed']['costs']))
    assert diff < 1e-3, f"Cost mismatch: {diff:.4g}"
    print("  centralized == distributed  ok")

if 'centralized_network' in results and 'distributed_network' in results:
    diff = abs(sum(results['centralized_network']['costs'])
               - sum(results['distributed_network']['costs']))
    assert diff < 1e-3, f"Network cost mismatch: {diff:.4g}"
    print("  centralized_network == distributed_network  ok")

print("\nAll checks passed.")