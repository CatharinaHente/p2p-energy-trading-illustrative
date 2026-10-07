# %% 1. Imports
from pathlib import Path
import json
import numpy as np
from algo import centralized, distributed
from network import evaluate_feeder
from plots import plot_for, plot_trades, plot_network, plot_balance, plot_devices
ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
 
# %% 2. Data
# Disable EV / HP by leaving capacity = 0; set > 0 to enable per prosumer.
scale            = 80.
battery_capacity = [3.*scale, 2.*scale, 2.*scale, 2.*scale]
_T = 4
 
data = dict(
    # grid / generation
    demand     = [[1.*scale]*4, [2.*scale]*4, [1.*scale]*4, [1.5*scale]*4],
    generation = [[0., 8.*scale, 5.*scale, 0.], [0., 0., 0., 0.],
                  [0., 0., 0., 0.],              [0., 0., 3.*scale, 0.]],
    edges      = [(0,1),(1,2),(2,3),(0,3)],
    agent_bus  = [5, 11, 17, 32],
    buy_price  = [.20, .12, .20, .45],
    sell_price = [.05]*4,
    p2p_price  = [.10, .08, .12, .25],
    trade_fee  = .001,
    dt         = 1.,
    grid_limit = 10.*scale,
    trade_limit= 5.*scale,
    line_limit_mva = 5.,
    # BESS
    battery_capacity = battery_capacity,
    battery_power    = [2.*scale, 1.*scale, 1.*scale, 1.*scale],
    battery_smax     = [2.*scale, 1.*scale, 1.*scale, 1.*scale],  # kVA converter rating
    battery_polygon_sides = 8,
    soc_min_fraction = .10,
    soc_max_fraction = .95,
    initial_soc      = [.3*c for c in battery_capacity],
    eta_charge       = .95,
    eta_discharge    = .95,
    battery_cost     = .005,
    # PV reactive power
    pf_pv_min = 0.90,   # REVIEW: EN 50549-1 / IEEE 1547-2018 §7.6
    # EV — all disabled
    ev_capacity          = [0.]*4,
    ev_cap_factor        = [1.0]*4,
    ev_power             = [0.]*4,
    ev_smax              = [0.]*4,
    ev_polygon_sides     = 8,
    ev_driving_load      = [[0.]*_T]*4,
    ev_soc_min_departure = [0.]*4,
    ev_dod_max           = 0.30,        # ref [16]
    ev_initial_soc       = [0.]*4,
    ev_availability      = [[1]*_T]*4,
    ev_eta_charge        = 0.95,
    ev_eta_discharge     = 0.95,
    ev_cost              = 0.005,
    ev_v2g               = True,
    # HP — all disabled
    hp_max_power  = [0.]*4,
    hp_base_power = [[0.]*_T]*4,
    hp_q_heat     = [[0.]*_T]*4,
    hp_T_init     = [20.]*4,
    hp_T_min      = 19.,
    hp_T_max      = 23.,
    hp_cost       = 0.0,
    # fixed power factors
    pf_hp   = 0.97,   # REVIEW: HP manufacturer data
    pf_load = 0.95,   # REVIEW: BDEW standard load profiles
    # Q regularisation — prevents free Q variables drifting to extremes in DSO QP
    # Same pattern as battery_cost / trade_fee; small enough not to change dispatch
    q_cost  = 1e-4,   # EUR/kVAR²·h
)
 
results = {}
print("Trading graph:", {i: sorted(j if i==a else a for a,j in data["edges"] if i in (a,j))
                         for i in range(len(data["demand"]))})
plot_for(data, ROOT)   # device FORs — run once after data is defined
 
 
# %% Helpers
 
def show(name, result):
    results[name] = result
    N = len(data["demand"])
    print(f"\n{name}  iter={result['iterations']}")
    for i, cost in enumerate(result["costs"]):
        print(f"  P{i+1}: {cost:.4f} EUR")
        for k in ["buy","sell","charge","discharge","soc"]:
            print(f"    {k}: {np.round(result[k][i],3)}")
        if data['ev_capacity'][i] > 0:
            for k in ["ev_charge","ev_discharge","ev_soc"]:
                if k in result: print(f"    {k}: {np.round(result[k][i],3)}")
        if data['hp_max_power'][i] > 0:
            for k in ["hp_power","temp"]:
                if k in result: print(f"    {k}: {np.round(result[k][i],3)}")
    out = ROOT / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    serial = {k: v.tolist() if isinstance(v, np.ndarray) else v
              for k,v in result.items() if k != "trades"}
    serial["trades"] = {f"{i}->{j}@{t}": v for (i,j,t),v in result["trades"].items()}
    (out / "summary.json").write_text(json.dumps(serial, indent=2))
 
 
def check_network(name, result):
    """Post-hoc feeder check; does not change the dispatch."""
    T, N   = len(data["buy_price"]), len(data["demand"])
    zeros  = np.zeros((N, T))
    p_inj  = (np.array(data["generation"]) - np.array(data["demand"])
              - result["charge"] + result["discharge"]
              - result.get("ev_charge",  zeros) + result.get("ev_discharge", zeros)
              - result.get("hp_power",   zeros))
    result["network"] = evaluate_feeder(
        p_inj, data["agent_bus"], T, data["line_limit_mva"],
        q_injections_kvar=result.get("q_injection"))
    net = result["network"]
    print(f"{name}: V_viol={net['any_voltage_violation']}  cong={net['any_congestion']}")
 
 
def _run(name, result):
    check_network(name, result)
    show(name, result)
    plot_trades(name, result, data, ROOT)
    plot_network(name, result, data, ROOT)
    plot_balance(name, result, data, ROOT)
    plot_devices(name, result, data, ROOT)
 
 
# %% 3–6. Four cases
_run("centralized",       centralized(data))
_run("distributed",       distributed(data))
_run("centralized_network", centralized(data, with_network=True))
_run("distributed_network", distributed(data, with_network=True))
 
# %% 7. Verification
T, N   = len(data['buy_price']), len(data['demand'])
zeros  = np.zeros((N, T))
 
for name, result in results.items():
    for i in range(N):
        for t in range(T):
            sale = sum(v for (a,b,h),v in result['trades'].items() if a==i and h==t)
            bal  = (data['generation'][i][t] + result['buy'][i,t]
                    + result['discharge'][i,t] + result.get('ev_discharge',zeros)[i,t]
                    - data['demand'][i][t] - result['sell'][i,t]
                    - result['charge'][i,t] - result.get('ev_charge',zeros)[i,t]
                    - result.get('hp_power',zeros)[i,t] - sale)
            assert abs(bal) < 1e-5, f"{name} P{i+1} t={t}: balance={bal:.3g}"
 
        exp = (result['soc'][i,:-1] + data['dt']
               * (data['eta_charge']*result['charge'][i]
                  - result['discharge'][i]/data['eta_discharge']))
        assert np.allclose(result['soc'][i,1:], exp, atol=1e-6)
        assert abs(result['soc'][i,0]  - data['initial_soc'][i]) < 1e-6
        assert abs(result['soc'][i,-1] - data['initial_soc'][i]) < 1e-6
        assert result['soc'][i].min() >= data['soc_min_fraction']*data['battery_capacity'][i]-1e-6
        assert result['soc'][i].max() <= data['soc_max_fraction']*data['battery_capacity'][i]+1e-6
 
        if data['ev_capacity'][i] > 0 and result.get('ev_soc') is not None:
            cap_eff = data['ev_capacity'][i] * data['ev_cap_factor'][i]
            drv     = np.array(data['ev_driving_load'][i])
            exp_ev  = (result['ev_soc'][i,:-1]
                       + data['dt']*(data['ev_eta_charge']*result['ev_charge'][i]
                                     - result['ev_discharge'][i]/data['ev_eta_discharge'])
                       - drv)
            assert np.allclose(result['ev_soc'][i,1:], exp_ev, atol=1e-6)
            assert result['ev_soc'][i].min() >= (1.-data['ev_dod_max'])*cap_eff-1e-6
 
    for i,j in data['edges']:
        for t in range(T):
            assert abs(result['trades'][i,j,t]+result['trades'][j,i,t]) < 1e-5
 
    if result['network']:
        net = result['network']
        vmin = min(net['voltage_pu'].values())
        vmax = max(net['voltage_pu'].values())
        print(f"  {name}: V=[{vmin:.4f},{vmax:.4f}] "
              f"viol={net['any_voltage_violation']} cong={net['any_congestion']}")
        if name.endswith('_network'):
            assert not net['any_voltage_violation']
            assert not net['any_congestion']
    print(f"  {name}: total cost {sum(result['costs']):.4f} EUR")
 
central   = results["centralized"]
local     = results["distributed"]
cn        = results["centralized_network"]
dn        = results["distributed_network"]
assert abs(sum(central['costs']) - sum(local['costs']))  < 1e-4
assert abs(sum(cn['costs'])      - sum(dn['costs']))     < 1e-4
assert central['charge'].sum() > .1
print("All four examples passed.")