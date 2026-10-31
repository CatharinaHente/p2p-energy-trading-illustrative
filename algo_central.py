"""algo_central.py — Centralized (social-cost) solver for the P2P market.

Exposes:
    centralized(data, with_network=False)  →  result dict

The shared Gurobi helpers (_model, _optimize, _result, _network_values)
are also imported by algo_ADMM.py, which is why they live here rather
than being duplicated.
"""
import gurobipy as gp
import numpy as np
from prosumer import add_prosumer
from network import add_network


# ── Gurobi helpers ────────────────────────────────────────────────────────────

def _model(name):
    m = gp.Model(name)
    m.Params.OutputFlag     = 0
    m.Params.FeasibilityTol = 1e-9
    m.Params.OptimalityTol  = 1e-9
    return m


def _optimize(model):
    model.optimize()
    if model.Status not in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL):
        raise RuntimeError(f'{model.ModelName}: status {model.Status}')
    if model.Status == gp.GRB.SUBOPTIMAL:
        print(f"  ⚠ {model.ModelName}: suboptimal solution accepted")


# ── Result assembly ───────────────────────────────────────────────────────────

def _result(agents, iterations, residual=0., network=None):
    """Extract solved variable arrays from all agents."""
    N = len(agents)
    T = len(agents[0]['buy'])

    result = {k: np.array([[v.X for v in a[k].values()] for a in agents])
              for k in ['buy', 'sell', 'charge', 'discharge', 'soc']}

    for k in ['ev_charge', 'ev_discharge', 'hp_power', 'q_batt', 'q_pv', 'ev_q_charger']:
        result[k] = (np.array([[v.X for v in a[k].values()] for a in agents])
                     if k in agents[0] else np.zeros((N, T)))

    for k in ['ev_soc', 'temp']:
        rows = []
        for a in agents:
            val = a.get(k)
            rows.append(np.zeros(T+1) if val is None
                        else np.array([v.X for v in val.values()]))
        result[k] = np.array(rows)

    result['q_injection'] = (
        np.array([[a['q_injection'][t].getValue() for t in range(T)] for a in agents])
        if 'q_injection' in agents[0] else np.zeros((N, T)))

    result.update(
        trades     = {(i,j,t): v.X for i,a in enumerate(agents)
                      for (j,t),v in a['trade'].items()},
        costs      = np.array([a['cost'].getValue() for a in agents]),
        iterations = iterations,
        residual   = residual,
        network    = network)
    return result


def _network_values(p_flow, q_flow, voltage, grid, line_limit_mva):
    """Pack solved network variables into a result dict."""
    vpu  = {k: v.X**.5 for k,v in voltage.items()}
    smva = {k: (p_flow[k].X**2 + q_flow[k].X**2)**.5 for k in p_flow}
    vviol = {}
    for (bus, t), vp in vpu.items():
        if vp < grid["voltage_min"][bus]-1e-9 or vp > grid["voltage_max"][bus]+1e-9:
            vviol[bus] = True
        else:
            vviol.setdefault(bus, False)
    cong = {}
    for (u, v, t), s in smva.items():
        if s > line_limit_mva+1e-9:
            cong[u, v] = True
        else:
            cong.setdefault((u, v), False)
    return dict(
        source                = grid["name"],
        p_flow_mw             = {str(k): v.X  for k,v in p_flow.items()},
        q_flow_mvar           = {str(k): v.X  for k,v in q_flow.items()},
        voltage_pu            = {str(k): v    for k,v in vpu.items()},
        apparent_mva          = {str(k): v    for k,v in smva.items()},
        line_limit_mva        = line_limit_mva,
        voltage_violation     = {str(k): v    for k,v in vviol.items()},
        congestion            = {str(k): v    for k,v in cong.items()},
        any_voltage_violation = any(vviol.values()),
        any_congestion        = any(cong.values()))


# ── Solver ────────────────────────────────────────────────────────────────────

def centralized(data, with_network=False):
    """Social-cost minimiser. Returns result dict.

    Builds a single Gurobi model containing all N prosumers, adds the
    reciprocity constraints (q_ij + q_ji = 0), and optionally the
    LinDistFlow network constraints, then minimises the sum of costs.

    Parameters
    ----------
    data         : dict   scenario data dict from make_scenario()
    with_network : bool   include LinDistFlow voltage / thermal constraints
    """
    model = _model('centralized')
    N, T  = len(data['demand']), len(data['buy_price'])
    try:
        agents = [add_prosumer(model, i, data) for i in range(N)]
        for i, j in data['edges']:
            for t in range(T):
                model.addConstr(agents[i]['trade'][j,t] + agents[j]['trade'][i,t] == 0)
        if with_network:
            p_inj = {(i,t): a['injection'][t]   for i,a in enumerate(agents) for t in range(T)}
            q_inj = {(i,t): a['q_injection'][t] for i,a in enumerate(agents) for t in range(T)}
            p_flow, q_flow, voltage, grid = add_network(
                model, p_inj, data['agent_bus'], T, data['line_limit_mva'],
                q_injections_kvar=q_inj)
        model.setObjective(gp.quicksum(a['cost'] for a in agents))
        _optimize(model)
        return _result(agents, 0, network=(
            _network_values(p_flow, q_flow, voltage, grid, data['line_limit_mva'])
            if with_network else None))
    finally:
        model.dispose()