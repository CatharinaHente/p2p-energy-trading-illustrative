"""Centralized and distributed ADMM P2P solvers.

centralized()  — single Gurobi model minimising sum of prosumer costs
distributed()  — consensus ADMM; market projection enforces reciprocity and
                 recovers a variational GNE (Behrunani et al. 2023, Baroche et al. 2019)
                 DSO projection enforces feeder feasibility on (P, Q) jointly

Both return the same result dict (see _result).
"""
import gurobipy as gp
import numpy as np
from prosumer import add_prosumer
from network import add_network


# --- Gurobi helpers ---

def _model(name):
    m = gp.Model(name)
    m.Params.OutputFlag = 0
    m.Params.FeasibilityTol = 1e-9
    m.Params.OptimalityTol  = 1e-9
    return m


def _optimize(model):
    model.optimize()
    if model.Status != gp.GRB.OPTIMAL:
        raise RuntimeError(f'{model.ModelName}: status {model.Status}')


# --- Result assembly ---

def _result(agents, iterations, residual=0., network=None):
    """Extract solved variable arrays from all agents.

    Returns dict with keys:
      buy, sell, charge, discharge, soc          — (N, T) or (N, T+1)
      ev_charge, ev_discharge, hp_power,
      q_batt, q_pv, ev_q_charger                 — (N, T), zeros if device absent
      ev_soc, temp                               — (N, T+1), zeros if device absent
      q_injection                                — (N, T) reactive power at bus
      trades                                     — dict {(i,j,t): float}
      costs                                      — (N,)
      iterations, residual, network
    """
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
    # violation flags — True if violated in ANY timestep
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
        source            = grid["name"],
        p_flow_mw         = {str(k): v.X  for k,v in p_flow.items()},
        q_flow_mvar       = {str(k): v.X  for k,v in q_flow.items()},
        voltage_pu        = {str(k): v    for k,v in vpu.items()},
        apparent_mva      = {str(k): v    for k,v in smva.items()},
        line_limit_mva    = line_limit_mva,
        voltage_violation = {str(k): v    for k,v in vviol.items()},
        congestion        = {str(k): v    for k,v in cong.items()},
        any_voltage_violation = any(vviol.values()),
        any_congestion        = any(cong.values()))


# --- Solvers ---

def centralized(data, with_network=False):
    """Social-cost minimiser. Returns result dict."""
    model = _model('centralized')
    N, T  = len(data['demand']), len(data['buy_price'])
    try:
        agents = [add_prosumer(model, i, data) for i in range(N)]
        for i,j in data['edges']:
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


def distributed(data, with_network=False,
                rho=.5, tolerance=1e-6, max_iterations=50000):
    """Consensus ADMM. Returns result dict.

    Iterations:
      1. Local step  — each prosumer minimises own cost + quadratic consensus penalty
      2. Market proj — z update enforces reciprocity; recovers variational GNE
      3. DSO proj    — (P,Q) joint QP projects injections onto feeder-feasible set
    Convergence: primal and dual residuals < tolerance (Boyd et al. 2011)
    """
    N, T   = len(data['demand']), len(data['buy_price'])
    models = [_model(f'p{i}') for i in range(N)]
    agents = [add_prosumer(m, i, data) for i,m in enumerate(models)]
    keys   = [(i,j,t) for i,a in enumerate(agents) for j,t in a['trade']]
    z, u   = dict.fromkeys(keys, 0.), dict.fromkeys(keys, 0.)
    s_p, w_p = np.zeros((N,T)), np.zeros((N,T))   # P consensus / dual
    s_q, w_q = np.zeros((N,T)), np.zeros((N,T))   # Q consensus / dual

    dso = None
    if with_network:
        dso   = _model('dso')
        inj_p = dso.addVars(N, T, lb=-gp.GRB.INFINITY, name='inj_p')
        inj_q = dso.addVars(N, T, lb=-gp.GRB.INFINITY, name='inj_q')
        p_flow, q_flow, voltage, grid = add_network(
            dso, inj_p, data['agent_bus'], T, data['line_limit_mva'],
            q_injections_kvar=inj_q)

    try:
        for iteration in range(1, max_iterations+1):

            # 1. Local step
            for i,(m,a) in enumerate(zip(models, agents)):
                pen = gp.quicksum((v - z[i,j,t] + u[i,j,t])**2
                                  for (j,t),v in a['trade'].items())
                if with_network:
                    pen += gp.quicksum(
                        (a['injection'][t]   - s_p[i,t] + w_p[i,t])**2
                        + (a['q_injection'][t] - s_q[i,t] + w_q[i,t])**2
                        for t in range(T))
                m.setObjective(a['cost'] + rho/2 * pen)
                _optimize(m)

            x      = {(i,j,t): agents[i]['trade'][j,t].X for i,j,t in keys}
            old_z  = z.copy()
            old_sp, old_sq = s_p.copy(), s_q.copy()

            # 2. Market projection — enforces q_ij + q_ji = 0
            for i,j in data['edges']:
                for t in range(T):
                    z[i,j,t] = (x[i,j,t] + u[i,j,t] - x[j,i,t] - u[j,i,t]) / 2
                    z[j,i,t] = -z[i,j,t]

            primal = [x[k] - z[k] for k in keys]
            dual   = [rho * (z[k] - old_z[k]) for k in keys]
            for k in keys: u[k] += x[k] - z[k]

            if with_network:
                p_loc = np.array([[a['injection'][t].getValue()   for t in range(T)] for a in agents])
                q_loc = np.array([[a['q_injection'][t].getValue() for t in range(T)] for a in agents])

                # 3. DSO projection — joint (P,Q) QP
                dso.setObjective(gp.quicksum(
                    (inj_p[i,t] - p_loc[i,t] - w_p[i,t])**2
                    + (inj_q[i,t] - q_loc[i,t] - w_q[i,t])**2
                    for i in range(N) for t in range(T)))
                _optimize(dso)
                s_p = np.array([[inj_p[i,t].X for t in range(T)] for i in range(N)])
                s_q = np.array([[inj_q[i,t].X for t in range(T)] for i in range(N)])
                w_p += p_loc - s_p
                w_q += q_loc - s_q
                primal.extend((p_loc - s_p).ravel()); primal.extend((q_loc - s_q).ravel())
                dual.extend((rho*(s_p - old_sp)).ravel()); dual.extend((rho*(s_q - old_sq)).ravel())

            residual = float(max(np.linalg.norm(primal), np.linalg.norm(dual)))
            if residual <= tolerance:
                return _result(agents, iteration, residual,
                               _network_values(p_flow, q_flow, voltage, grid, data['line_limit_mva'])
                               if with_network else None)

        raise RuntimeError(f'ADMM did not converge; residual={residual:.3g}')
    finally:
        for m in models: m.dispose()
        if dso is not None: dso.dispose()