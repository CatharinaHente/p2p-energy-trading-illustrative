"""Centralized and distributed (ADMM) P2P solvers.

[UPDATED] _result() now extracts new device variables from prosumer.py:
  ev_charge, ev_discharge, ev_soc, hp_power, temp, q_batt, ev_q_charger, q_injection.
  All additions use .get() guards for backward compatibility with older prosumer.py.

[UPDATED] distributed() DSO projection is now (P, Q)-aware.
  Each ADMM iteration tracks consensus copies (s_p, s_q) and scaled duals (w_p, w_q)
  for both active and reactive power.  The DSO solves a joint projection QP:
      min_{inj_p, inj_q feeder-feasible} sum_{i,t} (inj_p - p_local - w_p)^2
                                                   + (inj_q - q_local - w_q)^2
  where p_local = a['injection'][t].getValue() and q_local = a['q_injection'][t].getValue()
  are evaluated from the most recent local prosumer solve.
  The local penalty is extended symmetrically with a Q consensus term.
"""
import gurobipy as gp
import numpy as np
from prosumer import add_prosumer
from network import add_network


def _model(name):
    model = gp.Model(name)
    model.Params.OutputFlag     = 0
    model.Params.FeasibilityTol = 1e-9
    model.Params.OptimalityTol  = 1e-9
    return model


def _optimize(model):
    model.optimize()
    if model.Status != gp.GRB.OPTIMAL:
        raise RuntimeError(f'{model.ModelName}: Gurobi status {model.Status}')


def _result(agents, iterations, residual=0., network=None):
    """Collect optimised variable values from all agents into numpy arrays.

    [UPDATED] Extracts new device arrays alongside original ones.
    Variables absent in old prosumer.py default to zero arrays.
    ev_soc and temp may be None per-agent (device disabled) — replaced by zeros.
    """
    N = len(agents)
    T = len(agents[0]['buy'])

    # Original variables (T timesteps each)
    result = {key: np.array([[v.X for v in a[key].values()] for a in agents])
              for key in ['buy', 'sell', 'charge', 'discharge', 'soc']}

    # [NEW] New device variables — T timesteps; zero if key absent
    for key in ['ev_charge', 'ev_discharge', 'hp_power', 'q_batt', 'ev_q_charger']:
        if key in agents[0]:
            result[key] = np.array([[v.X for v in a[key].values()] for a in agents])
        else:
            result[key] = np.zeros((N, T))

    # [NEW] T+1 timestep arrays — ev_soc and temp may be None per agent
    for key in ['ev_soc', 'temp']:
        rows = []
        for a in agents:
            val = a.get(key)
            if val is None:
                rows.append(np.zeros(T + 1))
            else:
                rows.append(np.array([v.X for v in val.values()]))
        result[key] = np.array(rows)

    # [NEW] Reactive power injection (LinExpr evaluated after solve)
    if 'q_injection' in agents[0]:
        result['q_injection'] = np.array(
            [[a['q_injection'][t].getValue() for t in range(T)] for a in agents])
    else:
        result['q_injection'] = np.zeros((N, T))

    result.update(
        trades     = {(i,j,t): v.X
                      for i,a in enumerate(agents)
                      for (j,t),v in a['trade'].items()},
        costs      = np.array([a['cost'].getValue() for a in agents]),
        iterations = iterations,
        residual   = residual,
        network    = network,
    )
    return result


def _network_values(p_flow, q_flow, voltage, grid, line_limit_mva):
    """Assemble network result dict from solved Gurobi variables. Unchanged."""
    voltage_pu   = {k: v.X**.5 for k,v in voltage.items()}
    apparent_mva = {k: (p_flow[k].X**2 + q_flow[k].X**2)**.5 for k in p_flow}
    voltage_violation = {bus: False for bus,_ in voltage_pu}
    for (bus,t), vpu in voltage_pu.items():
        if vpu < grid["voltage_min"][bus]-1e-9 or vpu > grid["voltage_max"][bus]+1e-9:
            voltage_violation[bus] = True
    congestion = {(u,v): False for u,v,t in apparent_mva}
    for (u,v,t), s in apparent_mva.items():
        if s > line_limit_mva + 1e-9:
            congestion[u,v] = True
    return dict(
        source                = grid["name"],
        p_flow_mw             = {str(k): v.X for k,v in p_flow.items()},
        q_flow_mvar           = {str(k): v.X for k,v in q_flow.items()},
        voltage_pu            = {str(k): v for k,v in voltage_pu.items()},
        apparent_mva          = {str(k): v for k,v in apparent_mva.items()},
        line_limit_mva        = line_limit_mva,
        voltage_violation     = {str(bus):  flag for bus,  flag in voltage_violation.items()},
        congestion            = {str(edge): flag for edge, flag in congestion.items()},
        any_voltage_violation = any(voltage_violation.values()),
        any_congestion        = any(congestion.values()),
    )


def centralized(data, with_network=False):
    """Centralized social-cost minimiser.

    [UPDATED] Prosumer reactive power injections (q_injection) forwarded to
    add_network via q_injections_kvar so LinDistFlow Q balance and voltage
    drop equations reflect reactive power from BESS and EV chargers.
    """
    model = _model('centralized')
    N, T  = len(data['demand']), len(data['buy_price'])
    try:
        agents = [add_prosumer(model, i, data) for i in range(N)]
        for i,j in data['edges']:
            for t in range(T):
                model.addConstr(agents[i]['trade'][j,t] + agents[j]['trade'][i,t] == 0)
        if with_network:
            injections = {(i,t): a['injection'][t]
                          for i,a in enumerate(agents) for t in range(T)}
            # [NEW] Reactive power injections passed to add_network
            q_injections = {(i,t): a['q_injection'][t]
                            for i,a in enumerate(agents) for t in range(T)}
            p_flow, q_flow, voltage, grid = add_network(
                model, injections, data['agent_bus'], T, data['line_limit_mva'],
                q_injections_kvar=q_injections)
        model.setObjective(gp.quicksum(a['cost'] for a in agents))
        _optimize(model)
        return _result(
            agents, 0,
            network=(
                _network_values(p_flow, q_flow, voltage, grid, data['line_limit_mva'])
                if with_network else None
            )
        )
    finally:
        model.dispose()


def distributed(data, with_network=False,
                rho=.5, tolerance=1e-6, max_iterations=50000):
    """Distributed ADMM solver with (P, Q)-aware DSO projection.

    [UPDATED] The DSO projection now handles both active and reactive power
    jointly.  The structure mirrors the existing P-only approach exactly:

      P side (original):  consensus s_p, scaled dual w_p, DSO variable inj_p
      Q side [NEW]:       consensus s_q, scaled dual w_q, DSO variable inj_q

    The local step penalty is extended with a Q consensus term:
      rho/2 * (q_injection[t] - s_q[i,t] + w_q[i,t])^2
    q_injection[t] is a Gurobi LinExpr; squaring it produces a QuadExpr
    that Gurobi accepts in a quadratic objective.

    The DSO solves a joint QP each iteration:
      min_{inj_p, inj_q: feeder feasible}
          sum_{i,t} (inj_p - p_local - w_p)^2 + (inj_q - q_local - w_q)^2
    """
    N, T   = len(data['demand']), len(data['buy_price'])
    models = [_model(f'prosumer_{i}') for i in range(N)]
    agents = [add_prosumer(m, i, data) for i,m in enumerate(models)]
    keys   = [(i,j,t) for i,a in enumerate(agents) for j,t in a['trade']]
    z, u   = dict.fromkeys(keys, 0.), dict.fromkeys(keys, 0.)

    # P consensus and scaled dual (original, renamed for clarity)
    s_p, w_p = np.zeros((N,T)), np.zeros((N,T))
    # [NEW] Q consensus and scaled dual — same structure as P
    s_q, w_q = np.zeros((N,T)), np.zeros((N,T))

    dso = None
    if with_network:
        dso   = _model('dso_projection')
        inj_p = dso.addVars(N, T, lb=-gp.GRB.INFINITY, name='inj_p')
        inj_q = dso.addVars(N, T, lb=-gp.GRB.INFINITY, name='inj_q')   # [NEW]
        p_flow, q_flow, voltage, grid = add_network(
            dso, inj_p, data['agent_bus'], T, data['line_limit_mva'],
            q_injections_kvar=inj_q)                                     # [NEW]

    try:
        for iteration in range(1, max_iterations+1):

            # ---------------------------------------------------------------- #
            # Local step: each prosumer minimises its own cost plus consensus  #
            # penalties pulling trades, P injection and [NEW] Q injection      #
            # toward their respective DSO consensus copies.                    #
            # ---------------------------------------------------------------- #
            for i,(m,a) in enumerate(zip(models, agents)):
                penalty = gp.quicksum(
                    (v - z[i,j,t] + u[i,j,t])**2
                    for (j,t),v in a['trade'].items())
                if with_network:
                    penalty += gp.quicksum(
                        (a['injection'][t]   - s_p[i,t] + w_p[i,t])**2    # P (original)
                        + (a['q_injection'][t] - s_q[i,t] + w_q[i,t])**2  # [NEW] Q
                        for t in range(T))
                m.setObjective(a['cost'] + rho/2 * penalty)
                _optimize(m)

            x      = {(i,j,t): agents[i]['trade'][j,t].X for i,j,t in keys}
            old_z  = z.copy()
            old_sp = s_p.copy()
            old_sq = s_q.copy()   # [NEW]

            # Market projection: reciprocal bilateral contracts (unchanged)
            for i,j in data['edges']:
                for t in range(T):
                    z[i,j,t] = (x[i,j,t] + u[i,j,t] - x[j,i,t] - u[j,i,t]) / 2
                    z[j,i,t] = -z[i,j,t]

            primal = [x[k] - z[k] for k in keys]
            dual   = [rho * (z[k] - old_z[k]) for k in keys]
            for k in keys: u[k] += x[k] - z[k]

            if with_network:
                # Evaluate local P and Q injections after each prosumer solve
                p_local = np.array(
                    [[a['injection'][t].getValue()   for t in range(T)] for a in agents])
                q_local = np.array(
                    [[a['q_injection'][t].getValue() for t in range(T)] for a in agents])  # [NEW]

                # DSO projection: joint (P, Q) QP onto feeder-feasible set
                dso.setObjective(gp.quicksum(
                    (inj_p[i,t] - p_local[i,t] - w_p[i,t])**2
                    + (inj_q[i,t] - q_local[i,t] - w_q[i,t])**2          # [NEW]
                    for i in range(N) for t in range(T)))
                _optimize(dso)

                # Extract DSO consensus copies
                s_p = np.array([[inj_p[i,t].X for t in range(T)] for i in range(N)])
                s_q = np.array([[inj_q[i,t].X for t in range(T)] for i in range(N)])  # [NEW]

                # Dual updates — P (original) and Q [NEW]
                w_p += p_local - s_p
                w_q += q_local - s_q   # [NEW]

                # Residuals — P (original) and Q [NEW]
                primal.extend((p_local - s_p).ravel())
                primal.extend((q_local - s_q).ravel())                    # [NEW]
                dual.extend((rho * (s_p - old_sp)).ravel())
                dual.extend((rho * (s_q - old_sq)).ravel())               # [NEW]

            residual = float(max(np.linalg.norm(primal), np.linalg.norm(dual)))
            if residual <= tolerance:
                return _result(
                    agents, iteration, residual,
                    _network_values(p_flow, q_flow, voltage, grid, data['line_limit_mva'])
                    if with_network else None
                )

        raise RuntimeError(f'ADMM did not converge; residual={residual:.3g}')
    finally:
        for m in models: m.dispose()
        if dso is not None: dso.dispose()