"""Distributed ADMM P2P solver

distributed()        — consensus ADMM; recovers a variational GNE

Returns the same dict. as centralised solver
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


# ── Solvers ───────────────────────────────────────────────────────────────────

def distributed(data, with_network=False,
                rho=2.0, tolerance=1e-4, max_iterations=50000,
                adaptive_rho=True, mu=10.0, tau=2.0,
                log_history=False):
    """Consensus ADMM. Returns result dict.

    Iterations:
      1. Local step  — each prosumer minimises own cost + quadratic consensus penalty
      2. Market proj — z update enforces reciprocity; recovers variational GNE
      3. DSO proj    — (P,Q) joint QP projects injections onto feeder-feasible set
    Convergence: primal and dual residuals < tolerance (Boyd et al. 2011)

    Adaptive rho (Boyd et al. 2011, Section 3.4.1):
      rho is scaled up when primal residual dominates, down when dual dominates.
      This balances convergence speed and prevents oscillation near the optimum.
      Typically reduces iteration count by 10-100× vs fixed rho.

    Parameters
    ----------
    rho          : float  initial penalty parameter
    tolerance    : float  convergence threshold for primal and dual residuals
    adaptive_rho : bool   enable adaptive rho scaling (recommended)
    mu           : float  imbalance threshold — rescale when ratio > mu (default 10)
    tau          : float  scaling factor for rho update (default 2)
    """
    N, T   = len(data['demand']), len(data['buy_price'])
    models = [_model(f'p{i}') for i in range(N)]
    agents = [add_prosumer(m, i, data) for i,m in enumerate(models)]
    keys   = [(i,j,t) for i,a in enumerate(agents) for j,t in a['trade']]
    z, u   = dict.fromkeys(keys, 0.), dict.fromkeys(keys, 0.)
    s_p, w_p = np.zeros((N,T)), np.zeros((N,T))
    s_q, w_q = np.zeros((N,T)), np.zeros((N,T))

    dso = None
    if with_network:
        dso   = _model('dso')
        inj_p = dso.addVars(N, T, lb=-gp.GRB.INFINITY, name='inj_p')
        inj_q = dso.addVars(N, T, lb=-gp.GRB.INFINITY, name='inj_q')
        p_flow, q_flow, voltage, grid = add_network(
            dso, inj_p, data['agent_bus'], T, data['line_limit_mva'],
            q_injections_kvar=inj_q)

    history = [] if log_history else None

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
                primal.extend((p_loc - s_p).ravel())
                primal.extend((q_loc - s_q).ravel())
                dual.extend((rho*(s_p - old_sp)).ravel())
                dual.extend((rho*(s_q - old_sq)).ravel())

            r_norm = float(np.linalg.norm(primal))
            s_norm = float(np.linalg.norm(dual))
            residual = max(r_norm, s_norm)

            if log_history:
                history.append((iteration, residual, rho))

            if residual <= tolerance:
                result = _result(agents, iteration, residual,
                               _network_values(p_flow, q_flow, voltage, grid,
                                               data['line_limit_mva'])
                               if with_network else None)
                if log_history:
                    result['history'] = history
                return result

            # Adaptive rho — Boyd et al. (2011) Section 3.4.1
            # Rescale when primal and dual residuals are badly imbalanced.
            # When rho changes, dual variables u must be rescaled to keep
            # the augmented Lagrangian consistent: u_new = u_old * (rho_old/rho_new)
            if adaptive_rho and iteration % 10 == 0:
                rho_old = rho
                if r_norm > mu * s_norm:
                    rho = rho * tau          # primal too large → increase rho
                elif s_norm > mu * r_norm:
                    rho = rho / tau          # dual too large   → decrease rho

                if rho != rho_old:
                    scale = rho_old / rho
                    for k in keys:
                        u[k] *= scale        # rescale dual to match new rho
                    if with_network:
                        w_p *= scale
                        w_q *= scale

        # Did not converge — return partial result with history if requested
        if log_history:
            result = _result(agents, max_iterations, residual, None)
            result['history'] = history
            result['converged'] = False
            return result
        raise RuntimeError(f'ADMM did not converge; residual={residual:.3g}')
    finally:
        for m in models: m.dispose()
        if dso is not None: dso.dispose()