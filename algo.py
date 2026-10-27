"""Centralized and distributed ADMM P2P solvers.

centralized()        — single Gurobi model minimising sum of prosumer costs
distributed()        — consensus ADMM; recovers a variational GNE
distributed_window() — sliding-window wrapper for multi-day intervals
                       (not needed for single-day runs at 30-min resolution)

Both centralized and distributed return the same result dict.
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
    if model.Status != gp.GRB.OPTIMAL:
        raise RuntimeError(f'{model.ModelName}: status {model.Status}')


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


# ── Sliding-window wrappers ───────────────────────────────────────────────────

def centralized_window(data, window_h=3.0, with_network=False):
    """Sliding-window centralized solver — diagnostic mirror of distributed_window.

    Runs the same windowing logic as distributed_window but uses the centralized
    solver. If centralized is feasible in every window, the distributed solver
    should be too — any difference pinpoints a formulation bug in ADMM.
    If centralized is also infeasible, the bug is in state forwarding.

    Parameters
    ----------
    data      : dict   full-day scenario data
    window_h  : float  window size in hours
    with_network : bool enforce feeder constraints
    """
    T   = len(data['buy_price'])
    dt  = data['dt']
    N   = len(data['demand'])
    Tw  = int(round(window_h / dt))
    nw  = T // Tw

    assert nw * Tw == T, \
        f"window_h={window_h}h → Tw={Tw} steps, T={T} not divisible."

    print(f"Centralized sliding window: {nw} × {window_h}h ({Tw} steps)")

    soc_now    = list(data['initial_soc'])
    ev_soc_now = list(data.get('ev_initial_soc', [0.0]*N))
    T_now      = list(data.get('hp_T_init', [20.0]*N))

    window_results = []

    for w in range(nw):
        t0, t1  = w*Tw, (w+1)*Tw
        is_last = (w == nw-1)
        print(f"  Window {w+1}/{nw}  t=[{t0}:{t1})  "
              f"({'cyclic SOC' if is_last else 'free terminal'})")
        print(f"    soc_now:    {[round(v,3) for v in soc_now]}")
        print(f"    ev_soc_now: {[round(v,3) for v in ev_soc_now]}")
        print(f"    T_now:      {[round(v,3) for v in T_now]}")

        data_w = _slice_data(data, t0, t1, Tw)
        data_w['initial_soc']    = list(soc_now)
        data_w['ev_initial_soc'] = list(ev_soc_now)
        data_w['hp_T_init']      = list(T_now)
        data_w['cyclic_soc']     = is_last

        try:
            result_w = centralized(data_w, with_network=with_network)
        except Exception as e:
            print(f"    ✗ CENTRALIZED infeasible at window {w+1}: {e}")
            print(f"    → Bug is in state forwarding, not ADMM")
            return None

        window_results.append(result_w)

        # Forward state
        for i in range(N):
            cap = data['battery_capacity'][i]
            soc_now[i] = float(result_w['soc'][i, -1]) if cap > 0 else 0.0

            ev_cap = data.get('ev_capacity', [0]*N)[i]
            ev_soc_now[i] = float(result_w['ev_soc'][i, -1]) if ev_cap > 0 else 0.0

            hp_max_p = data.get('hp_max_power', [0]*N)[i]
            if hp_max_p > 0 and result_w['temp'] is not None:
                T_now[i] = float(result_w['temp'][i, -1])

        print(f"    ✓ cost={result_w['costs'].sum():.4f} EUR")

    result = _concat_results(window_results, N, T, Tw)
    print(f"\nCentralized window complete  "
          f"total_cost={result['costs'].sum():.4f} EUR")
    return result

def distributed_window(data, window_h=6.0, with_network=False,
                        rho=2.0, tolerance=1e-3, max_iterations=10000):
    """Sliding-window distributed ADMM for multi-day or long-horizon runs.

    NOT needed for single-day runs at 30-min resolution — use distributed()
    directly. This wrapper is kept for future multi-day scenario sweeps.

    Splits the horizon into non-overlapping windows of `window_h` hours,
    solves each with ADMM, and forwards device state between windows.

    Note: the sliding window introduces suboptimality vs the full-horizon
    solve due to limited foresight. The total cost will be slightly higher
    than centralized. This is a known property of MPC-style operation and
    should be documented when used.

    Parameters
    ----------
    data       : dict   scenario data dict
    window_h   : float  window size in hours
    with_network: bool  enforce feeder constraints
    rho        : float  ADMM step size (2.0 recommended for windowed solve)
    tolerance  : float  convergence threshold (1e-3 sufficient for windows)
    """
    T   = len(data['buy_price'])
    dt  = data['dt']
    N   = len(data['demand'])
    Tw  = int(round(window_h / dt))
    nw  = T // Tw

    assert nw * Tw == T, (
        f"window_h={window_h}h → Tw={Tw} steps, "
        f"but T={T} not divisible. Adjust window_h.")

    print(f"Sliding window: {nw} × {window_h}h ({Tw} steps) = {T} total")

    soc_now    = list(data['initial_soc'])
    ev_soc_now = list(data.get('ev_initial_soc', [0.0]*N))
    T_now      = list(data.get('hp_T_init', [20.0]*N))

    window_results = []

    for w in range(nw):
        t0, t1  = w*Tw, (w+1)*Tw
        is_last = (w == nw-1)
        print(f"  Window {w+1}/{nw}  t=[{t0}:{t1})  "
              f"({'cyclic SOC' if is_last else 'free terminal'})")

        data_w = _slice_data(data, t0, t1, Tw)
        data_w['initial_soc']    = list(soc_now)
        data_w['ev_initial_soc'] = list(ev_soc_now)
        data_w['hp_T_init']      = list(T_now)
        data_w['cyclic_soc']     = is_last

        result_w = distributed(data_w, with_network=with_network,
                               rho=rho, tolerance=tolerance,
                               max_iterations=max_iterations)
        window_results.append(result_w)

        for i in range(N):
            cap = data['battery_capacity'][i]
            soc_now[i] = float(result_w['soc'][i, -1]) if cap > 0 else 0.0

            ev_cap = data.get('ev_capacity', [0]*N)[i]
            ev_soc_now[i] = float(result_w['ev_soc'][i, -1]) if ev_cap > 0 else 0.0

            hp_max_p = data.get('hp_max_power', [0]*N)[i]
            if hp_max_p > 0 and result_w['temp'] is not None and result_w['temp'].any():
                T_now[i] = float(result_w['temp'][i, -1])

        print(f"    done  iter={result_w['iterations']}  "
              f"res={result_w['residual']:.2e}  "
              f"cost={result_w['costs'].sum():.4f} EUR")

    result = _concat_results(window_results, N, T, Tw)
    print(f"\nDone  total_iter={result['iterations']}  "
          f"total_cost={result['costs'].sum():.4f} EUR")
    return result


def _slice_data(data, t0, t1, Tw):
    """Slice time-series fields of data to window [t0, t1)."""
    N = len(data['demand'])
    d = {k: data[k][t0:t1]
         for k in ('buy_price', 'sell_price', 'p2p_price')}
    for k in ('demand', 'generation', 'hp_base_power', 'hp_q_heat',
              'ev_availability', 'ev_driving_load'):
        d[k] = [data[k][i][t0:t1] for i in range(N)]

    dep = []
    for i in range(N):
        orig  = data.get('ev_soc_min_departure', [0.0]*N)[i]
        avail = data['ev_availability'][i]
        has_dep = any(t+1 < len(avail) and avail[t] == 1 and avail[t+1] == 0
                      for t in range(t0, t1))
        dep.append(orig if has_dep else 0.0)
    d['ev_soc_min_departure'] = dep

    passthrough = {k: v for k, v in data.items()
                   if k not in d and k not in (
                       'buy_price', 'sell_price', 'p2p_price',
                       'demand', 'generation', 'hp_base_power', 'hp_q_heat',
                       'ev_availability', 'ev_driving_load',
                       'ev_soc_min_departure')}
    d.update(passthrough)
    return d


def _concat_results(windows, N, T, Tw):
    """Concatenate per-window results into a full-horizon result."""
    r = {}
    for k in ('buy', 'sell', 'charge', 'discharge', 'ev_charge', 'ev_discharge',
              'hp_power', 'q_batt', 'q_pv', 'ev_q_charger', 'q_injection'):
        r[k] = np.concatenate([w[k] for w in windows], axis=1)
    for k in ('soc', 'ev_soc', 'temp'):
        parts = [windows[0][k]]
        for w in windows[1:]:
            parts.append(w[k][:, 1:])
        r[k] = np.concatenate(parts, axis=1)
    trades = {}
    for wi, w in enumerate(windows):
        t_off = wi * Tw
        for (i,j,t), v in w['trades'].items():
            trades[i,j,t+t_off] = v
    r['trades']      = trades
    r['costs']       = sum(w['costs'] for w in windows)
    r['iterations']  = sum(w['iterations'] for w in windows)
    r['residual']    = max(w['residual'] for w in windows)
    r['network']     = windows[-1].get('network')
    return r