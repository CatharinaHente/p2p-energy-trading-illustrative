"""
fairness_optima.py — Fair optima, POO and POF for P2P electricity markets.

x^fair = argmax F(J(x)) s.t. x in X, computed centrally on raw costs J_i.
x^fair is NOT a Nash equilibrium — prosumers may deviate from it unilaterally.
Distinct from f-GNE (Hall et al. 2025), which is constrained to S^GNE.

POO = 1 - F(J(x^SO)) / F(J(x^fair))   how much fairness is lost at x^SO
POF = 1 - TW(x^fair) / TW(x^SO)       efficiency cost of pursuing x^fair

Both operate on J_i (raw costs) only. F is one of: NBS, MM, JI.
TW is used as the efficiency reference in POF (Bertsimas et al. 2011).

Note on p_cost: the quadratic regularisation that makes the LP strictly
convex forces S^GNE = {x^SO}, making f-GNE trivially equal to x^SO for
any metric. Remove p_cost when computing f-GNE via NashOpt or equivalent.
"""

import numpy as np
import gurobipy as gp
from prosumer import add_prosumer
from metrics  import tw, nbs, jains, savings


# ── Shared helpers ─────────────────────────────────────────────────────────────

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
        print(f'  ⚠ {model.ModelName}: suboptimal')


def _build_market(m, data):
    N, T   = len(data['demand']), len(data['buy_price'])
    agents = [add_prosumer(m, i, data) for i in range(N)]
    for i, j in data['edges']:
        for t in range(T):
            m.addConstr(agents[i]['trade'][j,t] + agents[j]['trade'][i,t] == 0)
    return agents, N, T


def _extract(agents):
    return {'costs': np.array([a['cost'].getValue() for a in agents])}


# ── Fair optima on J_i — NOT Nash equilibria ───────────────────────────────────

def fair_optimal_NBS(data, J_nt):
    """x^fair maximising NBS = Π_i (J_i^nt - J_i).

    CNC-invariant. Solved as max Σ log(J_i^nt - J_i) via Gurobi log constraint.
    """
    m = _model('fair_NBS')
    m.Params.NonConvex = 2
    try:
        agents, N, _ = _build_market(m, data)
        gains  = m.addVars(N, lb=1e-6, name='gain')
        log_g  = m.addVars(N, lb=-gp.GRB.INFINITY, name='log_gain')
        for i, a in enumerate(agents):
            m.addConstr(gains[i] == J_nt[i] - a['cost'])
            m.addGenConstrLog(gains[i], log_g[i])
        m.setObjective(gp.quicksum(log_g[i] for i in range(N)),
                       gp.GRB.MAXIMIZE)
        _optimize(m)
        return _extract(agents)
    finally:
        m.dispose()


def fair_optimal_MM(data, J_nt):
    """x^fair minimising max_i J_i (minimax on raw costs).

    Reformulated as: min u s.t. J_i(x) <= u for all i.
    Protects the highest-cost prosumer.
    """
    m = _model('fair_MM')
    try:
        agents, N, _ = _build_market(m, data)
        u = m.addVar(lb=0.0, name='u')
        for i, a in enumerate(agents):
            m.addQConstr(a['cost'] <= u, name=f'MM_{i}')
        m.setObjective(u, gp.GRB.MINIMIZE)
        _optimize(m)
        return _extract(agents)
    finally:
        m.dispose()


def fair_optimal_JI(data, J_nt):
    """x^fair maximising Jain's index on raw costs JI(J).

    Non-convex (ratio objective). Solved with Gurobi NonConvex=2.
    Note: JI on raw costs requires CFC comparability — interpret with care.
    """
    m = _model('fair_JI')
    m.Params.NonConvex = 2
    try:
        agents, N, _ = _build_market(m, data)
        sum_J  = m.addVar(lb=-gp.GRB.INFINITY, name='sum_J')
        sum_J2 = m.addVar(lb=0.0, name='sum_J2')
        m.addConstr(sum_J == gp.quicksum(a['cost'] for a in agents))
        m.addQConstr(sum_J2 == gp.quicksum(a['cost'] * a['cost']
                                            for a in agents))
        # Maximise sum_J^2 / (N * sum_J2) — add auxiliary for ratio
        ji = m.addVar(lb=0.0, ub=1.0, name='JI')
        m.addQConstr(ji * N * sum_J2 == sum_J * sum_J)
        m.setObjective(ji, gp.GRB.MAXIMIZE)
        _optimize(m)
        return _extract(agents)
    finally:
        m.dispose()


# ── POO and POF ────────────────────────────────────────────────────────────────

def poo(F_fn, J_so, J_fair, J_nt=None):
    """Price of Optimality: 1 - F(J^SO) / F(J^fair).

    F_fn  : fairness metric function on J_i
    J_so  : ndarray (N,) costs at x^SO
    J_fair: ndarray (N,) costs at x^fair
    J_nt  : ndarray (N,) no-trade costs (required for NBS)

    POO = 0: social optimum is already fair.
    POO > 0: fairness loss at social optimum.
    x^fair is not a Nash equilibrium.
    """
    kwargs = {'J_nt': J_nt} if J_nt is not None else {}
    f_so   = F_fn(J_so,   **kwargs)
    f_fair = F_fn(J_fair, **kwargs)
    return float('nan') if abs(f_fair) < 1e-12 else 1.0 - f_so / f_fair


def pof(result_so, result_fair, J_nt):
    """Price of Fairness: 1 - TW(x^fair) / TW(x^SO).

    Efficiency loss from clearing at x^fair instead of x^SO.
    POF >= 0 always; POF = 0 means x^fair is equally efficient.
    Ref: Bertsimas et al. (2011) OR 59(1):17-31, Definition 1.
    x^fair is not a Nash equilibrium.
    """
    tw_so   = tw(J_nt, np.array(result_so['costs']))
    tw_fair = tw(J_nt, np.array(result_fair['costs']))
    return float('nan') if abs(tw_so) < 1e-12 else 1.0 - tw_fair / tw_so


# ── Combined computation ───────────────────────────────────────────────────────

def compute_all(result_so, data, J_nt):
    """Compute all fair optima, POO and POF.

    Returns dict with fair optima results and POO/POF per metric.
    All POO computed on J_i (raw costs). POF uses TW throughout.
    """
    J_so = np.array(result_so['costs'])

    print("Computing x^fair (NBS)...")
    r_nbs = fair_optimal_NBS(data, J_nt)
    print("Computing x^fair (MM)...")
    r_mm  = fair_optimal_MM(data, J_nt)
    print("Computing x^fair (JI)...")
    r_ji  = fair_optimal_JI(data, J_nt)

    J_nbs = np.array(r_nbs['costs'])
    J_mm  = np.array(r_mm['costs'])
    J_ji  = np.array(r_ji['costs'])

    poo_vals = {
        'NBS': poo(lambda J, J_nt: nbs(J, J_nt), J_so, J_nbs, J_nt),
        'MM' : poo(lambda J, **_: -float(np.max(J)), J_so, J_mm),
        'JI' : poo(lambda J, **_: jains(J),          J_so, J_ji),
    }

    pof_vals = {
        'NBS': pof(result_so, r_nbs, J_nt),
        'MM' : pof(result_so, r_mm,  J_nt),
        'JI' : pof(result_so, r_ji,  J_nt),
    }

    return dict(
        fair_NBS=r_nbs, fair_MM=r_mm, fair_JI=r_ji,
        poo=poo_vals,   pof=pof_vals,
    )