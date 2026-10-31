"""
metrics.py — Fairness and optimality metrics for P2P electricity markets.

Follows Hall et al. (2025) "The Limits of Fairness of the v-GNE", Table I.

All fairness metrics are applied to the savings vector d ∈ R^N where:

    d_i = (J_i^nt - J_i(x)) / J_i^nt

d_i > 0 : prosumer i benefits from trading (saves money)
d_i = 0 : trading neither helps nor hurts
d_i < 0 : prosumer i is worse off with trading

We use d rather than raw costs J_i so that metrics are invariant to the
absolute size of each prosumer — a residential household and a commercial
office operate at very different cost scales, but their fractional savings
are directly comparable (CUC assumption, see Hall et al. Section III-B).

Workflow
--------
1.  J_nt = no_trade_costs(data)          individual costs without trading
2.  d    = savings(result_so, J_nt)      savings fractions at social optimum
3.  sw(d), mm(d), jains(d),              fairness metric values
    var_savings(d)
4.  compute x_fair (fair optimum) by
    optimising over fairness metric
5.  POO and POF (to be added in next step)
"""

import numpy as np
import gurobipy as gp
from prosumer import add_prosumer


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
        raise RuntimeError(f'{model.ModelName}: infeasible or unbounded '
                           f'(status {model.Status})')


# ── Step 1: No-trade baseline ─────────────────────────────────────────────────

def no_trade_costs(data):
    """Solve each prosumer independently with no P2P trading.

    Each prosumer i minimises its own cost J_i subject only to its own
    device constraints and the grid — no coupling to other prosumers.
    This gives the counterfactual: what would each prosumer pay if the
    P2P market did not exist?

    Parameters
    ----------
    data : dict   scenario data dict (from make_scenario)

    Returns
    -------
    J_nt : ndarray (N,)
        Individual optimal costs without trading [EUR].
    """
    N    = len(data['demand'])
    J_nt = np.zeros(N)

    # Empty edge list → no trade variables, no coupling constraints
    data_nt = {**data, 'edges': []}

    for i in range(N):
        m = _model(f'no_trade_P{i+1}')
        a = add_prosumer(m, i, data_nt)
        m.setObjective(a['cost'], gp.GRB.MINIMIZE)
        _optimize(m)
        J_nt[i] = a['cost'].getValue()
        m.dispose()
        print(f"  P{i+1} no-trade cost: {J_nt[i]:.4f} EUR")

    return J_nt


# ── Step 2: Savings fractions ─────────────────────────────────────────────────

def savings(result, J_nt):
    """Compute normalised savings d_i for each prosumer.

    Definition (Hall et al., eq. following Table I):

        d_i = (J_i^nt - J_i(x)) / J_i^nt

    d_i > 0 : i saves money by trading
    d_i = 0 : trading has no effect for i
    d_i < 0 : i is worse off trading (should not occur at social optimum
               with a well-defined no-trade baseline, but can occur due
               to trade fees, network constraints, or binding device limits)

    Parameters
    ----------
    result : dict      solver result (from centralized() or distributed())
    J_nt   : ndarray   (N,) no-trade costs from no_trade_costs()

    Returns
    -------
    d : ndarray (N,)   savings fractions ∈ (-∞, 1]
    """
    J_so  = np.array(result['costs'])
    denom = np.where(np.abs(J_nt) > 1e-9, J_nt, 1.0)   # guard zero J_nt
    return (J_nt - J_so) / denom


# ── Step 3: Fairness metrics (Hall et al. Table I) ────────────────────────────

def sw(d, J_nt=None, J_so=None):
    """Social Welfare / Utilitarianism (Hall et al. Table I, row 2).

    Paper definition (on costs):  f_SW(x) = Σ_{i∈I} J_i(x)
    Applied to savings:           SW(d)   = Σ_{i∈I} d_i

    Comparability: CUC (Cardinal Unit Comparability).

    Interpretation: total community savings from trading. Higher is better.

    Note: SW(d) = Σ dᵢ is NOT equivalent to the social optimum objective
    Σ Jᵢ. Maximising Σ dᵢ minimises a J_nt-weighted sum of costs,
    Σ Jᵢ/Jᵢⁿᵗ, which differs from Σ Jᵢ unless all Jᵢⁿᵗ are equal.
    For efficiency reporting use TW (total welfare) on raw costs instead.

    Returns scalar ∈ (-N, N].
    """
    return float(np.sum(d))


def tw(J_nt, J_so):
    """Total Welfare — community-level efficiency gain from trading.

    TW = (Σ Jᵢⁿᵗ - Σ Jᵢ(x_SO)) / Σ Jᵢⁿᵗ

    Applied to raw costs (EUR), not savings fractions.
    Comparability: CUC.

    Interpretation: fractional reduction in total community cost due to
    trading. Always ≥ 0 since x_SO minimises Σ Jᵢ and the no-trade
    solution is always feasible (set trades to zero).

    Returns scalar ∈ [0, 1].
    """
    sum_nt = float(np.sum(J_nt))
    sum_so = float(np.sum(J_so))
    if sum_nt < 1e-9:
        return float('nan')
    return (sum_nt - sum_so) / sum_nt


def mm(d):
    """Rawlsian Maximin (Hall et al. Table I, row 1).

    Paper definition (on costs):  f_MM(x) = max_{i∈I} J_i(x)
    Applied to savings:           MM(d)   = min_{i∈I} d_i

    Comparability: CUC on dᵢ — normalisation by Jᵢⁿᵗ makes absolute
    savings fractions comparable without requiring CFC on raw costs.

    Interpretation: the saving of the worst-off prosumer. Higher is fairer.
    A fair allocation under MM maximises the minimum savings fraction,
    protecting the prosumer who benefits least from trading.

    Returns scalar ∈ (-∞, 1].
    """
    return float(np.min(d))


def jains(d):
    """Jain's Fairness Index (Hall et al. Table I, row 5).

    Paper definition (on costs):
        f_JI(x) = (Σ_{i∈I} J_i(x))² / (M · Σ_{i∈I} J_i(x)²)

    Applied to savings:
        JI(d) = (Σ dᵢ)² / (N · Σ dᵢ²)

    Comparability: CUC on dᵢ — same argument as MM.

    Interpretation: higher is fairer. The range [1/N, 1] holds only when
    all dᵢ ≥ 0. With negative dᵢ the index can fall below 1/N and loses
    its usual interpretation — flag and interpret with care.

    Returns scalar (nominally ∈ [1/N, 1] but can fall below with dᵢ < 0).
    """
    N      = len(d)
    sum_d  = float(np.sum(d))
    sum_d2 = float(np.sum(d**2))
    if sum_d2 < 1e-12:
        return float('nan')    # all savings zero — undefined
    val = sum_d**2 / (N * sum_d2)
    if any(di < 0 for di in d):
        pass   # caller should be aware result may be < 1/N
    return val


def var_savings(d):
    """Variance of savings.

    Var(d) = (1/N) Σ_{i∈I} (dᵢ - d̄)²

    Not in Hall et al. table; natural dispersion measure on dᵢ.
    Comparability: CUC on dᵢ.

    Interpretation: lower is fairer (zero = all prosumers save equally).
    Population variance (ddof=0).

    Returns scalar ∈ [0, ∞).
    """
    return float(np.var(d, ddof=0))


# ── Precomputation: Fair-optimal allocations ──────────────────────────────────────────────────

def fair_optimal_SW(data, J_nt):
    """Allocation that maximises SW(d) = Σ d_i (utilitarian fairness optimum).

    Equivalent to minimising the J_nt-weighted cost Σ J_i(x)/J_nt_i, which
    gives more weight to prosumers with a lower no-trade cost, protecting them
    from subsidising high-cost participants.

    This is the denominator allocation for PoO_SW and the numerator for PoF_SW.

    Parameters
    ----------
    data  : dict     scenario data dict
    J_nt  : ndarray  (N,) no-trade costs from no_trade_costs()

    Returns
    -------
    dict with at least 'costs' key — sufficient for PoO and PoF computation.

    References
    ----------
    Bertsimas et al. (2011) Operations Research — utilitarian fairness.
    Student proposal eq. (4): PoO_W = 1 − W(x^SO) / max_{x∈X} W(x).
    """
    N, T  = len(data['demand']), len(data['buy_price'])
    m     = _model('fair_SW')
    try:
        agents = [add_prosumer(m, i, data) for i in range(N)]
        for i, j in data['edges']:
            for t in range(T):
                m.addConstr(agents[i]['trade'][j,t] + agents[j]['trade'][i,t] == 0)

        # Weighted objective: min Σ (1/J_nt_i) · J_i(x)
        # Since d_i = (J_nt_i − J_i)/J_nt_i, max Σ d_i ≡ min Σ J_i/J_nt_i
        weights = np.where(np.abs(J_nt) > 1e-9, 1.0 / J_nt, 1.0)
        m.setObjective(
            gp.quicksum(w * a['cost'] for w, a in zip(weights, agents)),
            gp.GRB.MINIMIZE)
        _optimize(m)
        return {'costs': np.array([a['cost'].getValue() for a in agents])}
    finally:
        m.dispose()


def fair_optimal_MM(data, J_nt):
    """Allocation that maximises MM(d) = min_i d_i (Rawlsian maximin optimum).

    Reformulation as a linear programme in τ (the guaranteed savings fraction):

        max  τ
        s.t. J_i(x) ≤ (1 − τ) · J_nt_i   ∀i ∈ I      [d_i ≥ τ]
             x ∈ X                                      [device + market constraints]

    Each constraint J_i(x) ≤ (1 − τ)·J_nt_i is a convex quadratic constraint
    (J_i is convex quadratic, J_nt_i is a positive constant) — Gurobi solves
    convex QCQPs to global optimality.

    Parameters
    ----------
    data  : dict     scenario data dict
    J_nt  : ndarray  (N,) no-trade costs from no_trade_costs()

    Returns
    -------
    dict with at least 'costs' key.

    References
    ----------
    Rawls (1971) A Theory of Justice — maximin principle.
    Bertsimas et al. (2011) — price of fairness under maximin.
    Student proposal eq. (4): PoO_W = 1 − W(x^SO) / max_{x∈X} W(x).
    """
    N, T  = len(data['demand']), len(data['buy_price'])
    m     = _model('fair_MM')
    m.Params.NonConvex = 0   # confirm problem is convex
    try:
        agents = [add_prosumer(m, i, data) for i in range(N)]
        for i, j in data['edges']:
            for t in range(T):
                m.addConstr(agents[i]['trade'][j,t] + agents[j]['trade'][i,t] == 0)

        # Slack variable τ: the minimum savings fraction to maximise
        # τ ≤ 1 always (cannot save more than 100 % of no-trade cost)
        # τ may be negative if some prosumer is worse off at the optimum
        tau = m.addVar(lb=-gp.GRB.INFINITY, ub=1.0, name='tau')

        # J_i(x) ≤ (1 − τ) · J_nt_i   ↔   d_i(x) ≥ τ
        denom = np.where(np.abs(J_nt) > 1e-9, J_nt, 1.0)
        for i, (a, jnt) in enumerate(zip(agents, denom)):
            # Rearranged: J_i(x) + jnt·τ ≤ jnt
            m.addQConstr(a['cost'] + jnt * tau <= jnt, name=f'MM_{i}')

        m.setObjective(tau, gp.GRB.MAXIMIZE)
        _optimize(m)
        return {'costs': np.array([a['cost'].getValue() for a in agents])}
    finally:
        m.dispose()


# ── Step 4: PoO and PoF ───────────────────────────────────────────────────────────────

def poo(W_fn, d_so, d_fair):
    """Price of Optimality for fairness metric W.

    Measures how much fairness is lost by clearing at the social optimum
    instead of the fairness optimum:

        PoO_W = 1 − W(d(x^SO)) / W(d(x^W-fair))
              = 1 − W(d_so) / W(d_fair)

    PoO = 0: social optimum is already fair (no loss).
    PoO = 1: social optimum achieves zero fairness relative to maximum.
    PoO > 1: social optimum is worse than the worst feasible allocation
             (possible when W can be negative, e.g. MM with d_i < 0).

    Parameters
    ----------
    W_fn   : callable   fairness metric, e.g. sw or mm from this module
    d_so   : ndarray    savings vector at social optimum
    d_fair : ndarray    savings vector at fairness optimum

    References
    ----------
    Sem. thesis proposal eq. (4): PoO_W = 1 − W(x^SO) / max_{x∈X} W(x).
    Bertsimas et al. (2011) — price of fairness (reversed direction of comparison).
    """
    w_so   = W_fn(d_so)
    w_fair = W_fn(d_fair)
    if abs(w_fair) < 1e-12:
        return float('nan')
    return 1.0 - w_so / w_fair


def pof(result_so, result_fair, J_nt):
    """Price of Fairness: efficiency loss from pursuing the fair allocation.

    Measures how much total welfare (efficiency) is sacrificed when the market
    clears at the fairness optimum instead of the social optimum:

        PoF = 1 − TW(x^fair) / TW(x^SO)
            = 1 − (ΣJ_nt − ΣJ_fair) / (ΣJ_nt − ΣJ_SO)

    PoF = 0: fair allocation is equally efficient as social optimum.
    PoF > 0: fair allocation wastes some total welfare for fairness.

    Parameters
    ----------
    result_so   : dict   social optimum result (needs 'costs' key)
    result_fair : dict   fairness-optimal result (needs 'costs' key)
    J_nt        : ndarray (N,) no-trade costs

    References
    ----------
    Bertsimas et al. (2011) Operations Research, Definition 1.
    Student proposal §3: price of optimality (reversed: here x^SO is efficient).
    """
    tw_so   = tw(J_nt, np.array(result_so['costs']))
    tw_fair = tw(J_nt, np.array(result_fair['costs']))
    if abs(tw_so) < 1e-12:
        return float('nan')
    return 1.0 - tw_fair / tw_so