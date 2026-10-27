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


# POF - Price of Fairness (W3)
   
# POO - Price of Optimality 


