# POF - Price of Fairness (W3)
"""
metrics.py — Fairness and optimality metrics for P2P electricity markets

(Reference: Hall et al. (2025) "The Limits of Fairness of the v-GNE", Table I)

--> fairness metrics are applied to the savings vector d ∈ R^N:

    d_i = (J_i^nt - J_i(x)) / J_i^nt

d_i > 0 : prosumer i benefits from trading (saves money)
d_i = 0 : trading neither helps nor hurts
d_i < 0 : prosumer i is worse off with trading

--> use d rather than raw costs J_i so that metrics are invariant to the
absolute size of each prosumer (fractional savings are comparable, totals are not) 
--> CUC assumption, see Hall et al. Section III-B

Workflow W3:
1.  J_nt = no_trade_costs(data)             individual costs without trading
2.  d    = savings(result_so, J_nt)         savings fractions at social optimum
3.  sw(d), mm(d), jains(d), var_savings(d)  fairness metric values 
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
    """
    Solve each prosumer independently, no P2P trading

    --> prosumer i minimises its own cost J_i subject only to its own
    device constraints and the grid, no coupling

    Parameters:
    data : dict   scenario data dict (from make_scenario)

    Returns:
    J_nt : ndarray (N,)
        Individual optimal costs without trading [EUR].
    """
    N    = len(data['demand'])
    J_nt = np.zeros(N)

    # empty list - no edges, no trades
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
    """
    --> computes normalised savings d_i for each prosumer

        d_i = (J_i^nt - J_i(x)) / J_i^nt

    d_i > 0 : i saves money by trading
    d_i = 0 : trading has no effect for i
    d_i < 0 : i is worse off trading (should not occur at social optimum
               with a well-defined no-trade baseline, but can occur due
               to numerical tolerances or network constraints)

    Parameters:
    result : dict   solver result (from centralized() or distributed())
    J_nt   : ndarray (N,)  no-trade costs from no_trade_costs()

    Returns:
    d : ndarray (N,)   savings fractions ∈ (-∞, 1]
    """
    J_so  = np.array(result['costs'])            # J_i(x_SO)
    denom = np.where(np.abs(J_nt) > 1e-9,
                     J_nt, 1.0)                  # guard against zero J_nt
    return (J_nt - J_so) / denom


# ── Step 3: Fairness metrics (Hall et al. Table I) ────────────────────────────

def sw(d):
    """
    Social Welfare / Utilitarianism (Hall et al. Table I, row 2)

    Paper definition (on costs):  f_SW(x) = Σ_{i∈I} J_i(x)
    Applied to savings:           SW(d)   = Σ_{i∈I} d_i

    Comparability: CUC (Cardinal Unit Comparability)

    Interpretation: total savings across all prosumers. Higher is better.
    This is MAXIMISED at the social optimum by construction — the social
    optimum minimises Σ J_i, which is equivalent to maximising Σ d_i
    given fixed J_nt.

    Returns scalar ∈ (-N, N]
    """
    return float(np.sum(d))


def mm(d):
    """Rawlsian Maximin (Hall et al. Table I, row 1).

    Paper definition (on costs):  f_MM(x) = max_{i∈I} J_i(x)
    Applied to savings:           MM(d)   = min_{i∈I} d_i

    Comparability: CFC (Cardinal Full Comparability).

    Interpretation: the saving of the worst-off prosumer. Higher is fairer.
    A fair allocation under MM maximises the minimum savings fraction —
    protecting the prosumer who benefits least from trading.

    Returns scalar ∈ (-∞, 1].
    """
    return float(np.min(d))


def jains(d):
    """
    Jain's Fairness Index (Hall et al. Table I, row 5)

    Paper definition (on costs):
        f_JI(x) = (Σ_{i∈I} J_i(x))² / (M · Σ_{i∈I} J_i(x)²)

    Applied to savings:
        JI(d) = (Σ d_i)² / (N · Σ d_i²)

    Comparability: CFC (Cardinal Full Comparability)

    Interpretation: ranges from 1/N (one prosumer captures all savings)
    to 1.0 (all prosumers save equally). Higher is fairer. Specifically
    developed for resource allocation in telecommunications [21].

    Note: undefined (returns NaN) when all d_i = 0

    Returns scalar ∈ [1/N, 1]
    """
    N      = len(d)
    sum_d  = float(np.sum(d))
    sum_d2 = float(np.sum(d**2))
    if sum_d2 < 1e-12:
        return float('nan')   # all savings are zero — metric undefined
    return sum_d**2 / (N * sum_d2)


def var_savings(d):
    """Variance of savings (natural dispersion measure, not in Hall table).

    Var(d) = (1/N) Σ_{i∈I} (d_i - d̄)²

    Interpretation: lower is fairer (zero = all prosumers save equally).
    Population variance (ddof=0) used here — consistent with the paper's
    notation of averaging over the full agent set I.

    Returns scalar ∈ [0, ∞).
    """
    return float(np.var(d, ddof=0))


# ── Summary printout ──────────────────────────────────────────────────────────

def print_metrics(d, J_nt, label='Social optimum'):
    """Print a clean table of all fairness metrics for a savings vector d."""
    N = len(d)
    print(f"\n{'─'*50}")
    print(f"Fairness metrics — {label}")
    print(f"{'─'*50}")
    print(f"  {'Prosumer':<20} {'J_nt [EUR]':>12} {'d_i':>10}")
    for i in range(N):
        print(f"  P{i+1:<19} {J_nt[i]:>12.4f} {d[i]:>10.4f}")
    print(f"{'─'*50}")
    print(f"  SW  (Σ d_i, higher=better):      {sw(d):>10.4f}")
    print(f"  MM  (min d_i, higher=better):     {mm(d):>10.4f}")
    print(f"  JI  (Jain's, higher=fairer):      {jains(d):>10.4f}  "
          f"[range: {1/N:.3f} – 1.000]")
    print(f"  Var (lower=fairer):               {var_savings(d):>10.6f}")
    print(f"{'─'*50}")
    
# POO - Price of Optimality 


