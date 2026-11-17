"""
metrics.py — Fairness metrics for P2P electricity markets.

Primary metrics on raw costs J_i (EUR):
    TW  : total welfare — efficiency reference (CUC)
    NBS : Nash Bargaining Solution — CNC-invariant, relative to x^nt
    MM  : maximin — worst-off prosumer's cost (CUC)
    JI  : Jain's fairness index on costs (CUC)
    Var : variance of costs (CUC)

Additional metrics on proportional savings d_i = (J_i^nt - J_i) / J_i^nt:
    d_i itself — reported as a metric (proportional fairness, design choice)
    MM(d), JI(d), Var(d) — proportional variants of the above

d_i is a modelling choice encoding proportional fairness across heterogeneous
prosumers, not a comparability claim. Stated as a design choice in thesis.

POO and POF live in fairness_optima.py and operate on J_i metrics only.
"""

import numpy as np
import gurobipy as gp
from prosumer import add_prosumer


# ── Gurobi helpers ─────────────────────────────────────────────────────────────

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
        print(f'  ⚠ {model.ModelName}: suboptimal solution accepted')


# ── No-trade baseline ──────────────────────────────────────────────────────────

def no_trade_costs(data):
    """Solve each prosumer independently with no P2P trading.

    Returns J_nt : ndarray (N,) individual no-trade costs [EUR].
    Disagreement point for NBS and normaliser for d_i.

    If a prosumer's standalone problem is infeasible (typically an EV departure
    SOC constraint that cannot be met on a specific day), retry with
    ev_soc_min_departure set to zero for that prosumer. This removes the
    departure requirement while keeping all other device constraints intact.
    The resulting J_nt_i is slightly optimistic but valid as a baseline.
    """
    N       = len(data['demand'])
    J_nt    = np.zeros(N)
    data_nt = {**data, 'edges': []}

    for i in range(N):
        m = _model(f'no_trade_P{i+1}')
        a = add_prosumer(m, i, data_nt)
        m.setObjective(a['cost'], gp.GRB.MINIMIZE)

        try:
            _optimize(m)
        except RuntimeError:
            print(f"  P{i+1}: infeasible — retrying with ev_soc_min_departure=0 ...")
            m.dispose()
            data_relax = {**data_nt,
                          'ev_soc_min_departure': [0.0] * N}
            m = _model(f'no_trade_P{i+1}_relax')
            a = add_prosumer(m, i, data_relax)
            m.setObjective(a['cost'], gp.GRB.MINIMIZE)
            _optimize(m)   # raises if still infeasible — genuine problem

        J_nt[i] = a['cost'].getValue()
        m.dispose()
        print(f"  P{i+1} no-trade cost: {J_nt[i]:.4f} EUR")

    return J_nt


# ── Proportional savings ───────────────────────────────────────────────────────

def savings(result, J_nt):
    """Proportional savings d_i = (J_i^nt - J_i) / J_i^nt.

    Design choice for heterogeneous prosumers — not a comparability claim.
    d_i > 0: benefits from trading; d_i < 0: worse off than no-trade.

    Returns d : ndarray (N,) in (-inf, 1].
    """
    J     = np.array(result['costs'])
    denom = np.where(np.abs(J_nt) > 1e-9, J_nt, 1.0)
    return (J_nt - J) / denom


# ── Metric functions — work on any cost/savings array ─────────────────────────

def tw(J_nt, J):
    """Total Welfare: (Σ J_i^nt - Σ J_i) / Σ J_i^nt. Higher = more efficient."""
    s = float(np.sum(J_nt))
    return float('nan') if s < 1e-9 else (s - float(np.sum(J))) / s


def nbs(J, J_nt):
    """Nash Bargaining Solution: Π_i (J_i^nt - J_i).

    CNC-invariant. Disagreement point x^nt gives NBS = 0 by definition.
    Higher = fairer. Returns 0 if any prosumer is worse off than x^nt.
    Ref: Hall et al. (2025) Table II; Facchinei & Kanzow (2010).
    """
    gains = np.array(J_nt) - np.array(J)
    return float(np.prod(gains)) if np.all(gains > 0) else 0.0


def mm(x):
    """Maximin: min_i x_i (or max_i x_i for costs — see sign convention).

    On J_i: returns max_i J_i (worst-off = highest cost). Lower is fairer.
    On d_i: returns min_i d_i (worst-off = smallest savings). Higher is fairer.
    Caller must interpret direction correctly.

    Returns scalar.
    """
    return float(np.min(x))


def jains(x):
    """Jain's fairness index: (Σ x_i)^2 / (N Σ x_i^2). Higher = fairer.

    On J_i: measures equality of cost distribution (CUC).
    On d_i: measures equality of proportional savings.
    In [1/N, 1] when all x_i > 0; loses interpretation with negative values.

    Returns scalar.
    """
    N, s, s2 = len(x), float(np.sum(x)), float(np.sum(x**2))
    return float('nan') if s2 < 1e-12 else s**2 / (N * s2)


def var(x):
    """Variance: (1/N) Σ (x_i - x_bar)^2. Lower = fairer (zero = equal).

    On J_i: dispersion of costs.
    On d_i: dispersion of proportional savings.

    Returns scalar >= 0.
    """
    return float(np.var(x, ddof=0))


# ── Summary ────────────────────────────────────────────────────────────────────

def compute_metrics(result, J_nt):
    """Compute all fairness metrics at a given allocation.

    Primary metrics on J_i, additional metrics on d_i.
    d_i itself reported as a vector and summarised by min/mean.

    Returns dict:
        J         : ndarray (N,) individual costs
        J_metrics : TW, NBS, MM, JI, Var  — primary (on J_i)
        d         : ndarray (N,) proportional savings
        d_metrics : MM, JI, Var, mean, min — additional (on d_i)
    """
    J = np.array(result['costs'])
    d = savings(result, J_nt)

    J_metrics = {
        'TW' : tw(J_nt, J),
        'NBS': nbs(J, J_nt),
        'MM' : float(np.max(J)),       # worst-off cost; lower is fairer
        'JI' : jains(J),
        'Var': var(J),
    }

    d_metrics = {
        'MM'  : mm(d),                 # min savings fraction; higher is fairer
        'JI'  : jains(d),
        'Var' : var(d),
        'mean': float(np.mean(d)),
        'min' : float(np.min(d)),
    }

    return dict(J=J, J_metrics=J_metrics, d=d, d_metrics=d_metrics)