"""Variational equilibrium benchmark for potential-form DynLQGameTV games.

If H=H.T and H is positive semidefinite, the VI solution is the convex QP
min 0.5 u'Hu + f'u  subject to A u <= b. This is NOT a general GNE solver.
Gurobi is used solely because the original study already uses it.
"""
import numpy as np
from scipy import sparse


def solve_vgne(bundle, *, verbose=False):
    import gurobipy as gp
    from dynect import MpAVI
    mp = MpAVI.from_game(bundle.game)
    avi = mp.at(bundle.x0)
    H = np.asarray(avi.H, dtype=float)
    f = np.asarray(avi.f, dtype=float).reshape(-1)
    symmetry_error = float(np.max(np.abs(H - H.T)))
    if symmetry_error > 1e-8:
        raise ValueError(f'Game is not a potential game (H symmetry error {symmetry_error:.2e}); '
                         'a convex potential QP would not necessarily give the vGNE')
    H = (H + H.T)/2
    # For a large game, requesting exact eigenvalues is unnecessarily costly.
    # A Cholesky check is cheap for the small benchmark, but Gurobi will also
    # reject nonconvex quadratic objectives unless NonConvex=2 is set (never here).
    m = gp.Model('p2p_vgne')
    m.Params.OutputFlag = int(verbose)
    z = m.addMVar(f.size, lb=-gp.GRB.INFINITY, name='u')
    m.addMConstr(sparse.csr_matrix(avi.A), z, '<', np.asarray(avi.b,float), name='feasible')
    m.setObjective(0.5 * (z @ sparse.csr_matrix(H) @ z) + f @ z, gp.GRB.MINIMIZE)
    m.optimize()
    if m.Status != gp.GRB.OPTIMAL:
        raise RuntimeError(f'vGNE potential QP not optimal, Gurobi status {m.Status}')
    u = np.asarray(z.X).reshape(-1)
    max_violation = float(np.max(avi.A @ u - avi.b))
    return dict(u=u, mp=mp, avi=avi, objective=float(m.ObjVal),
                max_violation=max_violation,
                gurobi_status=int(m.Status), symmetry_error=symmetry_error)
