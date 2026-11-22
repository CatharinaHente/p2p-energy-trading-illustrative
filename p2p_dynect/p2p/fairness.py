"""Fairness criteria using the actual first-day valued prosumer costs.

No GNE solver is run here. `selection_objective` returns a CasADi callback
suitable for a later OptimalGNEP, once a tractable GNE-set formulation exists.
The costs include all operating-cost terms and first-day EV/BESS valuations;
they do NOT attempt to reconstruct costs from the pseudo-gradient alone.
"""
from __future__ import annotations
import numpy as np


def metrics(J, J_nt):
    """Legacy fairness metrics from actual 24h costs, plus IR checks."""
    J = np.asarray(J,float).ravel()
    J_nt = np.asarray(J_nt,float).ravel()
    if J.shape != J_nt.shape or J.size == 0:
        raise ValueError('J and J_nt must have identical nonempty shapes')
    gain = J_nt - J
    d = gain / np.where(np.abs(J_nt)>1e-9,np.abs(J_nt),1.)
    def jain(v):
        denom=len(v)*np.dot(v,v)
        return float(np.sum(v)**2/denom) if denom>1e-12 else float('nan')
    return dict(
        total_gain=float(gain.sum()),
        TW=float(gain.sum()/J_nt.sum()) if J_nt.sum()>1e-9 else float('nan'),
        NBS=float(np.prod(gain)) if np.all(gain>0) else 0.,
        MM=float(np.max(J)),JI=jain(J),Var=float(np.var(J)),
        d_MM=float(np.min(d)),d_JI=jain(d),d_Var=float(np.var(d)),
        gains=gain, proportional_gains=d,
        n_losers=int(np.sum(gain<-1e-6)),
        IR=bool(np.all(gain>=-1e-6)),
    )


def symbolic_costs(bundle, u):
    """Exact *reported* first-day costs as CasADi expressions in stacked u.

    Uses the known game linear dynamics (state indexing is from setup.py).
    Excludes optional 1e-9 objective regularisation, as in evaluated_costs().
    """
    import casadi as ca
    data, T, N = bundle.data, bundle.T, bundle.N
    E = int(data.get('T_eval',T))
    dt=float(data['dt'])
    nu=bundle.game.nu
    starts=np.cumsum([0,*[T*n for n in nu]])
    def ui(i,t,name):
        return u[int(starts[i]+t*nu[i]+bundle.controls[i][name])]
    x=ca.DM(bundle.x0)
    J=[0 for _ in range(N)]
    for t in range(E):
        for i in range(N):
            trade=sum(ui(i,t,f'trade_{j}') for j in bundle.peers[i])
            trade_abs=sum(ui(i,t,f'trade_abs_{j}') for j in bundle.peers[i])
            c=ui(i,t,'charge'); d=ui(i,t,'discharge')
            ec=ui(i,t,'ev_charge'); ed=ui(i,t,'ev_discharge')
            hp=ui(i,t,'hp_power')
            J[i] += dt * (
                data['buy_price'][t]*ui(i,t,'buy') - data['sell_price'][t]*ui(i,t,'sell')
                -data['p2p_price'][t]*trade
                +data['battery_cost']*(c+d)+data.get('ev_cost',0.)*(ec+ed)
                +data.get('hp_cost',0.)*hp + data['trade_fee']*trade_abs
                +data.get('p_cost',0.)*(c*c+d*d+ec*ec+ed*ed
                  +sum(ui(i,t,f'trade_{j}')**2 for j in bundle.peers[i])))
        new_x=ca.mtimes(ca.DM(bundle.game.A[t]),x)+ca.DM(bundle.game.c[t])
        for i in range(N):
            ui_t=ca.vertcat(*[ui(i,t,n) for n in bundle.controls[i]])
            new_x+=ca.mtimes(ca.DM(bundle.game.B[t][i]),ui_t)
        x=new_x
    for i in range(N):
        if (i,'ev') in bundle.state_index:
            k=bundle.state_index[i,'ev']
            J[i]-=float(data['ev_terminal_price'])*(x[k]-bundle.x0[k])
        if (i,'bess') in bundle.state_index:
            k=bundle.state_index[i,'bess']
            battery_val=float(data['ev_terminal_price']*data['ev_eta_charge']/data['eta_charge'])
            J[i]-=battery_val*(x[k]-bundle.x0[k])
    return ca.vertcat(*J)


def selection_objective(bundle, criterion='social_cost', J_nt=None):
    """Return `phi(u)` suitable for `OptimalGNEP(gnep, phi)`.

    Criteria: social_cost, minimax_cost, cost_variance, minimax_loss,
    max_min_proportional_gain, jain_savings.
    Baseline-dependent objectives require first-day no-trade costs J_nt.
    These objectives may be nonsmooth/nonconvex and difficult to optimize.
    """
    import casadi as ca
    valid={'social_cost','minimax_cost','cost_variance','minimax_loss',
           'max_min_proportional_gain','jain_savings'}
    if criterion not in valid:
        raise ValueError(f'Unknown criterion {criterion!r}. Choose one of {sorted(valid)}')
    if criterion in {'minimax_loss','max_min_proportional_gain','jain_savings'}:
        if J_nt is None or np.asarray(J_nt).size!=bundle.N:
            raise ValueError(f'{criterion} needs first-day J_nt for all players')
        J_nt=np.asarray(J_nt,float).ravel()

    def phi(u):
        J=symbolic_costs(bundle,u)
        if criterion=='social_cost':
            return ca.sum1(J)
        if criterion=='minimax_cost':
            return ca.mmax(J)
        if criterion=='cost_variance':
            return ca.sumsqr(J-ca.sum1(J)/bundle.N)/bundle.N
        gain=ca.DM(J_nt)-J
        if criterion=='minimax_loss':
            return ca.mmax(-gain)
        d=ca.diag(ca.DM(1/np.where(np.abs(J_nt)>1e-9,np.abs(J_nt),1))) @ gain
        if criterion=='max_min_proportional_gain':
            return ca.mmax(-d)
        # Jain applied to signed normalized gains is NOT the canonical
        # nonnegative allocation fairness index; interpret cautiously.
        return -(ca.sum1(d)**2)/(bundle.N*ca.sumsqr(d)+1e-12)
    return phi
