"""Strategic GNE selection on the *original bilateral* P2P LQ game.

Requires the user's p2p/setup.py and dynect_py_p2p source. Gurobi is used
only for convex QP/QCQP solves; no ParametricDAQP region enumeration is done.

Main guarantees/limitations:
- Unilateral best responses retain ALL original shared constraints, fixing every
  other agent's full decision vector. A feasible profile with negligible best-
  response improvement is a numerical GNE of the exact supplied GameBundle.
- Weighted social welfare is a GNE *only if* agent cost gradients have no
  cross-player blocks (checked), and all weights are strictly positive.
- The code works on a reduced, same-resolution game only if each removed row
  has been independently certified redundant or equivalent.
- All comparisons of reported EUR costs vs the condensed LQ player costs are
  explicit. Nonzero regularization changes the game even if numerically tiny.
- A no-trade profile is used solely as a bargaining benchmark, not as a GNE.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


GAME_FIELDS = ('A','B','Q','R','P','c','q','r','p','C_x','b_x',
               'C_loc','b_loc','C_u','b_u')


def _arrays(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (list, tuple)):
        return [_arrays(y) for y in x]
    if isinstance(x, dict):
        return {str(k): _arrays(v) for k, v in x.items()}
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if hasattr(x, 'isoformat'):
        return x.isoformat()
    raise TypeError(f'Cannot serialize value of type {type(x)!r}; inspect scenario data')


def aggregate_to_hour(data: dict, target_dt: float = 1.0):
    """Aggregate 24h source data. This changes temporal resolution (not exact)."""
    d = copy.deepcopy(data)
    old_dt = float(d['dt'])
    factor = int(round(target_dt / old_dt))
    if factor < 1 or not np.isclose(factor * old_dt, target_dt, atol=1e-12):
        raise ValueError('Target timestep must be an integer multiple of source timestep')
    T0 = len(d['buy_price'])
    Te = int(d.get('T_eval', T0))
    if T0 % factor or Te % factor:
        raise ValueError('Horizon and evaluation horizon must divide target timestep')
    n = len(d['demand']); T = T0 // factor
    for key in ('demand', 'generation', 'hp_base_power', 'hp_q_heat'):
        d[key] = np.asarray(d[key],float).reshape(n,T,factor).mean(axis=2)
    d['ev_driving_load'] = np.asarray(d['ev_driving_load'],float).reshape(n,T,factor).sum(axis=2)
    av = np.asarray(d['ev_availability'],int).reshape(n,T,factor)
    mixed = (av.min(axis=2) != av.max(axis=2))
    d['ev_availability'] = av.min(axis=2).astype(int)  # conservative if mixed
    for key in ('buy_price', 'sell_price', 'p2p_price'):
        d[key] = np.asarray(d[key],float).reshape(T,factor).mean(axis=1)
    d.update(dt=float(target_dt),resolution_h=float(target_dt),T=T,T_eval=Te//factor)
    return d, mixed


def clean_local(A, b):
    """Exact duplicate-normal removal. No rounding of coefficients."""
    A = np.asarray(A, float); b = np.asarray(b, float).ravel()
    best = {}
    for row, rhs in zip(A, b):
        row = row.copy(); row[row == 0] = 0.0
        if not np.any(row):
            if rhs < 0:
                raise ValueError('Infeasible constant local row')
            continue
        key = row.tobytes()
        if key not in best or rhs < best[key][1]:
            best[key] = (row, float(rhs))
    if not best:
        return np.zeros((0,A.shape[1])), np.zeros(0)
    return np.stack([v[0] for v in best.values()]), np.array([v[1] for v in best.values()])


def injection_intervals(data):
    """Conservative individual injection box, not a solution-specific screen."""
    n, T = len(data['demand']), len(data['buy_price'])
    lo, hi = np.zeros((n,T)), np.zeros((n,T))
    base = np.asarray(data['generation'],float) - np.asarray(data['demand'],float)
    av = np.asarray(data['ev_availability'],float)
    bn = int(data.get('battery_polygon_sides',8))
    en = int(data.get('ev_polygon_sides',8))
    for i in range(n):
        b = min(float(data['battery_power'][i]),float(data['battery_smax'][i])*math.cos(math.pi/bn))
        e = min(float(data['ev_power'][i]),float(data['ev_smax'][i])*math.cos(math.pi/en))
        ev = e * av[i]
        lo[i] = base[i] - b - ev - float(data['hp_max_power'][i])
        hi[i] = base[i] + b + ev * float(bool(data.get('ev_v2g',True)))
    return lo, hi


def certify_feeder(case, margin: float = 1e-8):
    """Interval certificate for every IEEE33 line/voltage shared inequality.

    Only rows with worst-case violation < -margin may be removed globally.
    Conservative boxes can make the certificate inconclusive; retain such rows.
    """
    from p2p.setup import _feeder_maps
    if case.grid is None:
        raise ValueError('Cannot certify feeder limits without a feeder')
    g,d = case.grid,case.data
    lo,hi = injection_intervals(d)
    f0,L,drop = _feeder_maps(g,list(map(int,d['agent_bus'])))
    aa = L[:,:,None]*lo[None,:,:]; bb = L[:,:,None]*hi[None,:,:]
    fmin=f0[:,None]+np.minimum(aa,bb).sum(axis=1)
    fmax=f0[:,None]+np.maximum(aa,bb).sum(axis=1)
    M=-drop @ L
    v0=float(g['root_voltage'])**2-drop @ f0
    va=M[:,:,None]*lo[None,:,:]; vb=M[:,:,None]*hi[None,:,:]
    vmin=v0[:,None]+np.minimum(va,vb).sum(axis=1)
    vmax=v0[:,None]+np.maximum(va,vb).sum(axis=1)
    lim=float(d['line_limit_mva'])*math.cos(math.pi/8)
    bound={}
    for k,e in enumerate(g['branches']):
        bound[f'flow_{tuple(e)}_upper']=float(np.max(fmax[k]-lim))
        bound[f'flow_{tuple(e)}_lower']=float(np.max(-lim-fmin[k]))
    for bus in range(g['bus_count']):
        if bus==g['root']:
            continue
        bound[f'voltage_{bus}_upper']=float(np.max(vmax[bus]-float(g['voltage_max'][bus])**2))
        bound[f'voltage_{bus}_lower']=float(np.max(float(g['voltage_min'][bus])**2-vmin[bus]))
    records=[]
    for label in case.shared_labels:
        typ = 'trade' if label.startswith('edge_') else ('flow' if label.startswith('flow_') else 'voltage')
        if typ=='trade':
            records.append(dict(label=label,type=typ,worst_bound=None,certified_redundant=False))
        else:
            if label not in bound:
                raise KeyError(f'Unknown feeder label {label}')
            records.append(dict(label=label,type=typ,worst_bound=bound[label],
                                certified_redundant=bool(bound[label] < -margin)))
    return records


def rebuild_constraints(case, *, keep_labels=None, clean_locals=True,
                        localize_single_agent=True):
    """Preserve game dynamics/objectives; delete shared rows by certificate only."""
    from dynect import DynLQGameTV
    g=case.game
    kw={k:getattr(g,k) for k in GAME_FIELDS}
    labels=list(case.shared_labels)
    if keep_labels is None:
        keep_labels=labels
    if len(set(keep_labels))!=len(keep_labels) or not set(keep_labels).issubset(labels):
        raise ValueError('Shared row label selection is invalid')
    keep=set(keep_labels)
    ids=[j for j,label in enumerate(labels) if label in keep]
    owners={}; moved=[]
    if localize_single_agent:
        for j in ids:
            who=[i for i in range(case.N) if any(np.any(np.asarray(g.C_u[t][i][j])!=0)
                                            for t in range(case.T))]
            if len(who)<=1:
                if not who and any(float(g.b_u[t][j]) < 0 for t in range(case.T)):
                    raise ValueError(f'Infeasible constant shared row: {labels[j]}')
                owners[j]=who; moved.append(j)
        ids=[j for j in ids if j not in moved]
    locA=[];locb=[];shA=[];shb=[]
    for t in range(case.T):
        ai=[np.asarray(g.C_loc[t][i],float).copy() for i in range(case.N)]
        bi=[np.asarray(g.b_loc[t][i],float).copy() for i in range(case.N)]
        for j in moved:
            if owners[j]:
                i=owners[j][0]
                ai[i]=np.vstack([ai[i],np.asarray(g.C_u[t][i][j:j+1],float)])
                bi[i]=np.append(bi[i],float(g.b_u[t][j]))
        if clean_locals:
            cleaned=[clean_local(a,b) for a,b in zip(ai,bi)]
            ai=[x[0] for x in cleaned];bi=[x[1] for x in cleaned]
        locA.append(ai);locb.append(bi)
        shA.append([np.asarray(g.C_u[t][i],float)[ids].copy() for i in range(case.N)])
        shb.append(np.asarray(g.b_u[t],float)[ids].copy())
    kw.update(C_loc=locA,b_loc=locb,C_u=shA,b_u=shb)
    return dataclasses.replace(case,game=DynLQGameTV(**kw),
                               shared_labels=[labels[j] for j in ids]), [labels[j] for j in moved]


def lp_prune_local(case, *, tol=1e-9):
    """Sequential LP implication checks per local time block. Conservative on failures.

    A row is removed ONLY if its maximum over all other current local rows is
    no larger than its RHS + tol. This is numerical LP certification; record
    tolerance and unresolved solver statuses in the reduction report.
    """
    from dynect import DynLQGameTV
    g=case.game; kw={k:getattr(g,k) for k in GAME_FIELDS}
    newA=[];newb=[];report=[]
    for t in range(case.T):
        a_t=[]; b_t=[]
        for i in range(case.N):
            A=np.asarray(g.C_loc[t][i],float)
            b=np.asarray(g.b_loc[t][i],float)
            keep=list(range(len(b))); failures=0; worst_margin=-np.inf
            for j in range(len(b)):
                if j not in keep:
                    continue
                other=[k for k in keep if k!=j]
                if not other:
                    continue
                lp=linprog(-A[j],A_ub=A[other],b_ub=b[other],
                           bounds=[(None,None)]*A.shape[1],method='highs')
                if lp.status==0:
                    margin=float(-lp.fun-b[j]); worst_margin=max(worst_margin,margin)
                    if margin <= tol:
                        keep.remove(j)
                elif lp.status!=3:
                    failures+=1
            a_t.append(A[keep].copy());b_t.append(b[keep].copy())
            report.append(dict(t=t,agent=i,initial=int(len(b)),remaining=len(keep),
                               removed=len(b)-len(keep),unresolved=failures,
                               max_tested_margin=None if not np.isfinite(worst_margin) else worst_margin))
        newA.append(a_t);newb.append(b_t)
    kw.update(C_loc=newA,b_loc=newb)
    return dataclasses.replace(case,game=DynLQGameTV(**kw)),report


def model_max_violation(case, u):
    """Evaluate *dynamic* model rows, no extra condensed AVI allocated."""
    from p2p.setup import unpack_controls,rollout_states
    z=unpack_controls(case,u);x=rollout_states(case,u)
    worst=0.0
    for t in range(case.T):
        sh=np.zeros(len(case.game.b_u[t]))
        for i in range(case.N):
            ui=np.array([z[i][name][t] for name in case.controls[i]])
            a=np.asarray(case.game.C_loc[t][i],float)
            if len(a):
                worst=max(worst,float(np.max(a@ui-np.asarray(case.game.b_loc[t][i]))))
            sh+=np.asarray(case.game.C_u[t][i],float)@ui
        if len(sh):
            worst=max(worst,float(np.max(sh-np.asarray(case.game.b_u[t]))))
        if len(case.game.C_x[t]):
            worst=max(worst,float(np.max(np.asarray(case.game.C_x[t])@x[t+1]-np.asarray(case.game.b_x[t]))))
    return float(max(worst,0))


def reduction_counts(case):
    g=case.game
    return dict(N=case.N,T=case.T,dt=float(case.data['dt']),
                decisions=int(case.T*sum(g.nu)),shared_per_step=int(g.m_u),
                shared_total=int(case.T*g.m_u),
                local_rows=sum(len(g.b_loc[t][i]) for t in range(case.T) for i in range(case.N)),
                state_rows=sum(len(g.b_x[t]) for t in range(case.T)),
                beta=int((case.N-1)*case.T*g.m_u))


def build_reduced_model(date='2016-01-07', seed=42, *, target_dt=1.0,
                        source_dt=0.5, lp_prune=True, lp_tol=1e-9,
                        clearing_epsilon_kw=0.0,
                        input_regularization=0.0,state_regularization=0.0):
    """Full active-only game of the configured structure -> certified 24h reduced game.

    Returns (full_same_resolution, reduced, audit). All feeder rows are retained
    unless certified redundant. No edges/devices/horizon are dropped. Aggregation
    from source_dt to target_dt is a SEPARATE physical approximation.
    """
    from scenarios import make_scenario_horizon
    from p2p.setup import build_game,load_grid
    data0=make_scenario_horizon(date,seed=seed,resolution_h=source_dt,lookahead_h=0)
    data,mixed=aggregate_to_hour(data0,target_dt)
    if abs(len(data['buy_price'])*target_dt-24)>1e-8:
        raise ValueError('Expected the 24h first-day horizon (no lookahead)')
    grid=load_grid()
    choices=dict(clearing_epsilon_kw=clearing_epsilon_kw,
                 state_regularization=state_regularization,
                 input_regularization=input_regularization)
    full=build_game(data,grid=grid,**choices)
    certified=certify_feeder(full)
    keep=[r['label'] for r in certified if not r['certified_redundant']]
    # Retain stage-by-stage counts to make reductions scientifically auditable.
    cleaned,_=rebuild_constraints(full,clean_locals=True,localize_single_agent=False)
    screened,_=rebuild_constraints(cleaned,keep_labels=keep,clean_locals=True,
                                   localize_single_agent=False)
    localized,moved=rebuild_constraints(screened,clean_locals=True,
                                        localize_single_agent=True)
    reduced=localized
    stages=[dict(stage='Full feeder',**reduction_counts(full)),
            dict(stage='Exact local duplicates',**reduction_counts(cleaned)),
            dict(stage='Certified feeder screen',**reduction_counts(screened)),
            dict(stage='Move 1-player shared rows',**reduction_counts(localized))]
    pre_lp_counts=reduction_counts(reduced)
    local_report=[]
    if lp_prune:
        reduced,local_report=lp_prune_local(reduced,tol=lp_tol)
        stages.append(dict(stage='LP-implied local rows',**reduction_counts(reduced)))
    audit=dict(date=date,seed=seed,source_dt=source_dt,target_dt=target_dt,
               aggregation_is_exact=False,mixed_ev_availability_slots=int(mixed.sum()),
               choices=choices,lp_pruned=lp_prune,lp_tol=lp_tol,
               counts_full=reduction_counts(full),counts_pre_lp=pre_lp_counts,
               counts_reduced=reduction_counts(reduced),stages=stages,
               certified_feeder=certified,moved_to_local=moved,
               local_lp_report=local_report,
               keep_shared_labels=list(reduced.shared_labels))
    return full,reduced,audit


def save_reduced_model(path, case, audit):
    """Self-contained compressed game snapshot; no Python pickle is used.

    Reconstructs the real GameBundle from saved scenario+IEEE33 data and
    re-inserts EXACT saved reduced local constraints, while also storing H/f/A/b.
    No scenario regeneration or feeder/LP screening needed during reload.
    """
    from dynect import MpAVI
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    avi=MpAVI.from_game(case.game).at(case.x0)
    meta=dict(version=1,data=_arrays(case.data),grid=_arrays(case.grid),
              choices=_arrays(case.choices),labels=list(case.shared_labels),
              controls=_arrays(case.controls),audit=_arrays(audit),
              description='Same-resolution screened IEEE33 active-only bilateral GameBundle')
    contents=dict(metadata_json=np.array(json.dumps(meta,allow_nan=False)),
                  H=np.asarray(avi.H,float),f=np.asarray(avi.f,float),
                  A=np.asarray(avi.A,float),b=np.asarray(avi.b,float))
    for t in range(case.T):
        for i in range(case.N):
            contents[f'local_A_{t}_{i}']=np.asarray(case.game.C_loc[t][i],float)
            contents[f'local_b_{t}_{i}']=np.asarray(case.game.b_loc[t][i],float)
    # Passing a file handle prevents numpy from silently appending '.npz'.
    with path.open('wb') as fh:
        np.savez_compressed(fh,**contents)
    with np.load(path,allow_pickle=False) as written:
        assert written['H'].shape==contents['H'].shape
        assert json.loads(str(written['metadata_json'].item()))['audit']==_arrays(audit)
    return path


def load_reduced_model(path, *, check_matrices=True):
    """Reload the independently saved reduced GameBundle, without scenarios.py."""
    from dynect import DynLQGameTV
    from p2p.setup import build_game
    from dynect import MpAVI
    path=Path(path)
    with np.load(path,allow_pickle=False) as npz:
        meta=json.loads(str(npz['metadata_json'].item()))
        if meta.get('version')!=1:
            raise ValueError('Unsupported model snapshot version')
        g=meta['grid']
        if g is not None:
            g['branches']=[tuple(map(int,e)) for e in g['branches']]
        for key in ('r_ohm','p_load_mw','voltage_kv','voltage_min','voltage_max'):
            if g is not None and key in g:
                g[key]=np.asarray(g[key],float)
        d=meta['data']
        case=build_game(d,grid=g,**meta['choices'])
        # Recover certified subset, and exact stored local rows, including
        # single-agent migrated shared rows and locally LP-pruned rows.
        label_ids=[case.shared_labels.index(s) for s in meta['labels']]
        kw={k:getattr(case.game,k) for k in GAME_FIELDS}
        kw.update(C_u=[[np.asarray(case.game.C_u[t][i],float)[label_ids].copy()
                        for i in range(case.N)] for t in range(case.T)],
                  b_u=[np.asarray(case.game.b_u[t],float)[label_ids].copy() for t in range(case.T)],
                  C_loc=[[np.array(npz[f'local_A_{t}_{i}'],float) for i in range(case.N)]
                         for t in range(case.T)],
                  b_loc=[[np.array(npz[f'local_b_{t}_{i}'],float) for i in range(case.N)]
                         for t in range(case.T)])
        case=dataclasses.replace(case,game=DynLQGameTV(**kw),
                                 shared_labels=list(meta['labels']))
        if check_matrices:
            avi=MpAVI.from_game(case.game).at(case.x0)
            for k in ('H','f','A','b'):
                expected=np.asarray(npz[k],float)
                actual=np.asarray(getattr(avi,k),float)
                if actual.shape!=expected.shape or not np.allclose(actual,expected,atol=1e-8,rtol=1e-9):
                    raise RuntimeError(f'Reloaded condensed {k} differs from saved snapshot')
    return case,meta['audit']


def agent_slices(case):
    offsets=np.cumsum([0]+[case.T*int(k) for k in case.game.nu])
    return [slice(int(offsets[i]),int(offsets[i+1])) for i in range(case.N)]


def fairness_metrics(costs, no_trade_costs):
    """Fairness diagnostics in EUR; no canonical Jain score for signed savings."""
    J=np.asarray(costs,float); nt=np.asarray(no_trade_costs,float)
    if J.shape!=nt.shape or J.ndim!=1:
        raise ValueError('Costs and baseline must be same-length vectors')
    gain=nt-J; denom=np.maximum(np.abs(nt),1.0)
    p=gain/denom
    def jain_nonnegative(v):
        if np.any(v<0):
            return float('nan')
        total=float(np.dot(v,v))
        return float(v.sum()**2/(len(v)*total)) if total>1e-15 else float('nan')
    return dict(total_cost=float(J.sum()),total_gain=float(gain.sum()),
                worst_cost=float(J.max()),worst_loss=float((-gain).max()),
                min_proportional_gain=float(p.min()),
                cost_variance=float(np.var(J)),
                gain_variance=float(np.var(gain)),
                proportional_gain_variance=float(np.var(p)),
                jain_savings=jain_nonnegative(gain),
                jain_proportional_savings=jain_nonnegative(p),
                nbs_log=float(np.log(gain).sum()) if np.all(gain>0) else float('-inf'),
                n_losers=int((gain < -1e-6).sum()),
                individually_rational=bool(np.all(gain>=-1e-6)),
                gains=gain.tolist(),proportional_gains=p.tolist())


class StrategicGame:
    """Condensed game with independent per-player convex quadratic objectives.

    Keeps constraints in the EXACT supplied game, including both directions of
    each bilateral reciprocity inequality. No region enumeration is attempted.
    """
    def __init__(self, case, *, separability_tol=1e-8):
        from dynect import MpAVI
        from p2p.setup import evaluated_costs
        self.case=case
        avi=MpAVI.from_game(case.game).at(case.x0)
        self.gnep=None  # static GNEP is unnecessary for BR/fairness selection
        self.H=np.asarray(avi.H,float)
        self.f=np.asarray(avi.f,float).reshape(-1)
        self.A=sparse.csr_matrix(np.asarray(avi.A,float))
        self.b=np.asarray(avi.b,float).ravel()
        self.slices=agent_slices(case)
        self.n=len(self.f);self.N=case.N
        self.Hi=[];self.fi=[]
        for i,s in enumerate(self.slices):
            Hii=self.H[s,s]
            if np.max(np.abs(Hii-Hii.T)) > 1e-7:
                raise ValueError(f'Player {i} own Hessian is not symmetric')
            self.Hi.append(sparse.csr_matrix((Hii+Hii.T)*.5))
            self.fi.append(self.f[s].copy())
        self.separability_error=max((float(np.max(np.abs(self.H[s_i,s_j])))
                                     for i,s_i in enumerate(self.slices)
                                     for j,s_j in enumerate(self.slices) if i!=j),default=0.)
        self.separable=bool(self.separability_error<=separability_tol)
        # Additive constants matter for fairness metrics, but never for best responses.
        self.offsets=np.asarray(evaluated_costs(case,np.zeros(self.n),first_day=True),float)
        self.regularization=case.choices.copy()

    def _require_separable(self):
        if not self.separable:
            raise ValueError('Nonzero cross-player pseudo-gradient blocks; weighted QP is NOT a valid weighted VI')

    def model_costs(self,u):
        self._require_separable()
        u=np.asarray(u,float).ravel()
        return np.array([.5*float(u[s]@(self.Hi[i]@u[s]))+self.fi[i]@u[s]+self.offsets[i]
                         for i,s in enumerate(self.slices)])

    def physical_costs(self,u):
        from p2p.setup import evaluated_costs
        return np.asarray(evaluated_costs(self.case,u,first_day=True),float)

    def model_cost_discrepancy(self,u):
        return self.model_costs(u)-self.physical_costs(u)

    def feasibility(self,u):
        u=np.asarray(u,float).ravel()
        if u.size!=self.n or not np.all(np.isfinite(u)):
            raise ValueError(f'Expected finite decision vector of length {self.n}')
        raw=np.asarray(self.A@u-self.b).ravel()
        return dict(max_violation=float(max(0.,np.max(raw,initial=0.))),
                    number_violations=int((raw>1e-6).sum()))

    @staticmethod
    def _gp():
        try:
            import gurobipy as gp
        except ImportError as exc:
            raise ImportError('Gurobi Python API is required for solving. Install gurobipy in the notebook kernel.') from exc
        return gp

    def _new_model(self,name,verbose):
        gp=self._gp();m=gp.Model(name)
        m.Params.OutputFlag=int(verbose)
        m.Params.NonConvex=0  # disallow unintended nonconvex QP/QCQP
        return m

    def _player_expr(self,z,i,*,with_constant=False):
        s=self.slices[i]; zi=z[s]
        expr=.5*(zi @ self.Hi[i] @ zi)+self.fi[i]@zi
        return expr+self.offsets[i] if with_constant else expr

    def _trade_indices(self):
        edges=[tuple(sorted(map(int,e))) for e in self.case.data['edges']]
        lookup=[]
        for a,b in edges:
            ia=self.slices[a].start+self.case.controls[a][f'trade_{b}']
            ib=self.slices[b].start+self.case.controls[b][f'trade_{a}']
            # Agent-major blocks, T contiguous per agent with stage width nu.
            ia=np.arange(self.case.T)*self.case.game.nu[a]+ia
            ib=np.arange(self.case.T)*self.case.game.nu[b]+ib
            lookup.append((ia,ib))
        return edges,lookup

    def extract_trades(self,u):
        u=np.asarray(u,float).ravel()
        _,pairs=self._trade_indices()
        return np.vstack([u[ia] for ia,_ in pairs])

    def _set_trades(self,m,z,values):
        edges,pairs=self._trade_indices()
        q=np.asarray(values,float)
        if q.shape!=(len(edges),self.case.T):
            raise ValueError(f'fixed_trades expected shape {(len(edges),self.case.T)}, got {q.shape}')
        if not np.all(np.isfinite(q)):
            raise ValueError('Non-finite fixed trades')
        if abs(float(self.case.choices.get('clearing_epsilon_kw',0.)))>1e-12:
            raise ValueError('Trade-search routine is restricted to exact reciprocity (epsilon=0)')
        for k,(ia,ib) in enumerate(pairs):
            m.addConstr(z[ia]==q[k],name=f'trade_fixed_pos_{k}')
            m.addConstr(z[ib]==-q[k],name=f'trade_fixed_neg_{k}')

    def _optimize(self,m,z,*,tag):
        gp=self._gp();m.optimize()
        if m.Status!=gp.GRB.OPTIMAL:
            raise RuntimeError(f'{tag} not optimal: Gurobi status {m.Status}, SolCount={m.SolCount}')
        u=np.asarray(z.X,float).ravel()
        return dict(u=u,solver_status=int(m.Status),objective=float(m.ObjVal),
                    max_violation=self.feasibility(u)['max_violation'],
                    physical_costs=self.physical_costs(u).tolist(),
                    model_costs=self.model_costs(u).tolist())

    def solve_weighted(self,weights=None,*,fixed_trades=None,verbose=False):
        """Positive weighted potential QP; returns a GNE if costs separable."""
        self._require_separable()
        w=np.ones(self.N) if weights is None else np.asarray(weights,float).ravel()
        if w.shape!=(self.N,) or not np.all(np.isfinite(w)) or np.any(w<=0):
            raise ValueError('Weights must be finite and strictly positive for every agent')
        gp=self._gp();m=self._new_model('weighted_bilateral_GNE',verbose)
        z=m.addMVar(self.n,lb=-gp.GRB.INFINITY,name='u')
        m.addMConstr(self.A,z,'<',self.b,name='all_game_constraints')
        if fixed_trades is not None:
            self._set_trades(m,z,fixed_trades)
        obj=0.0
        for i in range(self.N):
            obj+=float(w[i])*self._player_expr(z,i)
        m.setObjective(obj,gp.GRB.MINIMIZE)
        result=self._optimize(m,z,tag='weighted GNE')
        result.update(method='weighted',weights=w.tolist(),
                      fixed_trades=fixed_trades is not None)
        return result

    def solve_no_trade(self,*,verbose=False):
        edges,_=self._trade_indices()
        return self.solve_weighted(np.ones(self.N),
                    fixed_trades=np.zeros((len(edges),self.case.T)),verbose=verbose)

    def best_response(self,u,i,*,verbose=False):
        """Solve exact constrained unilateral response (all other decisions FIXED)."""
        self._require_separable()  # physical-cost audit is defined for the original bilateral game
        u=np.asarray(u,float).ravel()
        if u.shape!=(self.n,):
            raise ValueError('Candidate shape mismatch')
        if not 0<=i<self.N:
            raise ValueError('Invalid prosumer')
        gp=self._gp();s=self.slices[i];Ai=self.A[:,s]
        rhs=self.b-np.asarray(self.A@u).ravel()+np.asarray(Ai@u[s]).ravel()
        m=self._new_model(f'best_response_{i}',verbose)
        zi=m.addMVar(s.stop-s.start,lb=-gp.GRB.INFINITY,name='ui')
        m.addMConstr(Ai,zi,'<',rhs,name='full_unilateral_constraints')
        # General pseudo-gradient may have cross-player terms: include them.
        grad_other=np.asarray(self.H[s,:]@u).ravel()-np.asarray(self.H[s,s]@u[s]).ravel()
        obj=.5*(zi@self.Hi[i]@zi)+(self.fi[i]+grad_other)@zi
        m.setObjective(obj,gp.GRB.MINIMIZE);m.optimize()
        if m.Status!=gp.GRB.OPTIMAL:
            raise RuntimeError(f'Player {i} best response did not solve, status {m.Status}')
        v=u.copy();v[s]=np.asarray(zi.X,float).ravel()
        return dict(agent=i,u=v,solver_status=int(m.Status),
                    gain_model=(.5*float(u[s]@(self.Hi[i]@u[s])) +
                        (self.fi[i]+grad_other)@u[s] - float(m.ObjVal)),
                    gain_physical=float(self.physical_costs(u)[i]-self.physical_costs(v)[i]),
                    max_violation=self.feasibility(v)['max_violation'])

    def verify(self,u,*,feas_tol=1e-6,gap_tol_eur=1e-5,
               relative_gap_tol=1e-7,verbose=False):
        """Numerically certify *any* candidate GNE, not just a vGNE.

        Accept if physically feasible and for every agent the true-cost
        best-response gain <= absolute + relative tolerance; also report the
        *regularized game* gain. Best responses use the game Hessian; with
        nonzero regularization that represents the regularized game's BR, so
        strict original-cost certification requires zero regularization.
        """
        self._require_separable()
        feasibility=self.feasibility(u)
        if feasibility['max_violation']>feas_tol:
            return dict(is_numerical_GNE=False,reason='infeasible candidate',
                        feasibility=feasibility,players=[])
        physical=self.physical_costs(u)
        report=[];pass_all=True
        for i in range(self.N):
            br=self.best_response(u,i,verbose=verbose)
            tol=gap_tol_eur+relative_gap_tol*max(1.,abs(float(physical[i])))
            ok=(br['gain_model']<=tol and br['gain_physical']<=tol
                and br['max_violation']<=feas_tol)
            pass_all &= bool(ok)
            report.append(dict(agent=i,candidate_cost_eur=float(physical[i]),
                               best_response_cost_eur=float(physical[i]-br['gain_physical']),
                               improvement_eur=br['gain_physical'],
                               improvement_regularized=br['gain_model'],
                               unilateral_feasibility=br['max_violation'],
                               tolerance_eur=tol,passes=bool(ok)))
        exact_costs=all(abs(float(self.case.choices.get(k,0)))<1e-15
                        for k in ('input_regularization','state_regularization'))
        return dict(is_numerical_GNE=bool(pass_all),
                    verifies_exact_reported_costs=bool(exact_costs),
                    reason='passed' if pass_all else 'profitable unilateral deviation',
                    feasibility=feasibility,players=report,
                    model_physical_cost_discrepancy=self.model_cost_discrepancy(u).tolist()
                      if self.separable else None)

    def solve_minimax(self,no_trade_costs,*,kind='absolute_cost',
                      tie_tol=1e-7,verbose=False):
        """Two-stage convex QCQP, Pareto efficiency as tie-breaker.

        kind: absolute_cost / absolute_loss / relative_loss.
        Uses model quadratic costs, including model reg if present; compare
        against evaluated EUR costs and set regularization=0 for exact matches.
        """
        self._require_separable()
        nt=np.asarray(no_trade_costs,float)
        if nt.shape!=(self.N,):
            raise ValueError('No-trade cost vector shape mismatch')
        if kind not in ('absolute_cost','absolute_loss','relative_loss'):
            raise ValueError('Unknown minimax kind')
        gp=self._gp();m=self._new_model('fair_minimax_GNE',verbose)
        z=m.addMVar(self.n,lb=-gp.GRB.INFINITY,name='u')
        m.addMConstr(self.A,z,'<',self.b)
        t=m.addVar(lb=-gp.GRB.INFINITY,name='worst')
        for i in range(self.N):
            expr=self._player_expr(z,i,with_constant=True)
            if kind=='absolute_cost':
                m.addConstr(expr<=t,name=f'minimax_{i}')
            elif kind=='absolute_loss':
                m.addConstr(expr-float(nt[i])<=t,name=f'minimax_{i}')
            else:
                d=max(1.,abs(float(nt[i])))
                m.addConstr((expr-float(nt[i]))/d<=t,name=f'minimax_{i}')
        m.setObjective(t,gp.GRB.MINIMIZE);m.optimize()
        if m.Status!=gp.GRB.OPTIMAL:
            raise RuntimeError(f'Minimax phase 1 failed, status {m.Status}')
        tstar=float(t.X)
        # Stage 2: positive weighted sum, holding the optimal worst score.
        m.addConstr(t<=tstar+tie_tol,name='lexicographic_bound')
        obj=0.0
        for i in range(self.N):
            obj+=self._player_expr(z,i)
        m.setObjective(obj,gp.GRB.MINIMIZE)
        result=self._optimize(m,z,tag='minimax tie-break')
        result.update(method='lexicographic_minimax',kind=kind,optimal_worst=tstar,
                      tie_tolerance=tie_tol)
        return result

    def _linear_oracle(self,gradient,*,verbose=False):
        gp=self._gp();m=self._new_model('nash_linear_oracle',verbose)
        z=m.addMVar(self.n,lb=-gp.GRB.INFINITY,name='u')
        m.addMConstr(self.A,z,'<',self.b)
        m.setObjective(np.asarray(gradient,float)@z,gp.GRB.MINIMIZE)
        m.optimize()
        if m.Status!=gp.GRB.OPTIMAL:
            raise RuntimeError(f'Linear oracle failed, status {m.Status}')
        return np.asarray(z.X,float).ravel()

    def solve_nash_bargaining(self,no_trade_costs,*,max_iter=80,
                             fw_gap_tol=1e-5,min_gain=1e-8,verbose=False):
        """Convex Nash-bargaining via conditional gradient (Frank-Wolfe).

        Globally optimal only WHEN FW gap meets tolerance and the original
        cost expressions match the mathematical game. Intermediate iterates
        are NOT asserted to be GNE. Requires strictly positive savings.
        """
        self._require_separable()
        nt=np.asarray(no_trade_costs,float)
        if nt.shape!=(self.N,):
            raise ValueError('Invalid baseline shape')
        # Maximise the minimum *absolute* gain to find an IR start.
        gp=self._gp();m=self._new_model('maxmin_savings_seed',verbose)
        z=m.addMVar(self.n,lb=-gp.GRB.INFINITY,name='u')
        m.addMConstr(self.A,z,'<',self.b)
        tau=m.addVar(lb=-gp.GRB.INFINITY,name='min_gain')
        for i in range(self.N):
            m.addConstr(self._player_expr(z,i,with_constant=True)+tau<=float(nt[i]))
        m.setObjective(tau,gp.GRB.MAXIMIZE);m.optimize()
        if m.Status!=gp.GRB.OPTIMAL:
            raise RuntimeError('Could not find a maximum-minimum-savings seed')
        if float(tau.X)<=min_gain:
            raise ValueError(f'No strictly positive common savings margin found: {tau.X:.4g} EUR')
        u=np.asarray(z.X,float).ravel()
        trace=[];converged=False
        for k in range(max_iter):
            gain=nt-self.model_costs(u)
            if np.min(gain)<=min_gain:
                raise RuntimeError('Lost strict positive savings: check cost calibration')
            grad=np.zeros(self.n)
            for i,s in enumerate(self.slices):
                grad[s]=(np.asarray(self.Hi[i]@u[s]).ravel()+self.fi[i])/gain[i]
            v=self._linear_oracle(grad,verbose=False)
            d=v-u
            fw_gap=float(grad@(u-v))
            log_nbs=float(np.log(gain).sum())
            trace.append(dict(iteration=k,fw_gap=fw_gap,log_nbs=log_nbs,
                              min_gain=float(np.min(gain))))
            if verbose and (k%10==0 or fw_gap<=fw_gap_tol):
                print(f'NBS iteration {k}: FW gap={fw_gap:.3g}, logNBS={log_nbs:.6g}')
            if fw_gap<=fw_gap_tol:
                converged=True;break
            # Exact 1-D convex line search for -sum_i log(gain_i).
            linear=np.array([(np.asarray(self.Hi[i]@u[s]).ravel()+self.fi[i])@d[s]
                              for i,s in enumerate(self.slices)])
            quad=np.array([float(d[s]@(self.Hi[i]@d[s])) for i,s in enumerate(self.slices)])
            def savings(alpha):
                return gain-alpha*linear-.5*alpha*alpha*quad
            hi=1.0
            for i in range(self.N):
                if savings(hi)[i] <= min_gain:
                    from scipy.optimize import brentq
                    boundary=brentq(lambda a: float(savings(a)[i]-min_gain),0,hi)
                    hi=min(hi,max(0.,boundary*(1-1e-9)))
            if hi<=1e-12:
                break
            def derivative(alpha):
                g=savings(alpha)
                return float(np.sum((linear+alpha*quad)/g))
            if derivative(hi)<=0:
                alpha=hi
            elif derivative(0)>=0:
                break
            else:
                from scipy.optimize import brentq
                alpha=brentq(derivative,0,hi)
            u=u+alpha*d
        return dict(u=u,method='nash_frank_wolfe',converged=converged,
                    iterations=len(trace),trace=trace,
                    fw_gap=trace[-1]['fw_gap'],
                    physical_costs=self.physical_costs(u).tolist(),
                    model_costs=self.model_costs(u).tolist(),
                    max_violation=self.feasibility(u)['max_violation'])

    def equality_rows(self, tol=1e-10):
        """Indices of rows r for which some row s has A_s=-A_r and b_s=-b_r (an equality)."""
        if getattr(self,'_eq_rows',None) is not None:
            return self._eq_rows
        A=self.A.toarray(); b=self.b
        keys={}
        for r in range(A.shape[0]):
            keys.setdefault(np.round(np.append(A[r],b[r])/tol).astype(np.int64).tobytes(),[]).append(r)
        eq=set()
        for r in range(A.shape[0]):
            if r in eq:
                continue
            mate=keys.get(np.round(-np.append(A[r],b[r])/tol).astype(np.int64).tobytes())
            if mate:
                eq.add(r); eq.update(mate)
        self._eq_rows=np.array(sorted(eq),dtype=int)
        return self._eq_rows

    def joint_equality_directions(self, tol=1e-9):
        """Orthonormal basis of JOINT moves (all players) that keep every equality row.

        Unlike per-player moves, this includes reciprocal trade changes,
        which differ between candidates and therefore matter for fairness metrics.
        """
        if getattr(self,'_joint_basis',None) is not None:
            return self._joint_basis
        E=self.A[self.equality_rows()].toarray()
        if not len(E):
            self._joint_basis=np.eye(self.n); return self._joint_basis
        _,s,vt=np.linalg.svd(E,full_matrices=True)
        rank=int(np.sum(s>tol*max(s[0],1.0)))
        self._joint_basis=vt[rank:].T
        return self._joint_basis

    def audit_costs(self, u=None, *, trials=4, amplitude=0.03,
                    seed=42, cross_tol=1e-9, physical_tol_eur=1e-7,
                    objective_tol_eur=1e-8, objective_rel_tol=1e-6,
                    calibration_step_kw=1.0):
        """Empirical separability and model/physical cost calibration check.

        Structural off-diagonal blocks are a necessary check; the independent
        cost perturbations catch additive cross-agent dependencies that do not
        appear in the pseudo-gradient. Perturbations may be infeasible; they
        test objective dependence, NOT feasible GNE deviations.

        Calibration (model quadratic cost vs reported EUR cost) is tested only
        along joint directions that keep every equality row of the game
        (paired opposite inequalities: cyclic SOC, energy balance, reciprocity).
        A term that is constant on that set (e.g. terminal BESS credit under
        cyclic SOC) cannot change any feasible comparison. Steps have
        `calibration_step_kw` RMS per coordinate; the test is relative:
        |dModel-dEUR| <= objective_tol_eur + objective_rel_tol*|dEUR|.
        """
        u = np.zeros(self.n) if u is None else np.asarray(u,float).ravel()
        if u.shape != (self.n,):
            raise ValueError('Audit decision vector has the wrong shape')
        rng=np.random.default_rng(seed)
        origin_p=self.physical_costs(u)
        origin_m=self.model_costs(u) if self.separable else None
        rows=[]
        for i,sl in enumerate(self.slices):
            ids=np.arange(sl.start,sl.stop)
            basis=self.joint_equality_directions()
            other=np.arange(self.N)!=i
            for k in range(trials):
                # (a) Independence: ARBITRARY own perturbation must not move others' costs.
                change=u.copy()
                choose=rng.choice(ids,size=min(8,len(ids)),replace=False)
                change[choose]+=rng.normal(0,amplitude,size=len(choose))
                pi=self.physical_costs(change)
                # (b) Calibration: player i's model vs EUR cost change along JOINT
                # directions that keep every equality of the game (cyclic SOC,
                # energy balance, trade reciprocity), so trade changes are tested too.
                # Terms that vanish on the equality set (e.g. terminal BESS credit
                # under cyclic SOC) cannot affect any feasible schedule.
                delta=None
                if self.separable and basis.shape[1]:
                    d=basis@rng.normal(0,1,basis.shape[1])
                    d*=calibration_step_kw*np.sqrt(self.n)/max(np.linalg.norm(d),1e-15)
                    calib=u+d
                    pc=self.physical_costs(calib); mc=self.model_costs(calib)
                    delta=float(abs((mc[i]-origin_m[i])-(pc[i]-origin_p[i])))
                    scale=float(abs(pc[i]-origin_p[i]))
                rows.append(dict(player=int(i),trial=int(k),
                    other_cost_change_eur=float(np.max(np.abs(pi[other]-origin_p[other]))),
                    own_model_physical_delta_eur=delta,
                    own_physical_change_eur=(scale if delta is not None else None),
                    calibration_ok=(None if delta is None else
                        bool(delta<=objective_tol_eur+objective_rel_tol*scale)),
                    equality_free_dims=int(basis.shape[1])))
        cross=float(self.separability_error)
        other=max((r['other_cost_change_eur'] for r in rows),default=0.)
        own=max((r['own_model_physical_delta_eur'] for r in rows
                 if r['own_model_physical_delta_eur'] is not None),default=float('inf'))
        checked=[r for r in rows if r['calibration_ok'] is not None]
        calib_ok=bool(checked) and all(r['calibration_ok'] for r in checked)
        rel=max((r['own_model_physical_delta_eur']/max(r['own_physical_change_eur'],1e-12)
                 for r in checked),default=None)
        reg={k:float(self.case.choices.get(k,0))
             for k in ('input_regularization','state_regularization')}
        exact=all(abs(v)<=1e-15 for v in reg.values())
        return dict(date=str(self.case.data.get('date','unspecified')),
            n_players=self.N,n_decisions=self.n,n_constraints=len(self.b),
            cross_player_H_max=cross,
            other_player_physical_cost_change_max_eur=other,
            own_model_physical_cost_delta_max_eur=(own if np.isfinite(own) else None),
            structural_separable=cross<=cross_tol,
            physical_separable_test=other<=physical_tol_eur,
            own_model_physical_relative_max=rel,
            calibration_directions='equality-preserving',
            objective_delta_test=bool(self.separable and calib_ok),
            zero_regularization=exact,regularization=reg,rows=rows,
            passed=bool(self.separable and cross<=cross_tol and other<=physical_tol_eur
                        and calib_ok and exact))

    def solve_pareto_refinement(self, reference_u, *, weights=None,
                                cost_slack_eur=1e-5,verbose=False):
        """Pareto-efficient GNE under upper bounds on reference player costs.

        For strictly positive weights and separable convex objectives, the
        weighted-sum minimizer with upper caps J_i<=J_i(reference)+slack is
        Pareto-efficient in the ORIGINAL feasible set: a dominating feasible
        allocation would still respect all caps. It is consequently a GNE.
        Does not guarantee to preserve non-monotone fairness metrics exactly.
        """
        self._require_separable()
        ref=np.asarray(reference_u,float).ravel()
        if self.feasibility(ref)['max_violation']>1e-6:
            raise ValueError('Reference for Pareto refinement must be jointly feasible')
        w=np.ones(self.N) if weights is None else np.asarray(weights,float).ravel()
        if w.shape!=(self.N,) or np.any(w<=0) or not np.all(np.isfinite(w)):
            raise ValueError('Pareto weights must be finite and strictly positive')
        slack=np.broadcast_to(np.asarray(cost_slack_eur,float),(self.N,))
        if np.any(slack<0):
            raise ValueError('Cost cap slack must be nonnegative')
        cap=self.model_costs(ref)+slack
        gp=self._gp();m=self._new_model('pareto_refinement',verbose)
        z=m.addMVar(self.n,lb=-gp.GRB.INFINITY,name='u')
        m.addMConstr(self.A,z,'<',self.b)
        objective=0
        for i in range(self.N):
            expr=self._player_expr(z,i,with_constant=True)
            m.addConstr(expr<=float(cap[i]),name=f'cost_cap_{i}')
            objective+=float(w[i])*expr
        m.setObjective(objective,gp.GRB.MINIMIZE)
        result=self._optimize(m,z,tag='Pareto refinement')
        result.update(method='pareto_cost_cap_refinement',
                      reference_model_costs=self.model_costs(ref).tolist(),
                      cost_caps=cap.tolist(),weights=w.tolist())
        return result

    def solve_robust_minimax(self, no_trade_costs, *, kind='absolute_loss',
                             tie_weight=1e-4,verbose=False):
        """Single-phase monotone minimax objective; avoids fragile t<=t* tie constraint.

        Minimizes worst metric + tie_weight * sum_i J_i. The positive weighted
        cost term makes the resulting optimum Pareto-efficient/GNE for separable
        objectives, but this is a regularized minimax, not exact lexicographic.
        """
        self._require_separable()
        if kind not in ('absolute_cost','absolute_loss','relative_loss'):
            raise ValueError('Unknown minimax kind')
        if not np.isfinite(tie_weight) or tie_weight<=0:
            raise ValueError('tie_weight must be strictly positive')
        nt=np.asarray(no_trade_costs,float).ravel()
        if nt.shape!=(self.N,):
            raise ValueError('No-trade cost vector shape mismatch')
        gp=self._gp();m=self._new_model('robust_minimax',verbose)
        z=m.addMVar(self.n,lb=-gp.GRB.INFINITY,name='u')
        m.addMConstr(self.A,z,'<',self.b)
        t=m.addVar(lb=-gp.GRB.INFINITY,name='worst')
        obj=t
        for i in range(self.N):
            expr=self._player_expr(z,i,with_constant=True)
            if kind=='absolute_cost':
                metric=expr
            elif kind=='absolute_loss':
                metric=expr-float(nt[i])
            else:
                metric=(expr-float(nt[i]))/max(1.,abs(float(nt[i])))
            m.addConstr(metric<=t,name=f'worst_{i}')
            obj+=float(tie_weight)*expr
        m.setObjective(obj,gp.GRB.MINIMIZE)
        result=self._optimize(m,z,tag='robust minimax')
        result.update(method='monotone_minimax',kind=kind,
                      worst_metric=float(t.X),tie_weight=float(tie_weight),
                      exact_lexicographic=False)
        return result

    def solve_robust_maxmin_savings(self,no_trade_costs, *,
                                    relative=False,tie_weight=1e-4,verbose=False):
        """Maximize worst savings (or proportional savings) with Pareto tie weight."""
        self._require_separable()
        nt=np.asarray(no_trade_costs,float).ravel()
        if nt.shape!=(self.N,) or tie_weight<=0:
            raise ValueError('Invalid baseline or tie_weight')
        gp=self._gp();m=self._new_model('maxmin_savings',verbose)
        z=m.addMVar(self.n,lb=-gp.GRB.INFINITY,name='u')
        m.addMConstr(self.A,z,'<',self.b)
        t=m.addVar(lb=-gp.GRB.INFINITY,name='minimum_savings')
        obj=-t
        for i in range(self.N):
            expr=self._player_expr(z,i,with_constant=True)
            scale=max(1.,abs(float(nt[i]))) if relative else 1.
            m.addConstr((float(nt[i])-expr)/scale>=t,name=f'savings_floor_{i}')
            obj+=float(tie_weight)*expr
        m.setObjective(obj,gp.GRB.MINIMIZE)
        result=self._optimize(m,z,tag='max-min savings')
        result.update(method='monotone_maxmin_savings',relative=bool(relative),
                      minimum_savings=float(t.X),tie_weight=float(tie_weight))
        return result

    def nash_derived_weights(self,u,no_trade_costs, *, use_physical=False):
        """Return positive normalized NBS marginal weights at a positive-gain point."""
        nt=np.asarray(no_trade_costs,float).ravel()
        J=self.physical_costs(u) if use_physical else self.model_costs(u)
        gain=nt-J
        if np.any(gain<=0):
            raise ValueError('NBS-derived weights require strictly positive gains')
        w=1.0/gain
        return w/np.exp(np.mean(np.log(w)))

    def search_weights(self,no_trade_costs,*,n_samples=12,seed=42,
                       log_weight_std=.6,metric='nbs_log',verify_top=3):
        """Sample positive weights; all solutions are weighted potential GNEs."""
        self._require_separable()
        rng=np.random.default_rng(seed)
        options=[np.ones(self.N)]
        for _ in range(n_samples):
            logw=rng.normal(0,log_weight_std,self.N)
            logw-=logw.mean()
            options.append(np.exp(logw))
        records=[]
        for w in options:
            try:
                sol=self.solve_weighted(w)
                score=fairness_metrics(sol['physical_costs'],no_trade_costs)
                records.append(dict(weights=w.tolist(),u=sol['u'],metrics=score,
                                    status='solved'))
            except RuntimeError as exc:
                records.append(dict(weights=w.tolist(),status=f'failed: {exc}'))
        good=[r for r in records if r['status']=='solved']
        reverse=metric in ('nbs_log','total_gain','min_proportional_gain',
                           'jain_savings','jain_proportional_savings')
        good.sort(key=lambda r:r['metrics'][metric],reverse=reverse)
        for r in good[:verify_top]:
            r['verification']=self.verify(r['u'])
        return records

    def search_trades(self,start_u,no_trade_costs,*,n_samples=8,
                      sigma_kw=.1,seed=42,verify=True):
        """Explore feasible exact bilateral trade schedules without weakening reciprocity.

        Local device decisions are re-optimized jointly at each fixed trade
        schedule; full BR verification is still performed. Infeasible targets
        are reported and skipped. No completeness or global-fairness claim.
        """
        if float(self.case.choices.get('clearing_epsilon_kw',0))!=0:
            raise ValueError('Exact bilateral reciprocity required for search_trades')
        rng=np.random.default_rng(seed)
        q0=self.extract_trades(start_u)
        records=[]
        for k in range(n_samples+1):
            q=q0.copy() if k==0 else q0+rng.normal(0,sigma_kw,q0.shape)
            try:
                sol=self.solve_weighted(fixed_trades=q)
            except RuntimeError as exc:
                records.append(dict(trial=k,status=f'infeasible or unsolved: {exc}'))
                continue
            rec=dict(trial=k,status='solved',trade_rms_kw=float(np.sqrt(np.mean((q-q0)**2))),
                     metrics=fairness_metrics(sol['physical_costs'],no_trade_costs),u=sol['u'])
            if verify:
                rec['verification']=self.verify(sol['u'])
            records.append(rec)
        return records



# --- Date-aware scenario snapshots (reusable across days) -------------------
# This belongs to the suite, not to notebook state. Each notebook can select a
# date independently; the demonstration creates a missing snapshot itself.
SCENARIO_CACHE_VERSION = 5
def configured_structure():
    """(structure_name, n_prosumers) of the project's config.py switch."""
    import sys
    repo = str(Path(__file__).resolve().parent.parent)   # suite lives in the repo
    if repo not in sys.path:
        sys.path.append(repo)
    import config
    return config.structure_name(), int(config.PROSUMER_SCENARIO)


DEFAULT_STRUCTURE_NAME = configured_structure()[0]
FAIR_GNE_FOLDER = 'fair_gne'


def find_simulation_root(repo_root, *, structure_name=DEFAULT_STRUCTURE_NAME,
                         override=None, cwd=None):
    """Locate the ORIGINAL repository's existing simulation results tree.

    No search in any separate notebook subproject and no parallel results tree.
    An explicit override can select a different pre-existing root, but never a
    nested deprecated project directory.
    """
    root = Path(repo_root).expanduser().resolve()
    base = root if override is None else Path(override).expanduser().resolve()
    if base.name == 'p2p_dynect' or 'p2p_dynect' in base.parts:
        raise ValueError('The legacy p2p_dynect directory is not an allowed save root. '
                         'Move the simulation results tree to the original repository first.')
    directory = base / 'structures' / structure_name / 'results'
    if not directory.is_dir():
        raise FileNotFoundError(
            f'The EXISTING simulation results directory was not found: {directory}\n'
            'No alternate structures/ tree was created. Move your original '
            'structures/ directory into the repository root, or explicitly set '
            'SIMULATION_ROOT to the parent of the existing structures/ folder.')
    return base

def test_results_write_access(paths):
    """Create/delete a round-trip probe in the intended fair_gne tree."""
    from uuid import uuid4
    root=Path(paths['simulation_dir'])
    if not root.parent.is_dir():
        raise FileNotFoundError(f'Expected existing simulation results directory: {root.parent}')
    fair=Path(paths['fair_dir'])
    fair.mkdir(parents=True,exist_ok=True)
    p=fair/f'.save_probe_{uuid4().hex}.txt'
    try:
        p.write_text('strategic_GNE_selection save check\n',encoding='utf8')
        if p.read_text(encoding='utf8')!='strategic_GNE_selection save check\n':
            raise IOError(f'Saved-file round-trip failed: {p}')
    finally:
        p.unlink(missing_ok=True)
    return fair.resolve()


def scenario_paths(base_dir, date, *, target_dt=1.0,
                   structure_name=DEFAULT_STRUCTURE_NAME):
    """Use the simulation tree, preserving the existing JSON result files.

    All new artifacts go below
      structures/<structure>/results/<date>/fair_gne/<dt>/{reduction,cache,selection}.
    ``base_dir`` points to the original repository root containing structures/.
    """
    from datetime import date as _date

    try:
        parsed = _date.fromisoformat(str(date))
    except ValueError as exc:
        raise ValueError('DATE must have ISO format YYYY-MM-DD') from exc
    if parsed.year != 2016:
        raise ValueError('The bundled SimBench daily profiles require a 2016 date')
    if str(date) != parsed.isoformat():
        raise ValueError('DATE must have ISO format YYYY-MM-DD')
    dt = float(target_dt)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('target_dt must be positive and finite')
    if not structure_name or any(x in str(structure_name) for x in ("/", "\\", "..")):
        raise ValueError('structure_name must be a single structure-folder name')
    dt_label = f'{dt:g}'.replace('.', 'p') + 'h'
    root = Path(base_dir).resolve()
    existing_day = root / 'structures' / str(structure_name) / 'results' / str(date)
    fair = existing_day / FAIR_GNE_FOLDER / dt_label
    return dict(model_path=fair / 'reduction' / 'reduced_game.npz',
                audit_path=fair / 'reduction' / 'reduction_audit.json',
                results_dir=fair / 'selection',
                cache_dir=fair / 'cache',
                fair_dir=fair,
                simulation_dir=existing_day,
                structure_name=str(structure_name))


def discover_prior_simulations(paths):
    """Index original same-date results WITHOUT claiming they match the reduced game.

    This is deliberately read-only. Original simulation runs may use a different
    horizon, time grid, constraints or objective, so their dispatch/costs are
    NOT silently used for GNE verification or 1-hour no-trade benchmarks.
    """
    out = []
    base = Path(paths['simulation_dir'])
    for kind in ('distributed', 'centralized', 'no_trade', 'fairness',
                 'distributed_network', 'centralized_network', 'infeasible'):
        f = base / (kind + '.json')
        if not f.exists():
            continue
        entry = dict(kind=kind, path=str(f), exists=True, json_valid=False)
        try:
            obj = json.loads(f.read_text(encoding='utf8'))
            entry['json_valid'] = True
            entry['root_keys'] = list(obj)[:20] if isinstance(obj, dict) else []
            entry['reported_status'] = (obj.get('status') or obj.get('reason') or '') if isinstance(obj, dict) else ''
        except (OSError, ValueError) as exc:
            entry['error'] = f'{type(exc).__name__}: {exc}'
        out.append(entry)
    return out


def _scenario_signature(config):
    """Fingerprint the reduction algorithm and external scenario/adapter source.

    Selection-only changes no longer force an expensive reduction rebuild.
    Rebuild explicitly when changing external profile files.
    """
    import hashlib, inspect
    from scenarios import make_scenario_horizon
    from p2p.setup import build_game
    reduction_functions=(aggregate_to_hour,clean_local,injection_intervals,
        certify_feeder,rebuild_constraints,lp_prune_local,build_reduced_model)
    code=''.join(inspect.getsource(fn) for fn in reduction_functions)
    for fn in (make_scenario_horizon,build_game):
        code+=inspect.getsource(fn)
    codehash=hashlib.sha256(code.encode('utf8')).hexdigest()
    encoded=json.dumps(dict(cache_version=SCENARIO_CACHE_VERSION,
                            config=config,reduction_code_sha256=codehash),
                       sort_keys=True,separators=(',',':'))
    return hashlib.sha256(encoded.encode('utf8')).hexdigest()


def ensure_reduced_scenario(date, *, base_dir='.', seed=42,
                            source_dt=0.5, target_dt=1.0, lp_prune=True,
                            lp_tol=1e-9, clearing_epsilon_kw=0.0,
                            input_regularization=0.0,
                            state_regularization=0.0,
                            force_rebuild=False, check_matrices=True, verbose=True,
                            structure_name=DEFAULT_STRUCTURE_NAME):
    """Load a matching reduced bilateral game or build, audit and save it.

    Returns ``(case, audit, info)``. This is the notebook's *single entry point*:
    change DATE and rerun; a missing/different-date/different-settings snapshot
    is never used silently. The .npz contains its own audit and is fully reloadable.

    Regenerate when external scenario DATA files change: set force_rebuild=True.
    """
    configured, n_expected = configured_structure()
    if structure_name != configured:
        raise ValueError(f'The game builder uses config.py ({configured}); '
                         f'refusing to label it {structure_name}. Change config.py instead.')
    paths = scenario_paths(base_dir, date, target_dt=target_dt,
                           structure_name=structure_name)
    config = dict(date=str(date), seed=int(seed), source_dt=float(source_dt),
                  target_dt=float(target_dt), lp_prune=bool(lp_prune),
                  lp_tol=float(lp_tol),
                  clearing_epsilon_kw=float(clearing_epsilon_kw),
                  input_regularization=float(input_regularization),
                  state_regularization=float(state_regularization),
                  structure_name=str(structure_name))
    signature = _scenario_signature(config)
    model_path = paths['model_path']
    reason = 'forced rebuild' if force_rebuild else 'no matching snapshot'
    if model_path.exists() and not force_rebuild:
        try:
            case, audit = load_reduced_model(model_path, check_matrices=check_matrices)
            valid = (audit.get('scenario_cache_signature') == signature
                     and audit.get('scenario_cache_config') == config
                     and audit.get('date') == str(date)
                     and case.N == n_expected
                     and int(round(24 / float(target_dt))) == case.T
                     and np.isclose(float(case.data['dt']), float(target_dt)))
            if valid:
                if not paths['audit_path'].exists():
                    paths['audit_path'].parent.mkdir(parents=True,exist_ok=True)
                    paths['audit_path'].write_text(json.dumps(audit, indent=2, default=str))
                if verbose:
                    print(f'Using cached {date} reduced model: {model_path}')
                return case, audit, dict(**paths, rebuilt=False,
                                         cache_reason='matched date, settings and suite version')
            reason = 'snapshot metadata/date/options no longer match'
        except Exception as exc:
            reason = f'snapshot failed validation/reload: {type(exc).__name__}: {exc}'
    if verbose:
        print(f'Building reduced bilateral scenario for {date} ({reason})')
    build_options={k:v for k,v in config.items() if k != 'structure_name'}
    _, case, audit = build_reduced_model(**build_options)
    audit = dict(audit, scenario_cache_config=config,
                 scenario_cache_signature=signature,
                 scenario_cache_version=SCENARIO_CACHE_VERSION)
    model_path.parent.mkdir(parents=True,exist_ok=True)
    save_reduced_model(model_path, case, audit)
    paths['audit_path'].parent.mkdir(parents=True,exist_ok=True)
    paths['audit_path'].write_text(json.dumps(audit, indent=2, default=str))
    # Validate the persisted model against its condensed matrices on every build.
    case_loaded, audit_loaded = load_reduced_model(model_path, check_matrices=True)
    if audit_loaded.get('scenario_cache_signature') != signature:
        raise RuntimeError('Reduced-model round-trip changed the scenario signature')
    if verbose:
        print(f'Saved and reload-verified: {model_path}')
    return case_loaded, audit_loaded, dict(**paths, rebuilt=True, cache_reason=reason)


def save_candidate(path,case,u,*,method,verification=None,extra=None):
    """Save independently inspectable dispatch + a lightweight JSON manifest."""
    from p2p.setup import evaluated_costs
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    encoded=json.dumps(_arrays(dict(method=method,
                            verification=verification,extra=extra)),allow_nan=False)
    with path.open('wb') as fh:
        np.savez_compressed(fh,u=np.asarray(u,float),
                            original_costs=np.asarray(evaluated_costs(case,u),float),
                            meta_json=np.array(encoded))
    with np.load(path,allow_pickle=False) as saved:
        if saved['u'].shape!=np.asarray(u).shape:
            raise IOError(f'Candidate save round-trip failed: {path}')
    return path



# --- Reuse results from the SAME certified reduced model ------------------
RESULT_CACHE_VERSION = 1


def _cached_result_key(model_path, method, options=None):
    """Hash the actual saved model bytes and all optimization choices."""
    import hashlib
    model_path = Path(model_path)
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    payload = dict(version=RESULT_CACHE_VERSION,
                   model_sha256=model_hash, method=str(method),
                   options=_arrays({} if options is None else options))
    key = hashlib.sha256(json.dumps(payload,sort_keys=True,
                        separators=(',', ':')).encode('utf8')).hexdigest()
    return key, model_hash


def run_or_load_result(game, path, model_path, *, method, solve,
                       options=None, verify=False, force=False,
                       feasibility_tol=1e-6, cost_tol=1e-4):
    """Execute an optimization or load a matching prior result, never by date alone.

    Requires the same reduced-model SHA256, method and parameters. Independently
    rechecks the cached vector's feasibility and reported EUR costs. A cached
    GNE report is reused only when it was recorded for this exact model hash.
    Set force=True to recompute. Returns (result_dict, cache_hit).
    """
    path=Path(path)
    key, model_hash=_cached_result_key(model_path,method,options)
    if path.exists() and not force:
        try:
            with np.load(path,allow_pickle=False) as data:
                meta=json.loads(str(data['meta_json'].item()))
                extra=meta.get('extra') or {}
                if extra.get('result_cache_key') != key:
                    raise ValueError('model or solver options differ')
                u=np.asarray(data['u'],float).ravel()
                saved_J=np.asarray(data['original_costs'],float).ravel()
            if len(u)!=game.n or not np.all(np.isfinite(u)):
                raise ValueError('wrong length or nonfinite decision vector')
            if game.feasibility(u)['max_violation'] > feasibility_tol:
                raise ValueError('cached solution is infeasible')
            live_J=np.asarray(game.physical_costs(u),float).ravel()
            if not np.allclose(saved_J,live_J,atol=cost_tol,rtol=0):
                raise ValueError('physical cost mismatch')
            result=dict(extra.get('solver_result') or {})
            result['u']=u
            result['physical_costs']=live_J.tolist()
            result['model_costs']=game.model_costs(u).tolist()
            if verify and not meta.get('verification'):
                raise ValueError('cached result lacks GNE verification')
            if meta.get('verification') is not None:
                result['verification']=meta['verification']
            return result,True
        except (OSError, KeyError, ValueError, TypeError, RuntimeError) as exc:
            print(f'Cache miss ({method}): {exc}; solving again')
    result=solve()
    if 'u' not in result:
        raise ValueError('Solver did not return a decision vector')
    result=dict(result)
    u=np.asarray(result['u'],float).ravel()
    if verify:
        result['verification']=game.verify(u)
    if game.feasibility(u)['max_violation'] > feasibility_tol:
        raise RuntimeError(f'Not caching infeasible solution from {method}')
    extra=dict(result_cache_key=key, model_sha256=model_hash,
               options=_arrays({} if options is None else options),
               solver_result=_arrays({k:v for k,v in result.items()
                                      if k not in ('u','verification')}))
    save_candidate(path,game.case,u,method=method,
                   verification=result.get('verification'),extra=extra)
    return result,False


__all__=['StrategicGame','build_reduced_model','save_reduced_model',
         'load_reduced_model','aggregate_to_hour','clean_local','certify_feeder',
         'rebuild_constraints','lp_prune_local','model_max_violation',
         'reduction_counts','fairness_metrics','save_candidate','agent_slices',
         'scenario_paths','ensure_reduced_scenario','discover_prior_simulations',
         'run_or_load_result','find_simulation_root','test_results_write_access']
