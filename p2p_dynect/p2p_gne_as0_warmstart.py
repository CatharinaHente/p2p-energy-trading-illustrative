"""Warm-start ParametricDAQP from a certified active set of the Nabetani mpAVI.

Diagnostic helper for the user's local P2P notebook. No Julia code or installations
are patched. Fixed-theta QP needs symmetric positive-definite H.
"""
import os
import sys
import json
import time
import pickle
import signal
import subprocess
import hashlib
import importlib
from pathlib import Path
from datetime import datetime
import numpy as np
import gurobipy as gp
from scipy import sparse
from scipy.optimize import linprog
from dynect import MpAVI
from p2p.setup import to_static_gnep

def warm_start_from_fixed_parameter(game_case, u_hint=None, beta=2., tol=2e-6):
    _,_,gnep=to_static_gnep(game_case)
    nb=(gnep.N-1)*gnep.m_sh
    p=MpAVI.nabetani(gnep,beta_lb=np.full(nb,-beta),beta_ub=np.full(nb,beta))
    H=np.asarray(p.H,float); F=np.asarray(p.F,float); f=np.asarray(p.f,float).ravel()
    A=np.asarray(p.A,float);B=np.asarray(p.B,float);b=np.asarray(p.b,float).ravel()
    C=np.asarray(p.C,float);d=np.asarray(p.d,float).ravel()
    lb=np.asarray(p.lb,float).ravel();ub=np.asarray(p.ub,float).ravel()
    n=H.shape[0];k=F.shape[1]
    assert H.shape==(n,n) and A.shape[1]==n and B.shape==(len(b),k)
    assert len(lb)==len(ub)==k
    th=np.zeros(k)
    if not (np.all(th>=lb-1e-9) and np.all(th<=ub+1e-9) and
            np.all(C@th<=d+1e-9)):
        raise ValueError('θ=0 is outside the Nabetani parameter domain. '
                         'Do not use the original vGNE without a feasible θ.')
    skew=float(np.max(np.abs(H-H.T)))
    Hs=(H+H.T)/2
    mineig=float(np.linalg.eigvalsh(Hs)[0])
    print(f'Transformed dimensions: n={n}, inequalities={len(b)}, θ={k}, β={nb}')
    print(f'Symmetric H error={skew:.3e}; min eig(Hsym)={mineig:.3e}')
    if skew>1e-7 or mineig<1e-9:
        raise ValueError('Transformed VI is not demonstrably a strictly convex QP. '
                         'Cannot infer its solution with a symmetric-QP solver.')
    rhs=b+B@th; gradient_const=f+F@th
    m=gp.Model('Nabetani_fixed_theta_for_AS0');m.Params.OutputFlag=0
    z=m.addMVar(n,lb=-gp.GRB.INFINITY,name='z')
    constr=m.addMConstr(sparse.csr_matrix(A),z,'<',rhs,name='ineq')
    m.setObjective(.5*(z@sparse.csr_matrix(Hs)@z)+gradient_const@z,gp.GRB.MINIMIZE)
    m.optimize()
    if m.Status!=gp.GRB.OPTIMAL:
        raise RuntimeError(f'Fixed-parameter QP did not solve: Gurobi status={m.Status}')
    x=np.asarray(z.X,float).ravel()
    slack=rhs-A@x
    grad=H@x+gradient_const
    primal_violation=float(max(0.,-slack.min()))
    active=np.flatnonzero(slack<=tol)
    # A basic nonnegative LP solution can remove redundant tight rows.
    lam=np.zeros(len(b))
    if len(active):
        lp=linprog(1.+1e-9*np.arange(len(active)),A_eq=A[active,:].T,
                   b_eq=-grad,bounds=(0,None),method='highs')
        if not lp.success:
            raise RuntimeError('Cannot certify nonnegative KKT multipliers on tight '
                               'transformed rows: '+lp.message)
        lam[active]=lp.x
    elif np.max(np.abs(grad))>1e-6:
        raise RuntimeError('No active rows but gradient is nonzero.')
    active_support=np.flatnonzero(lam>max(1e-8,1e-7*np.max(lam)))
    KKT=float(np.max(np.abs(grad+A.T@lam)))
    compl=float(np.max(np.abs(lam*slack)))
    rank=(int(np.linalg.matrix_rank(A[active_support],tol=1e-8))
          if len(active_support) else 0)
    print('Fixed-θ QP solved; max primal violation:',f'{primal_violation:.2e}')
    print('Tight rows:',len(active),'positive-dual rows:',len(active_support),
          'rank:',rank)
    print('Stationarity residual:',f'{KKT:.2e}',
          'complementarity residual:',f'{compl:.2e}')
    if u_hint is not None:
        u_hint=np.asarray(u_hint,float).ravel()
        if len(u_hint)==len(x):
            print('RMS difference to existing vGNE:',
                  f'{np.sqrt(np.mean((x-u_hint)**2)):.3e}')
            hint_slack=rhs-A@u_hint
            print('vGNE max violation of Nabetani θ=0 constraints:',
                  f'{max(0.,-float(hint_slack.min())):.3e}')
        else:
            print('vGNE length differs from transformed QP:',len(u_hint),'vs',len(x))
    if not (primal_violation<tol*10 and KKT<3e-4 and compl<3e-4
            and rank==len(active_support)):
        raise RuntimeError('AS0 failed feasibility/KKT/independence checks. '
                           'Do not pass it to Julia without investigating.')
    return dict(as0=active_support.tolist(),theta=th,x=x,mpp=p,
                n_beta=nb, gnep=gnep,primal_violation=primal_violation,
                stationarity=KKT,rank=rank,
                A_sha=hashlib.sha256(np.ascontiguousarray(A).tobytes()).hexdigest())

def run_with_as0(case, warm, *, beta=2.0, region_limit=25,
                 timeout_seconds=180, verbose=2):
    BETA=float(beta)
    REGION_LIMIT=int(region_limit)
    TIMEOUT_SECONDS=int(timeout_seconds)
    VERBOSE=int(verbose)
    p2p_root=Path(importlib.import_module('p2p.setup').__file__).resolve().parents[1]
    dynect_root=Path(importlib.import_module('dynect').__file__).resolve().parents[1]
    env=os.environ.copy()
    env['PYTHONPATH']=os.pathsep.join([str(p2p_root),str(dynect_root),env.get('PYTHONPATH','')])
    env['PYTHONUNBUFFERED']='1'
    folder=p2p_root/'.gne_experiments'/datetime.now().strftime('AS0_%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True,exist_ok=False)
    inp,meta,worker,logpath,summ,out=[folder/k for k in
       ('case.pkl','warm.json','worker.py','solver.log','summary.json','gne_set.pkl')]
    with inp.open('wb') as f:pickle.dump(case,f)
    meta.write_text(json.dumps({'as0':warm['as0'],'A_sha':warm['A_sha'],
                                'beta':BETA,'limit':REGION_LIMIT,'verbose':VERBOSE}))
    worker.write_text("import json, sys, time, pickle, traceback, hashlib\nfrom pathlib import Path\nimport numpy as np\nfrom p2p.setup import to_static_gnep\nfrom dynect import MpAVI\nimport dynect.solvers.gne as gm\n\ninp, meta, summ, out = map(Path, sys.argv[1:5])\nsettings = json.loads(meta.read_text())\ndef log(*xs):\n    print(time.strftime('%H:%M:%S'), *xs, flush=True)\ntry:\n    with inp.open('rb') as f:\n        case = pickle.load(f)\n    _, _, gnep = to_static_gnep(case)\n    nb = (gnep.N - 1) * gnep.m_sh\n    beta = settings['beta']\n    p = MpAVI.nabetani(gnep, beta_lb=np.full(nb, -beta), beta_ub=np.full(nb, beta))\n    sha = hashlib.sha256(np.ascontiguousarray(np.asarray(p.A, float)).tobytes()).hexdigest()\n    if sha != settings['A_sha']:\n        raise ValueError('Nabetani A changed across processes: AS0 indices invalid')\n    AS0 = settings['as0']\n    log('Nabetani ready: decisions=', p.H.shape[0], 'rows=', p.A.shape[0],\n        'beta=', nb, 'AS0 size=', len(AS0))\n    log('ENTER mpsolve, warm start AS0, region_limit=', settings['limit'])\n    tic = time.monotonic()\n    raw, status = gm._julia.mpsolve(\n        p.H, p.F, p.f, p.A, p.B, p.b, p.C, p.d, p.lb, p.ub,\n        opts={'verbose': settings['verbose'], 'region_limit': settings['limit']},\n        AS0=AS0,\n    )\n    log('EXIT mpsolve:', status, 'elapsed=', round(time.monotonic()-tic,2), 's')\n    regions = gm.denormalised_regions(raw) if raw is not None else []\n    gneset = gm.gne_regions(regions, gnep)\n    result = {'status':str(status), 'critical_regions':len(regions),\n              'gne_regions':len(gneset.regions), 'as0_size':len(AS0)}\n    summ.write_text(json.dumps(result,indent=2))\n    try:\n        with out.open('wb') as f:\n            pickle.dump(gneset,f)\n        log('Saved gne_set.pkl')\n    except Exception as exc:\n        log('GNE set could not be pickled:',repr(exc))\n    log('SUMMARY',result)\nexcept BaseException as exc:\n    log('ERROR',repr(exc))\n    traceback.print_exc()\n    summ.write_text(json.dumps({'error':repr(exc)},indent=2))\n    sys.exit(1)\n")
    started=time.monotonic();pos=0;last=started;timeout=False
    print('Experiment:',folder,flush=True)
    with logpath.open('w') as lf:
        proc=subprocess.Popen([sys.executable,'-u',str(worker),str(inp),str(meta),str(summ),str(out)],
            cwd=p2p_root,env=env,stdout=lf,stderr=subprocess.STDOUT,start_new_session=True)
        while proc.poll() is None:
            time.sleep(.5);elapsed=time.monotonic()-started
            if elapsed>=TIMEOUT_SECONDS:
                timeout=True
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                break
            if time.monotonic()-last>=10:
                print(f'[{elapsed:.0f} s] worker active',flush=True)
                with logpath.open(errors='replace') as f:
                    f.seek(pos);s=f.read();pos=f.tell()
                if s:print(s[-2000:],flush=True)
                last=time.monotonic()
        proc.wait()
    print('Exit:',proc.returncode,'timed out:',timeout,'elapsed:',round(time.monotonic()-started,1),'s')
    print('Summary:',summ.read_text() if summ.exists() else '(not returned)')
    print('Log tail:',logpath.read_text(errors='replace')[-4000:])

# Short names for notebook use.
prepare_as0 = warm_start_from_fixed_parameter
