
import json, pickle, sys, time, traceback
from pathlib import Path
import numpy as np
from juliacall import Main as jl
from p2p.setup import to_static_gnep
from dynect import MpAVI
from dynect.solvers.gne import GNESetSolver

def log(*a): print(time.strftime('%H:%M:%S'),*a,flush=True)
try:
    inpath, outpath = map(Path,sys.argv[1:3])
    beta=float(sys.argv[3]); limit=int(sys.argv[4]); verbose=int(sys.argv[5])
    log('Julia smoke test:',jl.seval('1+1'),'version',jl.seval('string(VERSION)'))
    with inpath.open('rb') as f: case=pickle.load(f)
    start=time.perf_counter(); _,avi,gnep=to_static_gnep(case)
    log('StaticGNEP ready in',round(time.perf_counter()-start,3),'s',
        'decisions',sum(gnep.n),'m_sh',gnep.m_sh)
    nb=(gnep.N-1)*gnep.m_sh
    start=time.perf_counter()
    p=MpAVI.nabetani(gnep,beta_lb=np.full(nb,-beta),beta_ub=np.full(nb,beta))
    log('Nabetani ready in',round(time.perf_counter()-start,3),'s',
        'H',p.H.shape,'A',p.A.shape,'C',p.C.shape)
    log('ENTER GNESetSolver (Julia region search)')
    start=time.perf_counter()
    solver=GNESetSolver(gnep,beta_lb=np.full(nb,-beta),beta_ub=np.full(nb,beta),
                        verbose=verbose,region_limit=limit)
    summary={'status':str(solver.status),'gne_regions':len(solver.gne_set.regions),
             'elapsed_solver_s':round(time.perf_counter()-start,2),
             'decisions':sum(gnep.n),'beta':nb}
    outpath.write_text(json.dumps(summary,indent=2))
    log('DONE:',summary)
except Exception as e:
    log('ERROR:',repr(e));traceback.print_exc()
    sys.exit(1)
