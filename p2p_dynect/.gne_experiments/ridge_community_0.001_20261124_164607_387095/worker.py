
import sys,json,time,pickle,traceback
from pathlib import Path
import numpy as np
from p2p.setup import to_static_gnep
from dynect.solvers.gne import GNESetSolver

def log(*xs): print(time.strftime('%H:%M:%S'), *xs, flush=True)
input_file,summary_file,out_file=map(Path,sys.argv[1:4])
beta=float(sys.argv[4]); region_limit=int(sys.argv[5]); verbose=int(sys.argv[6])
try:
    log('Worker Python started; loading game...')
    with input_file.open('rb') as f: case=pickle.load(f)
    tic=time.monotonic()
    log('Condensing to StaticGNEP...')
    _, _, gnep=to_static_gnep(case)
    nb=(gnep.N-1)*gnep.m_sh
    log('Condensed in',round(time.monotonic()-tic,3),'s; decisions=',sum(gnep.n),
        'shared=',gnep.m_sh,'beta=',nb)
    log('ENTER GNESetSolver; region_limit=',region_limit,'beta=±',beta,'verbose=',verbose)
    tic=time.monotonic()
    solver=GNESetSolver(gnep,beta_lb=np.full(nb,-beta),beta_ub=np.full(nb,beta),
                        verbose=verbose,region_limit=region_limit)
    log('GNESetSolver RETURNED after',round(time.monotonic()-tic,2),'s')
    summary=dict(status=str(solver.status),gne_regions=len(solver.gne_set.regions),
                 beta_dimension=nb,region_limit=region_limit)
    summary_file.write_text(json.dumps(summary,indent=2))
    with out_file.open('wb') as f: pickle.dump(solver.gne_set,f)
    log('RESULT',summary)
except BaseException as e:
    log('WORKER ERROR',repr(e));traceback.print_exc()
    summary_file.write_text(json.dumps({'error':repr(e)},indent=2))
    sys.exit(1)
