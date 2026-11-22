
import importlib, json, pickle, sys, time
from pathlib import Path
import numpy as np
from p2p.setup import to_static_gnep
gm = importlib.import_module('dynect.solvers.gne')

def log(*args):
    print(time.strftime('%H:%M:%S'), *args, flush=True)

# Trace the two expensive phases without modifying the installed package.
if hasattr(gm, '_julia') and hasattr(gm._julia, 'mpsolve'):
    real_mpsolve = gm._julia.mpsolve
    def traced_mpsolve(*args, **kwargs):
        log('ENTER Julia ParametricDAQP.mpsolve (region exploration)')
        t0 = time.monotonic()
        result = real_mpsolve(*args, **kwargs)
        log('EXIT Julia mpsolve after', round(time.monotonic()-t0,1), 's')
        return result
    gm._julia.mpsolve = traced_mpsolve
if hasattr(gm, 'gne_regions'):
    real_filter = gm.gne_regions
    def traced_filter(*args, **kwargs):
        log('ENTER GNE-region filtering')
        t0 = time.monotonic()
        result = real_filter(*args, **kwargs)
        log('EXIT filtering after', round(time.monotonic()-t0,1), 's',
            'GNE regions:', len(result.regions))
        return result
    gm.gne_regions = traced_filter

input_path, output_path, summary_path = map(Path, sys.argv[1:4])
beta, limit = float(sys.argv[4]), int(sys.argv[5])
log('Constructing StaticGNEP...')
with input_path.open('rb') as f:
    case = pickle.load(f)
t0 = time.monotonic()
_, _, gnep = to_static_gnep(case)
n_beta = (gnep.N-1)*gnep.m_sh
log('StaticGNEP ready after', round(time.monotonic()-t0,1), 's:',
    'n=',sum(gnep.n), 'm_sh=',gnep.m_sh, 'beta=',n_beta)
assert n_beta <= 300
log('Starting GNESetSolver; verbose=2, region_limit=', limit)
solver = gm.GNESetSolver(gnep,
    beta_lb=np.full(n_beta,-beta), beta_ub=np.full(n_beta,beta),
    region_limit=limit, verbose=2)
summary = dict(status=str(solver.status), regions=len(solver.gne_set.regions),
               beta_dimension=n_beta, limit=limit)
summary_path.write_text(json.dumps(summary,indent=2))
log('Solver returned:', summary)
with output_path.open('wb') as f:
    pickle.dump(solver.gne_set,f)
log('Saved region set:', output_path)
