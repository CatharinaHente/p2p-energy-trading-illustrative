import json, sys, time, pickle, traceback, hashlib
from pathlib import Path
import numpy as np
from p2p.setup import to_static_gnep
from dynect import MpAVI
import dynect.solvers.gne as gm

inp, meta, summ, out = map(Path, sys.argv[1:5])
settings = json.loads(meta.read_text())
def log(*xs):
    print(time.strftime('%H:%M:%S'), *xs, flush=True)
try:
    with inp.open('rb') as f:
        case = pickle.load(f)
    _, _, gnep = to_static_gnep(case)
    nb = (gnep.N - 1) * gnep.m_sh
    beta = settings['beta']
    p = MpAVI.nabetani(gnep, beta_lb=np.full(nb, -beta), beta_ub=np.full(nb, beta))
    sha = hashlib.sha256(np.ascontiguousarray(np.asarray(p.A, float)).tobytes()).hexdigest()
    if sha != settings['A_sha']:
        raise ValueError('Nabetani A changed across processes: AS0 indices invalid')
    AS0 = settings['as0']
    log('Nabetani ready: decisions=', p.H.shape[0], 'rows=', p.A.shape[0],
        'beta=', nb, 'AS0 size=', len(AS0))
    log('ENTER mpsolve, warm start AS0, region_limit=', settings['limit'])
    tic = time.monotonic()
    raw, status = gm._julia.mpsolve(
        p.H, p.F, p.f, p.A, p.B, p.b, p.C, p.d, p.lb, p.ub,
        opts={'verbose': settings['verbose'], 'region_limit': settings['limit']},
        AS0=AS0,
    )
    log('EXIT mpsolve:', status, 'elapsed=', round(time.monotonic()-tic,2), 's')
    regions = gm.denormalised_regions(raw) if raw is not None else []
    gneset = gm.gne_regions(regions, gnep)
    result = {'status':str(status), 'critical_regions':len(regions),
              'gne_regions':len(gneset.regions), 'as0_size':len(AS0)}
    summ.write_text(json.dumps(result,indent=2))
    try:
        with out.open('wb') as f:
            pickle.dump(gneset,f)
        log('Saved gne_set.pkl')
    except Exception as exc:
        log('GNE set could not be pickled:',repr(exc))
    log('SUMMARY',result)
except BaseException as exc:
    log('ERROR',repr(exc))
    traceback.print_exc()
    summ.write_text(json.dumps({'error':repr(exc)},indent=2))
    sys.exit(1)
