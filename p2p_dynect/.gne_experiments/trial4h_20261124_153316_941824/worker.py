
import json, pickle, sys, time, importlib
from pathlib import Path
import numpy as np
from p2p.setup import to_static_gnep

gm = importlib.import_module("dynect.solvers.gne")

input_path, summary_path, output_path = map(Path, sys.argv[1:4])
beta = float(sys.argv[4])
region_limit = int(sys.argv[5])

def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)

# Report entry and exit of Julia's multiparametric solver.
original_mpsolve = gm._julia.mpsolve

def traced_mpsolve(*args, **kwargs):
    log("ENTER Julia ParametricDAQP.mpsolve")
    t0 = time.monotonic()
    result = original_mpsolve(*args, **kwargs)
    log("EXIT Julia mpsolve after",
        round(time.monotonic() - t0, 1), "seconds")
    return result

gm._julia.mpsolve = traced_mpsolve

with input_path.open("rb") as f:
    case = pickle.load(f)

log("Constructing StaticGNEP...")

mp, avi, gnep = to_static_gnep(case)

n_beta = (gnep.N - 1) * gnep.m_sh

log(
    "StaticGNEP ready:",
    "decisions =", sum(gnep.n),
    "shared =", gnep.m_sh,
    "beta =", n_beta
)

assert n_beta == 144

log(
    "Starting GNESetSolver:",
    "verbose=2",
    "region_limit =", region_limit,
    "beta =", beta
)

solver = gm.GNESetSolver(
    gnep,
    beta_lb=np.full(n_beta, -beta),
    beta_ub=np.full(n_beta, beta),
    region_limit=region_limit,
    verbose=2,
)

summary = {
    "status": str(solver.status),
    "gne_regions": len(solver.gne_set.regions),
    "beta_dimension": n_beta,
}

summary_path.write_text(json.dumps(summary, indent=2))

log("SOLVER RETURNED:", summary)

with output_path.open("wb") as f:
    pickle.dump(solver.gne_set, f)

log("GNE SET SAVED:", output_path)
