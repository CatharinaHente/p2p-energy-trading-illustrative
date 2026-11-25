
import json
import pickle
import sys
import time
import traceback

from pathlib import Path
import numpy as np

from p2p.setup import to_static_gnep
from dynect.solvers.gne import GNESetSolver

def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)

input_file, summary_file, output_file = map(Path, sys.argv[1:4])
beta = float(sys.argv[4])
region_limit = int(sys.argv[5])
verbose = int(sys.argv[6])

try:
    log("Worker started")
    log("Loading dynamic game")

    with input_file.open("rb") as f:
        case = pickle.load(f)

    tic = time.perf_counter()

    log("Constructing StaticGNEP")
    _, _, gnep = to_static_gnep(case)

    nb = (gnep.N - 1) * gnep.m_sh

    log(
        "StaticGNEP ready",
        "decisions =", sum(gnep.n),
        "shared =", gnep.m_sh,
        "beta =", nb,
        "time =", round(time.perf_counter() - tic, 3), "s"
    )

    log("ENTER GNESetSolver / Julia")
    log("Region limit =", region_limit, "verbose =", verbose)

    tic = time.perf_counter()

    solver = GNESetSolver(
        gnep,
        beta_lb=np.full(nb, -beta),
        beta_ub=np.full(nb, beta),
        region_limit=region_limit,
        verbose=verbose,
    )

    log("GNESetSolver returned in",
        round(time.perf_counter() - tic, 2), "seconds")

    summary = {
        "status": str(solver.status),
        "gne_regions": len(solver.gne_set.regions),
        "beta_dimension": nb,
        "region_limit": region_limit
    }

    summary_file.write_text(json.dumps(summary, indent=2))

    with output_file.open("wb") as f:
        pickle.dump(solver.gne_set, f)

    log("RESULT:", summary)
    log("Saved GNE set")

except Exception as e:
    log("ERROR:", repr(e))
    traceback.print_exc()
    summary_file.write_text(json.dumps({"error": repr(e)}, indent=2))
    sys.exit(1)
