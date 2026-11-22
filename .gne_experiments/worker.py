
import sys, json, pickle
import numpy as np
from p2p.setup import to_static_gnep
from dynect.solvers.gne import GNESetSolver

input_path, output_path, summary_path = sys.argv[1:4]
with open(input_path, "rb") as f:
    case = pickle.load(f)

_, _, gnep = to_static_gnep(case)

n_beta = (gnep.N - 1) * gnep.m_sh
print(f"Condensed decisions: {sum(gnep.n)}", flush=True)
print(f"Shared rows: {gnep.m_sh}", flush=True)
print(f"Beta dimension: {n_beta}", flush=True)

assert n_beta <= 300

solver = GNESetSolver(
    gnep,
    beta_lb=np.full(n_beta, -5.0),
    beta_ub=np.full(n_beta, 5.0),
    region_limit=25,
    verbose=0,
)

with open(output_path, "wb") as f:
    pickle.dump(solver.gne_set, f)

summary = {
    "status": str(solver.status),
    "regions": len(solver.gne_set.regions),
    "beta_dimension": n_beta,
}

with open(summary_path, "w") as f:
    json.dump(summary, f)

print(summary, flush=True)
