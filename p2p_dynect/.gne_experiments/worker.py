
import sys
import json
import pickle
import numpy as np

from pathlib import Path
from p2p.setup import to_static_gnep
from dynect.solvers.gne import GNESetSolver

input_path, output_path, summary_path = sys.argv[1:4]
beta = float(sys.argv[4])
region_limit = int(sys.argv[5])

with open(input_path, "rb") as f:
    case = pickle.load(f)

print("Building static GNEP...", flush=True)

mp, avi, gnep = to_static_gnep(case)

n_beta = (gnep.N - 1) * gnep.m_sh

print(f"Condensed decision variables: {sum(gnep.n)}", flush=True)
print(f"Shared constraints: {gnep.m_sh}", flush=True)
print(f"Beta dimension: {n_beta}", flush=True)
print(f"Region limit: {region_limit}", flush=True)

assert n_beta <= 300, "Unexpectedly large beta dimension"

print("Starting GNESetSolver...", flush=True)

solver = GNESetSolver(
    gnep,
    beta_lb=np.full(n_beta, -beta),
    beta_ub=np.full(n_beta, beta),
    region_limit=region_limit,
    verbose=1,
)

gne_set = solver.gne_set

summary = {
    "status": str(solver.status),
    "gne_regions": len(gne_set.regions),
    "beta_dimension": n_beta,
    "region_limit": region_limit,
}

# Save the summary before attempting to serialize the region set
Path(summary_path).write_text(json.dumps(summary, indent=2))

print("\n=== RESULT ===", flush=True)
print(summary, flush=True)

try:
    with open(output_path, "wb") as f:
        pickle.dump(gne_set, f)
    print("GNE set saved successfully", flush=True)
except Exception as exc:
    print(f"Could not pickle GNE set: {exc}", flush=True)
