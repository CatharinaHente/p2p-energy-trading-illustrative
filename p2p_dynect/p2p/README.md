# Time-varying DyNECT game for the existing P2P energy study

Copy the entire `p2p/` directory into the existing research repository, next to
`scenarios.py`, `config.py` and `network.py`. **Do not replace** the original
`prosumer.py`, `algo_central.py`, or `network.py`.

## First run

```python
from p2p.setup import build_for_day, to_static_gnep
from p2p.vgne import solve_vgne

# config.py must select NETWORK_TOPOLOGY='ieee33' and PROSUMER_SCENARIO=4.
case = build_for_day('2016-01-07')
print(case.N, case.T, case.game.nx, case.game.nu)

# Builds the condensed VI and computes the potential-game vGNE with Gurobi.
sol = solve_vgne(case)
print('Max feasibility violation:', sol['max_violation'])
```

The notebook `p2p_study_validation.ipynb` guides this process and benchmarks
against the saved centralized JSON for the same date and population.

## Modeling fidelity

Retained from `scenarios.py` and `prosumer.py`:

* Actual 2016 SimBench demand, PV and device profiles; price arrays from
  `pricing.py`; trading edges and bus assignments from the configured population.
* 48-hour horizon at 30-min resolution; first 24h valued-cost evaluation.
* Separate grid purchase and sale; signed directed P2P trades, reciprocal
  clearing; trade absolute-value epigraph and associated fee.
* BESS SOC, initial/terminal cycling, power/energy bounds and efficiencies.
* EV SOC, availability-dependent charging and V2G, driving load, departure
  energy requirement, season-dependent limits, and terminal value.
* HP indoor-temperature state dynamics and upper/lower comfort bounds.
* Original linear prices, cycling costs, P2P trade fees, and `p_cost` quadratic
  regularisation. The original energy balance is encoded as paired LOCAL
  inequalities and P2P clearing as paired SHARED inequalities.
* Original case33bw static bus loads, line resistances, real-power flows,
  voltage-squared limits and line limits. The physical line/voltage constraints
  are SHARED because all prosumers' injections affect them.

**Deliberately changed:** no reactive variables, no reactive base load, no
Q-component of the voltage drop, and no P-Q device flexibility polygons.
At Q=0 the original inscribed 8-sided converter and line polygons imply
`|P_net| <= S_max*cos(pi/8)` (unless the device's individual power bound is
tighter). We **retain these active-only slices** rather than silently enlarging
the permitted active power range.
Not an exact voltage model; compare with the original AC/LinDistFlow checks
before interpreting voltage feasibility.

**Potential numerical change:** `state_regularization=input_regularization=1e-9`
adds tiny strictly convex costs to every state/input. Set to zero in
`build_game` if exact legacy objective coefficients are required; then a
potential QP may lose strict convexity. `evaluated_costs` always reports the
**legacy** cost, excluding tiny extra regularization.

**Network scope:** this first implementation intentionally uses IEEE33 only.
The existing `network.py` loads `case33bw` even if config selects LV3/LV5/LV6.
`build_for_day` fails explicitly for other feeders, rather than silently using
the wrong topology. The `assemble_game(data, grid=...)` input can later accept
an appropriately extracted radial SimBench feeder dictionary.

**Feeder baseline:** the original formulation includes `case33bw`'s static
load plus the added prosumer demand. This is retained for comparability but
may double count loads if your prosumers conceptually replace existing buses.

**Parameterization:** `MpAVI.from_game(game)` keeps x0 as its parameter.
`to_static_gnep` evaluates at the actual x0 for GNE computation; player
states remain local after condensation, line limits and trades stay shared.

**P2P clearing:** `clearing_epsilon_kw=0` matches exact legacy reciprocity but
`GNESetSolver` explicitly does not enumerate shared equalities represented
by opposite inequalities. Use a small positive epsilon only for a *different*
relaxed game and label any resulting GNE set partial, bounded, and approximate.

**Scalability:** the full horizon has about 96 steps; state elimination and
shared feeder constraints produce a very large, dense AVI. The notebook does
not run `GNESetSolver` automatically. Validate the vGNE and physical costs
first on four prosumers and one date, and only then experiment with shortened
horizons for GNE-region enumeration.

**No equality elimination by substitution** was performed. Such a substitution
can alter an individual agent's feasible unilateral deviations and hence the
GNE set, even if it preserves central feasibility.

## Reusable fairness plug-ins

`p2p/fairness.py` contains `metrics(J, J_nt)` for the legacy fairness
statistics, `symbolic_costs(bundle,u)` for *actual* first-day evaluated
individual costs, and `selection_objective(bundle, criterion, J_nt)` which
returns a CasADi callback for `OptimalGNEP`. Available initial criteria:
`social_cost`, `minimax_cost`, `cost_variance`, `minimax_loss`,
`max_min_proportional_gain`, and `jain_savings` (signed savings caveat).

General GNE-set computation and selection are **not run by default**.
Existing `GNESetSolver` explicitly does not enumerate shared exact
equalities encoded as opposite inequalities. Your 4-32 agent network
case is substantially larger than the earlier toy; verify vGNE first.

Nash bargaining is intentionally not offered as a plug-in yet: its log
objective requires strictly positive individual gains and thus additional
IR-domain constraints. This cannot be implemented correctly by simply
clipping losses.
