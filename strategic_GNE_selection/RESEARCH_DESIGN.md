# Research design memo — fair GNE selection in P2P trading

## Objective

Study whether fairness-oriented equilibrium selection improves the distribution of electricity-trading benefits, without exhaustive parametric critical-region enumeration. Focus on the cyclic four-prosumer IEEE33 active-only bilateral game, 24-hour horizon and 1-hour resolution, and subsequently test promising candidates at 30-minute resolution.

## Mathematical foundation

For a jointly feasible set K and player-separable objectives J_i(u_i), a Pareto-efficient point u* in K is a generalized Nash equilibrium. Proof: a feasible unilateral deviation that strictly reduced player i's cost and fixed u_{-i} would leave every other J_j unchanged, contradicting Pareto efficiency. The reverse implication generally fails. This proof **does not** apply when the game includes direct cross-player cost terms (such as a community mismatch penalty) or when the selected fairness optimum is not Pareto-efficient.

For every strictly positive weight vector w, globally minimizing sum_i w_i J_i(u_i) over K yields a Pareto-efficient GNE, provided the costs are convex and the feasible set is jointly convex. NBS maximization (sum of log positive individual savings) and Pareto-monotone minimax/tie objectives can also produce fair GNEs, subject to exact solver feasibility/optimality and cost calibration.

Exact bilateral trading reciprocity means changing a trade unilaterally may be impossible. Consequently, many trade schedules can be GNE after private device optimization while still failing joint efficiency or pairwise renegotiation stability. This is a structural property of the chosen game, not necessarily a desirable market mechanism.

## Proposed experiment families

1. Cost audit: check off-diagonal pseudo-gradient blocks, independent perturbations, objective regularization and original EUR cost calibration; halt if inconsistent.
2. Certified reduction: document exact duplicate cleanup, feeder interval bound certificates, relocation of single-owner rows and optional local LP implication. Distinguish these from time aggregation, which is an approximation.
3. Baseline: equal-weight vGNE, independent full unilateral best responses and the no-trade benchmark.
4. Direct fair optimization: positive weighted social welfare; Pareto-monotone regularized minimax; max-min savings. Re-evaluate full unilateral responses.
5. Nash bargaining: report its own convergence gap, individual positivity of gains, and numerical GNE acceptance separately; use positive marginal weights to refine approximate iterates when appropriate.
6. Alternative trade schedules: preserve reciprocity by construction; reoptimize private operating decisions and verify the full unilateral constraints.
7. Cross-resolution validation: reoptimize lifted trade schedules on the 30-minute, 24-hour model; do not assert equivalence of GNE sets from close vGNE costs.

## Outputs

Per date, the demo writes `structures/ieee33_4_cyclic/results/<date>/fair_gne/1h/selection/` with `comparison.csv` (all candidate fairness scores and GNE status), `experiment_manifest.json` (model SHA256, settings, paths), the candidate `.npz` dispatches and `diagnostics/cost_audit.json`. Existing same-day simulation JSON files remain unchanged and are only used as context.

## Key cautions

- Numerical best-response verification is a *conditional* GNE check of the saved model only; it does not prove global fairness optimality or full-set coverage.
- The 1-hour active-only reduction is not the original 30-minute AC/reactive-power game.
- Metrics requiring positive savings (NBS and certain Jain definitions) have a restricted domain. Do not rank non-admissible candidates by a synthetic NBS score.
- Approximate Nash iterates, even with good log product, can fail GNE verification and must remain distinctly labeled.
- Scientific reference: Hall, Nax and Bolognani (2026), *Fair Selection of Generalized Nash Equilibria: A Comparability Perspective*, arXiv:2611.09060.
