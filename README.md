# P2P electricity trading with storage

Four prosumers, four hourly periods, solar generation, batteries and a
prosumer trading graph.

| File | Contents |
| --- | --- |
| `main.py` | Explicit inputs, four examples and checks in `# %%` cells |
| `prosumer.py` | One `add_prosumer()` function: energy balance, trades, batteries and cost |
| `network.py` | Imports pandapower `case33bw()` and builds multi-period LinDistFlow constraints |
| `algo.py` | Centralized optimization and distributed consensus ADMM |

## Run

Python 3.10+ and an available Gurobi license:

```bash
python -m pip install -r requirements.txt
python main.py
```

For interactive use, open this folder in VS Code with the Python and Jupyter
extensions, select the Python environment, and run the `# %%` cells in order.
After setup, each of the four example cells can run separately.

Each of the four example cells produces image output. Every case exports a
heatmap of bilateral P2P trades: each row is one stored edge orientation,
positive values follow the displayed arrow and negative values flow in the
reverse direction. It is displayed inline in an interactive session and
exported to `results/<case>/<case>_p2p.{pdf,svg,tiff,png}`, with the plotted
values written to `<case>_p2p_source_data.csv` in the same folder.

Every case also exports a feeder voltage profile — one line per period across
all 33 `case33bw` buses, with the `[0.90, 1.10]` p.u. bounds and the prosumer
buses marked — to `results/<case>/<case>_voltage_profile.{pdf,svg,tiff,png}`,
with source data in `<case>_voltage_profile_source_data.csv`. For the
`centralized_network`/`distributed_network` cases the feeder is a hard
constraint of the optimization, so the profile is exactly what the solver
enforced. For the plain `centralized`/`distributed` cases (no network
constraints), `network.evaluate_feeder()` runs a deterministic LinDistFlow
forward/backward sweep *after* the fact, from the buy/sell/battery schedule's
implied bus injections, purely to check whether the unconstrained trades would
have violated the feeder — it never changes the schedule. Both paths also flag
line thermal congestion (`|S| > line_limit_mva`) alongside the voltage check;
`main.py`'s final cell prints the voltage range and both flags for every case
and asserts no violation/congestion only for the two cases where the network
was actually enforced. `scale` (currently 80) is deliberately picked past the
point where the plain `centralized`/`distributed` cases start sagging below
the `0.90` p.u. floor, so that check is illustrative rather than always
`False`: `centralized_network`/`distributed_network` stay feasible across
this range by actively holding voltage at exactly `0.90` p.u. (the
constraint binds) instead of sagging further — the point of running the
network-aware cases at all.

Every case also exports a per-prosumer net-flow chart: one panel per
prosumer, each with two net bars per period — grid (buy minus sell) and P2P
(import minus export), positive above zero for a net purchase, negative below
for a net sale — plus battery state of charge, as a line on a second axis, to
`results/<case>/<case>_energy_balance.{pdf,svg,tiff,png}`, with source data
(one row per prosumer per period, including `soc_start_pct`/`soc_end_pct`) in
`<case>_energy_balance_source_data.csv`. SOC is plotted as a percentage of
that prosumer's own `battery_capacity` rather than kWh, so the right axis is
the same `0-100%` scale on every panel even though the four prosumers have
different capacities; the left (power) axis is still scaled independently per
panel, since net grid/P2P flows aren't naturally comparable across prosumers
the way SOC is.

This went through several revisions. It started as a stacked area against
only a system-wide demand line, adapted from the "P2P impact" chart in
[d-vf/P2PEnergyTrading](https://github.com/d-vf/P2PEnergyTrading) — whose P2P
trades bypass the external grid and so are additive there, unlike the
reciprocal, net-zero trades here — but that hid the battery-charging term and
implied a continuous ramp between four discrete hours. An aggregate diverging
bar chart (generation/discharge/buy vs. demand/charge/sell) fixed both, but
summed away the one thing that matters for a P2P market — which prosumer is a
net buyer or seller. A per-prosumer version of that same 8-category chart
fixed that, but was too busy to read at a glance. The current version keeps
only what a P2P reader needs per prosumer — grid vs. peer reliance, and the
battery level that reliance is buying — with generation/demand/charge/sell
still fully recoverable from `summary.json` and the CSV.

## Mathematical model

The example has four prosumers and four one-hour periods. Demand and solar
generation are fixed inputs; grid exchanges, peer trades and battery schedules
are decisions. Edit the inputs in `main.py`. Demand/generation rows represent
prosumers and columns represent periods.

### Variables and notation

Let $i$ index prosumers, $j\in\mathcal N_i$ their trading neighbours, and
$t=0,\ldots,T-1$ the periods. The interval length is $\Delta t$ hours.

| Symbol | Code in `prosumer.py` | Meaning |
| --- | --- | --- |
| $b_{i,t}\ge0$ | `buy[t]` | Grid purchase, kW |
| $s_{i,t}\ge0$ | `sell[t]` | Grid sale, kW |
| $q_{ij,t}$ | `trade[j,t]` | Signed peer trade: positive means $i$ sells to $j$, kW |
| $c_{i,t},d_{i,t}\ge0$ | `charge[t]`, `discharge[t]` | Battery charging/discharging power, kW |
| $e_{i,t}\ge0$ | `soc[t]` | Stored energy, kWh (not a percentage) |
| $G_{i,t},D_{i,t}$ | `generation[i][t]`, `demand[i][t]` | Fixed generation and demand, kW |

### Cost function

Each prosumer's total cost over the horizon is

$$
J_i=\Delta t\sum_{t=0}^{T-1}\left[
\lambda_t^{\mathrm{buy}}b_{i,t}
-\lambda_t^{\mathrm{sell}}s_{i,t}
-\pi_t\sum_{j\in\mathcal N_i}q_{ij,t}
+\kappa(c_{i,t}+d_{i,t})
+\mu\sum_{j\in\mathcal N_i}|q_{ij,t}|\right].
$$

The code uses `buy_price`, `sell_price`, `p2p_price` and `battery_cost` for
$\lambda^{\mathrm{buy}}$, $\lambda^{\mathrm{sell}}$, $\pi$ and $\kappa$.
The parameter `trade_fee` is $\mu$. All are in EUR/kWh. The last two terms
include a linear battery throughput cost and a small absolute-volume trade fee.
A positive peer sale reduces the seller's cost; the buyer pays the same amount.
Prices are specified inputs, not optimized market-clearing prices.

The centralized problem minimizes $\sum_i J_i$ subject to the constraints
below. Since each pair uses the same price and reciprocal trades, peer payments
cancel in the aggregate objective; they still affect individual costs.

### Prosumer constraints — `add_prosumer()`

Energy balance, for each prosumer and period:

$$
G_{i,t}+b_{i,t}+d_{i,t}
=D_{i,t}+s_{i,t}+c_{i,t}+\sum_{j\in\mathcal N_i}q_{ij,t}.
$$

Battery dynamics and boundary conditions:

$$
e_{i,t+1}=e_{i,t}+\Delta t\left(\eta_c c_{i,t}-\frac{d_{i,t}}{\eta_d}\right),
\qquad e_{i,0}=e_i^{\mathrm{init}},\qquad e_{i,T}=e_i^{\mathrm{init}}.
$$

The terminal condition prevents using initial battery energy for free without
restoring it. In the supplied example, batteries start and end at 30% of
`battery_capacity` (`initial_soc_fraction` in `main.py`) rather than empty —
`initial_soc=0` was an earlier, unrealistic choice of input (real batteries
keep a depth-of-discharge margin and are not dispatched to a literal 0%
state of charge); the cyclic terminal condition itself is unchanged and is a
standard, deliberate assumption (the schedule must be repeatable day to
day), not something specific to the empty-battery choice.

Variable bounds:

$$
\begin{aligned}
0\le b_{i,t},s_{i,t}&\le\overline P^{\mathrm{grid}},\\
-\overline Q\le q_{ij,t}&\le\overline Q,\\
0\le c_{i,t},d_{i,t}&\le\overline P_i^{\mathrm{bat}},\\
\underline\alpha\,\overline E_i\le e_{i,t}&\le\overline\alpha\,\overline E_i.
\end{aligned}
$$

The corresponding inputs are `grid_limit`, `trade_limit`, `battery_power`,
`battery_capacity`, `initial_soc`, `soc_min_fraction` ($\underline\alpha$),
`soc_max_fraction` ($\overline\alpha$), `eta_charge`, `eta_discharge` and
`dt`. The supplied example keeps `soc` within `[10%, 95%]` of capacity
(`soc_min_fraction=.1`, `soc_max_fraction=.95` in `main.py`) rather than the
full `[0%, 100%]` range — real batteries are not run down to a literal 0% or
charged to a literal 100% state of charge (depth-of-discharge/overcharge
margin for cycle life), and `initial_soc` (30% of capacity) must lie inside
this tighter range. Set capacity, battery power and initial energy to zero to
disable storage. The battery model is continuous and convex: simultaneous
charging/discharging is not explicitly forbidden, nor is an asymmetric
charge/discharge power limit modeled (`battery_power` bounds both
directions equally). Simultaneous grid buying/selling is also not explicitly
forbidden; the supplied import tariff exceeds the export tariff.

### Peer-market constraints — `algo.py`

Each trading edge requires reciprocal contracts in every period:

$$
q_{ij,t}+q_{ji,t}=0.
$$

The supplied undirected trading graph is the four-node cycle

$$
\mathcal E_{\mathrm{P2P}}=\{(0,1),(1,2),(2,3),(0,3)\}.
$$

An edge means that the two prosumers may trade; it does not prescribe the
direction. For the stored orientation $(i,j)$, $q_{ij,t}>0$ means $i$ sells to
$j$, while $q_{ij,t}<0$ means $j$ sells to $i$. The small absolute-volume
`trade_fee` discourages unnecessary circulating trades when several allocations
have the same physical system cost. This trading graph is independent of the
33-bus electrical feeder topology.
The centralized solver adds these as equality constraints. The distributed
solver enforces them through edge-consensus ADMM.

### Optional pandapower 33-bus constraints — `network.py`

The network is loaded with `pandapower.networks.case33bw()`. The code does not
contain a handwritten list of buses, lines, impedances or loads. It reads:

- bus nominal voltages and voltage limits from `net.bus`;
- the slack bus and voltage from `net.ext_grid`;
- active/reactive demands from `net.load`;
- line endpoints, lengths, resistance and reactance from `net.line`.

Only in-service lines are used. In `case33bw()` this gives 33 buses and 32
radial branches; the five normally open tie lines are excluded. The original
case loads remain in place, and the four prosumers are additional resources
at zero-based bus indices `[5, 11, 17, 32]`.

The physical injection of prosumer $i$ is

$$
p_{i,t}=G_{i,t}-D_{i,t}-c_{i,t}+d_{i,t}
=s_{i,t}-b_{i,t}+\sum_jq_{ij,t}.
$$

Physical injections are aggregated by bus. Peer contracts are not line flows.
Let $P_{uv,t}$ and $Q_{uv,t}$ denote active and reactive branch flow in MW and
Mvar, $v_{b,t}$ squared voltage in p.u., $P_b^L,Q_b^L$ the imported
`case33bw()` load, and $\beta(i)$ prosumer $i$'s bus. Prosumers operate at unity
power factor in this example. The implemented lossless LinDistFlow equations are

$$
\begin{aligned}
P_{uv,t}&=\sum_{w:(v,w)\in\mathcal E}P_{vw,t}
+P_v^L-10^{-3}\sum_{i:\beta(i)=v}p_{i,t},\\
Q_{uv,t}&=\sum_{w:(v,w)\in\mathcal E}Q_{vw,t}+Q_v^L,\\
v_{v,t}&=v_{u,t}-\frac{2(r_{uv}P_{uv,t}+x_{uv}Q_{uv,t})}{V_{\mathrm{base},u}^2}.
\end{aligned}
$$

The factor $10^{-3}$ converts prosumer injections from kW to MW. Line
resistance and reactance are obtained as the imported per-kilometre values
multiplied by imported line length.

The limits are

$$
P_{uv,t}^2+Q_{uv,t}^2\le\overline S^2,
\qquad \underline V_v^2\le v_{v,t}\le\overline V_v^2,
\qquad v_{0,t}=1.
$$

Voltage bounds come from `net.bus`. `case33bw()` uses placeholder line-current
ratings (`max_i_ka=99999`), so they are not meaningful thermal limits. The
example therefore exposes `line_limit_mva=5.0` in `main.py` as a study setting
for $\overline S$, rather than presenting the placeholder as physical data.

This is the standard balanced Baran--Wu 33-bus case represented by a lossless
LinDistFlow approximation inside Gurobi. It is not a nonlinear AC power flow.
Battery decisions change physical injections, so network constraints can affect
the battery schedule.

## Algorithms and results

Both solvers call the same prosumer function. The centralized solver adds
reciprocity and optional network constraints. Distributed ADMM alternates
local optimization, reciprocal-trade projection and, for network cases, a
DSO projection of physical injections onto feeder constraints. Battery
schedules therefore participate in network coordination.

`results/<case>/summary.json` contains purchases, sales, trades, charging,
discharging, SOC, costs and network results. Variables also remain available
in the interactive session. `reference_results/` contains checked outputs.
The final cell checks energy balance, battery dynamics, trade reciprocity and
aggregate cost agreement. Individual schedules can differ at equal cost.

### Distributed local subproblem — `algo.py`'s `distributed()`

Consensus ADMM decomposes the same problem `centralized()` solves (minimize
$\sum_iJ_i$ subject to $q_{ij,t}+q_{ji,t}=0$, plus the feeder equations when
`with_network=True`) into a per-agent local step, a market projection, and
— for network cases — a DSO projection, repeated until both the primal and
dual residuals fall below `tolerance`. Let $z_{ij,t}$ be the shared
consensus value for edge $(i,j)$ at period $t$, $u_{ij,t}$ agent $i$'s own
scaled dual for its trade with $j$, and $\rho$ the penalty parameter (`rho`
in code, default $0.5$).

**Local step** (agent $i$; one small independent Gurobi model per agent —
the `for i,(m,a) in enumerate(...)` loop): minimize the agent's own
operating cost plus a quadratic penalty pulling its trade decisions toward
last iteration's consensus target, subject only to that agent's own
constraints from `add_prosumer()` — no other agent's variables appear:

$$
\min_{q_{ij,t},\,b_{i,t},\,s_{i,t},\,c_{i,t},\,d_{i,t},\,e_{i,t}}\quad
J_i+\frac{\rho}{2}\sum_{j\in\mathcal N_i,\,t}\bigl(q_{ij,t}-z_{ij,t}+u_{ij,t}\bigr)^2
\;\Bigl[+\frac{\rho}{2}\sum_t\bigl(p_{i,t}-\sigma_{i,t}+\omega_{i,t}\bigr)^2\Bigr].
$$

The bracketed term (and $\sigma,\omega$ below) appears only when
`with_network=True`; $\sigma_{i,t}$ is the DSO's consensus copy of the
physical injection $p_{i,t}$ and $\omega_{i,t}$ its scaled dual (code
`s`/`w`), distinct from the grid-sale variable $s_{i,t}$.

**Market projection** (closed form, no optimization — the `# Market
projection` step): average the two agents' local proposals for each edge
into one shared value, so the reciprocity constraint gets a single common
multiplier instead of two agent-specific ones:

$$
z_{ij,t}\leftarrow\tfrac12\bigl(q_{ij,t}+u_{ij,t}-q_{ji,t}-u_{ji,t}\bigr),
\qquad z_{ji,t}\leftarrow-z_{ij,t}.
$$

This symmetric averaging is what makes the fixed point a *variational* GNE
of the underlying bilateral-trading game rather than a general one: without
it, $i$ and $j$ could settle on different shadow prices for the same
reciprocity constraint.

**DSO projection** (network case only, one small QP — the `dso_projection`
model): project the agents' reported injections onto the feeder-feasible
set defined by `add_network()`:

$$
\sigma_t\leftarrow\arg\min_{\sigma_t\ \text{feeder-feasible}}
\sum_i\bigl(\sigma_{i,t}-p_{i,t}-\omega_{i,t}\bigr)^2.
$$

**Dual update:**

$$
u_{ij,t}\leftarrow u_{ij,t}+q_{ij,t}-z_{ij,t},
\qquad
\omega_{i,t}\leftarrow\omega_{i,t}+p_{i,t}-\sigma_{i,t}.
$$

Iteration stops once both the primal residual ($\|q-z\|$, and
$\|p-\sigma\|$ for network cases) and the dual residual
($\rho\|z^{(k+1)}-z^{(k)}\|$, similarly for $\sigma$) drop below
`tolerance`.

**Why individual schedules can differ from the centralized solution at equal
cost.** The linear `trade_fee` term in $J_i$ is convex but not *strictly*
convex in the trade variables: rerouting a constant amount of trade around a
cycle in $\mathcal E_{\mathrm{P2P}}$ can leave every agent's net position —
and therefore every term of every agent's cost — unchanged. The centralized
solver and ADMM can each land on a different point of that flat optimal
face; both are correct, and `main.py`'s final cell only asserts agreement on
the *aggregate* cost, not the individual allocation. A small strictly convex
penalty added to $J_i$ (e.g. $\epsilon\sum_{j,t}q_{ij,t}^2$) would select a
unique point on that face and make the two solvers agree at the individual
level too, at the cost of a small, $\epsilon$-proportional perturbation to
the true optimum — but ADMM's convergence rate turned out to be noticeably
more sensitive to `rho` once that term is added, so it is not implemented
here.
