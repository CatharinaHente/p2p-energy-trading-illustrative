"""Build a time-varying DyNECT game from the existing P2P study data.

This module does not edit the legacy Gurobi formulation. It retains its
active-power decision variables, device states, settlement, and network
constraints; all Q/kVAR/MVAR variables are deliberately omitted.

Use build_for_day('2016-01-07') from the original repository root, after
config.py selects the desired population and IEEE33 network.

IMPORTANT: the game objective is the legacy full-horizon cost (with tiny,
explicit optional regularisation); fairness uses the first-day valued costs.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass
class GameBundle:
    game: Any
    x0: np.ndarray
    data: dict
    grid: dict | None
    state_names: list[str]
    state_index: dict[tuple[int, str], int]
    controls: list[dict[str, int]]
    peers: list[list[int]]
    shared_labels: list[str]
    choices: dict

    @property
    def N(self):
        return self.game.N

    @property
    def T(self):
        return self.game.T_hor


def load_ieee33():
    """Extract the original radial IEEE33 feeder (not an AC power flow)."""
    from pandapower.networks import case33bw
    net = case33bw()
    lines = net.line.loc[net.line.in_service.astype(bool)]
    branches = [tuple(map(int, r)) for r in lines[['from_bus', 'to_bus']].to_numpy()]
    nb = len(net.bus)
    root = int(net.ext_grid.iloc[0].bus)
    assert len(branches) == nb - 1, 'Expected n_bus - 1 energized lines'
    assert len(set(net.bus.index)) == nb and set(map(int, net.bus.index)) == set(range(nb)), \
        'IEEE33 buses must be numbered 0..nb-1'
    parent = {v: u for u, v in branches}
    assert root not in parent and len(parent) == nb - 1, 'Unexpected feeder directions'
    for v in range(nb):
        seen = set()
        while v != root:
            if v in seen or v not in parent:
                raise ValueError('Network is not a directed radial feeder from the root')
            seen.add(v)
            v = parent[v]
    p_load_mw = np.zeros(nb)
    for ld in net.load.itertuples():
        if ld.in_service:
            p_load_mw[int(ld.bus)] += float(ld.p_mw * ld.scaling)
    return dict(
        name='IEEE33 (active-only LinDistFlow)', root=root, bus_count=nb,
        branches=branches,
        r_ohm=(lines.r_ohm_per_km * lines.length_km).to_numpy(float),
        p_load_mw=p_load_mw,
        voltage_kv=net.bus.vn_kv.to_numpy(float),
        voltage_min=net.bus.min_vm_pu.to_numpy(float),
        voltage_max=net.bus.max_vm_pu.to_numpy(float),
        root_voltage=float(net.ext_grid.iloc[0].vm_pu),
    )


def load_grid(topology: str | None = None):
    """Active-only feeder of config.NETWORK_TOPOLOGY (IEEE33 or SimBench LV).

    Uses the study's own network.load_network(), so bus numbering matches the
    prosumer populations. For 'ieee33' it is identical to load_ieee33().
    """
    import network
    g = network.load_network(topology)
    return dict(name=f"{g['name']} (active-only LinDistFlow)", root=int(g['root']),
                bus_count=int(g['bus_count']), branches=[tuple(map(int, e)) for e in g['branches']],
                r_ohm=np.asarray(g['r_ohm'], float), p_load_mw=np.asarray(g['p_load_mw'], float),
                voltage_kv=np.asarray(g['voltage_kv'], float),
                voltage_min=np.asarray(g['voltage_min'], float),
                voltage_max=np.asarray(g['voltage_max'], float),
                root_voltage=float(g['root_voltage']))


def _feeder_maps(grid: dict, agent_bus: list[int]):
    """Sparse linear maps: flow = f0 + L @ injection[kW]."""
    branches = grid['branches']
    root = grid['root']
    children = {v: [] for v in range(grid['bus_count'])}
    for u, v in branches:
        children[u].append(v)
    order, queue = [], [root]
    while queue:
        v = queue.pop(0)
        order.append(v)
        queue.extend(children[v])
    if len(order) != grid['bus_count']:
        raise ValueError('Feeder does not connect all buses to its root')
    subtree = {v: {v} for v in reversed(order)}
    for v in reversed(order):
        for child in children[v]:
            subtree[v] |= subtree[child]
    E, N = len(branches), len(agent_bus)
    flow0 = np.array([sum(grid['p_load_mw'][k] for k in subtree[v]) for u, v in branches])
    # Positive injection reduces upstream flow, original units are kW.
    L = np.array([[-.001 if agent_bus[i] in subtree[v] else 0. for i in range(N)]
                  for u, v in branches])
    # v_squared = vroot_squared - Drop @ flow[M W].
    drop = np.zeros((grid['bus_count'], E))
    path = {root: []}
    for v in order:
        for child in children[v]:
            k = branches.index((v, child))
            path[child] = path[v] + [k]
            for k2 in path[child]:
                u2, _ = branches[k2]
                drop[child, k2] = 2 * float(grid['r_ohm'][k2]) / float(grid['voltage_kv'][u2])**2
    return flow0, L, drop


def _layout_for_agent(i, peers):
    names = ['buy', 'sell', 'charge', 'discharge', 'ev_charge', 'ev_discharge', 'hp_power']
    for j in peers:
        names.extend([f'trade_{j}', f'trade_abs_{j}'])
    return {name: k for k, name in enumerate(names)}


def assemble_game(data: dict, *, grid: dict | None = None,
                  clearing_epsilon_kw: float = 0.0,
                  state_regularization: float = 1e-9,
                  input_regularization: float = 1e-9) -> dict:
    """Assemble keyword arguments matching DynLQGameTV without constructing it.

    Passing grid=None leaves out feeder limits. The normal study case should
    pass load_ieee33(), so all active-only IEEE33 feeder constraints are included.
    An epsilon > 0 changes the game and is for partial GNE-set exploration.
    """
    if clearing_epsilon_kw < 0 or state_regularization < 0 or input_regularization < 0:
        raise ValueError('Tolerances/regularisers must be nonnegative')
    N = len(data['demand'])
    T = len(data['buy_price'])
    dt = float(data['dt'])
    if N < 2 or T < 1:
        raise ValueError('Need at least two agents and one timestep')
    for key in ['demand', 'generation', 'ev_driving_load', 'ev_availability',
                'hp_base_power', 'hp_q_heat']:
        a = np.asarray(data[key], dtype=float)
        if a.shape != (N, T):
            raise ValueError(f'{key}: expected shape {(N,T)}, got {a.shape}')
    for key in ['sell_price', 'p2p_price']:
        if np.asarray(data[key]).shape != (T,):
            raise ValueError(f'{key}: expected shape {(T,)}')
    edges = [tuple(map(int, e)) for e in data['edges']]
    if any(i == j or i not in range(N) or j not in range(N) for i,j in edges):
        raise ValueError('Invalid edge endpoints')
    if len({tuple(sorted(e)) for e in edges}) != len(edges):
        raise ValueError('Duplicate or reversed-duplicate trading edges')
    edges = [tuple(sorted(e)) for e in edges]
    peers = [sorted(j if i == a else a for a,b in edges for j in [b] if i in (a,b))
             for i in range(N)]
    controls = [_layout_for_agent(i, peers[i]) for i in range(N)]
    nu = [len(c) for c in controls]
    T_eval = int(data.get('T_eval', T))
    if not 1 <= T_eval <= T:
        raise ValueError('Invalid T_eval')
    state_names, state_index, x0 = [], {}, []
    for i in range(N):
        if float(data['battery_capacity'][i]) > 0:
            state_index[i, 'bess'] = len(x0)
            state_names.append(f'P{i+1} BESS SOC [kWh]')
            x0.append(float(data['initial_soc'][i]))
        if float(data['ev_capacity'][i]) > 0:
            state_index[i, 'ev'] = len(x0)
            state_names.append(f'P{i+1} EV SOC [kWh]')
            x0.append(float(data['ev_initial_soc'][i]))
        if float(data['hp_max_power'][i]) > 0:
            state_index[i, 'hp'] = len(x0)
            state_names.append(f'P{i+1} indoor temperature [C]')
            x0.append(float(data['hp_T_init'][i]))
    nx = len(x0)
    if nx == 0:
        raise ValueError('The chosen population has no dynamic device states')
    x0 = np.asarray(x0, dtype=float)
    eye = np.eye(nx)
    # In this game the state variables belong to individual prosumers; their
    # bounds become local after condensing, unlike feeder/trade constraints.
    A = [eye.copy() for _ in range(T)]
    B, c = [], []
    Q, q, R, r, C_x, b_x, C_loc, b_loc, C_u, b_u = ([] for _ in range(10))
    P = [state_regularization * eye.copy() for _ in range(N)]
    p = [np.zeros(nx) for _ in range(N)]
    for i in range(N):
        if (i, 'ev') in state_index and not data.get('ev_cyclic_soc', False):
            p[i][state_index[i, 'ev']] = -float(data['ev_terminal_price'])
    grid_limit = float(data['grid_limit'])
    trade_limit = float(data['trade_limit'])
    for t in range(T):
        B_t = [np.zeros((nx, nu[i])) for i in range(N)]
        c_t = np.zeros(nx)
        R_t = [[np.zeros((nu[i], nu[j])) for j in range(N)] for i in range(N)]
        r_t = [np.zeros(nu[i]) for i in range(N)]
        C_loc_t, b_loc_t = [], []
        for i in range(N):
            ix = controls[i]
            if (i, 'bess') in state_index:
                row = state_index[i, 'bess']
                B_t[i][row, ix['charge']] = dt * float(data['eta_charge'])
                B_t[i][row, ix['discharge']] = -dt / float(data['eta_discharge'])
            if (i, 'ev') in state_index:
                row = state_index[i, 'ev']
                B_t[i][row, ix['ev_charge']] = dt * float(data['ev_eta_charge'])
                B_t[i][row, ix['ev_discharge']] = -dt / float(data['ev_eta_discharge'])
                c_t[row] = -float(data['ev_driving_load'][i][t])
            if (i, 'hp') in state_index:
                row = state_index[i, 'hp']
                base = float(data['hp_base_power'][i][t])
                heat = float(data['hp_q_heat'][i][t])
                if base > 0:
                    B_t[i][row, ix['hp_power']] = dt * heat / base
                    c_t[row] = -dt * heat
            # Exactly the legacy linear operating costs (EUR per kWh).
            r_t[i][ix['buy']] = dt * float(data['buy_price'][t])
            r_t[i][ix['sell']] = -dt * float(data['sell_price'][t])
            r_t[i][ix['charge']] = r_t[i][ix['discharge']] = dt * float(data['battery_cost'])
            r_t[i][ix['ev_charge']] = r_t[i][ix['ev_discharge']] = dt * float(data['ev_cost'])
            r_t[i][ix['hp_power']] = dt * float(data.get('hp_cost', 0.))
            # All input blocks are positive definite when input_regularization > 0.
            R_t[i][i] = 2 * dt * input_regularization * np.eye(nu[i])
            pcost = float(data.get('p_cost', 0.))
            for name in ['charge','discharge','ev_charge','ev_discharge']:
                R_t[i][i][ix[name], ix[name]] += 2 * dt * pcost
            for j in peers[i]:
                a, aa = ix[f'trade_{j}'], ix[f'trade_abs_{j}']
                r_t[i][a] = -dt * float(data['p2p_price'][t])
                r_t[i][aa] = dt * float(data['trade_fee'])
                R_t[i][i][a, a] += 2 * dt * pcost
            # Local constraints: own input bounds + own power balance + |trade| epigraph.
            bounds = {
                'buy':(0., grid_limit), 'sell':(0., grid_limit),
                'charge':(0., float(data['battery_power'][i])),
                'discharge':(0., float(data['battery_power'][i])),
                'ev_charge':(0., float(data['ev_power'][i]) * float(data['ev_availability'][i][t])),
                'ev_discharge':(0., float(data['ev_power'][i]) * float(data['ev_availability'][i][t])
                                * float(bool(data.get('ev_v2g', True)))),
                'hp_power':(0., float(data['hp_max_power'][i])),
            }
            for j in peers[i]:
                bounds[f'trade_{j}'] = (-trade_limit, trade_limit)
                bounds[f'trade_abs_{j}'] = (0., trade_limit)
            lr, lb = [], []
            def row_constraint(coefs, rhs):
                rr = np.zeros(nu[i])
                for name, v in coefs.items():
                    rr[ix[name]] += v
                lr.append(rr)
                lb.append(float(rhs))
            for name, (lo, hi) in bounds.items():
                row_constraint({name:1.}, hi)
                row_constraint({name:-1.}, -lo)
            balance = {'buy':1.,'sell':-1., 'charge':-1.,'discharge':1.,
                       'ev_charge':-1.,'ev_discharge':1.,'hp_power':-1.}
            for j in peers[i]:
                balance[f'trade_{j}'] = -1.
                row_constraint({f'trade_{j}':1., f'trade_abs_{j}':-1.}, 0.)
                row_constraint({f'trade_{j}':-1., f'trade_abs_{j}':-1.}, 0.)
            # Preserve the Q=0 cross-section of the original 8-sided
            # BESS/EV active-reactive FOR polygons. At Q=0 the polygon
            # limits |P_net| to S_max*cos(pi/n), NOT to S_max.
            b_n = int(data.get('battery_polygon_sides',8))
            b_s = float(data.get('battery_smax',data['battery_power'])[i])
            b_active = b_s * math.cos(math.pi/b_n)
            row_constraint({'discharge':1.,'charge':-1.}, b_active)
            row_constraint({'discharge':-1.,'charge':1.}, b_active)
            e_n = int(data.get('ev_polygon_sides',8))
            e_s = float(data.get('ev_smax',data['ev_power'])[i])
            e_active = e_s * math.cos(math.pi/e_n)
            row_constraint({'ev_discharge':1.,'ev_charge':-1.}, e_active)
            row_constraint({'ev_discharge':-1.,'ev_charge':1.}, e_active)
            bal_rhs = float(data['demand'][i][t] - data['generation'][i][t])
            row_constraint(balance, bal_rhs)
            row_constraint({key:-v for key,v in balance.items()}, -bal_rhs)
            C_loc_t.append(np.vstack(lr))
            b_loc_t.append(np.asarray(lb))
        B.append(B_t)
        c.append(c_t)
        R.append(R_t)
        r.append(r_t)
        C_loc.append(C_loc_t)
        b_loc.append(b_loc_t)
        if t < T-1:
            Q.append([state_regularization * eye.copy() for _ in range(N)])
            q.append([np.zeros(nx) for _ in range(N)])
        # State bounds act on x[t+1]. Departure constraints update EV floors.
        xr, xb = [], []
        for (i, kind), idx in state_index.items():
            if kind == 'bess':
                cap = float(data['battery_capacity'][i])
                lo = float(data['soc_min_fraction']) * cap
                hi = float(data['soc_max_fraction']) * cap
            elif kind == 'ev':
                cap = float(data['ev_capacity'][i]) * float(data['ev_cap_factor'][i])
                lo = (.30 if data.get('season') == 'winter' else .20) * cap
                hi = cap
                if t + 1 < T and data['ev_availability'][i][t] == 1 and data['ev_availability'][i][t+1] == 0:
                    lo = max(lo, float(data['ev_soc_min_departure'][i]) * cap)
            else:
                lo = float(data['hp_T_min'])
                hi = float(data['hp_T_max'])
            xr.extend((eye[idx], -eye[idx]))
            xb.extend((hi, -lo))
        # Exact terminal SOC conditions, as in the legacy 48h optimization.
        # These rows occur only at the final step (Julia TV conversion accepts
        # time-varying state-constraint counts via block-diagonal C_x[t]).
        if t == T - 1:
            for (i, kind), idx in state_index.items():
                if (kind == 'bess' and data.get('cyclic_soc', True)) or (kind == 'ev' and data.get('ev_cyclic_soc', False)):
                    xr.extend((eye[idx], -eye[idx]))
                    xb.extend((x0[idx], -x0[idx]))
        C_x.append(np.vstack(xr))
        b_x.append(np.asarray(xb))
        # Shared constraints: each directed trading pair, followed by feeder rows.
        # Fixed row count m_u for all t (required by the Python constructor).
        shr, shb = [], []
        labels = []
        for a, b in edges:
            rr = [np.zeros(nu[i]) for i in range(N)]
            rr[a][controls[a][f'trade_{b}']] = 1.
            rr[b][controls[b][f'trade_{a}']] = 1.
            shr.append(rr)
            shb.append(float(clearing_epsilon_kw))
            labels.append(f'edge_{a}_{b}_pos')
            shr.append([-x for x in rr])
            shb.append(float(clearing_epsilon_kw))
            labels.append(f'edge_{a}_{b}_neg')
        if grid is not None:
            if any(int(b) not in range(grid['bus_count']) for b in data['agent_bus']):
                raise ValueError('Agent bus index is not a bus in the feeder')
            f0, L, drop = _feeder_maps(grid, list(map(int, data['agent_bus'])))
            injection_constant = np.asarray(data['generation'],float)[:,t] - np.asarray(data['demand'],float)[:,t]
            const_flow = f0 + L @ injection_constant
            # Injection_i = generation-demand -charge+discharge-ev_charge+ev_discharge-hp.
            inj_coef = []
            for i in range(N):
                ci = np.zeros(nu[i])
                for name, sign in [('charge',-1),('discharge',1),('ev_charge',-1),
                                   ('ev_discharge',1),('hp_power',-1)]:
                    ci[controls[i][name]] = sign
                inj_coef.append(ci)
            flow_coeff = [[L[k,i]*inj_coef[i] for i in range(N)] for k in range(len(f0))]
            line_limit = float(data['line_limit_mva']) * math.cos(math.pi/8) # original 8-sided polygon at Q=0
            for k, edge in enumerate(grid['branches']):
                shr.append(flow_coeff[k]); shb.append(line_limit - const_flow[k]); labels.append(f'flow_{edge}_upper')
                shr.append([-v for v in flow_coeff[k]]); shb.append(line_limit + const_flow[k]); labels.append(f'flow_{edge}_lower')
            for bus in range(grid['bus_count']):
                if bus == grid['root']:
                    continue
                voltage_const = float(grid['root_voltage'])**2 - drop[bus] @ const_flow
                voltage_coef = [-sum(drop[bus,k] * flow_coeff[k][i] for k in range(len(f0)))
                                for i in range(N)]
                v_hi = float(grid['voltage_max'][bus]) ** 2
                v_lo = float(grid['voltage_min'][bus]) ** 2
                shr.append(voltage_coef); shb.append(v_hi - voltage_const); labels.append(f'voltage_{bus}_upper')
                shr.append([-v for v in voltage_coef]); shb.append(voltage_const - v_lo); labels.append(f'voltage_{bus}_lower')
        C_u.append([np.vstack([rr[i] for rr in shr]) for i in range(N)])
        b_u.append(np.asarray(shb))
        if t == 0:
            shared_labels = labels
        elif labels != shared_labels:
            raise AssertionError('Shared constraint ordering changed across timesteps')
    kwargs = dict(A=A,B=B,Q=Q,R=R,P=P,c=c,q=q,r=r,p=p,
                  C_x=C_x,b_x=b_x,C_loc=C_loc,b_loc=b_loc,C_u=C_u,b_u=b_u)
    return dict(kwargs=kwargs, x0=x0, state_index=state_index, state_names=state_names,
                controls=controls, peers=peers, shared_labels=shared_labels,
                choices=dict(clearing_epsilon_kw=clearing_epsilon_kw,
                             state_regularization=state_regularization,
                             input_regularization=input_regularization), grid=grid)


def build_game(data: dict, *, grid: dict | None = None,
               clearing_epsilon_kw: float = 0.,
               state_regularization: float = 1e-9,
               input_regularization: float = 1e-9) -> GameBundle:
    """Construct a *genuine* DynLQGameTV from existing scenario data."""
    from dynect import DynLQGameTV
    a = assemble_game(data, grid=grid, clearing_epsilon_kw=clearing_epsilon_kw,
                      state_regularization=state_regularization,
                      input_regularization=input_regularization)
    game = DynLQGameTV(**a['kwargs'])
    return GameBundle(game, a['x0'], data, grid, a['state_names'], a['state_index'],
                      a['controls'], a['peers'], a['shared_labels'], a['choices'])


def build_for_day(date_str: str, *, seed: int = 42, resolution_h: float = 0.5,
                  lookahead_h: int = 24, clearing_epsilon_kw: float = 0.) -> GameBundle:
    """The real scenario: config.py controls population and trading topology.

    The first use may download SimBench/DWD/price data, as in the original code.
    The feeder follows config.NETWORK_TOPOLOGY (see load_grid).
    """
    from scenarios import make_scenario_horizon
    data = make_scenario_horizon(date_str, seed=seed, resolution_h=resolution_h,
                                 lookahead_h=lookahead_h)
    return build_game(data, grid=load_grid(), clearing_epsilon_kw=clearing_epsilon_kw)


def unpack_controls(bundle: GameBundle, u) -> list[dict[str, np.ndarray]]:
    """Turn agent-major stacked decisions into named T-length arrays."""
    u = np.asarray(u, float).reshape(-1)
    expected = bundle.T * sum(bundle.game.nu)
    if u.size != expected:
        raise ValueError(f'Expected {expected} decisions, got {u.size}')
    result = []
    cursor = 0
    for ctl in bundle.controls:
        mat = u[cursor:cursor+bundle.T*len(ctl)].reshape(bundle.T, len(ctl))
        result.append({name: mat[:, idx].copy() for name, idx in ctl.items()})
        cursor += bundle.T * len(ctl)
    return result


def rollout_states(bundle: GameBundle, u) -> np.ndarray:
    """Simulate x[0:T] from game dynamics for any stacked control vector."""
    us = unpack_controls(bundle, u)
    x = np.zeros((bundle.T+1, bundle.game.nx))
    x[0] = bundle.x0
    for t in range(bundle.T):
        x[t+1] = bundle.game.A[t] @ x[t] + bundle.game.c[t]
        for i in range(bundle.N):
            ui = np.array([us[i][name][t] for name in bundle.controls[i]])
            x[t+1] += bundle.game.B[t][i] @ ui
    return x


def evaluated_costs(bundle: GameBundle, u, *, first_day: bool = True) -> np.ndarray:
    """Legacy operating cost and terminal valuation, per prosumer [EUR].

    This is NOT reconstructed from the VI pseudo-gradient, which lacks the
    cross-player-only terms in individual state costs. It uses physical flows.
    Tiny DyNECT regularisation (if used) is intentionally excluded here.
    """
    dt, data = float(bundle.data['dt']), bundle.data
    T_end = int(data.get('T_eval',bundle.T)) if first_day else bundle.T
    control = unpack_controls(bundle, u)
    states = rollout_states(bundle, u)
    costs = np.zeros(bundle.N)
    for i in range(bundle.N):
        z = control[i]
        for t in range(T_end):
            trade = sum(z[f'trade_{j}'][t] for j in bundle.peers[i])
            trade_abs = sum(z[f'trade_abs_{j}'][t] for j in bundle.peers[i])
            costs[i] += dt * (
                data['buy_price'][t] * z['buy'][t]
                - data['sell_price'][t] * z['sell'][t]
                - data['p2p_price'][t] * trade
                + data['battery_cost'] * (z['charge'][t] + z['discharge'][t])
                + data.get('ev_cost',0.) * (z['ev_charge'][t] + z['ev_discharge'][t])
                + data.get('hp_cost',0.) * z['hp_power'][t]
                + data.get('p_cost',0.) * (z['charge'][t]**2 + z['discharge'][t]**2
                    + z['ev_charge'][t]**2 + z['ev_discharge'][t]**2
                    + sum(z[f'trade_{j}'][t]**2 for j in bundle.peers[i]))
                + data['trade_fee'] * trade_abs)
        if (i,'ev') in bundle.state_index:
            k = bundle.state_index[i,'ev']
            if first_day or not data.get('ev_cyclic_soc',False):
                costs[i] -= data['ev_terminal_price'] * (states[T_end,k] - states[0,k])
        if first_day and (i,'bess') in bundle.state_index:
            k = bundle.state_index[i,'bess']
            battery_value = (data['ev_terminal_price'] * data['ev_eta_charge']
                             / data['eta_charge'])
            costs[i] -= battery_value * (states[T_end,k] - states[0,k])
    return costs


def feeder_results(bundle: GameBundle, u) -> dict:
    """Recompute the active-only feeder for an allocation (no reactive power)."""
    if bundle.grid is None:
        raise ValueError('No feeder configured')
    z = unpack_controls(bundle, u)
    T, N = bundle.T, bundle.N
    inj = np.zeros((N,T))
    for i in range(N):
        inj[i] = (np.asarray(bundle.data['generation'][i]) - np.asarray(bundle.data['demand'][i])
                  - z[i]['charge'] + z[i]['discharge'] - z[i]['ev_charge']
                  + z[i]['ev_discharge'] - z[i]['hp_power'])
    f0, L, drop = _feeder_maps(bundle.grid, bundle.data['agent_bus'])
    pf = f0[:,None] + L @ inj
    vsq = float(bundle.grid['root_voltage'])**2 - drop @ pf
    vp = np.sqrt(np.maximum(vsq,0))
    return dict(injection_kw=inj, line_p_mw=pf, voltage_pu=vp,
                max_abs_line_mw=float(np.abs(pf).max()),
                min_voltage_pu=float(vp.min()), max_voltage_pu=float(vp.max()),
                max_congestion_mw=float(np.maximum(np.abs(pf)-bundle.data['line_limit_mva']*math.cos(math.pi/8),0).max()),
                max_voltage_violation_pu=float(max(np.max(bundle.grid['voltage_min'][:,None]-vp),
                                                  np.max(vp-bundle.grid['voltage_max'][:,None]),0)))


def to_static_gnep(bundle: GameBundle):
    """Condense the *TV* game to a static GNEP at the actual initial state.

    The condensed state and local input constraints must each depend on just
    one player's controls. This holds here (device states are independent).
    Feeder and P2P coupling remains SHARED; do not assign it to local rows.
    """
    from dynect import MpAVI, StaticGNEP
    mp = MpAVI.from_game(bundle.game)
    avi = mp.at(bundle.x0)
    T, N = bundle.T, bundle.N
    n = [T*k for k in bundle.game.nu]
    off = np.cumsum([0,*n]); sl = [slice(off[i],off[i+1]) for i in range(N)]
    m = T * bundle.game.m_u
    ash, bsh = avi.A[:m], avi.b[:m]
    remaining, rem_b = avi.A[m:], avi.b[m:]
    rows = [[] for _ in range(N)]
    for k, r in enumerate(remaining):
        owners = [i for i in range(N) if np.any(np.abs(r[sl[i]])>1e-10)]
        if len(owners) != 1:
            if not owners and rem_b[k] >= -1e-9:
                continue  # trivially satisfied constant constraint
            raise ValueError(f'Condensed non-shared constraint {k} has owners {owners}.')
        rows[owners[0]].append(k)
    gne = StaticGNEP(
        Q=[[avi.H[sl[i],sl[j]].copy() for j in range(N)] for i in range(N)],
        q=[avi.f[sl[i]].copy() for i in range(N)],
        A_loc=[remaining[rows[i],sl[i]] for i in range(N)],
        b_loc=[rem_b[rows[i]] for i in range(N)],
        A_sh=[ash[:,sl[i]] for i in range(N)], b_sh=bsh,
    )
    return mp, avi, gne
