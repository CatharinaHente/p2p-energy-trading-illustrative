"""Radial LinDistFlow network model (IEEE33 case33bw or SimBench LV feeders).

load_network()    — reads topology, impedances and loads of config.NETWORK_TOPOLOGY
add_network()     — adds LinDistFlow constraints to a Gurobi model
evaluate_feeder() — post-hoc forward sweep; reports violations, never enforces

Sign convention (P and Q): positive injection = prosumer exports → reduces upstream flow.
Units: prosumer inputs in kW / kVAR; network variables in MW / MVAR (÷1000 internally).

Every loader returns the same dict: buses numbered 0..bus_count-1, `branches`
oriented parent → child from `root`, per-branch r/x [Ω] and length [km], static
bus loads, voltage limits, and `line_limit_mva` (the feeder's cable rating).
"""
import copy
import functools
import math
import gurobipy as gp
import numpy as np
from config import NETWORK_TOPOLOGY, SIMBENCH_CODES
from pandapower.networks import case33bw


def load_network(topology=None):
    """Load the configured feeder (default: config.NETWORK_TOPOLOGY)."""
    topology = NETWORK_TOPOLOGY if topology is None else topology
    if topology == 'ieee33':
        return _load_ieee33()
    if topology in SIMBENCH_CODES:
        return copy.deepcopy(_load_simbench_lv(SIMBENCH_CODES[topology]))
    raise ValueError(f"Unknown network topology: {topology!r}. "
                     f"Options: 'ieee33', {sorted(SIMBENCH_CODES)}")


def _load_ieee33():
    """Load case33bw topology, impedances, loads and voltage limits."""
    net    = case33bw()
    lines  = net.line[net.line.in_service.astype(bool)]
    length_km = lines.length_km.to_numpy(float)
    branches = [tuple(map(int, r)) for r in lines[["from_bus","to_bus"]].to_numpy()]
    if len(branches) != len(net.bus) - 1:
        raise ValueError("case33bw must be radial")

    p_load = np.zeros(len(net.bus))
    q_load = np.zeros(len(net.bus))
    for ld in net.load.itertuples():
        if ld.in_service:
            p_load[int(ld.bus)] += float(ld.p_mw   * ld.scaling)
            q_load[int(ld.bus)] += float(ld.q_mvar * ld.scaling)

    return dict(
        name         = "pandapower case33bw",
        root         = int(net.ext_grid.iloc[0].bus),
        root_voltage = float(net.ext_grid.iloc[0].vm_pu),
        bus_count    = len(net.bus),
        branches     = branches,
        r_ohm        = (lines.r_ohm_per_km * lines.length_km).to_numpy(float),
        x_ohm        = (lines.x_ohm_per_km * lines.length_km).to_numpy(float),
        p_load_mw    = p_load,
        q_load_mvar  = q_load,
        voltage_kv   = net.bus.vn_kv.to_numpy(float),
        voltage_min  = net.bus.min_vm_pu.to_numpy(float),
        voltage_max  = net.bus.max_vm_pu.to_numpy(float),
        length_km    = length_km,
        line_limit_mva = 5.0)


@functools.lru_cache(maxsize=None)
def _load_simbench_lv(code):
    """SimBench LV feeder below its MV/LV transformer, renumbered from the LV busbar.

    The transformer is not modelled: the LV busbar is the root, held at 1.0 pu
    (as the IEEE33 root). Only the LV side is kept; buses are renumbered 0..B-1
    in breadth-first order from the root and every line is oriented parent → child.
    `pp_bus` maps internal index → original pandapower bus id, `load_buses` lists
    the internal indices of buses with SimBench household/commercial loads.
    """
    import simbench as sb
    net = sb.get_simbench_net(code)
    if len(net.trafo) != 1 or len(net.switch):
        raise ValueError(f"{code}: expected one MV/LV transformer and no switches")
    root_pp = int(net.trafo.lv_bus.iloc[0])
    lines = net.line[net.line.in_service.astype(bool)]
    lv_buses = set(map(int, net.bus.index[net.bus.vn_kv < 1.0]))
    adj = {b: [] for b in lv_buses}
    for k, (a, b) in enumerate(lines[['from_bus', 'to_bus']].to_numpy(int)):
        adj[int(a)].append((int(b), k)); adj[int(b)].append((int(a), k))
    order, parent_line, seen = [root_pp], {}, {root_pp}
    for u in order:                                       # BFS from the LV busbar
        for v, k in adj[u]:
            if v not in seen:
                seen.add(v); parent_line[v] = (u, k); order.append(v)
    if len(order) != len(lv_buses) or len(lines) != len(lv_buses) - 1:
        raise ValueError(f"{code}: LV network is not a connected radial tree")
    index = {b: i for i, b in enumerate(order)}
    branches, r, x, length = [], [], [], []
    for v in order[1:]:
        u, k = parent_line[v]
        ln = lines.iloc[k]
        branches.append((index[u], index[v]))
        r.append(float(ln.r_ohm_per_km * ln.length_km))
        x.append(float(ln.x_ohm_per_km * ln.length_km))
        length.append(float(ln.length_km))
    B = len(order)
    p_load, q_load = np.zeros(B), np.zeros(B)
    for ld in net.load.itertuples():
        if ld.in_service and int(ld.bus) in index:
            p_load[index[int(ld.bus)]] += float(ld.p_mw * ld.scaling)
            q_load[index[int(ld.bus)]] += float(ld.q_mvar * ld.scaling)
    bus = net.bus.loc[order]
    vn = bus.vn_kv.to_numpy(float)
    rating = float((np.sqrt(3) * lines.max_i_ka * net.bus.loc[lines.from_bus, 'vn_kv'].to_numpy()).min())
    return dict(
        name         = f"SimBench {code} (LV side)",
        root         = 0,
        root_voltage = 1.0,
        bus_count    = B,
        branches     = branches,
        r_ohm        = np.array(r),
        x_ohm        = np.array(x),
        p_load_mw    = p_load,
        q_load_mvar  = q_load,
        voltage_kv   = vn,
        voltage_min  = bus.min_vm_pu.fillna(0.9).to_numpy(float),
        voltage_max  = bus.max_vm_pu.fillna(1.1).to_numpy(float),
        length_km    = np.array(length),
        line_limit_mva = rating,
        pp_bus       = list(order),
        load_buses   = sorted(index[int(b)] for b in net.load.bus.unique() if int(b) in index))


def add_network(model, injections_kw, agent_bus, T,
                line_limit_mva=5.0, q_injections_kvar=None):
    """Add lossless LinDistFlow constraints to a Gurobi model.

    injections_kw    : {(i,t): LinExpr}  active power [kW], positive = export
    q_injections_kvar: {(i,t): LinExpr}  reactive power [kVAR], positive = capacitive
                       None → prosumer Q ignored (backward-compatible)
    Returns p_flow, q_flow, voltage, grid.
    """
    grid     = load_network()
    branches = grid["branches"]

    p_flow  = model.addVars(branches, range(T), lb=-line_limit_mva, ub=line_limit_mva, name="p_mw")
    q_flow  = model.addVars(branches, range(T), lb=-line_limit_mva, ub=line_limit_mva, name="q_mvar")
    voltage = model.addVars(range(grid["bus_count"]), range(T), name="v_sq_pu")

    for t in range(T):
        model.addConstr(voltage[grid["root"], t] == grid["root_voltage"]**2)
        for bus in range(grid["bus_count"]):
            model.addConstr(voltage[bus, t] >= grid["voltage_min"][bus]**2)
            model.addConstr(voltage[bus, t] <= grid["voltage_max"][bus]**2)

        for k, (u, v) in enumerate(branches):
            cp = gp.quicksum(p_flow[a, b, t] for a, b in branches if a == v)
            cq = gp.quicksum(q_flow[a, b, t] for a, b in branches if a == v)

            # prosumer injections at bus v [kW/kVAR → MW/MVAR]
            pp = gp.quicksum(injections_kw[i,t] / 1000.0
                             for i,bus in enumerate(agent_bus) if bus == v)
            qp = (gp.quicksum(q_injections_kvar[i,t] / 1000.0
                              for i,bus in enumerate(agent_bus) if bus == v)
                  if q_injections_kvar is not None else 0.0)

            model.addConstr(p_flow[u,v,t] == cp + grid["p_load_mw"][v]   - pp)  # P balance
            model.addConstr(q_flow[u,v,t] == cq + grid["q_load_mvar"][v] - qp)  # Q balance

            base_kv = grid["voltage_kv"][u]
            model.addConstr(                                                       # voltage drop
                voltage[v,t] == voltage[u,t]
                - 2.0*(grid["r_ohm"][k]*p_flow[u,v,t]
                       + grid["x_ohm"][k]*q_flow[u,v,t]) / base_kv**2)

            # Thermal limit — polygon approx of P²+Q²≤S² (same as BESS/EV FOR)
            # Keeps problem LP or convex QP; avoids QCQP numerical instability.
            # n=8 gives 92.4% circle area — same accuracy as device FORs.
            _n = 8
            _s = line_limit_mva * math.cos(math.pi / _n)
            for _k in range(_n):
                _a = 2.0 * math.pi * _k / _n
                model.addConstr(
                    math.cos(_a)*p_flow[u,v,t] + math.sin(_a)*q_flow[u,v,t] <= _s)

    return p_flow, q_flow, voltage, grid


def evaluate_feeder(injections_kw, agent_bus, T,
                    line_limit_mva=5.0, q_injections_kvar=None):
    """Forward/backward LinDistFlow sweep on fixed injections.

    injections_kw    : (N,T) array [kW]
    q_injections_kvar: (N,T) array [kVAR] or None
    Returns dict with voltage_pu, p_flow_mw, q_flow_mvar, apparent_mva,
    voltage_violation, congestion, any_voltage_violation, any_congestion.
    """
    grid     = load_network()
    branches = grid["branches"]
    children = {}
    for u,v in branches:
        children.setdefault(u, []).append(v)

    # aggregate prosumer injections per bus [kW/kVAR → MW/MVAR]
    p_inj = {bus: np.zeros(T) for bus in set(agent_bus)}
    q_inj = {bus: np.zeros(T) for bus in set(agent_bus)}
    for i,bus in enumerate(agent_bus):
        p_inj[bus] = p_inj[bus] + np.asarray(injections_kw[i]) / 1000.0
    if q_injections_kvar is not None:
        q_arr = np.asarray(q_injections_kvar)
        for i,bus in enumerate(agent_bus):
            q_inj[bus] = q_inj[bus] + q_arr[i] / 1000.0

    # topological order (root → leaves)
    order, stack = [], [grid["root"]]
    while stack:
        node = stack.pop()
        order.append(node)
        stack.extend(children.get(node, []))

    # subtree sums (backward sweep)
    sp = {n: np.full(T, grid["p_load_mw"][n])   - p_inj.get(n, np.zeros(T)) for n in order}
    sq = {n: np.full(T, grid["q_load_mvar"][n]) - q_inj.get(n, np.zeros(T)) for n in order}
    for node in reversed(order):
        for child in children.get(node, []):
            sp[node] += sp[child]
            sq[node] += sq[child]

    p_flow = {(u,v): sp[v] for u,v in branches}
    q_flow = {(u,v): sq[v] for u,v in branches}
    smva   = {e: np.hypot(p_flow[e], q_flow[e]) for e in p_flow}

    # forward voltage sweep
    vsq = {grid["root"]: np.full(T, grid["root_voltage"]**2)}
    for k,(u,v) in enumerate(branches):
        vsq[v] = vsq[u] - 2.0*(grid["r_ohm"][k]*p_flow[u,v]
                                + grid["x_ohm"][k]*q_flow[u,v]) / grid["voltage_kv"][u]**2
    vpu = {bus: np.sqrt(np.maximum(v, 0.)) for bus,v in vsq.items()}

    # violations — any timestep
    vviol = {bus: bool(np.any(vpu[bus] < grid["voltage_min"][bus]-1e-9)
                       or np.any(vpu[bus] > grid["voltage_max"][bus]+1e-9))
             for bus in vpu}
    cong  = {e: bool(np.any(smva[e] > line_limit_mva+1e-9)) for e in smva}

    return dict(
        source                = grid["name"],
        p_flow_mw             = {str((u,v,t)): float(p_flow[u,v][t]) for u,v in branches for t in range(T)},
        q_flow_mvar           = {str((u,v,t)): float(q_flow[u,v][t]) for u,v in branches for t in range(T)},
        voltage_pu            = {str((bus,t)): float(vpu[bus][t])     for bus in vpu      for t in range(T)},
        apparent_mva          = {str((u,v,t)): float(smva[u,v][t])   for u,v in branches for t in range(T)},
        line_limit_mva        = line_limit_mva,
        voltage_violation     = {str(bus): flag for bus,flag in vviol.items()},
        congestion            = {str(e):   flag for e,  flag in cong.items()},
        any_voltage_violation = any(vviol.values()),
        any_congestion        = any(cong.values()))