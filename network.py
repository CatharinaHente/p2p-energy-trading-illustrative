"""Import pandapower case33bw and add a multi-period LinDistFlow model."""
import gurobipy as gp
import numpy as np
from pandapower.networks import case33bw


def load_network():
    """Read topology, impedances, loads and voltage bounds from case33bw()."""
    net = case33bw()
    lines = net.line[net.line.in_service.astype(bool)]
    branches = [tuple(map(int, row)) for row in lines[["from_bus", "to_bus"]].to_numpy()]
    if len(branches) != len(net.bus) - 1:
        raise ValueError("The in-service case33bw network must be radial")

    p_load = np.zeros(len(net.bus))
    q_load = np.zeros(len(net.bus))
    for load in net.load.itertuples():
        if load.in_service:
            p_load[int(load.bus)] += float(load.p_mw * load.scaling)
            q_load[int(load.bus)] += float(load.q_mvar * load.scaling)

    return {
        "name": "pandapower case33bw",
        "root": int(net.ext_grid.iloc[0].bus),
        "root_voltage": float(net.ext_grid.iloc[0].vm_pu),
        "bus_count": len(net.bus),
        "branches": branches,
        "r_ohm": (lines.r_ohm_per_km * lines.length_km).to_numpy(float),
        "x_ohm": (lines.x_ohm_per_km * lines.length_km).to_numpy(float),
        "p_load_mw": p_load,
        "q_load_mvar": q_load,
        "voltage_kv": net.bus.vn_kv.to_numpy(float),
        "voltage_min": net.bus.min_vm_pu.to_numpy(float),
        "voltage_max": net.bus.max_vm_pu.to_numpy(float),
    }


def add_network(model, injections_kw, agent_bus, T, line_limit_mva=5.0):
    """Add balanced LinDistFlow constraints using imported case33bw data.

    ``injections_kw[i,t]`` is positive when prosumer i injects active power.
    The original case loads remain in the network; prosumers are additional
    resources at the buses listed in ``agent_bus``.
    """
    grid = load_network()
    branches = grid["branches"]
    p_flow = model.addVars(branches, range(T), lb=-line_limit_mva,
                           ub=line_limit_mva, name="p_flow_mw")
    q_flow = model.addVars(branches, range(T), lb=-line_limit_mva,
                           ub=line_limit_mva, name="q_flow_mvar")
    voltage = model.addVars(range(grid["bus_count"]), range(T),
                            name="voltage_sq_pu")

    for t in range(T):
        root = grid["root"]
        model.addConstr(voltage[root, t] == grid["root_voltage"] ** 2)
        for bus in range(grid["bus_count"]):
            model.addConstr(voltage[bus, t] >= grid["voltage_min"][bus] ** 2)
            model.addConstr(voltage[bus, t] <= grid["voltage_max"][bus] ** 2)

        for k, (u, v) in enumerate(branches):
            children_p = gp.quicksum(p_flow[a, b, t] for a, b in branches if a == v)
            children_q = gp.quicksum(q_flow[a, b, t] for a, b in branches if a == v)
            prosumer_p_mw = gp.quicksum(
                injections_kw[i, t] / 1000.0
                for i, bus in enumerate(agent_bus) if bus == v
            )
            model.addConstr(
                p_flow[u, v, t] == children_p + grid["p_load_mw"][v] - prosumer_p_mw
            )
            model.addConstr(q_flow[u, v, t] == children_q + grid["q_load_mvar"][v])

            base_kv = grid["voltage_kv"][u]
            model.addConstr(
                voltage[v, t] == voltage[u, t]
                - 2.0 * (grid["r_ohm"][k] * p_flow[u, v, t]
                         + grid["x_ohm"][k] * q_flow[u, v, t]) / base_kv**2
            )
            model.addQConstr(
                p_flow[u, v, t] * p_flow[u, v, t]
                + q_flow[u, v, t] * q_flow[u, v, t]
                <= line_limit_mva**2
            )
    return p_flow, q_flow, voltage, grid


def evaluate_feeder(injections_kw, agent_bus, T, line_limit_mva=5.0):
    """Deterministic LinDistFlow check of fixed injections against case33bw.

    Unlike ``add_network``, this is a forward/backward sweep over the radial
    topology, not an optimization: voltage and thermal bounds are not
    enforced, only reported. Use it to check whether a schedule obtained
    without network constraints would violate the feeder's voltage limits or
    congest a line. ``injections_kw[i]`` is a length-``T`` array, positive
    when prosumer ``i`` injects active power (same sign convention as
    ``add_network``).
    """
    grid = load_network()
    branches = grid["branches"]
    children = {}
    for u, v in branches:
        children.setdefault(u, []).append(v)

    prosumer_p_mw = {bus: np.zeros(T) for bus in set(agent_bus)}
    for i, bus in enumerate(agent_bus):
        prosumer_p_mw[bus] = prosumer_p_mw[bus] + np.asarray(injections_kw[i]) / 1000.0

    order, stack = [], [grid["root"]]
    while stack:
        node = stack.pop()
        order.append(node)
        stack.extend(children.get(node, []))

    subtree_p = {node: np.full(T, grid["p_load_mw"][node]) - prosumer_p_mw.get(node, np.zeros(T))
                for node in order}
    subtree_q = {node: np.full(T, grid["q_load_mvar"][node]) for node in order}
    for node in reversed(order):
        for child in children.get(node, []):
            subtree_p[node] = subtree_p[node] + subtree_p[child]
            subtree_q[node] = subtree_q[node] + subtree_q[child]
    p_flow = {(u, v): subtree_p[v] for u, v in branches}
    q_flow = {(u, v): subtree_q[v] for u, v in branches}
    apparent_mva = {edge: np.hypot(p_flow[edge], q_flow[edge]) for edge in p_flow}

    voltage_sq = {grid["root"]: np.full(T, grid["root_voltage"] ** 2)}
    for k, (u, v) in enumerate(branches):
        base_kv = grid["voltage_kv"][u]
        voltage_sq[v] = (voltage_sq[u] - 2.0 * (grid["r_ohm"][k] * p_flow[u, v]
                         + grid["x_ohm"][k] * q_flow[u, v]) / base_kv**2)
    voltage_pu = {bus: np.sqrt(np.maximum(value, 0.)) for bus, value in voltage_sq.items()}

    voltage_violation = {bus: bool(np.any(voltage_pu[bus] < grid["voltage_min"][bus] - 1e-9)
                                   or np.any(voltage_pu[bus] > grid["voltage_max"][bus] + 1e-9))
                         for bus in voltage_pu}
    congestion = {edge: bool(np.any(apparent_mva[edge] > line_limit_mva + 1e-9))
                 for edge in apparent_mva}

    return dict(
        source=grid["name"],
        p_flow_mw={str((u, v, t)): float(p_flow[u, v][t]) for u, v in branches for t in range(T)},
        q_flow_mvar={str((u, v, t)): float(q_flow[u, v][t]) for u, v in branches for t in range(T)},
        voltage_pu={str((bus, t)): float(voltage_pu[bus][t]) for bus in voltage_pu for t in range(T)},
        apparent_mva={str((u, v, t)): float(apparent_mva[u, v][t]) for u, v in branches for t in range(T)},
        line_limit_mva=line_limit_mva,
        voltage_violation={str(bus): flag for bus, flag in voltage_violation.items()},
        congestion={str(edge): flag for edge, flag in congestion.items()},
        any_voltage_violation=any(voltage_violation.values()),
        any_congestion=any(congestion.values()),
    )
