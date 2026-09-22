"""Centralized optimization and two-block consensus ADMM with a DSO projection."""
import gurobipy as gp
import numpy as np
from prosumer import add_prosumer
from network import add_network


def _model(name):
    model = gp.Model(name)
    model.Params.OutputFlag = 0
    model.Params.FeasibilityTol = 1e-9
    model.Params.OptimalityTol = 1e-9
    return model


def _optimize(model):
    model.optimize()
    if model.Status != gp.GRB.OPTIMAL:
        raise RuntimeError(f'{model.ModelName}: Gurobi status {model.Status}')


def _result(agents, iterations, residual=0., network=None):
    result = {key:np.array([[v.X for v in a[key].values()] for a in agents])
              for key in ['buy','sell','charge','discharge','soc']}
    result.update(trades={(i,j,t):v.X for i,a in enumerate(agents) for (j,t),v in a['trade'].items()},
                  costs=np.array([a['cost'].getValue() for a in agents]),
                  iterations=iterations, residual=residual, network=network)
    return result


def _network_values(p_flow, q_flow, voltage, grid, line_limit_mva):
    voltage_pu = {k:v.X**.5 for k,v in voltage.items()}
    apparent_mva = {k:(p_flow[k].X**2+q_flow[k].X**2)**.5 for k in p_flow}
    voltage_violation = {bus:False for bus,_ in voltage_pu}
    for (bus,t),vpu in voltage_pu.items():
        if vpu < grid["voltage_min"][bus]-1e-9 or vpu > grid["voltage_max"][bus]+1e-9:
            voltage_violation[bus] = True
    congestion = {(u,v):False for u,v,t in apparent_mva}
    for (u,v,t),s in apparent_mva.items():
        if s > line_limit_mva+1e-9:
            congestion[u,v] = True
    return dict(source=grid["name"],
                p_flow_mw={str(k):v.X for k,v in p_flow.items()},
                q_flow_mvar={str(k):v.X for k,v in q_flow.items()},
                voltage_pu={str(k):v for k,v in voltage_pu.items()},
                apparent_mva={str(k):v for k,v in apparent_mva.items()},
                line_limit_mva=line_limit_mva,
                voltage_violation={str(bus):flag for bus,flag in voltage_violation.items()},
                congestion={str(edge):flag for edge,flag in congestion.items()},
                any_voltage_violation=any(voltage_violation.values()),
                any_congestion=any(congestion.values()))


def centralized(data, with_network=False):
    model = _model('centralized')
    N, T = len(data['demand']),len(data['buy_price'])
    try:
        agents = [add_prosumer(model,i,data) for i in range(N)]
        for i,j in data['edges']:
            for t in range(T):
                model.addConstr(agents[i]['trade'][j,t]+agents[j]['trade'][i,t] == 0)
        if with_network:
            injections = {(i,t):a['injection'][t] for i,a in enumerate(agents) for t in range(T)}
            p_flow,q_flow,voltage,grid = add_network(
                model,injections,data['agent_bus'],T,data['line_limit_mva'])
        model.setObjective(gp.quicksum(a['cost'] for a in agents))
        _optimize(model)
        return _result(agents,0,network=(
            _network_values(p_flow,q_flow,voltage,grid,data['line_limit_mva'])
            if with_network else None))
    finally:
        model.dispose()


def distributed(data, with_network=False, rho=.5, tolerance=1e-6, max_iterations=50000):
    N,T = len(data['demand']),len(data['buy_price'])
    models = [_model(f'prosumer_{i}') for i in range(N)]
    agents = [add_prosumer(m,i,data) for i,m in enumerate(models)]
    keys = [(i,j,t) for i,a in enumerate(agents) for j,t in a['trade']]
    z,u = dict.fromkeys(keys,0.),dict.fromkeys(keys,0.)
    s,w = np.zeros((N,T)),np.zeros((N,T))
    dso = None
    if with_network:
        dso = _model('dso_projection')
        injection = dso.addVars(N,T,lb=-gp.GRB.INFINITY,name='injection')
        p_flow,q_flow,voltage,grid = add_network(
            dso,injection,data['agent_bus'],T,data['line_limit_mva'])
    try:
        for iteration in range(1,max_iterations+1):
            # Local step: operating cost plus trade and network consensus penalties.
            for i,(m,a) in enumerate(zip(models,agents)):
                penalty = gp.quicksum((v-z[i,j,t]+u[i,j,t])**2 for (j,t),v in a['trade'].items())
                if with_network:
                    penalty += gp.quicksum((a['injection'][t]-s[i,t]+w[i,t])**2 for t in range(T))
                m.setObjective(a['cost']+rho/2*penalty)
                _optimize(m)
            x = {(i,j,t):agents[i]['trade'][j,t].X for i,j,t in keys}
            old_z,old_s = z.copy(),s.copy()
            # Market projection: reciprocal bilateral contracts.
            for i,j in data['edges']:
                for t in range(T):
                    z[i,j,t] = (x[i,j,t]+u[i,j,t]-x[j,i,t]-u[j,i,t])/2
                    z[j,i,t] = -z[i,j,t]
            primal = [x[k]-z[k] for k in keys]
            dual = [rho*(z[k]-old_z[k]) for k in keys]
            for k in keys: u[k] += x[k]-z[k]
            if with_network:
                q = np.array([[a['injection'][t].getValue() for t in range(T)] for a in agents])
                # DSO projects physical injections onto feeder feasibility.
                dso.setObjective(gp.quicksum((injection[i,t]-q[i,t]-w[i,t])**2
                                            for i in range(N) for t in range(T)))
                _optimize(dso)
                s = np.array([[injection[i,t].X for t in range(T)] for i in range(N)])
                w += q-s
                primal.extend((q-s).ravel())
                dual.extend((rho*(s-old_s)).ravel())
            residual = float(max(np.linalg.norm(primal),np.linalg.norm(dual)))
            if residual <= tolerance:
                return _result(agents,iteration,residual,
                               _network_values(p_flow,q_flow,voltage,grid,data['line_limit_mva'])
                               if with_network else None)
        raise RuntimeError(f'ADMM did not converge; residual={residual:.3g}')
    finally:
        for m in models: m.dispose()
        if dso is not None: dso.dispose()
