"""Multi-period P2P prosumer with a continuous battery model."""
import gurobipy as gp


def add_prosumer(model, i, data):
    """Positive peer trade is a sale; power kW, stored energy kWh."""
    T = len(data['buy_price'])
    peers = [b if a == i else a for a,b in data['edges'] if i in (a,b)]
    buy = model.addVars(T, ub=data['grid_limit'], name=f'buy_{i}')
    sell = model.addVars(T, ub=data['grid_limit'], name=f'sell_{i}')
    trade = model.addVars(peers, range(T), lb=-data['trade_limit'], ub=data['trade_limit'], name=f'trade_{i}')
    trade_abs = model.addVars(peers, range(T), name=f'trade_abs_{i}')
    charge = model.addVars(T, ub=data['battery_power'][i], name=f'charge_{i}')
    discharge = model.addVars(T, ub=data['battery_power'][i], name=f'discharge_{i}')
    # A real battery keeps a depth-of-discharge/overcharge margin instead of
    # cycling between literally empty and literally full.
    soc = model.addVars(T+1, lb=data['soc_min_fraction']*data['battery_capacity'][i],
                        ub=data['soc_max_fraction']*data['battery_capacity'][i], name=f'soc_{i}')
    model.addConstr(soc[0] == data['initial_soc'][i])
    model.addConstr(soc[T] == soc[0])
    for j in peers:
        for t in range(T):
            model.addConstr(trade_abs[j,t] >= trade[j,t])
            model.addConstr(trade_abs[j,t] >= -trade[j,t])
    injection = {}
    for t in range(T):
        model.addConstr(data['generation'][i][t] + buy[t] + discharge[t]
                        == data['demand'][i][t] + sell[t] + charge[t] + trade.sum('*',t))
        model.addConstr(soc[t+1] == soc[t] + data['dt'] *
                        (data['eta_charge']*charge[t] - discharge[t]/data['eta_discharge']))
        injection[t] = data['generation'][i][t]-data['demand'][i][t]-charge[t]+discharge[t]
    trade_volume = trade_abs.sum()
    cost = data['dt'] * gp.quicksum(data['buy_price'][t]*buy[t]-data['sell_price'][t]*sell[t]
            -data['p2p_price'][t]*trade.sum('*',t)
            +data['battery_cost']*(charge[t]+discharge[t]) for t in range(T)) \
            + data['dt'] * data['trade_fee'] * trade_volume
    return dict(buy=buy, sell=sell, trade=trade, charge=charge, discharge=discharge,
                soc=soc, injection=injection, cost=cost)
