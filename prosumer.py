"""Multi-period P2P prosumer.

Devices: grid buy/sell | PV (triangular Q FOR) | BESS (polygon FOR) | EV V2G | HP | P2P trades

Sources: Huaman-Rivera et al. (2024) Electronics 13(12):2259 [EV eqs 1-4]
         Fornier et al. flexibility paper Sec A [BESS/HP/PV FOR]

Required data keys (beyond original grid/PV/BESS/trade keys)
-------------------------------------------------------------
BESS    battery_smax [kVA], battery_polygon_sides int=8
PV Q    pf_pv_min float=0.90                         # REVIEW: cite grid code (EN 50549 / IEEE 1547)
EV      ev_capacity [kWh], ev_cap_factor [], ev_power [kW], ev_smax [kVA],
        ev_polygon_sides int=8, ev_driving_load [[kWh]], ev_soc_min_departure [],
        ev_dod_max float=0.30, ev_initial_soc [kWh], ev_availability [[0|1]],
        ev_eta_charge, ev_eta_discharge, ev_cost [EUR/kWh], ev_v2g bool
HP      hp_max_power [kW], hp_base_power [[kW]], hp_q_heat [[°C/h]],
        hp_T_init [°C], hp_T_min °C, hp_T_max °C, hp_cost [EUR/kWh]
pf      pf_hp float=0.97    # REVIEW: cite HP manufacturer data
        pf_load float=0.95  # REVIEW: cite standard load model (e.g. BDEW)
"""
import math
import gurobipy as gp


def _tan(pf):
    return math.tan(math.acos(max(1e-6, min(1.0, pf))))


def _polygon(model, p_expr, q_var, s_max, n):
    """Inner polygon approx of P²+Q² ≤ S². Flexibility paper Sec A.1."""
    c = s_max * math.cos(math.pi / n)
    for k in range(n):
        a = 2.0 * math.pi * k / n
        model.addConstr(math.cos(a) * p_expr + math.sin(a) * q_var <= c)


def add_prosumer(model, i, data):
    """Positive trade = sale from i to j. Power kW, energy kWh, temp °C."""
    T     = len(data['buy_price'])
    N     = len(data['demand'])
    peers = [b if a == i else a for a, b in data['edges'] if i in (a, b)]

    # --- Grid ---
    buy  = model.addVars(T, ub=data['grid_limit'],  name=f'buy_{i}')
    sell = model.addVars(T, ub=data['grid_limit'],  name=f'sell_{i}')

    # --- Trades ---
    trade     = model.addVars(peers, range(T), lb=-data['trade_limit'],
                              ub=data['trade_limit'], name=f'trade_{i}')
    trade_abs = model.addVars(peers, range(T), name=f'trade_abs_{i}')
    for j in peers:
        for t in range(T):
            model.addConstr(trade_abs[j, t] >=  trade[j, t])
            model.addConstr(trade_abs[j, t] >= -trade[j, t])

    # --- BESS ---
    # (a) cell rate limits; (b) converter FOR polygon — flexibility paper Sec A.1
    s_batt = data['battery_smax'][i]
    n_batt = data.get('battery_polygon_sides', 8)
    charge    = model.addVars(T, ub=data['battery_power'][i], name=f'charge_{i}')
    discharge = model.addVars(T, ub=data['battery_power'][i], name=f'discharge_{i}')
    q_batt    = model.addVars(T, lb=-s_batt, ub=s_batt,       name=f'q_batt_{i}')
    soc = model.addVars(T + 1,
                        lb=data['soc_min_fraction'] * data['battery_capacity'][i],
                        ub=data['soc_max_fraction'] * data['battery_capacity'][i],
                        name=f'soc_{i}')
    model.addConstr(soc[0] == data['initial_soc'][i])
    model.addConstr(soc[T] == soc[0])
    for t in range(T):
        _polygon(model, discharge[t] - charge[t], q_batt[t], s_batt, n_batt)
        model.addConstr(                                               # lossless SOC
            soc[t+1] == soc[t] + data['dt'] * (
                data['eta_charge'] * charge[t] - discharge[t] / data['eta_discharge']))

    # --- PV reactive power ---
    # Triangular FOR: |Q_pv| ≤ P_pv · tan(arccos(pf_pv_min)) — flexibility paper Fig 1b
    pf_pv_min = data.get('pf_pv_min', 0.90)                          # REVIEW: EN 50549 / IEEE 1547
    tan_pv    = _tan(pf_pv_min)
    q_pv = model.addVars(T, lb=0, ub=0, name=f'q_pv_{i}')           # bounds set per t below
    for t in range(T):
        q_max = data['generation'][i][t] * tan_pv
        model.addConstr(q_pv[t] <=  q_max)
        model.addConstr(q_pv[t] >= -q_max)

    # --- EV ---
    ev_cap_nom = data.get('ev_capacity', [0] * N)[i]
    has_ev     = ev_cap_nom > 0
    if has_ev:
        cap_factor   = data.get('ev_cap_factor', [1.0] * N)[i]
        ev_cap_eff   = ev_cap_nom * cap_factor                        # Huaman-Rivera Sec 3.3
        dod_max      = data.get('ev_dod_max', 0.30)                   # ref [16]
        ev_soc_floor = (1.0 - dod_max) * ev_cap_eff
        s_ev         = data.get('ev_smax', [ev_cap_nom] * N)[i]
        n_ev         = data.get('ev_polygon_sides', 8)
        v2g          = data.get('ev_v2g', True)

        ev_charge    = model.addVars(T, ub=data['ev_power'][i],             name=f'ev_charge_{i}')
        ev_discharge = model.addVars(T, ub=(data['ev_power'][i] if v2g else 0.0), name=f'ev_discharge_{i}')
        ev_q         = model.addVars(T, lb=-s_ev, ub=s_ev,                 name=f'ev_q_{i}')
        ev_soc       = model.addVars(T + 1, lb=ev_soc_floor, ub=ev_cap_eff, name=f'ev_soc_{i}')
        model.addConstr(ev_soc[0] == data['ev_initial_soc'][i])
        model.addConstr(ev_soc[T] == ev_soc[0])

        for t in range(T):
            av = data['ev_availability'][i][t]
            model.addConstr(ev_charge[t]   <= av * data['ev_power'][i])
            model.addConstr(ev_discharge[t] <= av * data['ev_power'][i])
            model.addConstr(ev_q[t]  <=  av * s_ev)
            model.addConstr(ev_q[t]  >= -av * s_ev)
            _polygon(model, ev_discharge[t] - ev_charge[t], ev_q[t], s_ev, n_ev)
            model.addConstr(                                           # Huaman-Rivera eq (1)
                ev_soc[t+1] == ev_soc[t]
                + data['dt'] * (data['ev_eta_charge'] * ev_charge[t]
                                - ev_discharge[t] / data['ev_eta_discharge'])
                - data['ev_driving_load'][i][t])

        soc_dep = data.get('ev_soc_min_departure', [0.0] * N)[i] * ev_cap_eff
        for t in range(T):                                             # Huaman-Rivera eq (2)
            if data['ev_availability'][i][t] == 1 and \
               data['ev_availability'][i][(t+1) % T] == 0:
                model.addConstr(ev_soc[t+1] >= soc_dep)
    else:
        ev_charge = ev_discharge = model.addVars(T, ub=0, name=f'ev_off_{i}')
        ev_q      = model.addVars(T, ub=0, lb=0, name=f'ev_q_off_{i}')
        ev_soc    = None

    # --- HP ---
    # Controllable load, fixed pf (line FOR) — flexibility paper Sec A.3
    hp_max = data.get('hp_max_power', [0] * N)[i]
    has_hp = hp_max > 0
    if has_hp:
        hp_power = model.addVars(T, ub=hp_max, name=f'hp_{i}')
        temp     = model.addVars(T + 1, lb=data['hp_T_min'], ub=data['hp_T_max'], name=f'temp_{i}')
        model.addConstr(temp[0] == data['hp_T_init'][i])
        for t in range(T):
            base = data['hp_base_power'][i][t]
            q    = data['hp_q_heat'][i][t]
            if base > 0:                                               # flexibility paper eq (2)
                model.addConstr(
                    temp[t+1] == temp[t] + hp_power[t] * (q / base) * data['dt'] - q * data['dt'])
            else:
                model.addConstr(temp[t+1] == temp[t])
    else:
        hp_power = model.addVars(T, ub=0, name=f'hp_off_{i}')
        temp     = None

    # --- Power factors (fixed) ---
    tan_hp   = _tan(data.get('pf_hp',   0.97))   # REVIEW: cite HP manufacturer data
    tan_load = _tan(data.get('pf_load', 0.95))   # REVIEW: cite BDEW or similar

    # --- Energy balance and bus injections ---
    injection   = {}
    q_injection = {}
    for t in range(T):
        model.addConstr(
            data['generation'][i][t] + buy[t] + discharge[t] + ev_discharge[t]
            == data['demand'][i][t] + sell[t] + charge[t] + ev_charge[t]
            + hp_power[t] + trade.sum('*', t))

        injection[t] = (
            data['generation'][i][t] - data['demand'][i][t]
            - charge[t]    + discharge[t]
            - ev_charge[t] + ev_discharge[t]
            - hp_power[t])

        q_injection[t] = (
              q_pv[t]                               # PV: triangular FOR (decision var)
            + q_batt[t]                             # BESS: polygon FOR (decision var)
            + ev_q[t]                               # EV: polygon FOR (decision var)
            - hp_power[t] * tan_hp                  # HP: fixed pf, inductive
            - data['demand'][i][t] * tan_load)      # load: fixed pf, inductive

    # --- Cost ---
    q_cost = data.get('q_cost', 0.0)   # EUR/kVAR²·h — regularises Q, prevents extreme values
    cost = (
        data['dt'] * gp.quicksum(
            data['buy_price'][t]  * buy[t]
            - data['sell_price'][t] * sell[t]
            - data['p2p_price'][t]  * trade.sum('*', t)
            + data['battery_cost']  * (charge[t] + discharge[t])
            + data.get('ev_cost',  0.0) * (ev_charge[t] + ev_discharge[t])
            + data.get('hp_cost',  0.0) * hp_power[t]
            + q_cost * (q_batt[t]*q_batt[t] + q_pv[t]*q_pv[t] + ev_q[t]*ev_q[t])
            for t in range(T))
        + data['dt'] * data['trade_fee'] * trade_abs.sum())

    return dict(
        buy=buy, sell=sell, trade=trade,
        charge=charge, discharge=discharge, soc=soc,
        q_batt=q_batt, q_pv=q_pv,
        ev_charge=ev_charge, ev_discharge=ev_discharge, ev_soc=ev_soc, ev_q_charger=ev_q,
        hp_power=hp_power, temp=temp,
        injection=injection, q_injection=q_injection,
        cost=cost)