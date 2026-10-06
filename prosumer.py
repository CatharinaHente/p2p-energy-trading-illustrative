"""Multi-period P2P prosumer: BESS (polygon FOR), EV (V2G, paper eqs 1-4), HP, reactive power.
 
Sources
-------
EV model   : Huaman-Rivera et al. (2024), Electronics 13(12):2259, eqs (1)-(4), Section 3.3
BESS / HP  : Fornier et al. flexibility paper, Section A
 
Device summary
--------------
Original  : grid buy/sell, PV (fixed), BESS, P2P trades
[UPDATED] BESS — apparent power polygon (P,Q) replaces fixed P bound; Q now decision variable
[NEW]     EV   — V2G bidirectional; SOC depletion from driving (eq 1); departure SOC (eq 2);
                 DoD constraint [16]; seasonal capacity derating; reactive power from charger
[NEW]     HP   — controllable load with intertemporal thermal comfort (unchanged from previous)
 
 
Required data keys (additions beyond original)
----------------------------------------------
BESS:
  battery_smax          list[float]   kVA converter apparent power, one per prosumer
  battery_polygon_sides int           polygon sides (default 8)
  battery_power         list[float]   kW rate limit on charge/discharge (separate from S_max)
 
EV (set ev_capacity[i]=0 or omit to disable):
  ev_capacity           list[float]   kWh nominal capacity
  ev_cap_factor         list[float]   seasonal temperature derating in [0,1] (eq 1 / Sec 3.3)
  ev_power              list[float]   kW max charge/discharge rate (P_C in eqs 3-4)
  ev_smax               list[float]   kVA charger apparent power (for Q polygon)
  ev_polygon_sides      int           charger polygon sides (default 8)
  ev_driving_load       list[list]    kWh depleted by driving [prosumer][timestep]
                                      (0 when plugged in, >0 when away — encodes eq 1)
                                      Seasonal: use 17 kWh/100km winter, 15 kWh/100km summer
  ev_soc_min_departure  list[float]   fraction — min SOC before next trip (eq 2 rearranged)
  ev_dod_max            float         max DoD; default 0.30 → SoC_min = 0.70 (ref [16])
  ev_initial_soc        list[float]   kWh initial SOC
  ev_availability       list[list]    1=plugged in, 0=away [prosumer][timestep]
  ev_eta_charge         float         η_c charging efficiency (eq 3)
  ev_eta_discharge      float         η_d discharging efficiency
  ev_cost               float         EUR/kWh throughput degradation cost
  ev_v2g                bool          True=V2G bidirectional, False=V1G charge-only
 
HP (set hp_max_power[i]=0 or omit to disable):
  hp_max_power          list[float]   kW
  hp_base_power         list[list]    kW baseline [prosumer][timestep]
  hp_q_heat             list[list]    degC/h heating demand [prosumer][timestep]
  hp_T_init             list[float]   degC initial indoor temperature
  hp_T_min              float         degC comfort lower bound
  hp_T_max              float         degC comfort upper bound
  hp_cost               float         EUR/kWh (optional, default 0)
 
Reactive power factors (all optional, defaults shown):
  pf_pv                 float         0.95
  pf_hp                 float         0.97
  pf_load               float         0.95
 
"""
import math
import gurobipy as gp
 
 
# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
 
def _tan(pf: float) -> float:
    """tan(arccos(pf)) — Q/P ratio for a fixed-power-factor device."""
    return math.tan(math.acos(max(1e-6, min(1.0, pf))))
 
 
def _add_polygon(model, p_expr, q_var, s_max: float, n_sides: int):
    """Inner polygon approximation of the apparent power circle P² + Q² ≤ S_max².
 
    Flexibility paper Section A.1 / [5]: k-th face of an n-sided regular polygon:
        cos(2πk/n)·P + sin(2πk/n)·Q  ≤  S_max · cos(π/n)
 
    n=8 (octagon) gives a 92.4% area ratio — accurate for distribution studies.
    This is called once per timestep for each converter (BESS, EV charger).
    """
    scale = s_max * math.cos(math.pi / n_sides)
    for k in range(n_sides):
        angle = 2.0 * math.pi * k / n_sides
        model.addConstr(
            math.cos(angle) * p_expr + math.sin(angle) * q_var <= scale
        )
 
 
# --------------------------------------------------------------------------- #
# Main prosumer function                                                        #
# --------------------------------------------------------------------------- #
 
def add_prosumer(model, i, data):
    """Add one prosumer's variables and constraints to *model*.
 
    Positive peer trade = sale from prosumer i to peer j.
    Power in kW, stored energy in kWh, temperature in degC.
    """
    T     = len(data['buy_price'])
    N     = len(data['demand'])
    peers = [b if a == i else a for a, b in data['edges'] if i in (a, b)]
 
    # ----------------------------------------------------------------------- #
    # Grid buy / sell  (original)                                              #
    # ----------------------------------------------------------------------- #
    buy  = model.addVars(T, ub=data['grid_limit'], name=f'buy_{i}')
    sell = model.addVars(T, ub=data['grid_limit'], name=f'sell_{i}')
 
    # ----------------------------------------------------------------------- #
    # P2P trades  (original)                                                   #
    # ----------------------------------------------------------------------- #
    trade = model.addVars(peers, range(T),
                          lb=-data['trade_limit'], ub=data['trade_limit'],
                          name=f'trade_{i}')
    trade_abs = model.addVars(peers, range(T), name=f'trade_abs_{i}')
    for j in peers:
        for t in range(T):
            model.addConstr(trade_abs[j, t] >=  trade[j, t])
            model.addConstr(trade_abs[j, t] >= -trade[j, t])
 
    # ----------------------------------------------------------------------- #
    # [UPDATED] BESS with apparent-power polygon FOR                           #
    #                                                                          #
    # Two separate physical constraints:                                       #
    #   (a) Rate limits  — battery_power[i] bounds charge and discharge        #
    #       individually (physical limit on how fast the cell can charge).     #
    #   (b) Converter apparent power — polygon on (P_net, Q_bess) where        #
    #       P_net = discharge - charge.  Q_bess is a free decision variable   #
    #       (the converter can supply reactive power independently of SOC).    #
    #                                                                          #
    # The two constraints are complementary: (a) limits each direction alone;  #
    # (b) limits the joint (P,Q) operating region of the power electronics.   #
    # ----------------------------------------------------------------------- #
    s_max_batt = data['battery_smax'][i]
    n_poly     = data.get('battery_polygon_sides', 8)
 
    charge    = model.addVars(T, ub=data['battery_power'][i], name=f'charge_{i}')
    discharge = model.addVars(T, ub=data['battery_power'][i], name=f'discharge_{i}')
 
    # [UPDATED] BESS reactive power — decision variable, bounded by converter rating
    q_batt = model.addVars(T, lb=-s_max_batt, ub=s_max_batt, name=f'q_batt_{i}')
 
    soc = model.addVars(T + 1,
                        lb=data['soc_min_fraction'] * data['battery_capacity'][i],
                        ub=data['soc_max_fraction'] * data['battery_capacity'][i],
                        name=f'soc_{i}')
    model.addConstr(soc[0] == data['initial_soc'][i])
    model.addConstr(soc[T] == soc[0])   # cyclic terminal: schedule repeatable day to day
 
    for t in range(T):
        # [UPDATED] (b) Apparent-power polygon on (P_net, Q_bess)
        _add_polygon(model,
                     p_expr=discharge[t] - charge[t],
                     q_var=q_batt[t],
                     s_max=s_max_batt,
                     n_sides=n_poly)
        # BESS SOC dynamics (original, lossless model → flexibility paper eq A.1 spirit)
        model.addConstr(
            soc[t + 1] == soc[t] + data['dt'] * (
                data['eta_charge'] * charge[t]
                - discharge[t] / data['eta_discharge']
            )
        )
 
    # ----------------------------------------------------------------------- #
    # [NEW] EV — V2G model following Huaman-Rivera et al. (2024) eqs (1)-(4)  #
    #                                                                          #
    # Eq (1): SOC depletion from driving encoded as ev_driving_load[i][t]     #
    #         (kWh depleted per timestep; 0 when plugged in, >0 when away).   #
    #         Seasonal values: 17 kWh/100km winter, 15 kWh/100km summer [38]. #
    #                                                                          #
    # Eq (2): At departure: ev_soc ≥ SoC_desired × C_B_eff                   #
    #         (must hold enough energy for the upcoming trip).                 #
    #                                                                          #
    # Eqs (3)-(4): Implicit in SOC dynamics with efficiency η_c, η_d.        #
    #                                                                          #
    # DoD [16]: SoC_min = (1 - DoD_max) × C_B_eff;  default DoD_max = 0.30. #
    #                                                                          #
    # Temperature derating (Section 3.3): C_B_eff = C_B_nominal × cap_factor.#
    #                                                                          #
    # V2G: EV charger has apparent power S_charger. When plugged in the       #
    #      charger can simultaneously control active power (charge or          #
    #      discharge in V2G mode) and reactive power. Joint constraint is the  #
    #      same polygon as BESS, applied to (P_ev_net, Q_ev_charger).         #
    # ----------------------------------------------------------------------- #
    ev_cap_nominal = data.get('ev_capacity', [0] * N)[i]
    has_ev         = ev_cap_nominal > 0
 
    if has_ev:
        # Temperature derating — Section 3.3, [38]: reduces effective range in cold
        cap_factor  = data.get('ev_cap_factor', [1.0] * N)[i]
        ev_cap_eff  = ev_cap_nominal * cap_factor        # kWh effective capacity
 
        # DoD constraint from [16]: max depth of discharge = 30% → floor = 70%
        dod_max      = data.get('ev_dod_max', 0.30)
        ev_soc_floor = (1.0 - dod_max) * ev_cap_eff     # absolute kWh floor
 
        # Charger apparent power rating (for polygon on P_ev_net and Q_ev)
        s_max_ev   = data.get('ev_smax', [ev_cap_nominal] * N)[i]
        n_poly_ev  = data.get('ev_polygon_sides', 8)
 
        # Active power: charge always available when plugged in
        ev_charge = model.addVars(T, ub=data['ev_power'][i], name=f'ev_charge_{i}')
 
        # V2G flag: V1G fixes discharge to zero; V2G keeps it free when plugged in
        v2g = data.get('ev_v2g', True)
        ev_discharge = model.addVars(
            T,
            ub=(data['ev_power'][i] if v2g else 0.0),   # V1G: ub=0 → no discharge
            name=f'ev_discharge_{i}'
        )
 
        # [NEW] Reactive power from EV charger — available in V2G mode (and optionally V1G)
        # The charger's power electronics can supply Q independently of SOC (STATCOM-like)
        ev_q_charger = model.addVars(
            T,
            lb=-s_max_ev, ub=s_max_ev,
            name=f'ev_q_charger_{i}'
        )
 
        # SOC bounded by DoD floor and effective capacity ceiling
        ev_soc = model.addVars(T + 1,
                               lb=ev_soc_floor,
                               ub=ev_cap_eff,
                               name=f'ev_soc_{i}')
        model.addConstr(ev_soc[0] == data['ev_initial_soc'][i])
        model.addConstr(ev_soc[T] == ev_soc[0])   # cyclic: EV ends with same SOC as start
 
        for t in range(T):
            avail        = data['ev_availability'][i][t]     # 1=plugged in, 0=away
            driving_load = data['ev_driving_load'][i][t]     # kWh depleted by driving (eq 1)
 
            # Availability gates active and reactive power (only when plugged in)
            model.addConstr(ev_charge[t]     <= avail * data['ev_power'][i])
            model.addConstr(ev_discharge[t]  <= avail * data['ev_power'][i])
            model.addConstr(ev_q_charger[t]  <= avail * s_max_ev)
            model.addConstr(ev_q_charger[t]  >= -avail * s_max_ev)
 
            # [NEW] Joint apparent-power polygon on charger (P_ev_net, Q_ev_charger)
            # Same structure as BESS polygon — flexibility paper Section A.1
            _add_polygon(model,
                         p_expr=ev_discharge[t] - ev_charge[t],
                         q_var=ev_q_charger[t],
                         s_max=s_max_ev,
                         n_sides=n_poly_ev)
 
            # SOC dynamics — eqs (3)-(4) implicit; eq (1) driving depletion explicit
            model.addConstr(
                ev_soc[t + 1] == ev_soc[t]
                + data['dt'] * (
                    data['ev_eta_charge'] * ev_charge[t]
                    - ev_discharge[t] / data['ev_eta_discharge']
                )
                - driving_load                             # eq (1): (d/D)×C_B_eff depleted
            )
 
        # Departure SOC constraint — eq (2): SoC_desired = SoC_init + E_demand/C_B
        # Rearranged: at the last plugged-in timestep before an away period,
        # SOC must be at least soc_min_departure × effective_capacity
        soc_dep_frac = data.get('ev_soc_min_departure', [0.0] * N)[i]
        soc_dep_abs  = soc_dep_frac * ev_cap_eff          # absolute kWh target (eq 2)
        for t in range(T):
            next_avail = data['ev_availability'][i][(t + 1) % T]
            if data['ev_availability'][i][t] == 1 and next_avail == 0:
                # Last plugged-in hour before departure: enforce eq (2)
                model.addConstr(ev_soc[t + 1] >= soc_dep_abs)
 
    else:
        # No EV: zero upper bounds make all EV variables trivially zero
        ev_charge    = model.addVars(T, ub=0, name=f'ev_charge_{i}')
        ev_discharge = model.addVars(T, ub=0, name=f'ev_discharge_{i}')
        ev_q_charger = model.addVars(T, ub=0, lb=0, name=f'ev_q_charger_{i}')
        ev_soc       = None
 
    # ----------------------------------------------------------------------- #
    # [NEW] Heat pump with intertemporal thermal comfort model (unchanged)     #
    # Fornier et al. / flexibility paper eq (2):                              #
    #   T[t+1] = T[t] + (hp_power[t]/hp_base[t] - 1) × q_heat[t] × dt       #
    # Linearised: T[t+1] = T[t] + hp_power[t]×(q/base)×dt - q×dt            #
    # ----------------------------------------------------------------------- #
    hp_max = data.get('hp_max_power', [0] * N)[i]
    has_hp = hp_max > 0
 
    if has_hp:
        hp_power = model.addVars(T, ub=hp_max, name=f'hp_power_{i}')
        temp = model.addVars(T + 1,
                             lb=data['hp_T_min'],
                             ub=data['hp_T_max'],
                             name=f'temp_{i}')
        model.addConstr(temp[0] == data['hp_T_init'][i])
 
        for t in range(T):
            base = data['hp_base_power'][i][t]
            q    = data['hp_q_heat'][i][t]
            if base > 0:
                model.addConstr(
                    temp[t + 1] == temp[t]
                    + hp_power[t] * (q / base) * data['dt']
                    - q * data['dt']
                )
            else:
                model.addConstr(temp[t + 1] == temp[t])
    else:
        hp_power = model.addVars(T, ub=0, name=f'hp_power_{i}')
        temp     = None
 
    # ----------------------------------------------------------------------- #
    # Reactive power factors — fixed pf for PV, HP and base load              #
    # BESS and EV use decision variables (q_batt, ev_q_charger) via polygons  #
    # ----------------------------------------------------------------------- #
    tan_pv   = _tan(data.get('pf_pv',   0.95))
    tan_hp   = _tan(data.get('pf_hp',   0.97))
    tan_load = _tan(data.get('pf_load', 0.95))
 
    # ----------------------------------------------------------------------- #
    # Energy balance and bus injections                                        #
    # ----------------------------------------------------------------------- #
    injection   = {}   # active power injection at bus (kW)
    q_injection = {}   # reactive power injection at bus (kVAR)
 
    for t in range(T):
        # Energy balance — all active power devices
        model.addConstr(
            data['generation'][i][t] + buy[t] + discharge[t] + ev_discharge[t]
            == data['demand'][i][t] + sell[t] + charge[t] + ev_charge[t]
            + hp_power[t] + trade.sum('*', t)
        )
 
        # Active power injection at prosumer bus (feeds LinDistFlow P equation)
        injection[t] = (
            data['generation'][i][t]
            - data['demand'][i][t]
            - charge[t]    + discharge[t]          # BESS net
            - ev_charge[t] + ev_discharge[t]       # EV net
            - hp_power[t]                           # HP load
        )
 
        # Reactive power injection at prosumer bus (feeds LinDistFlow Q equation)
        # BESS and EV use decision variables; others use fixed power factor
        q_injection[t] = (
              data['generation'][i][t] * tan_pv    # PV: fixed pf (capacitive)
            + q_batt[t]                             # [UPDATED] BESS: decision variable
            + ev_q_charger[t]                       # [NEW] EV charger: decision variable
            - hp_power[t] * tan_hp                  # HP: inductive
            - data['demand'][i][t] * tan_load       # base load: inductive
        )
 
    # ----------------------------------------------------------------------- #
    # Cost function                                                            #
    # ----------------------------------------------------------------------- #
    trade_volume = trade_abs.sum()
 
    cost = (
        data['dt'] * gp.quicksum(
            data['buy_price'][t]  * buy[t]
            - data['sell_price'][t] * sell[t]
            - data['p2p_price'][t]  * trade.sum('*', t)
            + data['battery_cost']  * (charge[t] + discharge[t])
            + data.get('ev_cost',  0.0) * (ev_charge[t] + ev_discharge[t])
            + data.get('hp_cost',  0.0) * hp_power[t]
            for t in range(T)
        )
        + data['dt'] * data['trade_fee'] * trade_volume
    )
 
    return dict(
        # Original
        buy=buy, sell=sell, trade=trade,
        charge=charge, discharge=discharge, soc=soc,
        injection=injection, cost=cost,
        # [UPDATED] BESS reactive power
        q_batt=q_batt,
        # [NEW] EV
        ev_charge=ev_charge, ev_discharge=ev_discharge, ev_soc=ev_soc,
        ev_q_charger=ev_q_charger,
        # [NEW] HP
        hp_power=hp_power, temp=temp,
        # [NEW] Q injection
        q_injection=q_injection,
    )
