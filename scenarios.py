"""scenarios.py — Extract SimBench + DWD prosumer profiles 

Builds a data dict compatible with main.py to add prosumers to grid

Usage
-----
from scenarios import make_scenario, prosumer_summary
data = make_scenario(season='winter')

Sources
-------
Demand / HP / EV  : SimBench LoadProfile.csv (2016, 15-min, 96 steps/day)
PV generation     : SimBench RESProfile.csv  (2016, 15-min, 96 steps/day)
HP thermal demand : DWD 2016 hourly temp → upsampled 15-min
                    station 2564 Hanover / 2928 Lübeck
"""
import io
import re
import random
import urllib.request
import zipfile

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
_SB_BASE = ('/opt/anaconda3/envs/p2p/lib/python3.10/site-packages/'
            'simbench/networks/1-complete_data-mixed-all-0-sw/')
_LP_PATH  = _SB_BASE + 'LoadProfile.csv'
_RP_PATH  = _SB_BASE + 'RESProfile.csv'

# ── Season slices (index into 35136-step 2016 year, 96 steps/day) ─────────────
_SEASONS = {
    'winter': slice(9*96,  10*96),    # Jan 10
    'spring': slice((31+29+31+9)*96,  (31+29+31+10)*96),   # Apr 10
    'summer': slice((31+29+31+30+31+30+9)*96,
                    (31+29+31+30+31+30+10)*96),              # Jul 10
    'autumn': slice((31+29+31+30+31+30+31+31+30+9)*96,
                    (31+29+31+30+31+30+31+31+30+10)*96),     # Oct 10
}

# ── Profile pool ───────────────────────────────────────────────────────────────
DEMAND_PROFILES = {
    'H0-A': dict(label='Residential low (seasonal)',  peak_kw=6,  sector='residential'),
    'H0-B': dict(label='Residential typical',          peak_kw=8,  sector='residential'),
    'H0-C': dict(label='Residential evening peak',     peak_kw=7,  sector='residential'),
    'H0-G': dict(label='Residential rural',            peak_kw=5,  sector='residential'),
    'H0-L': dict(label='Residential agricultural',     peak_kw=5,  sector='residential'),
    'G1-A': dict(label='Small shop',                   peak_kw=15, sector='commercial'),
    'G1-B': dict(label='Shop alt.',                    peak_kw=18, sector='commercial'),
    'G2-A': dict(label='Evening facility',              peak_kw=20, sector='commercial'),
    'G3-A': dict(label='Food service / restaurant',    peak_kw=20, sector='commercial'),
    'G4-A': dict(label='School',                       peak_kw=30, sector='commercial'),
    'G4-B': dict(label='Office',                       peak_kw=25, sector='commercial'),
    'G5-A': dict(label='Early facility (bakery etc)',  peak_kw=15, sector='commercial'),
    'G6-A': dict(label='Evening commercial',            peak_kw=12, sector='commercial'),
    'L1-A': dict(label='Farm (summer heavy)',           peak_kw=25, sector='agricultural'),
    'L2-A': dict(label='Farm (evening peak)',           peak_kw=20, sector='agricultural'),
}

HP_PROFILES = {
    # name → (dwd_station, description)
    'Air_Semi-Parallel_1': ('hannover', 'Air semi-parallel, Hanover'),
    'Air_Alternative_1':   ('hannover', 'Air alternative, Hanover'),
    'Air_Parallel_1':      ('hannover', 'Air parallel, Hanover'),
    'Soil_Alternative_1':  ('hannover', 'Soil alternative, Hanover'),
    'Air_Semi-Parallel_2': ('luebeck',  'Air semi-parallel, Lübeck'),
    'Air_Alternative_2':   ('luebeck',  'Air alternative, Lübeck'),
    'Air_Parallel_2':      ('luebeck',  'Air parallel, Lübeck'),
    'Soil_Alternative_2':  ('luebeck',  'Soil alternative, Lübeck'),
}

PV_PROFILES = {
    'PV1': ('east',  'Hanover', 'Peak 9h'),
    'PV2': ('east',  'Lübeck',  'Peak 9h'),
    'PV3': ('south', 'Hanover', 'Peak 12h'),
    'PV4': ('south', 'Hanover', 'Peak 12h'),
    'PV5': ('south', 'Hanover', 'Peak 11h'),
    'PV6': ('south', 'Hanover', 'Peak 10h'),
    'PV7': ('west',  'Lübeck',  'Peak 14h'),
    'PV8': ('south', 'Lübeck',  'Peak 12h'),
}

EV_HOME_PROFILES = {
    'HLS_A_3.7':  dict(power_kw=3.7,  label='Home A 3.7kW'),
    'HLS_B_3.7':  dict(power_kw=3.7,  label='Home B 3.7kW'),
    'HLS_C_3.7':  dict(power_kw=3.7,  label='Home C 3.7kW'),
    'HLS_A_11.0': dict(power_kw=11.0, label='Home A 11kW'),
    'HLS_B_11.0': dict(power_kw=11.0, label='Home B 11kW'),
    'HLS_A_22.0': dict(power_kw=22.0, label='Home A 22kW'),
}

EV_WORK_PROFILES = {
    # APLS_A_22.0 excluded — anomalous S/W ratio (10.05)
    'APLS_A_3.7':  dict(power_kw=3.7,  label='Work A 3.7kW'),
    'APLS_B_3.7':  dict(power_kw=3.7,  label='Work B 3.7kW'),
    'APLS_A_11.0': dict(power_kw=11.0, label='Work A 11kW'),
    'APLS_B_11.0': dict(power_kw=11.0, label='Work B 11kW'),
    'APLS_A_50.0': dict(power_kw=50.0, label='Work A 50kW'),
}

# ── Prosumer type definitions ──────────────────────────────────────────────────
PROSUMER_TYPES = {
    1: dict(
        name='Residential ++',
        description='High DER residential — PV + BESS + HP + EV',
        demand_profile='H0-B',       # typical, balanced seasons S/W=0.80
        pv_profile='PV5',            # south Hanover, peak 11h, S/W=2.58
        pv_capacity_kwp=12.0,
        bess_kwh=15.0, bess_kw=7.0,
        has_hp=True,  hp_rated_kw=10.0,
        ev_profile='HLS_A_11.0',     # home 11kW
        ev_capacity_kwh=60.0, ev_type='home',
    ),
    2: dict(
        name='Residential --',
        description='Low DER residential — small EV only',
        demand_profile='H0-C',       # evening peak 21h
        pv_profile=None, pv_capacity_kwp=0.0,
        bess_kwh=0.0, bess_kw=0.0,
        has_hp=False, hp_rated_kw=0.0,
        ev_profile='HLS_B_3.7',      # small home charger
        ev_capacity_kwh=40.0, ev_type='home',
    ),
    3: dict(
        name='Residential PV only',
        description='PV + BESS, no HP or EV — self-consumption focus',
        demand_profile='H0-A',       # low seasonal S/W=0.18 → large PV surplus summer
        pv_profile='PV7',            # west Lübeck, peak 14h
        pv_capacity_kwp=8.0,
        bess_kwh=10.0, bess_kw=5.0,
        has_hp=False, hp_rated_kw=0.0,
        ev_profile=None, ev_capacity_kwh=0.0, ev_type=None,
    ),
    4: dict(
        name='Residential HP only',
        description='HP only — rural household, no generation',
        demand_profile='H0-G',       # rural, low base S/W=0.40
        pv_profile=None, pv_capacity_kwp=0.0,
        bess_kwh=0.0, bess_kw=0.0,
        has_hp=True, hp_rated_kw=8.0,
        ev_profile=None, ev_capacity_kwh=0.0, ev_type=None,
    ),
    5: dict(
        name='Restaurant',
        description='Food service — midday peak, PV on roof',
        demand_profile='G3-A',       # food service, peak 11h, S/W=1.03
        pv_profile='PV4',            # south Hanover, peak 12h
        pv_capacity_kwp=15.0,
        bess_kwh=10.0, bess_kw=5.0,
        has_hp=False, hp_rated_kw=0.0,
        ev_profile=None, ev_capacity_kwh=0.0, ev_type=None,
    ),
    6: dict(
        name='Office building',
        description='Business hours, roof PV, workplace EV charging',
        demand_profile='G4-B',       # office, peak 9h, S/W=1.56
        pv_profile='PV1',            # east Hanover, peak 9h — matches morning demand
        pv_capacity_kwp=30.0,
        bess_kwh=20.0, bess_kw=10.0,
        has_hp=False, hp_rated_kw=0.0,
        ev_profile='APLS_A_11.0',    # workplace charging, peak 6–8h
        ev_capacity_kwh=40.0, ev_type='work',
    ),
    7: dict(
        name='School',
        description='School hours, winter heavy S/W=0.39, large PV roof',
        demand_profile='G4-A',       # school, peak 8h
        pv_profile='PV5',            # south Hanover, peak 11h — midday surplus
        pv_capacity_kwp=40.0,
        bess_kwh=30.0, bess_kw=15.0,
        has_hp=False, hp_rated_kw=0.0,
        ev_profile=None, ev_capacity_kwh=0.0, ev_type=None,
    ),
    8: dict(
        name='Day-round facility',
        description='Hospital/care home — flat 24h demand, evening peak',
        demand_profile='G2-A',       # peak 18h, flat S/W=1.10
        pv_profile='PV3',            # south Hanover, midday
        pv_capacity_kwp=10.0,
        bess_kwh=25.0, bess_kw=12.0,
        has_hp=False, hp_rated_kw=0.0,
        ev_profile=None, ev_capacity_kwh=0.0, ev_type=None,
    ),
    9: dict(
        name='Agricultural farm',
        description='Farm — summer heavy S/W=1.33, large PV, big BESS',
        demand_profile='L1-A',       # farm, peak 17h
        pv_profile='PV6',            # south Hanover, large roof
        pv_capacity_kwp=50.0,
        bess_kwh=50.0, bess_kw=20.0,
        has_hp=False, hp_rated_kw=0.0,
        ev_profile=None, ev_capacity_kwh=0.0, ev_type=None,
    ),
    10: dict(
        name='Small industry',
        description='Industrial — high flat consumption, BESS peak shaving, EV fleet',
        demand_profile='G1-B',       # commercial, peak 9h
        pv_profile='PV2',            # east Lübeck
        pv_capacity_kwp=60.0,
        bess_kwh=100.0, bess_kw=40.0,
        has_hp=False, hp_rated_kw=0.0,
        ev_profile='APLS_B_11.0',    # workplace fleet
        ev_capacity_kwh=40.0, ev_type='work',
    ),
}

# ── Prosumer-to-bus assignment (types → case33bw buses) ───────────────────────
# T1 Residential ++   bus  5 — rich flexibility, close to slack
# T6 Office           bus 11 — commercial, different peak hours
# T4 Residential HP   bus 17 — purely flexible load, no generation
# T3 Residential PV   bus 32 — west PV → afternoon voltage rise at weak bus
_PROSUMERS = [
    {**PROSUMER_TYPES[1], 'bus': 5},
    {**PROSUMER_TYPES[6], 'bus': 11},
    {**PROSUMER_TYPES[4], 'bus': 17},
    {**PROSUMER_TYPES[3], 'bus': 32},
]


# ── Data loading helpers ───────────────────────────────────────────────────────

def _load_profiles():
    lp = pd.read_csv(_LP_PATH, sep=';', index_col=0)
    rp = pd.read_csv(_RP_PATH, sep=';', index_col=0)
    return lp, rp


def _dwd_temperature(station_id, year=2016):
    """Download DWD hourly temperature and upsample to 15-min."""
    base = ('https://opendata.dwd.de/climate_environment/CDC/'
            'observations_germany/climate/hourly/air_temperature/historical/')
    with urllib.request.urlopen(base) as r:
        html = r.read().decode()
    fname = re.findall(
        rf'stundenwerte_TU_{station_id:05d}_\d+_\d+_hist\.zip', html)
    if not fname:
        raise ValueError(f'DWD station {station_id} not found')
    with urllib.request.urlopen(base + fname[0]) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
        df_file = [f for f in z.namelist() if f.startswith('produkt_')][0]
        df = pd.read_csv(z.open(df_file), sep=';')
    df['datetime'] = pd.to_datetime(
        df['MESS_DATUM'].astype(str).str.strip(),
        format='%Y%m%d%H', errors='coerce')
    df['temp_C'] = pd.to_numeric(
        df['TT_TU'].astype(str).str.strip(), errors='coerce')
    df = (df[df['datetime'].dt.year == year][['datetime', 'temp_C']]
          .dropna().set_index('datetime'))
    return df.resample('15min').interpolate('linear')


# ── Main scenario builder ──────────────────────────────────────────────────────

def make_scenario(season='winter', seed=42):
    """Build a data dict from SimBench + DWD for case33bw.

    Parameters
    ----------
    season : 'winter' | 'spring' | 'summer' | 'autumn'
    seed   : random seed for HP profile selection

    Returns
    -------
    data dict compatible with prosumer.py / algo.py (T=96, dt=0.25h)
    """
    assert season in _SEASONS, f"season must be one of {list(_SEASONS)}"
    random.seed(seed)

    slc = _SEASONS[season]
    T   = 96
    dt  = 0.25

    print(f"Building {season} scenario (T={T}, dt={dt}h)...")
    lp, rp = _load_profiles()

    print("  Downloading DWD temperature...")
    temp_han = _dwd_temperature(2564)['temp_C'].values   # Hanover
    temp_lub = _dwd_temperature(2928)['temp_C'].values   # Lübeck

    # Electricity prices — 24 hourly values repeated ×4 → 96 steps
    if season == 'winter':
        _buy_h  = [0.28]*6 + [0.35]*3 + [0.22]*6 + [0.35]*4 + [0.28]*3 + [0.22]*2
        _sell_h = [0.08]*24
        _p2p_h  = [0.15]*6 + [0.20]*3 + [0.12]*6 + [0.20]*4 + [0.15]*3 + [0.12]*2
    elif season == 'summer':
        _buy_h  = [0.22]*6 + [0.28]*3 + [0.18]*6 + [0.25]*4 + [0.22]*3 + [0.20]*2
        _sell_h = [0.08]*24
        _p2p_h  = [0.12]*6 + [0.16]*3 + [0.10]*6 + [0.14]*4 + [0.12]*3 + [0.11]*2
    else:
        _buy_h  = [0.25]*24
        _sell_h = [0.08]*24
        _p2p_h  = [0.14]*24

    buy_price  = np.repeat(_buy_h,  4).tolist()
    sell_price = np.repeat(_sell_h, 4).tolist()
    p2p_price  = np.repeat(_p2p_h,  4).tolist()

    # Per-prosumer arrays — populated in the loop below
    demand     = []
    generation = []
    batt_cap, batt_pow, batt_smax, init_soc = [], [], [], []
    hp_max, hp_T_init, hp_base, hp_q        = [], [], [], []
    ev_cap, ev_pow, ev_smax, ev_cap_fac     = [], [], [], []
    ev_drive, ev_avail, ev_soc_dep, ev_init = [], [], [], []

    for p in _PROSUMERS:
        dp    = DEMAND_PROFILES[p['demand_profile']]
        scale = dp['peak_kw']
        print(f"  P{_PROSUMERS.index(p)+1}: {p['name']} (bus {p['bus']})")

        # ── Demand ───────────────────────────────────────────────────────────
        d_96 = lp[p['demand_profile'] + '_pload'].values[slc] * scale
        demand.append(d_96.tolist())

        # ── PV ───────────────────────────────────────────────────────────────
        if p['pv_profile']:
            g_96 = rp[p['pv_profile']].values[slc] * p['pv_capacity_kwp']
        else:
            g_96 = np.zeros(T)
        generation.append(g_96.tolist())

        # ── BESS ─────────────────────────────────────────────────────────────
        cap = p['bess_kwh']
        batt_cap.append(cap)
        batt_pow.append(p['bess_kw'])
        batt_smax.append(p['bess_kw'])      # converter rating = power rating
        init_soc.append(0.3 * cap)

        # ── HP ───────────────────────────────────────────────────────────────
        hp_max.append(p['hp_rated_kw'])
        hp_T_init.append(20.0)

        if p['has_hp']:
            # Random HP profile from pool — any type occurs in communities
            hp_name   = random.choice(list(HP_PROFILES.keys()))
            hp_loc    = HP_PROFILES[hp_name][0]
            hp_96     = lp[hp_name + '_pload'].values[slc] * p['hp_rated_kw']
            temp_96   = temp_han[slc] if hp_loc == 'hannover' else temp_lub[slc]
            # RC model: q_heat [°C/h] = deficit / (R×C), R=8°C/kW, C=50kWh/°C
            q_96      = (np.maximum(0.0, 20.0 - temp_96) / (8.0 * 50.0)).tolist()
            hp_base.append(hp_96.tolist())
            hp_q.append(q_96)
            print(f"    HP: {hp_name} ({HP_PROFILES[hp_name][1]})")
        else:
            hp_base.append([0.0] * T)
            hp_q.append([0.0] * T)

        # ── EV ───────────────────────────────────────────────────────────────
        cap_eff = p['ev_capacity_kwh'] * (0.80 if season == 'winter' else 1.00)
        ev_cap.append(p['ev_capacity_kwh'])
        if p.get('ev_profile'):
            prof = p['ev_profile']
            if prof in EV_HOME_PROFILES:
                ev_kw = EV_HOME_PROFILES[prof]['power_kw']
            elif prof in EV_WORK_PROFILES:
                ev_kw = EV_WORK_PROFILES[prof]['power_kw']
            else:
                ev_kw = float(prof.split('_')[2])  # fallback: extract from name
        else:
            ev_kw = 0.0
        ev_pow.append(ev_kw)
        ev_smax.append(ev_kw)
        ev_cap_fac.append(0.80 if season == 'winter' else 1.00)
        ev_init.append(0.90 * cap_eff)   # 90% — realistic starting charge

        if p.get('ev_profile'):
            ev_col  = p['ev_profile'] + '_pload'
            ev_96   = lp[ev_col].values[slc]
            avail = (ev_96 > 0.001).astype(int).tolist()
            
            # Fallback if no activity detected on this day
            if sum(avail) == 0:
                print(f"    ⚠ No EV activity detected — using business hours fallback")
                if p['ev_type'] == 'work':
                    # Plugged in at workplace 8h–18h = steps 32–72
                    avail = [0]*32 + [1]*40 + [0]*24
                else:
                    # Home: plugged in overnight 0–7h and 18–24h
                    avail = [1]*28 + [0]*40 + [1]*28
                    
            away    = [i for i, a in enumerate(avail) if a == 0]
            # Driving energy: 35km/day × seasonal consumption rate
            consump = 0.17 if season == 'winter' else 0.15  # kWh/km
            daily_e = 35.0 * consump
            driving = [0.0] * T
            if away:
                per_step = daily_e / len(away)
                for i in away:
                    driving[i] = per_step
            # Departure SOC: enough energy for next trip
            trip_kwh  = daily_e
            soc_dep   = min(trip_kwh / cap_eff, 0.8) if cap_eff > 0 else 0.0
            ev_avail.append(avail)
            ev_drive.append(driving)
            ev_soc_dep.append(soc_dep)
        else:
            ev_avail.append([1] * T)
            ev_drive.append([0.0] * T)
            ev_soc_dep.append(0.0)

    # ── Assemble data dict ────────────────────────────────────────────────────
    N          = len(_PROSUMERS)
    max_demand = max(max(d) for d in demand)

    data = dict(
        # Grid / market
        demand        = demand,
        generation    = generation,
        edges         = [(0,1),(1,2),(2,3),(0,3)],
        agent_bus     = [p['bus'] for p in _PROSUMERS],
        buy_price     = buy_price,
        sell_price    = sell_price,
        p2p_price     = p2p_price,
        trade_fee     = 0.001,
        dt            = dt,
        grid_limit    = 10.0 * max_demand,
        trade_limit   = 5.0  * max_demand,
        line_limit_mva= 5.0,
        # BESS
        battery_capacity      = batt_cap,
        battery_power         = batt_pow,
        battery_smax          = batt_smax,
        battery_polygon_sides = 8,
        soc_min_fraction      = 0.10,
        soc_max_fraction      = 0.95,
        initial_soc           = init_soc,
        eta_charge            = 0.95,
        eta_discharge         = 0.95,
        battery_cost          = 0.005,
        # PV reactive
        pf_pv_min             = 0.90,    # REVIEW: EN 50549-1 / IEEE 1547
        # HP
        hp_max_power          = hp_max,
        hp_base_power         = hp_base,
        hp_q_heat             = hp_q,
        hp_T_init             = hp_T_init,
        hp_T_min              = 19.0,
        hp_T_max              = 23.0,
        hp_cost               = 0.0,
        pf_hp                 = 0.97,    # REVIEW: manufacturer data
        # EV
        ev_capacity           = ev_cap,
        ev_cap_factor         = ev_cap_fac,
        ev_power              = ev_pow,
        ev_smax               = ev_smax,
        ev_polygon_sides      = 8,
        ev_driving_load       = ev_drive,
        ev_soc_min_departure  = ev_soc_dep,
        ev_dod_max            = 0.30,    # ref [16]
        ev_initial_soc        = ev_init,
        ev_availability       = ev_avail,
        ev_eta_charge         = 0.95,
        ev_eta_discharge      = 0.95,
        ev_cost               = 0.005,
        ev_v2g                = True,
        # Load reactive
        pf_load               = 0.95,    # REVIEW: BDEW standard load profiles
        # Q regularisation
        q_cost                = 1e-4,
        # Metadata
        season                = season,
        prosumer_names        = [p['name'] for p in _PROSUMERS],
    )

    print(f"  Done. N={N}, T={T}, max_demand={max_demand:.1f} kW")
    return data


# ── Summary table ──────────────────────────────────────────────────────────────

def prosumer_summary():
    rows = []
    for i, p in enumerate(_PROSUMERS):
        dp = DEMAND_PROFILES[p['demand_profile']]
        rows.append({
            'Prosumer':      f"P{i+1} (bus {p['bus']})",
            'Type':          p['name'],
            'Demand':        p['demand_profile'],
            'Peak kW':       dp['peak_kw'],
            'PV':            f"{p['pv_profile']} {p['pv_capacity_kwp']}kWp"
                             if p['pv_profile'] else '—',
            'BESS':          f"{p['bess_kwh']}kWh/{p['bess_kw']}kW"
                             if p['bess_kwh'] > 0 else '—',
            'HP':            f"random {p['hp_rated_kw']}kW"
                             if p['has_hp'] else '—',
            'EV':            f"{p['ev_profile']} {p['ev_capacity_kwh']}kWh"
                             if p.get('ev_profile') else '—',
        })
    return pd.DataFrame(rows).set_index('Prosumer')


if __name__ == '__main__':
    print(prosumer_summary().to_string())
    data = make_scenario('winter')