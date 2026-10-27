"""scenarios.py — Extract SimBench + DWD prosumer profiles for case33bw.

Builds a data dict compatible with main.py / algo.py for 4 heterogeneous
prosumers placed at buses 5, 11, 17, 32 of case33bw.

Usage
-----
from scenarios import make_scenario, prosumer_summary
data = make_scenario(season='winter', resolution_h=0.5)  # 30-min, T=48

Resolution options
------------------
0.25h → 15-min (T=96)  full resolution, centralized only
0.5h  → 30-min (T=48)  recommended for distributed
1.0h  → 1-hour (T=24)  fastest, coarser dynamics

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

# ── Season slices (index into 35136-step 2016 year at 15-min) ─────────────────
_SEASONS = {
    'winter': slice(9*96,  10*96),                                  # Jan 10
    'spring': slice((31+29+31+9)*96,   (31+29+31+10)*96),          # Apr 10
    'summer': slice((31+29+31+30+31+30+9)*96,
                    (31+29+31+30+31+30+10)*96),                     # Jul 10
    'autumn': slice((31+29+31+30+31+30+31+31+30+9)*96,
                    (31+29+31+30+31+30+31+31+30+10)*96),            # Oct 10
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
        demand_profile='H0-B', pv_profile='PV5', pv_capacity_kwp=12.0,
        bess_kwh=15.0, bess_kw=7.0,
        has_hp=True, hp_rated_kw=10.0,
        ev_profile='HLS_A_11.0', ev_capacity_kwh=60.0, ev_type='home',
    ),
    2: dict(
        name='Residential --',
        description='Low DER residential — small EV only',
        demand_profile='H0-C', pv_profile=None, pv_capacity_kwp=0.0,
        bess_kwh=0.0, bess_kw=0.0,
        has_hp=False, hp_rated_kw=0.0,
        ev_profile='HLS_B_3.7', ev_capacity_kwh=40.0, ev_type='home',
    ),
    3: dict(
        name='Residential PV only',
        description='PV + BESS, no HP or EV — self-consumption focus',
        demand_profile='H0-A', pv_profile='PV7', pv_capacity_kwp=8.0,
        bess_kwh=10.0, bess_kw=5.0,
        has_hp=False, hp_rated_kw=0.0,
        ev_profile=None, ev_capacity_kwh=0.0, ev_type=None,
    ),
    4: dict(
        name='Residential HP only',
        description='HP only — rural household, no generation',
        demand_profile='H0-G', pv_profile=None, pv_capacity_kwp=0.0,
        bess_kwh=0.0, bess_kw=0.0,
        has_hp=True, hp_rated_kw=8.0,
        ev_profile=None, ev_capacity_kwh=0.0, ev_type=None,
    ),
    5: dict(
        name='Restaurant',
        description='Food service — midday peak, PV on roof',
        demand_profile='G3-A', pv_profile='PV4', pv_capacity_kwp=15.0,
        bess_kwh=10.0, bess_kw=5.0,
        has_hp=False, hp_rated_kw=0.0,
        ev_profile=None, ev_capacity_kwh=0.0, ev_type=None,
    ),
    6: dict(
        name='Office building',
        description='Business hours, roof PV, workplace EV charging',
        demand_profile='G4-B', pv_profile='PV1', pv_capacity_kwp=30.0,
        bess_kwh=20.0, bess_kw=10.0,
        has_hp=False, hp_rated_kw=0.0,
        ev_profile='APLS_A_11.0', ev_capacity_kwh=40.0, ev_type='work',
    ),
    7: dict(
        name='School',
        description='School hours, winter heavy S/W=0.39, large PV roof',
        demand_profile='G4-A', pv_profile='PV5', pv_capacity_kwp=40.0,
        bess_kwh=30.0, bess_kw=15.0,
        has_hp=False, hp_rated_kw=0.0,
        ev_profile=None, ev_capacity_kwh=0.0, ev_type=None,
    ),
    8: dict(
        name='Day-round facility',
        description='Hospital/care home — flat 24h demand, evening peak',
        demand_profile='G2-A', pv_profile='PV3', pv_capacity_kwp=10.0,
        bess_kwh=25.0, bess_kw=12.0,
        has_hp=False, hp_rated_kw=0.0,
        ev_profile=None, ev_capacity_kwh=0.0, ev_type=None,
    ),
    9: dict(
        name='Agricultural farm',
        description='Farm — summer heavy S/W=1.33, large PV, big BESS',
        demand_profile='L1-A', pv_profile='PV6', pv_capacity_kwp=50.0,
        bess_kwh=50.0, bess_kw=20.0,
        has_hp=False, hp_rated_kw=0.0,
        ev_profile=None, ev_capacity_kwh=0.0, ev_type=None,
    ),
    10: dict(
        name='Small industry',
        description='Industrial — high flat consumption, BESS peak shaving, EV fleet',
        demand_profile='G1-B', pv_profile='PV2', pv_capacity_kwp=60.0,
        bess_kwh=100.0, bess_kw=40.0,
        has_hp=False, hp_rated_kw=0.0,
        ev_profile='APLS_B_11.0', ev_capacity_kwh=40.0, ev_type='work',
    ),
}

# ── Prosumer-to-bus assignment ─────────────────────────────────────────────────
_PROSUMERS = [
    {**PROSUMER_TYPES[1], 'bus': 5},   # Residential ++ — rich flexibility
    {**PROSUMER_TYPES[6], 'bus': 11},  # Office          — commercial, different peak
    {**PROSUMER_TYPES[4], 'bus': 17},  # Residential HP  — flexible load, no generation
    {**PROSUMER_TYPES[3], 'bus': 32},  # Residential PV  — west PV, weak bus
]


# ── Data loading helpers ───────────────────────────────────────────────────────

def _load_profiles():
    lp = pd.read_csv(_LP_PATH, sep=';', index_col=0)
    rp = pd.read_csv(_RP_PATH, sep=';', index_col=0)
    return lp, rp


def _dwd_temperature(station_id, year=2016):
    """Download DWD hourly temperature and upsample to 15-min (35136 steps)."""
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
        df['MESS_DATUM'].astype(str).str.strip(), format='%Y%m%d%H',
        errors='coerce')
    df['temp_C'] = pd.to_numeric(
        df['TT_TU'].astype(str).str.strip(), errors='coerce')
    df = (df[df['datetime'].dt.year == year][['datetime', 'temp_C']]
          .dropna().set_index('datetime'))
    return df.resample('15min').interpolate('linear')


# ── Main scenario builder ──────────────────────────────────────────────────────

def make_scenario(season='winter', seed=42, resolution_h=0.5):
    """Build a data dict from SimBench + DWD for case33bw.

    Parameters
    ----------
    season       : 'winter' | 'spring' | 'summer' | 'autumn'
    seed         : random seed for HP profile selection
    resolution_h : timestep size in hours
                   0.25 → 15-min T=96  (centralized only, distributed too slow)
                   0.5  → 30-min T=48  (recommended — distributed terminates)
                   1.0  → 1-hour T=24  (fastest, coarser EV/BESS dynamics)

    Returns
    -------
    data dict compatible with prosumer.py / algo.py
    """
    assert season in _SEASONS, f"season must be one of {list(_SEASONS)}"
    assert resolution_h in (0.25, 0.5, 1.0), \
        "resolution_h must be 0.25, 0.5 or 1.0"

    random.seed(seed)

    # Resampling factor relative to 15-min base
    factor = int(round(resolution_h / 0.25))   # 1, 2, or 4
    T      = 96 // factor
    dt     = resolution_h
    slc    = _SEASONS[season]

    # ── Resampling helpers (operate on 96-step 15-min day slices) ─────────────
    def _mean(arr):
        """Average blocks of `factor` steps → T values. For power profiles."""
        return np.array(arr).reshape(T, factor).mean(axis=1).tolist()

    def _sum(arr):
        """Sum blocks of `factor` steps → T values. For energy quantities."""
        return np.array(arr).reshape(T, factor).sum(axis=1).tolist()

    def _avail(arr):
        """EV availability: 1 if any step in block is plugged in."""
        return (np.array(arr).reshape(T, factor).max(axis=1) > 0.001
                ).astype(int).tolist()

    print(f"Building {season} scenario (T={T}, dt={dt}h, "
          f"resolution={int(resolution_h*60)}min)...")
    lp, rp = _load_profiles()

    print("  Downloading DWD temperature...")
    temp_han = _dwd_temperature(2564)['temp_C'].values   # 35136 × 15-min
    temp_lub = _dwd_temperature(2928)['temp_C'].values

    # ── Electricity prices — defined hourly, repeated to resolution ────────────
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

    steps_per_hour = 4 // factor   # 4 for 15-min, 2 for 30-min, 1 for hourly
    buy_price  = np.repeat(_buy_h,  steps_per_hour).tolist()
    sell_price = np.repeat(_sell_h, steps_per_hour).tolist()
    p2p_price  = np.repeat(_p2p_h,  steps_per_hour).tolist()

    # ── Per-prosumer arrays ───────────────────────────────────────────────────
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
        demand.append(_mean(d_96))

        # ── PV ───────────────────────────────────────────────────────────────
        if p['pv_profile']:
            g_96 = rp[p['pv_profile']].values[slc] * p['pv_capacity_kwp']
        else:
            g_96 = np.zeros(96)
        generation.append(_mean(g_96))

        # ── BESS ─────────────────────────────────────────────────────────────
        cap = p['bess_kwh']
        batt_cap.append(cap)
        batt_pow.append(p['bess_kw'])
        batt_smax.append(p['bess_kw'])
        init_soc.append(0.3 * cap)

        # ── HP ───────────────────────────────────────────────────────────────
        hp_max.append(p['hp_rated_kw'])
        hp_T_init.append(20.0)

        if p['has_hp']:
            hp_name = random.choice(list(HP_PROFILES.keys()))
            hp_loc  = HP_PROFILES[hp_name][0]
            hp_96   = lp[hp_name + '_pload'].values[slc] * p['hp_rated_kw']
            temp_96 = temp_han[slc] if hp_loc == 'hannover' else temp_lub[slc]
            q_96    = np.maximum(0.0, 20.0 - temp_96) / (8.0 * 50.0)
            hp_base.append(_mean(hp_96))
            hp_q.append(_mean(q_96))
            print(f"    HP: {hp_name} ({HP_PROFILES[hp_name][1]})")
        else:
            hp_base.append([0.0] * T)
            hp_q.append([0.0] * T)

        # ── EV ───────────────────────────────────────────────────────────────
        cap_eff = p['ev_capacity_kwh'] * (0.80 if season == 'winter' else 1.00)
        ev_cap.append(p['ev_capacity_kwh'])
        if p.get('ev_profile'):
            prof = p['ev_profile']
            all_ev = {**EV_HOME_PROFILES, **EV_WORK_PROFILES}
            ev_kw  = all_ev[prof]['power_kw'] if prof in all_ev else float(prof.split('_')[2])
        else:
            ev_kw  = 0.0
        ev_pow.append(ev_kw)
        ev_smax.append(ev_kw)
        ev_cap_fac.append(0.80 if season == 'winter' else 1.00)
        ev_init.append(0.90 * cap_eff)

        if p.get('ev_profile'):
            ev_col   = p['ev_profile'] + '_pload'
            ev_96    = lp[ev_col].values[slc]

            # Fallback if no EV activity detected on this day
            avail_96 = (ev_96 > 0.001).astype(int)
            if avail_96.sum() == 0:
                print(f"    ⚠ No EV activity detected — using fallback schedule")
                if p['ev_type'] == 'work':
                    avail_96 = np.array([0]*32 + [1]*40 + [0]*24)
                else:
                    avail_96 = np.array([1]*28 + [0]*40 + [1]*28)

            avail = _avail(avail_96)

            # Driving load: 35km/day × seasonal kWh/km distributed over away steps
            consump  = 0.17 if season == 'winter' else 0.15
            daily_e  = 35.0 * consump
            away_96  = np.where(avail_96 == 0)[0]
            drive_96 = np.zeros(96)
            if len(away_96) > 0:
                drive_96[away_96] = daily_e / len(away_96)

            # Resample driving load: sum energy within each block
            ev_drive.append(_sum(drive_96))
            ev_avail.append(avail)

            trip_kwh = daily_e
            ev_soc_dep.append(min(trip_kwh / cap_eff, 0.8) if cap_eff > 0 else 0.0)
        else:
            ev_avail.append([1] * T)
            ev_drive.append([0.0] * T)
            ev_soc_dep.append(0.0)

    # ── Assemble data dict ────────────────────────────────────────────────────
    N          = len(_PROSUMERS)
    max_demand = max(max(d) for d in demand)

    data = dict(
        demand        = demand,
        generation    = generation,
        edges         = [(0,1),(1,2),(2,3), (3,0)], 
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
        pf_pv_min             = 0.90,
        # HP
        hp_max_power          = hp_max,
        hp_base_power         = hp_base,
        hp_q_heat             = hp_q,
        hp_T_init             = hp_T_init,
        hp_T_min              = 19.0,
        hp_T_max              = 23.0,
        hp_cost               = 0.0,
        pf_hp                 = 0.97,
        # EV
        ev_capacity           = ev_cap,
        ev_cap_factor         = ev_cap_fac,
        ev_power              = ev_pow,
        ev_smax               = ev_smax,
        ev_polygon_sides      = 8,
        ev_driving_load       = ev_drive,
        ev_soc_min_departure  = ev_soc_dep,
        ev_dod_max            = 0.30,
        ev_initial_soc        = ev_init,
        ev_availability       = ev_avail,
        ev_eta_charge         = 0.95,
        ev_eta_discharge      = 0.95,
        ev_cost               = 0.005,
        ev_v2g                = True,
        # Load reactive
        pf_load               = 0.95,
        # Regularisation
        q_cost                = 1e-3,   # EUR/kVAR²·h — breaks Q degeneracy
        p_cost                = 1e-3,   # EUR/kW²·h  — breaks P degeneracy
        # Metadata
        season                = season,
        resolution_h          = resolution_h,
        prosumer_names        = [p['name'] for p in _PROSUMERS],
    )

    print(f"  Done. N={N}, T={T}, dt={dt}h, max_demand={max_demand:.1f} kW")
    return data


# ── Summary table ──────────────────────────────────────────────────────────────

def prosumer_summary():
    rows = []
    for i, p in enumerate(_PROSUMERS):
        dp = DEMAND_PROFILES[p['demand_profile']]
        rows.append({
            'Prosumer': f"P{i+1} (bus {p['bus']})",
            'Type':     p['name'],
            'Demand':   p['demand_profile'],
            'Peak kW':  dp['peak_kw'],
            'PV':       f"{p['pv_profile']} {p['pv_capacity_kwp']}kWp"
                        if p['pv_profile'] else '—',
            'BESS':     f"{p['bess_kwh']}kWh/{p['bess_kw']}kW"
                        if p['bess_kwh'] > 0 else '—',
            'HP':       f"random {p['hp_rated_kw']}kW"
                        if p['has_hp'] else '—',
            'EV':       f"{p['ev_profile']} {p['ev_capacity_kwh']}kWh"
                        if p.get('ev_profile') else '—',
        })
    return pd.DataFrame(rows).set_index('Prosumer')


if __name__ == '__main__':
    print(prosumer_summary().to_string())
    data = make_scenario('winter', resolution_h=0.5)