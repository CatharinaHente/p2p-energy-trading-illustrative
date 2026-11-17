# ── Prosumer type definitions ──────────────────────────────────────────────────

# ── Scenario 4: original prosumer configuration ────────────────────────────────

_PROSUMERS_4 = [
    dict(
        name='Residential ++',
        description='High DER residential — PV + BESS + HP + EV',
        demand_profile='H0-B', pv_profile='PV5', pv_capacity_kwp=12.0,
        bess_kwh=15.0, bess_kw=7.0,
        has_hp=True, hp_rated_kw=10.0,
        ev_profile='HLS_A_11.0', ev_capacity_kwh=60.0, ev_type='home',
        bus=5,
    ),
    dict(
        name='Office building',
        description='Business hours, roof PV, workplace EV charging',
        demand_profile='G4-B', pv_profile='PV1', pv_capacity_kwp=30.0,
        bess_kwh=20.0, bess_kw=10.0,
        has_hp=False, hp_rated_kw=0.0,
        ev_profile='APLS_A_11.0', ev_capacity_kwh=40.0, ev_type='work',
        bus=11,
    ),
    dict(
        name='Residential HP only',
        description='HP only — rural household, no generation',
        demand_profile='H0-G', pv_profile=None, pv_capacity_kwp=0.0,
        bess_kwh=0.0, bess_kw=0.0,
        has_hp=True, hp_rated_kw=8.0,
        ev_profile=None, ev_capacity_kwh=0.0, ev_type=None,
        bus=17,
    ),
    dict(
        name='Residential PV only',
        description='PV + BESS, no HP or EV — self-consumption focus',
        demand_profile='H0-A', pv_profile='PV7', pv_capacity_kwp=8.0,
        bess_kwh=10.0, bess_kw=5.0,
        has_hp=False, hp_rated_kw=0.0,
        ev_profile=None, ev_capacity_kwh=0.0, ev_type=None,
        bus=32,
    ),
]


# ── Scenario 32: full prosumer population ──────────────────────────────────────

_PROSUMERS_32 = [

    # ── Residential ++ (7 prosumers) ─────────────────────────────────────────

    dict(
        name='Residential ++',
        description='PV + BESS + HP + EV',
        demand_profile='H0-B', pv_profile='PV5', pv_capacity_kwp=12.0,
        bess_kwh=15.0, bess_kw=7.0,
        has_hp=True, hp_rated_kw=10.0,
        ev_profile='HLS_A_11.0', ev_capacity_kwh=60.0, ev_type='home',
        bus=32,
    ),
    dict(
        name='Residential ++',
        description='PV + BESS + HP + EV',
        demand_profile='H0-A', pv_profile='PV7', pv_capacity_kwp=10.0,
        bess_kwh=12.0, bess_kw=6.0,
        has_hp=True, hp_rated_kw=9.0,
        ev_profile='HLS_A_11.0', ev_capacity_kwh=50.0, ev_type='home',
        bus=20,
    ),
    dict(
        name='Residential ++',
        description='PV + BESS + HP + EV',
        demand_profile='H0-C', pv_profile='PV3', pv_capacity_kwp=8.0,
        bess_kwh=15.0, bess_kw=7.0,
        has_hp=True, hp_rated_kw=10.0,
        ev_profile='HLS_B_11.0', ev_capacity_kwh=40.0, ev_type='home',
        bus=8,
    ),
    dict(
        name='Residential ++',
        description='PV + BESS + HP + EV',
        demand_profile='H0-B', pv_profile='PV5', pv_capacity_kwp=14.0,
        bess_kwh=18.0, bess_kw=8.0,
        has_hp=True, hp_rated_kw=8.0,
        ev_profile='HLS_A_11.0', ev_capacity_kwh=60.0, ev_type='home',
        bus=28,
    ),
    dict(
        name='Residential ++',
        description='PV + BESS + HP + EV',
        demand_profile='H0-G', pv_profile='PV4', pv_capacity_kwp=10.0,
        bess_kwh=12.0, bess_kw=6.0,
        has_hp=True, hp_rated_kw=10.0,
        ev_profile='HLS_A_11.0', ev_capacity_kwh=60.0, ev_type='home',
        bus=27,
    ),
    dict(
        name='Residential ++',
        description='PV + BESS + HP + EV',
        demand_profile='H0-A', pv_profile='PV6', pv_capacity_kwp=12.0,
        bess_kwh=15.0, bess_kw=7.0,
        has_hp=True, hp_rated_kw=9.0,
        ev_profile='HLS_B_11.0', ev_capacity_kwh=60.0, ev_type='home',
        bus=19,
    ),
    dict(
        name='Residential ++',
        description='PV + BESS + HP + EV',
        demand_profile='H0-B', pv_profile='PV7', pv_capacity_kwp=15.0,
        bess_kwh=20.0, bess_kw=8.0,
        has_hp=True, hp_rated_kw=10.0,
        ev_profile='HLS_A_11.0', ev_capacity_kwh=40.0, ev_type='home',
        bus=6,
    ),

    # ── Residential -- (5 prosumers) ─────────────────────────────────────────

    dict(
        name='Residential --',
        description='EV only',
        demand_profile='H0-C', pv_profile=None, pv_capacity_kwp=0.0,
        bess_kwh=0.0, bess_kw=0.0,
        has_hp=False, hp_rated_kw=0.0,
        ev_profile='HLS_B_3.7', ev_capacity_kwh=40.0, ev_type='home',
        bus=23,
    ),
    dict(
        name='Residential --',
        description='EV only',
        demand_profile='H0-B', pv_profile=None, pv_capacity_kwp=0.0,
        bess_kwh=0.0, bess_kw=0.0,
        has_hp=False, hp_rated_kw=0.0,
        ev_profile='HLS_A_3.7', ev_capacity_kwh=30.0, ev_type='home',
        bus=29,
    ),
    dict(
        name='Residential --',
        description='EV + small BESS',
        demand_profile='H0-A', pv_profile=None, pv_capacity_kwp=0.0,
        bess_kwh=5.0, bess_kw=3.0,
        has_hp=False, hp_rated_kw=0.0,
        ev_profile='HLS_B_3.7', ev_capacity_kwh=40.0, ev_type='home',
        bus=11,
    ),
    dict(
        name='Residential --',
        description='EV only',
        demand_profile='H0-C', pv_profile=None, pv_capacity_kwp=0.0,
        bess_kwh=0.0, bess_kw=0.0,
        has_hp=False, hp_rated_kw=0.0,
        ev_profile='HLS_B_3.7', ev_capacity_kwh=50.0, ev_type='home',
        bus=25,
    ),
    dict(
        name='Residential --',
        description='EV only',
        demand_profile='H0-G', pv_profile=None, pv_capacity_kwp=0.0,
        bess_kwh=0.0, bess_kw=0.0,
        has_hp=False, hp_rated_kw=0.0,
        ev_profile='HLS_A_3.7', ev_capacity_kwh=60.0, ev_type='home',
        bus=24,
    ),

    # ── Residential PV only (6 prosumers) ────────────────────────────────────

    dict(
        name='Residential PV only',
        description='PV + BESS',
        demand_profile='H0-A', pv_profile='PV7', pv_capacity_kwp=8.0,
        bess_kwh=10.0, bess_kw=5.0,
        has_hp=False, hp_rated_kw=0.0,
        ev_profile=None, ev_capacity_kwh=0.0, ev_type=None,
        bus=21,
    ),
    dict(
        name='Residential PV only',
        description='PV + BESS',
        demand_profile='H0-B', pv_profile='PV3', pv_capacity_kwp=6.0,
        bess_kwh=8.0, bess_kw=4.0,
        has_hp=False, hp_rated_kw=0.0,
        ev_profile=None, ev_capacity_kwh=0.0, ev_type=None,
        bus=10,
    ),
    dict(
        name='Residential PV only',
        description='PV + BESS',
        demand_profile='H0-A', pv_profile='PV5', pv_capacity_kwp=10.0,
        bess_kwh=12.0, bess_kw=6.0,
        has_hp=False, hp_rated_kw=0.0,
        ev_profile=None, ev_capacity_kwh=0.0, ev_type=None,
        bus=7,
    ),
    dict(
        name='Residential PV only',
        description='PV + BESS',
        demand_profile='H0-G', pv_profile='PV7', pv_capacity_kwp=8.0,
        bess_kwh=10.0, bess_kw=5.0,
        has_hp=False, hp_rated_kw=0.0,
        ev_profile=None, ev_capacity_kwh=0.0, ev_type=None,
        bus=17,
    ),
    dict(
        name='Residential PV only',
        description='PV + BESS',
        demand_profile='H0-C', pv_profile='PV4', pv_capacity_kwp=7.0,
        bess_kwh=8.0, bess_kw=4.0,
        has_hp=False, hp_rated_kw=0.0,
        ev_profile=None, ev_capacity_kwh=0.0, ev_type=None,
        bus=4,
    ),
    dict(
        name='Residential PV only',
        description='PV + BESS',
        demand_profile='H0-B', pv_profile='PV6', pv_capacity_kwp=12.0,
        bess_kwh=15.0, bess_kw=7.0,
        has_hp=False, hp_rated_kw=0.0,
        ev_profile=None, ev_capacity_kwh=0.0, ev_type=None,
        bus=1,
    ),

    # ── Residential HP only (4 prosumers) ────────────────────────────────────

    dict(
        name='Residential HP only',
        description='HP only, no generation',
        demand_profile='H0-G', pv_profile=None, pv_capacity_kwp=0.0,
        bess_kwh=0.0, bess_kw=0.0,
        has_hp=True, hp_rated_kw=8.0,
        ev_profile=None, ev_capacity_kwh=0.0, ev_type=None,
        bus=16,
    ),
    dict(
        name='Residential HP only',
        description='HP only, no generation',
        demand_profile='H0-B', pv_profile=None, pv_capacity_kwp=0.0,
        bess_kwh=0.0, bess_kw=0.0,
        has_hp=True, hp_rated_kw=6.0,
        ev_profile=None, ev_capacity_kwh=0.0, ev_type=None,
        bus=18,
    ),
    dict(
        name='Residential HP only',
        description='HP only, no generation',
        demand_profile='H0-A', pv_profile=None, pv_capacity_kwp=0.0,
        bess_kwh=0.0, bess_kw=0.0,
        has_hp=True, hp_rated_kw=10.0,
        ev_profile=None, ev_capacity_kwh=0.0, ev_type=None,
        bus=26,
    ),
    dict(
        name='Residential HP only',
        description='HP only, no generation',
        demand_profile='H0-C', pv_profile=None, pv_capacity_kwp=0.0,
        bess_kwh=0.0, bess_kw=0.0,
        has_hp=True, hp_rated_kw=7.0,
        ev_profile=None, ev_capacity_kwh=0.0, ev_type=None,
        bus=13,
    ),

    # ── Residential nothing (4 prosumers) ────────────────────────────────────

    dict(
        name='Residential nothing',
        description='Bare residential demand',
        demand_profile='H0-A', pv_profile=None, pv_capacity_kwp=0.0,
        bess_kwh=0.0, bess_kw=0.0,
        has_hp=False, hp_rated_kw=0.0,
        ev_profile=None, ev_capacity_kwh=0.0, ev_type=None,
        bus=22,
    ),
    dict(
        name='Residential nothing',
        description='Bare residential demand',
        demand_profile='H0-B', pv_profile=None, pv_capacity_kwp=0.0,
        bess_kwh=0.0, bess_kw=0.0,
        has_hp=False, hp_rated_kw=0.0,
        ev_profile=None, ev_capacity_kwh=0.0, ev_type=None,
        bus=12,
    ),
    dict(
        name='Residential nothing',
        description='Bare residential demand',
        demand_profile='H0-C', pv_profile=None, pv_capacity_kwp=0.0,
        bess_kwh=0.0, bess_kw=0.0,
        has_hp=False, hp_rated_kw=0.0,
        ev_profile=None, ev_capacity_kwh=0.0, ev_type=None,
        bus=15,
    ),
    dict(
        name='Residential nothing',
        description='Bare residential demand',
        demand_profile='H0-G', pv_profile=None, pv_capacity_kwp=0.0,
        bess_kwh=0.0, bess_kw=0.0,
        has_hp=False, hp_rated_kw=0.0,
        ev_profile=None, ev_capacity_kwh=0.0, ev_type=None,
        bus=31,
    ),

    # ── Commercial (6 prosumers) ─────────────────────────────────────────────

    dict(
        name='Day-round facility',
        description='Hospital/care home — day-round demand',
        demand_profile='G2-A', pv_profile='PV3', pv_capacity_kwp=10.0,
        bess_kwh=25.0, bess_kw=12.0,
        has_hp=False, hp_rated_kw=0.0,
        ev_profile=None, ev_capacity_kwh=0.0, ev_type=None,
        bus=3,
    ),
    dict(
        name='Day-round facility (smaller)',
        description='Smaller day-round facility',
        demand_profile='G2-A', pv_profile='PV3', pv_capacity_kwp=8.0,
        bess_kwh=20.0, bess_kw=10.0,
        has_hp=False, hp_rated_kw=0.0,
        ev_profile=None, ev_capacity_kwh=0.0, ev_type=None,
        bus=5,
    ),
    dict(
        name='Restaurant large',
        description='Food service — large PV system',
        demand_profile='G3-A', pv_profile='PV4', pv_capacity_kwp=15.0,
        bess_kwh=10.0, bess_kw=5.0,
        has_hp=False, hp_rated_kw=0.0,
        ev_profile=None, ev_capacity_kwh=0.0, ev_type=None,
        bus=30,
    ),
    dict(
        name='Restaurant medium',
        description='Food service — medium PV system',
        demand_profile='G3-A', pv_profile='PV4', pv_capacity_kwp=10.0,
        bess_kwh=8.0, bess_kw=4.0,
        has_hp=False, hp_rated_kw=0.0,
        ev_profile=None, ev_capacity_kwh=0.0, ev_type=None,
        bus=2,
    ),
    dict(
        name='Restaurant small',
        description='Food service — small PV system',
        demand_profile='G3-A', pv_profile='PV3', pv_capacity_kwp=8.0,
        bess_kwh=6.0, bess_kw=3.0,
        has_hp=False, hp_rated_kw=0.0,
        ev_profile=None, ev_capacity_kwh=0.0, ev_type=None,
        bus=14,
    ),
    dict(
        name='Small office',
        description='Business hours, roof PV, no EV',
        demand_profile='G4-B', pv_profile='PV1', pv_capacity_kwp=15.0,
        bess_kwh=10.0, bess_kw=5.0,
        has_hp=False, hp_rated_kw=0.0,
        ev_profile=None, ev_capacity_kwh=0.0, ev_type=None,
        bus=9,
    ),
]


# ── Bus lookup for intermediate scenarios ──────────────────────────────────────

_BY_BUS = {p['bus']: p for p in _PROSUMERS_32}


# ── Scenario 8: original 4 + 4 additional prosumers ────────────────────────────

_PROSUMERS_8 = _PROSUMERS_4 + [
    _BY_BUS[20],  # Residential ++
    _BY_BUS[23],  # Residential --
    _BY_BUS[21],  # Residential PV only
    _BY_BUS[30],  # Restaurant large
]


# ── Scenario 16: scenario 8 + 8 additional prosumers ───────────────────────────

_PROSUMERS_16 = _PROSUMERS_8 + [
    _BY_BUS[8],   # Residential ++
    _BY_BUS[28],  # Residential ++
    _BY_BUS[29],  # Residential --
    _BY_BUS[10],  # Residential PV only
    _BY_BUS[16],  # Residential HP only
    _BY_BUS[22],  # Residential nothing
    _BY_BUS[3],   # Day-round facility
    _BY_BUS[9],   # Small office
]


# ── Scenario registry ─────────────────────────────────────────────────────────

_PROSUMER_SCENARIOS = {
    4:  _PROSUMERS_4,
    8:  _PROSUMERS_8,
    16: _PROSUMERS_16,
    32: _PROSUMERS_32,
}