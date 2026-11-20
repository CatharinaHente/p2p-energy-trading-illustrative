# config.py

# ── Physical network ─────────────────────────────────────────────────────────

NETWORK_TOPOLOGY = 'ieee33'

SIMBENCH_CODES = {
    'lv1': '1-LV-rural1--0-no_sw',
    'lv2': '1-LV-rural2--0-no_sw',
    'lv3': '1-LV-rural3--0-no_sw',
    'lv4': '1-LV-semiurb4--0-no_sw',
    'lv5': '1-LV-semiurb5--0-no_sw',
    'lv6': '1-LV-urban6--0-no_sw',
}

# ── Prosumer population ──────────────────────────────────────────────────────

PROSUMER_SCENARIO = 32

# ── Trading graph ────────────────────────────────────────────────────────────

TRADING_TOPOLOGY = 'cyclic'  # Options: 'cyclic', 'knn'
K_NEIGHBOURS = 3

# ── Lookahead ────────────────────────────────────────────────────────────────

LOOKAHEAD_H = 24
EVALUATION_H = 24