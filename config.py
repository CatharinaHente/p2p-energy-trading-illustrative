# config.py — the scenario switch. Change these values, then run batch_main.py
# (or the strategic_GNE_selection notebooks). Results go to
# structures/<structure_name()>/results/<date>/.

# ── Physical network ─────────────────────────────────────────────────────────
#   'ieee33' — IEEE 33-bus (pandapower case33bw), original reference
#   'lv3'    — SimBench LV rural3
#   'lv5'    — SimBench LV semi-urban5
#   'lv6'    — SimBench LV urban6

NETWORK_TOPOLOGY = 'ieee33'

SIMBENCH_CODES = {
    'lv1': '1-LV-rural1--0-no_sw',
    'lv2': '1-LV-rural2--0-no_sw',
    'lv3': '1-LV-rural3--0-no_sw',
    'lv4': '1-LV-semiurb4--0-no_sw',
    'lv5': '1-LV-semiurb5--0-no_sw',
    'lv6': '1-LV-urban6--0-no_sw',
}

# Networks with a prosumer population (populations/<module>.py).
NETWORKS = ('ieee33', 'lv3', 'lv5', 'lv6')

# ── Prosumer population ──────────────────────────────────────────────────────
#   4, 8, 16 or 32 in every network (nested: 4 ⊂ 8 ⊂ 16; same prosumers as IEEE33)

PROSUMER_SCENARIO = 4

POPULATION_SIZES = (4, 8, 16, 32)

# ── Trading graph ────────────────────────────────────────────────────────────
#   'cyclic'        — ring over prosumer index order
#   'knn'           — k nearest neighbours by feeder distance (K_NEIGHBOURS)
#   'sparse_hetero' — sparse random graph with heterogeneous degrees
#                     (preferential attachment, mean degree SPARSE_MEAN_DEGREE)

TRADING_TOPOLOGY = 'cyclic'
K_NEIGHBOURS = 3
SPARSE_MEAN_DEGREE = 2.5
TRADING_SEED = 42

# ── Analysis days ────────────────────────────────────────────────────────────
#   Base days: 7th, 14th, 21st of every month of 2016 (36 days). Each network
#   shifts them by its own offset so the study covers more distinct days;
#   IEEE33 keeps the original days (+0).

BASE_DAYS_OF_MONTH = (7, 14, 21)
DATE_OFFSET_DAYS = {'ieee33': 0, 'lv3': 1, 'lv5': 2, 'lv6': 3}

# ── Lookahead ────────────────────────────────────────────────────────────────

LOOKAHEAD_H = 24
EVALUATION_H = 24


def trading_label(topology=None):
    """Short tag for the trading graph used in folder names."""
    topology = TRADING_TOPOLOGY if topology is None else topology
    if topology == 'cyclic':
        return 'cyclic'                                   # keeps existing folders
    if topology == 'knn':
        return f'knn{K_NEIGHBOURS}'
    if topology == 'sparse_hetero':
        return f'sparsehet{SPARSE_MEAN_DEGREE:g}s{TRADING_SEED}'
    raise ValueError(f"Unknown trading topology: {topology!r}")


def structure_name(network=None, n_prosumers=None, trading=None):
    """Folder name under structures/, e.g. 'ieee33_4_cyclic', 'lv5_16_knn3'."""
    network = NETWORK_TOPOLOGY if network is None else network
    n_prosumers = PROSUMER_SCENARIO if n_prosumers is None else n_prosumers
    return f'{network}_{n_prosumers}_{trading_label(trading)}'


def analysis_dates(network=None):
    """The 36 analysis days of a network, e.g. lv3 → 2016-01-08, 2016-01-15, ..."""
    network = NETWORK_TOPOLOGY if network is None else network
    shift = DATE_OFFSET_DAYS[network]
    return [f'2016-{m:02d}-{d + shift:02d}'
            for m in range(1, 13) for d in BASE_DAYS_OF_MONTH]
