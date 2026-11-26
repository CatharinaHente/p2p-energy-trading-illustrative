"""SimBench Rural (1-LV-rural3--0-no_sw) prosumer populations: N = 4, 8, 16, 32.

Same prosumers as populations/ieee33.py. Their locations were drawn ONCE by a
random permutation (generated 2026-11-26) and are frozen below; nothing is
random at run time. Larger populations (48, 64, 96) are not implemented yet.
"""
from populations._simbench import build_population

# IEEE33 location (bus id used in populations/ieee33.py)
#   -> SimBench pandapower bus id (net.bus.index of 1-LV-rural3--0-no_sw)
ALLOCATION = {
     1:  21,
     2:  90,
     3: 127,
     4:  40,
     5:  35,
     6:  28,
     7:  91,
     8:   9,
     9:  34,
    10: 120,
    11:  59,
    12:  41,
    13:  93,
    14:  77,
    15:  31,
    16:  94,
    17: 121,
    18: 101,
    19:  22,
    20: 100,
    21:  54,
    22:  18,
    23:  78,
    24: 118,
    25: 125,
    26:  88,
    27: 116,
    28:  39,
    29:  36,
    30:  49,
    31: 114,
    32:  76,
}

_PROSUMER_SCENARIOS = build_population('lv3', ALLOCATION)
