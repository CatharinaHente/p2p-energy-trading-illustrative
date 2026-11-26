"""SimBench Semi-urban (1-LV-semiurb5--0-no_sw) prosumer populations: N = 4, 8, 16, 32.

Same prosumers as populations/ieee33.py. Their locations were drawn ONCE by a
random permutation (generated 2026-11-26) and are frozen below; nothing is
random at run time. Larger populations (48, 64, 96) are not implemented yet.
"""
from populations._simbench import build_population

# IEEE33 location (bus id used in populations/ieee33.py)
#   -> SimBench pandapower bus id (net.bus.index of 1-LV-semiurb5--0-no_sw)
ALLOCATION = {
     1:  56,
     2:  86,
     3:  40,
     4:  92,
     5:  51,
     6:  96,
     7:  57,
     8:  82,
     9:  53,
    10:  63,
    11:  19,
    12:  27,
    13:  25,
    14:  16,
    15:  95,
    16: 104,
    17:  55,
    18:  12,
    19:  67,
    20:  93,
    21:  45,
    22:  59,
    23:   0,
    24: 102,
    25:  43,
    26:   7,
    27:  22,
    28:  80,
    29:  58,
    30:  68,
    31:  31,
    32:  23,
}

_PROSUMER_SCENARIOS = build_population('lv5', ALLOCATION)
