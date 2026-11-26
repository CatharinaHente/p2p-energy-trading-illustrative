"""SimBench Urban (1-LV-urban6--0-no_sw) prosumer populations: N = 4, 8, 16, 32.

Same prosumers as populations/ieee33.py. Their locations were drawn ONCE by a
random permutation (generated 2026-11-26) and are frozen below; nothing is
random at run time. Larger populations (48, 64, 96) are not implemented yet.
"""
from populations._simbench import build_population

# IEEE33 location (bus id used in populations/ieee33.py)
#   -> SimBench pandapower bus id (net.bus.index of 1-LV-urban6--0-no_sw)
ALLOCATION = {
     1:   2,
     2:  52,
     3:  56,
     4:  31,
     5:  44,
     6:   5,
     7:   7,
     8:  49,
     9:  34,
    10:  28,
    11:  55,
    12:  36,
    13:  25,
    14:   3,
    15:  37,
    16:  53,
    17:   8,
    18:  14,
    19:   4,
    20:  32,
    21:  30,
    22:  39,
    23:   9,
    24:  29,
    25:  35,
    26:  33,
    27:  21,
    28:  27,
    29:  22,
    30:  45,
    31:  12,
    32:  54,
}

_PROSUMER_SCENARIOS = build_population('lv6', ALLOCATION)
