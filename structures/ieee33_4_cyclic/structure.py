"""Base structure — 4 prosumers on case33bw."""
from scenarios import PROSUMER_TYPES

NAME = 'base'

PROSUMERS = [
    {**PROSUMER_TYPES[1], 'bus': 5},   # Residential ++
    {**PROSUMER_TYPES[6], 'bus': 11},  # Office
    {**PROSUMER_TYPES[4], 'bus': 17},  # Residential HP only
    {**PROSUMER_TYPES[3], 'bus': 32},  # Residential PV only
]

EDGES = [(0, 1), (1, 2), (2, 3), (0, 3)]