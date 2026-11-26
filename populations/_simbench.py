"""Shared builder for the SimBench LV prosumer populations.

The prosumers are exactly the IEEE33 ones (populations/ieee33.py) for every
population size 4, 8, 16, 32. Only their location changes: each IEEE33
location (bus id 1..32 used by any IEEE33 population) is mapped to a distinct
SimBench load bus. That mapping (ALLOCATION in populations/simbench_lv*.py)
was drawn once by a random permutation and is hardcoded, so it never changes.

Consequences:
  * The same IEEE33 prosumer sits at the same SimBench bus for every N, so
    populations stay nested exactly as in IEEE33 (4 ⊂ 8 ⊂ 16).
  * The allocation is not a copy of the IEEE33 pattern: neighbours in IEEE33
    are generally not neighbours in the LV feeder.
  * ALLOCATION uses original pandapower bus ids (stable); they are converted to
    the internal indices of network.load_network(topology) here.
"""
from network import load_network
from populations.ieee33 import _PROSUMER_SCENARIOS as _IEEE33_SCENARIOS


def build_population(topology, allocation):
    """Return {N: [prosumer dicts with internal SimBench bus index]}."""
    grid = load_network(topology)
    internal = {pp: i for i, pp in enumerate(grid['pp_bus'])}
    locations = {p['bus'] for pop in _IEEE33_SCENARIOS.values() for p in pop}
    if set(allocation) != locations:
        raise ValueError(f"{topology}: ALLOCATION must cover IEEE33 locations {sorted(locations)}")
    if len(set(allocation.values())) != len(allocation):
        raise ValueError(f"{topology}: two prosumer locations share a SimBench bus")
    load_buses = set(grid['load_buses'])
    relocate = {}
    for loc, pp in allocation.items():
        if pp not in internal or internal[pp] not in load_buses:
            raise ValueError(f"{topology}: pandapower bus {pp} is not an LV load bus")
        relocate[loc] = internal[pp]
    return {n: [{**p, 'bus': relocate[p['bus']]} for p in pop]
            for n, pop in _IEEE33_SCENARIOS.items()}
