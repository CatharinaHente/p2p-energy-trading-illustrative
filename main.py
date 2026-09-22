# %% [markdown]
# # Simple P2P electricity trading with Gurobi
# Four prosumers, four hours, batteries and a P2P trading graph.

# %% 1. Imports
from pathlib import Path
import ast
import csv
import json
import matplotlib as mpl
if "get_ipython" not in globals():
    mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from algo import centralized, distributed
from network import evaluate_feeder
ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# %% 2. Four hourly periods with solar generation and time-varying prices
# Rows are prosumers, columns are hours. Batteries start and end at 30% of
# capacity, not empty: `soc[T]==soc[0]` is a cyclic terminal constraint (the
# schedule must be repeatable day to day), and a real battery is not run
# down to a literal 0% state of charge (depth-of-discharge/degradation
# margin), so `initial_soc=0` was an unrealistic choice of input, not
# something the storage model itself requires.
# `edges` is the undirected P2P trading graph; it is separate from case33bw.
# Prosumers are sized at ~80x a household (light-industrial, 100s of kW) so
# their aggregate is a non-trivial fraction of case33bw's 3.715 MW native
# load: at household scale (kW) the feeder's own loading and voltage sag
# dominate, and evaluate_feeder()/the network constraints never bind.
# Deliberately picked past the point (~scale 68) where the plain
# centralized/distributed cases start violating the 0.90 p.u. floor (down to
# ~0.898 p.u. at scale 80) -- with the depth-of-discharge-limited battery
# above, centralized_network/distributed_network stay feasible across this
# whole range by actively holding voltage at exactly the 0.90 p.u. limit
# (the constraint binds) instead of sagging further, which is the actual
# point of running the network-aware cases: check_network()'s post-hoc
# check on the plain cases now shows a real violation instead of never
# triggering, while the enforced cases show what correcting it costs.
scale = 80.
battery_capacity = [3.*scale, 2.*scale, 2.*scale, 2.*scale]
initial_soc_fraction = .3
soc_min_fraction = .1   # depth-of-discharge floor
soc_max_fraction = .95  # overcharge ceiling
data = dict(demand=[[1.*scale]*4, [2.*scale]*4,
                    [1.*scale]*4, [1.5*scale]*4],
            generation=[[0.,8.*scale,5.*scale,0.], [0.,0.,0.,0.],
                        [0.,0.,0.,0.], [0.,0.,3.*scale,0.]],
            edges=[(0,1),(1,2),(2,3),(0,3)], agent_bus=[5,11,17,32],
            buy_price=[.20,.12,.20,.45], sell_price=[.05]*4,
            p2p_price=[.10,.08,.12,.25], trade_fee=.001, dt=1.,
            battery_capacity=battery_capacity,
            battery_power=[2.*scale,1.*scale,1.*scale,1.*scale],
            soc_min_fraction=soc_min_fraction, soc_max_fraction=soc_max_fraction,
            initial_soc=[initial_soc_fraction*c for c in battery_capacity],
            eta_charge=.95, eta_discharge=.95,
            battery_cost=.005, grid_limit=10.*scale, trade_limit=5.*scale,
            line_limit_mva=5.)
results = {}
trading_neighbors = {
    i: sorted(j if i == a else a for a,j in data["edges"] if i in (a,j))
    for i in range(len(data["demand"]))
}
print("P2P trading graph:", trading_neighbors)


def show(name, result):
    results[name] = result
    print(f"\n{name}, iterations: {result['iterations']}")
    for i,cost in enumerate(result["costs"]):
        print(f"Prosumer {i+1}: cost={cost:.4f} EUR")
        for key in ["buy","sell","charge","discharge","soc"]:
            print(f"  {key}: {np.round(result[key][i],4)}")
    out = ROOT / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    serial = {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in result.items() if k != "trades"}
    serial["trades"] = {f"{i}->{j}@{t}":v for (i,j,t),v in result["trades"].items()}
    (out / "summary.json").write_text(json.dumps(serial, indent=2))


def check_network(name, result):
    """Attach a post-hoc feeder voltage/congestion check to a result that was
    computed without network constraints, using the injections implied by its
    buy/sell/battery schedule. This never changes the schedule; it only
    reports whether the unconstrained trades would violate the feeder."""
    T = len(data["buy_price"])
    injections = (np.array(data["generation"]) - np.array(data["demand"])
                  - result["charge"] + result["discharge"])
    result["network"] = evaluate_feeder(injections, data["agent_bus"], T,
                                        data["line_limit_mva"])
    net = result["network"]
    print(f"{name}: feeder check (not enforced) -> "
          f"voltage_violation={net['any_voltage_violation']}, "
          f"congestion={net['any_congestion']}")


def plot_trades(name, result):
    """Plot signed bilateral trades and export the plotted source data."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    periods = np.arange(len(data["buy_price"]))
    trades = np.array([
        [result["trades"][i,j,t] for t in periods]
        for i,j in data["edges"]
    ])

    fig, ax = plt.subplots(figsize=(3.5, 2.4), constrained_layout=True)
    trade_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "signed_trade", ["#C65D3B", "#F7F7F5", "#2F6FA3"]
    )
    limit = max(1., float(np.abs(trades).max()))
    image = ax.imshow(trades, cmap=trade_cmap, vmin=-limit, vmax=limit,
                      aspect="auto", interpolation="nearest")

    for row in range(trades.shape[0]):
        for column in range(trades.shape[1]):
            value = trades[row, column]
            label = "0" if abs(value) < .005 else f"{value:.2f}"
            color = "white" if abs(value) > .58*limit else "#222222"
            ax.text(column, row, label, ha="center", va="center",
                    fontsize=7, color=color, fontweight="bold")

    ax.set_xticks(periods, [str(t) for t in periods])
    ax.set_yticks(np.arange(len(data["edges"])),
                  [f"{i+1} → {j+1}" for i,j in data["edges"]])
    ax.set(xlabel="Period", ylabel="Trading edge (+ follows arrow)",
           title=f"{name.replace('_',' ').title()} P2P trades")
    ax.set_xticks(np.arange(-.5, len(periods), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(data["edges"]), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=.045, pad=.035)
    colorbar.set_label("Trade (kW)")
    colorbar.outline.set_visible(False)

    out = ROOT / "results" / name
    rows = zip(periods, *trades)
    with (out / f"{name}_p2p_source_data.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["period"]
                        + [f"p2p_{i+1}_to_{j+1}_kw" for i,j in data["edges"]])
        writer.writerows(rows)
    fig.savefig(out / f"{name}_p2p.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}_p2p.svg", bbox_inches="tight")
    fig.savefig(out / f"{name}_p2p.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(out / f"{name}_p2p.png", dpi=300, bbox_inches="tight")
    if "get_ipython" in globals():
        plt.show()
    plt.close(fig)


def plot_network(name, result):
    """Plot the feeder voltage profile and export the plotted source data."""
    network = result["network"]
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    T = len(data["buy_price"])
    bus_count = max(bus for bus,_ in
                    (ast.literal_eval(k) for k in network["voltage_pu"])) + 1
    voltage = np.zeros((bus_count, T))
    for key, value in network["voltage_pu"].items():
        bus, t = ast.literal_eval(key)
        voltage[bus, t] = value
    buses = np.arange(bus_count)

    fig, ax = plt.subplots(figsize=(3.5, 2.4), constrained_layout=True)
    period_cmap = mpl.colormaps["viridis"].resampled(T)
    for t in range(T):
        ax.plot(buses, voltage[:,t], marker=".", markersize=3,
                linewidth=1., color=period_cmap(t), label=f"t={t}")
    ax.axhline(.90, color="#C65D3B", linewidth=.8, linestyle="--")
    ax.axhline(1.10, color="#C65D3B", linewidth=.8, linestyle="--")
    for bus in data["agent_bus"]:
        ax.axvline(bus, color="#999999", linewidth=.5, linestyle=":")
    ax.set(xlabel="Bus", ylabel="Voltage (p.u.)",
           title=f"{name.replace('_',' ').title()} voltage profile")
    ax.legend(frameon=False, fontsize=5, ncol=2, loc="lower left")

    out = ROOT / "results" / name
    with (out / f"{name}_voltage_profile_source_data.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bus"] + [f"t{t}_pu" for t in range(T)])
        writer.writerows(zip(buses, *voltage.T))
    fig.savefig(out / f"{name}_voltage_profile.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}_voltage_profile.svg", bbox_inches="tight")
    fig.savefig(out / f"{name}_voltage_profile.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(out / f"{name}_voltage_profile.png", dpi=300, bbox_inches="tight")
    if "get_ipython" in globals():
        plt.show()
    plt.close(fig)

def plot_balance(name, result):
    """Per-prosumer net flows, one panel per prosumer: two net bars per
    period -- grid (buy minus sell) and P2P (import minus export), positive
    above zero for a net purchase, negative below for a net sale -- plus
    battery state of charge as a line on a second axis. This is a third,
    deliberately reduced version of this plot: the first two broke supply
    and consumption down into every component (generation, discharge, buy,
    charge, sell, trades), which was accurate but busy; battery flow in
    particular is easier to read as the *level* it leaves behind (SOC) than
    as two separate charge/discharge power bars. Only the two things a P2P
    market cares about -- how much a prosumer leans on the grid vs. on its
    peers -- are kept as bars.
    """
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    T = len(data["buy_price"])
    N = len(data["demand"])
    periods = np.arange(T)
    soc_x = np.arange(T+1) - .5
    width = .32
    rows = []
    fig, axes = plt.subplots(1, N, figsize=(4.*N, 4.4), constrained_layout=True)
    for i, ax in enumerate(axes):
        net_export = np.array([
            sum(v for (a,b,t),v in result["trades"].items() if a == i and t == period)
            for period in range(T)])
        net_grid = result["buy"][i] - result["sell"][i]
        net_p2p = -net_export
        soc_pct = 100. * result["soc"][i] / max(1e-9, data["battery_capacity"][i])
        rows.append(dict(net_grid_kw=net_grid, net_p2p_kw=net_p2p, soc_pct=soc_pct))

        ax.bar(periods-width/2, net_grid, width, color="#5C7A99", label="Grid (net)")
        ax.bar(periods+width/2, net_p2p, width, color="#8B6BB1", label="P2P (net)")
        ax.axhline(0, color="#222222", linewidth=1.)
        ax.set_xticks(periods, [str(t) for t in periods])
        ax.set_xlim(soc_x[0], soc_x[-1])
        ax.set_title(f"Prosumer {i+1}", fontweight="bold", pad=10)
        ax.set_xlabel("Period")
        ax.set_ylabel("Net power (kW)")
        ax.margins(y=.15)

        ax2 = ax.twinx()
        ax2.plot(soc_x, soc_pct, color="#2E6F52", linewidth=2., marker="o",
                markersize=5, label="Battery SOC")
        ax2.set_ylim(0, 105)
        ax2.set_ylabel("SOC (%)")
    fig.suptitle(f"{name.replace('_',' ').title()} net flows", fontsize=15, fontweight="bold")
    fig.legend([plt.Rectangle((0,0),1,1,color="#5C7A99"),
               plt.Rectangle((0,0),1,1,color="#8B6BB1"),
               plt.Line2D([0],[0],color="#2E6F52",marker="o",markersize=5)],
              ["Grid (net, + = buy)","P2P (net, + = buy)","Battery SOC (right axis)"],
              frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.02))

    out = ROOT / "results" / name
    with (out / f"{name}_energy_balance_source_data.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["prosumer","period","net_grid_kw","net_p2p_kw",
                         "soc_start_pct","soc_end_pct"])
        for i, row in enumerate(rows):
            for t in range(T):
                writer.writerow([i+1, t, row["net_grid_kw"][t], row["net_p2p_kw"][t],
                                 row["soc_pct"][t], row["soc_pct"][t+1]])
    fig.savefig(out / f"{name}_energy_balance.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}_energy_balance.svg", bbox_inches="tight")
    fig.savefig(out / f"{name}_energy_balance.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(out / f"{name}_energy_balance.png", dpi=300, bbox_inches="tight")
    if "get_ipython" in globals():
        plt.show()
    plt.close(fig)

# %% 3. Centralized P2P
central = centralized(data)
check_network("centralized", central)
show("centralized", central)
plot_trades("centralized", central)
plot_network("centralized", central)
plot_balance("centralized", central)

# %% 4. Distributed P2P
local = distributed(data)
check_network("distributed", local)
show("distributed", local)
plot_trades("distributed", local)
plot_network("distributed", local)
plot_balance("distributed", local)

# %% 5. Centralized P2P with pandapower case33bw
central_network = centralized(data, with_network=True)
show("centralized_network", central_network)
plot_trades("centralized_network", central_network)
plot_network("centralized_network", central_network)
plot_balance("centralized_network", central_network)

# %% 6. Distributed P2P with pandapower case33bw
local_network = distributed(data, with_network=True)
show("distributed_network", local_network)
plot_trades("distributed_network", local_network)
plot_network("distributed_network", local_network)
plot_balance("distributed_network", local_network)

# %% 7. Check battery dynamics, balance and centralized/distributed costs
T = len(data['buy_price'])
N = len(data['demand'])
for name, result in results.items():
    for i in range(N):
        for t in range(T):
            sale = sum(v for (a,b,h),v in result['trades'].items() if a == i and h == t)
            balance = (data['generation'][i][t]+result['buy'][i,t]+result['discharge'][i,t]
                       -data['demand'][i][t]-result['sell'][i,t]-result['charge'][i,t]-sale)
            assert abs(balance) < 1e-5
        expected = result['soc'][i,:-1]+data['dt']*(data['eta_charge']*result['charge'][i]
                    -result['discharge'][i]/data['eta_discharge'])
        assert np.allclose(result['soc'][i,1:],expected,atol=1e-6)
        assert abs(result['soc'][i,0]-data['initial_soc'][i]) < 1e-6
        assert abs(result['soc'][i,-1]-data['initial_soc'][i]) < 1e-6
        assert result['soc'][i].min() >= data['soc_min_fraction']*data['battery_capacity'][i]-1e-6
        assert result['soc'][i].max() <= data['soc_max_fraction']*data['battery_capacity'][i]+1e-6
    for i,j in data['edges']:
        for t in range(T):
            assert abs(result['trades'][i,j,t]+result['trades'][j,i,t]) < 1e-5
    if result['network']:
        net = result['network']
        v_min, v_max = min(net['voltage_pu'].values()), max(net['voltage_pu'].values())
        print(f"  Feeder voltage range: [{v_min:.4f}, {v_max:.4f}] p.u., "
              f"violation={net['any_voltage_violation']}, congestion={net['any_congestion']}")
        if name.endswith('_network'):
            # These two cases optimize with the feeder enforced as hard constraints.
            assert not net['any_voltage_violation']
            assert not net['any_congestion']
    print('Total cost:', round(sum(result['costs']),6), 'EUR')
# Individual allocations may differ between equally optimal solutions.
assert abs(sum(central['costs'])-sum(local['costs'])) < 1e-4
assert abs(sum(central_network['costs'])-sum(local_network['costs'])) < 1e-4
assert central['charge'].sum() > .1
print('All four examples passed.')
