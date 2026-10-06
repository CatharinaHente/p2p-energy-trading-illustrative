# %% [markdown]
# # Simple P2P electricity trading with Gurobi
# Four prosumers, four hours, batteries, EVs, heat pumps and a P2P trading graph.

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

# %% 2. Input data
# Existing parameters unchanged.  All [NEW] device parameters are set to zero /
# disabled so the four original cases run exactly as before.  Enable devices by
# setting ev_capacity[i] > 0  or  hp_max_power[i] > 0  for prosumer i.

scale             = 80.
battery_capacity  = [3.*scale, 2.*scale, 2.*scale, 2.*scale]
initial_soc_fraction = .3
soc_min_fraction  = .1    # depth-of-discharge floor
soc_max_fraction  = .95   # overcharge ceiling

# [NEW] Number of timesteps — keep in sync with buy_price list below.
_T = 4

# [NEW] BESS apparent power rating for polygon FOR (flexibility paper, Section A.1).
# Set equal to battery_power here (conservative: polygon matches existing P limit).
# Increase battery_smax[i] independently if the converter can handle more kVA than kW.
_battery_smax = [2.*scale, 1.*scale, 1.*scale, 1.*scale]

# [NEW] EV parameters — all disabled (ev_capacity = 0 for every prosumer).
# To enable EV for prosumer i: set ev_capacity[i] > 0, ev_power[i] > 0, ev_smax[i] > 0,
# fill ev_driving_load[i] from mobility data (eq 1: 17 kWh/100km winter, 15 summer),
# set ev_availability[i] schedule (1=plugged in, 0=away), ev_initial_soc[i], etc.
_ev_capacity           = [0.,  0.,  0.,  0.]       # kWh nominal; 0 = disabled
_ev_cap_factor         = [1.0, 1.0, 1.0, 1.0]     # temperature derating (0.80 winter)
_ev_power              = [0.,  0.,  0.,  0.]        # kW max charge/discharge rate (P_C)
_ev_smax               = [0.,  0.,  0.,  0.]        # kVA charger apparent power
_ev_driving_load       = [[0.]*_T for _ in range(4)]  # kWh depleted per timestep (eq 1)
_ev_soc_min_departure  = [0.,  0.,  0.,  0.]       # fraction SOC required before departure (eq 2)
_ev_initial_soc        = [0.,  0.,  0.,  0.]       # kWh initial SOC
_ev_availability       = [[1]*_T for _ in range(4)]   # 1=plugged in, 0=away

# [NEW] HP parameters — all disabled (hp_max_power = 0 for every prosumer).
# To enable HP for prosumer i: set hp_max_power[i] > 0, fill hp_base_power[i] from
# heating load profile, hp_q_heat[i] from outdoor temperature (flexibility paper eq 2).
_hp_max_power  = [0.,  0.,  0.,  0.]          # kW; 0 = disabled
_hp_base_power = [[0.]*_T for _ in range(4)]  # kW baseline per timestep
_hp_q_heat     = [[0.]*_T for _ in range(4)]  # degC/h heating demand per timestep
_hp_T_init     = [20., 20., 20., 20.]         # degC initial indoor temperature
_hp_T_min      = 19.                           # degC comfort lower bound
_hp_T_max      = 23.                           # degC comfort upper bound

data = dict(
    # ------------------------------------------------------------------ #
    # Original keys — unchanged                                           #
    # ------------------------------------------------------------------ #
    demand=[[1.*scale]*4, [2.*scale]*4,
            [1.*scale]*4, [1.5*scale]*4],
    generation=[[0., 8.*scale, 5.*scale, 0.], [0., 0., 0., 0.],
                [0., 0., 0., 0.],             [0., 0., 3.*scale, 0.]],
    edges=[(0,1),(1,2),(2,3),(0,3)], agent_bus=[5,11,17,32],
    buy_price=[.20,.12,.20,.45], sell_price=[.05]*4,
    p2p_price=[.10,.08,.12,.25], trade_fee=.001, dt=1.,
    battery_capacity=battery_capacity,
    battery_power=[2.*scale, 1.*scale, 1.*scale, 1.*scale],
    soc_min_fraction=soc_min_fraction, soc_max_fraction=soc_max_fraction,
    initial_soc=[initial_soc_fraction*c for c in battery_capacity],
    eta_charge=.95, eta_discharge=.95,
    battery_cost=.005, grid_limit=10.*scale, trade_limit=5.*scale,
    line_limit_mva=5.,
    # ------------------------------------------------------------------ #
    # [NEW] BESS polygon apparent power                                   #
    # ------------------------------------------------------------------ #
    battery_smax=_battery_smax,
    battery_polygon_sides=8,
    # ------------------------------------------------------------------ #
    # [NEW] EV (all disabled — see comments above to enable)             #
    # ------------------------------------------------------------------ #
    ev_capacity=_ev_capacity,
    ev_cap_factor=_ev_cap_factor,
    ev_power=_ev_power,
    ev_smax=_ev_smax,
    ev_polygon_sides=8,
    ev_driving_load=_ev_driving_load,
    ev_soc_min_departure=_ev_soc_min_departure,
    ev_dod_max=0.30,                   # max DoD from [16]; SoC_min = 0.70 × C_eff
    ev_initial_soc=_ev_initial_soc,
    ev_availability=_ev_availability,
    ev_eta_charge=0.95,                # η_c, eq (3)
    ev_eta_discharge=0.95,             # η_d, eq (3)
    ev_cost=0.005,                     # EUR/kWh throughput degradation
    ev_v2g=True,                       # True=V2G bidirectional, False=V1G charge-only
    # ------------------------------------------------------------------ #
    # [NEW] HP (all disabled — see comments above to enable)             #
    # ------------------------------------------------------------------ #
    hp_max_power=_hp_max_power,
    hp_base_power=_hp_base_power,
    hp_q_heat=_hp_q_heat,
    hp_T_init=_hp_T_init,
    hp_T_min=_hp_T_min,
    hp_T_max=_hp_T_max,
    hp_cost=0.0,
    # ------------------------------------------------------------------ #
    # [NEW] Reactive power factors                                        #
    # ------------------------------------------------------------------ #
    pf_pv=0.95,
    pf_hp=0.97,
    pf_load=0.95,
)

results = {}
trading_neighbors = {
    i: sorted(j if i == a else a for a,j in data["edges"] if i in (a,j))
    for i in range(len(data["demand"]))
}
print("P2P trading graph:", trading_neighbors)


# %% Helper functions

def show(name, result):
    results[name] = result
    N = len(data["demand"])
    print(f"\n{name}, iterations: {result['iterations']}")
    for i, cost in enumerate(result["costs"]):
        print(f"Prosumer {i+1}: cost={cost:.4f} EUR")
        for key in ["buy", "sell", "charge", "discharge", "soc"]:
            print(f"  {key}: {np.round(result[key][i], 4)}")
        # [NEW] EV summary — only when device is enabled for this prosumer
        if data.get('ev_capacity', [0]*N)[i] > 0:
            for key in ["ev_charge", "ev_discharge", "ev_soc"]:
                if key in result:
                    print(f"  {key}: {np.round(result[key][i], 4)}")
        # [NEW] HP summary — only when device is enabled for this prosumer
        if data.get('hp_max_power', [0]*N)[i] > 0:
            for key in ["hp_power", "temp"]:
                if key in result:
                    print(f"  {key}: {np.round(result[key][i], 4)}")
    out = ROOT / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    serial = {k: v.tolist() if isinstance(v, np.ndarray) else v
              for k, v in result.items() if k != "trades"}
    serial["trades"] = {f"{i}->{j}@{t}": v
                        for (i,j,t), v in result["trades"].items()}
    (out / "summary.json").write_text(json.dumps(serial, indent=2))


def check_network(name, result):
    """Post-hoc feeder voltage/congestion check (does not change the schedule).
    [UPDATED] Active power injection now includes EV net flow and HP load.
    """
    T  = len(data["buy_price"])
    N  = len(data["demand"])
    # [UPDATED] EV and HP terms added; default to zeros if keys absent (old algo.py)
    _zeros = np.zeros((N, T))
    injections = (
        np.array(data["generation"]) - np.array(data["demand"])
        - result["charge"]                              + result["discharge"]
        - result.get("ev_charge",    _zeros)            + result.get("ev_discharge", _zeros)  # [NEW]
        - result.get("hp_power",     _zeros)                                                   # [NEW]
    )
    result["network"] = evaluate_feeder(
        injections, data["agent_bus"], T, data["line_limit_mva"],
        q_injections_kvar=result.get('q_injection'))             # [NEW-network]
    net = result["network"]
    print(f"{name}: feeder check (not enforced) -> "
          f"voltage_violation={net['any_voltage_violation']}, "
          f"congestion={net['any_congestion']}")


def plot_trades(name, result):
    """Plot signed bilateral trades and export the plotted source data."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    periods = np.arange(len(data["buy_price"]))
    trades = np.array([[result["trades"][i,j,t] for t in periods]
                       for i,j in data["edges"]])
    fig, ax = plt.subplots(figsize=(3.5, 2.4), constrained_layout=True)
    trade_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "signed_trade", ["#C65D3B", "#F7F7F5", "#2F6FA3"])
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
    for spine in ax.spines.values(): spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=.045, pad=.035)
    colorbar.set_label("Trade (kW)")
    colorbar.outline.set_visible(False)
    out = ROOT / "results" / name
    rows = zip(periods, *trades)
    with (out / f"{name}_p2p_source_data.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["period"] + [f"p2p_{i+1}_to_{j+1}_kw" for i,j in data["edges"]])
        writer.writerows(rows)
    for ext, kw in [(".pdf",{}),(".svg",{}),(".tiff",{"dpi":600}),(".png",{"dpi":300})]:
        fig.savefig(out / f"{name}_p2p{ext}", bbox_inches="tight", **kw)
    if "get_ipython" in globals(): plt.show()
    plt.close(fig)


def plot_network(name, result):
    """Plot the feeder voltage profile and export source data."""
    network = result["network"]
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "svg.fonttype": "none",
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
    for ext, kw in [(".pdf",{}),(".svg",{}),(".tiff",{"dpi":600}),(".png",{"dpi":300})]:
        fig.savefig(out / f"{name}_voltage_profile{ext}", bbox_inches="tight", **kw)
    if "get_ipython" in globals(): plt.show()
    plt.close(fig)


def plot_balance(name, result):
    """Per-prosumer net flows: grid bar, P2P bar, BESS SOC line.
    [NEW] EV SOC dashed line added on second axis when EV is enabled.
    """
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 13, "axes.labelsize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 11, "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    T  = len(data["buy_price"])
    N  = len(data["demand"])
    periods = np.arange(T)
    soc_x   = np.arange(T+1) - .5
    width   = .32
    rows    = []
    fig, axes = plt.subplots(1, N, figsize=(4.*N, 4.4), constrained_layout=True)
    for i, ax in enumerate(axes):
        net_export = np.array([
            sum(v for (a,b,t),v in result["trades"].items() if a == i and t == period)
            for period in range(T)])
        net_grid = result["buy"][i] - result["sell"][i]
        net_p2p  = -net_export
        soc_pct  = 100. * result["soc"][i] / max(1e-9, data["battery_capacity"][i])
        row = dict(net_grid_kw=net_grid, net_p2p_kw=net_p2p, soc_pct=soc_pct)

        # [NEW] EV SOC — only when device enabled for this prosumer
        ev_enabled = data.get('ev_capacity', [0]*N)[i] > 0
        ev_soc_pct = None
        if ev_enabled and result.get('ev_soc') is not None:
            ev_cap_eff = (data['ev_capacity'][i]
                          * data.get('ev_cap_factor', [1.]*N)[i])
            ev_soc_pct = 100. * result['ev_soc'][i] / max(1e-9, ev_cap_eff)
            row['ev_soc_pct'] = ev_soc_pct
        rows.append(row)

        ax.bar(periods - width/2, net_grid, width, color="#5C7A99", label="Grid (net)")
        ax.bar(periods + width/2, net_p2p,  width, color="#8B6BB1", label="P2P (net)")
        ax.axhline(0, color="#222222", linewidth=1.)
        ax.set_xticks(periods, [str(t) for t in periods])
        ax.set_xlim(soc_x[0], soc_x[-1])
        ax.set_title(f"Prosumer {i+1}", fontweight="bold", pad=10)
        ax.set_xlabel("Period")
        ax.set_ylabel("Net power (kW)")
        ax.margins(y=.15)

        ax2 = ax.twinx()
        ax2.plot(soc_x, soc_pct, color="#2E6F52", linewidth=2., marker="o",
                 markersize=5, label="BESS SOC")
        # [NEW] EV SOC on same axis, dashed
        if ev_soc_pct is not None:
            ax2.plot(soc_x, ev_soc_pct, color="#B5720A", linewidth=2., marker="s",
                     markersize=5, linestyle="--", label="EV SOC")
        ax2.set_ylim(0, 105)
        ax2.set_ylabel("SOC (%)")

    fig.suptitle(f"{name.replace('_',' ').title()} net flows",
                 fontsize=15, fontweight="bold")
    # [UPDATED] legend — EV SOC patch added conditionally
    legend_handles = [
        plt.Rectangle((0,0),1,1,color="#5C7A99"),
        plt.Rectangle((0,0),1,1,color="#8B6BB1"),
        plt.Line2D([0],[0],color="#2E6F52",marker="o",markersize=5),
    ]
    legend_labels = ["Grid (net, + = buy)", "P2P (net, + = buy)", "BESS SOC (right axis)"]
    if any(data.get('ev_capacity', [0]*N)[i] > 0 for i in range(N)):  # [NEW]
        legend_handles.append(
            plt.Line2D([0],[0],color="#B5720A",marker="s",markersize=5,linestyle="--"))
        legend_labels.append("EV SOC (right axis)")
    fig.legend(legend_handles, legend_labels,
               frameon=False, ncol=len(legend_handles),
               loc="upper center", bbox_to_anchor=(.5, -.02))

    out = ROOT / "results" / name
    with (out / f"{name}_energy_balance_source_data.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        # [UPDATED] CSV header — include ev_soc columns if any EV enabled
        any_ev = any(data.get('ev_capacity', [0]*N)[i] > 0 for i in range(N))
        header = ["prosumer","period","net_grid_kw","net_p2p_kw",
                  "soc_start_pct","soc_end_pct"]
        if any_ev: header += ["ev_soc_start_pct","ev_soc_end_pct"]   # [NEW]
        writer.writerow(header)
        for i, row in enumerate(rows):
            for t in range(T):
                line = [i+1, t, row["net_grid_kw"][t], row["net_p2p_kw"][t],
                        row["soc_pct"][t], row["soc_pct"][t+1]]
                if any_ev:
                    ep = row.get("ev_soc_pct")
                    line += [ep[t] if ep is not None else 0.,
                             ep[t+1] if ep is not None else 0.]
                writer.writerow(line)
    for ext, kw in [(".pdf",{}),(".svg",{}),(".tiff",{"dpi":600}),(".png",{"dpi":300})]:
        fig.savefig(out / f"{name}_energy_balance{ext}", bbox_inches="tight", **kw)
    if "get_ipython" in globals(): plt.show()
    plt.close(fig)


def plot_devices(name, result):
    """[NEW] HP indoor temperature and EV charge/discharge profiles.

    Only runs when at least one prosumer has an enabled EV or HP.
    Two sub-plots per enabled prosumer:
      - Top panel : HP indoor temperature over time (degC), with comfort bounds.
      - Bottom panel: EV charge (positive) / discharge (negative) power (kW).
    """
    N = len(data["demand"])
    T = len(data["buy_price"])
    has_hp = [data.get('hp_max_power', [0]*N)[i] > 0 for i in range(N)]
    has_ev = [data.get('ev_capacity',  [0]*N)[i] > 0 for i in range(N)]
    if not any(has_hp) and not any(has_ev):
        return   # nothing to plot — skip silently when all devices disabled

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    periods = np.arange(T)
    soc_x   = np.arange(T+1) - .5
    rows_data = []

    fig, axes_grid = plt.subplots(
        2, N, figsize=(4.*N, 5.5), constrained_layout=True, sharex=True)
    # Make axes_grid always 2-D even for N=1
    if N == 1:
        axes_grid = np.array(axes_grid).reshape(2, 1)

    for i in range(N):
        ax_top = axes_grid[0, i]
        ax_bot = axes_grid[1, i]
        row    = {}

        # -- HP indoor temperature ----------------------------------------
        if has_hp[i] and result.get('temp') is not None:
            temp = result['temp'][i]           # T+1 values
            ax_top.plot(soc_x, temp, color="#C65D3B", linewidth=2.,
                        marker="o", markersize=4, label="Indoor temp")
            ax_top.axhline(data['hp_T_min'], color="#999999", linewidth=.8,
                           linestyle="--", label=f"T_min={data['hp_T_min']}°C")
            ax_top.axhline(data['hp_T_max'], color="#444444", linewidth=.8,
                           linestyle="--", label=f"T_max={data['hp_T_max']}°C")
            ax_top.set_ylabel("Temp (°C)")
            row['temp'] = temp
        else:
            ax_top.set_visible(False)

        # -- EV charge / discharge ----------------------------------------
        if has_ev[i] and result.get('ev_charge') is not None:
            ev_c = result['ev_charge'][i]
            ev_d = result['ev_discharge'][i]
            avail = np.array(data['ev_availability'][i])
            # Shade away periods
            for t in range(T):
                if avail[t] == 0:
                    ax_bot.axvspan(t - .5, t + .5, color="#EEEEEE", zorder=0)
            ax_bot.bar(periods - .2, ev_c,  .35, color="#2F6FA3", label="EV charge")
            ax_bot.bar(periods + .2, -ev_d, .35, color="#C65D3B", label="EV discharge")
            ax_bot.axhline(0, color="#222222", linewidth=.8)
            ax_bot.set_ylabel("EV power (kW)\n+ = charge, - = discharge")
            ax_bot.legend(frameon=False, fontsize=7)
            row['ev_charge'] = ev_c
            row['ev_discharge'] = ev_d
        else:
            ax_bot.set_visible(False)

        ax_bot.set_xlabel("Period")
        ax_top.set_title(f"Prosumer {i+1}", fontweight="bold")
        rows_data.append(row)

    fig.suptitle(f"{name.replace('_',' ').title()} device profiles",
                 fontsize=13, fontweight="bold")
    out = ROOT / "results" / name
    # Export source CSV
    with (out / f"{name}_devices_source_data.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["prosumer","period","temp_degC",
                         "ev_charge_kw","ev_discharge_kw"])
        for i, row in enumerate(rows_data):
            t_vals = row.get('temp', [None]*(T+1))
            ec     = row.get('ev_charge', [None]*T)
            ed     = row.get('ev_discharge', [None]*T)
            for t in range(T):
                writer.writerow([i+1, t,
                                 t_vals[t] if t_vals[t] is not None else "",
                                 ec[t]     if ec[t]     is not None else "",
                                 ed[t]     if ed[t]     is not None else ""])
    for ext, kw in [(".pdf",{}),(".svg",{}),(".tiff",{"dpi":600}),(".png",{"dpi":300})]:
        fig.savefig(out / f"{name}_devices{ext}", bbox_inches="tight", **kw)
    if "get_ipython" in globals(): plt.show()
    plt.close(fig)


# %% 3. Centralized P2P
central = centralized(data)
check_network("centralized", central)
show("centralized", central)
plot_trades("centralized", central)
plot_network("centralized", central)
plot_balance("centralized", central)
plot_devices("centralized", central)      # [NEW] no-op while all devices disabled

# %% 4. Distributed P2P
local = distributed(data)
check_network("distributed", local)
show("distributed", local)
plot_trades("distributed", local)
plot_network("distributed", local)
plot_balance("distributed", local)
plot_devices("distributed", local)        # [NEW]

# %% 5. Centralized P2P with pandapower case33bw
central_network = centralized(data, with_network=True)
show("centralized_network", central_network)
plot_trades("centralized_network", central_network)
plot_network("centralized_network", central_network)
plot_balance("centralized_network", central_network)
plot_devices("centralized_network", central_network)   # [NEW]

# %% 6. Distributed P2P with pandapower case33bw
local_network = distributed(data, with_network=True)
show("distributed_network", local_network)
plot_trades("distributed_network", local_network)
plot_network("distributed_network", local_network)
plot_balance("distributed_network", local_network)
plot_devices("distributed_network", local_network)     # [NEW]

# %% 7. Verification
T = len(data['buy_price'])
N = len(data['demand'])
_zeros = np.zeros((N, T))

for name, result in results.items():
    for i in range(N):
        for t in range(T):
            sale = sum(v for (a,b,h),v in result['trades'].items()
                       if a == i and h == t)
            # [UPDATED] Energy balance includes EV net flow and HP load
            balance = (
                data['generation'][i][t]
                + result['buy'][i,t]
                + result['discharge'][i,t]
                + result.get('ev_discharge', _zeros)[i,t]          # [NEW]
                - data['demand'][i][t]
                - result['sell'][i,t]
                - result['charge'][i,t]
                - result.get('ev_charge', _zeros)[i,t]             # [NEW]
                - result.get('hp_power',  _zeros)[i,t]             # [NEW]
                - sale
            )
            assert abs(balance) < 1e-5, \
                f"{name} prosumer {i+1} period {t}: balance={balance:.3g}"

        # BESS SOC dynamics (original)
        expected_soc = (result['soc'][i,:-1]
                        + data['dt'] * (data['eta_charge'] * result['charge'][i]
                                        - result['discharge'][i] / data['eta_discharge']))
        assert np.allclose(result['soc'][i,1:], expected_soc, atol=1e-6)
        assert abs(result['soc'][i,0]  - data['initial_soc'][i]) < 1e-6
        assert abs(result['soc'][i,-1] - data['initial_soc'][i]) < 1e-6
        assert result['soc'][i].min() >= data['soc_min_fraction']*data['battery_capacity'][i] - 1e-6
        assert result['soc'][i].max() <= data['soc_max_fraction']*data['battery_capacity'][i] + 1e-6

        # [NEW] EV SOC dynamics — only verified when EV is enabled for this prosumer
        if data.get('ev_capacity', [0]*N)[i] > 0 and result.get('ev_soc') is not None:
            ev_cap_eff = (data['ev_capacity'][i]
                          * data.get('ev_cap_factor', [1.]*N)[i])
            dod_max    = data.get('ev_dod_max', 0.30)
            driving    = np.array(data['ev_driving_load'][i])
            expected_ev = (result['ev_soc'][i,:-1]
                           + data['dt'] * (data['ev_eta_charge'] * result['ev_charge'][i]
                                           - result['ev_discharge'][i] / data['ev_eta_discharge'])
                           - driving)
            assert np.allclose(result['ev_soc'][i,1:], expected_ev, atol=1e-6), \
                f"{name} prosumer {i+1}: EV SOC dynamics violated"
            assert result['ev_soc'][i].min() >= (1. - dod_max)*ev_cap_eff - 1e-6, \
                f"{name} prosumer {i+1}: EV DoD constraint violated"

    for i,j in data['edges']:
        for t in range(T):
            assert abs(result['trades'][i,j,t] + result['trades'][j,i,t]) < 1e-5

    if result['network']:
        net = result['network']
        v_min = min(net['voltage_pu'].values())
        v_max = max(net['voltage_pu'].values())
        print(f"  Feeder voltage range: [{v_min:.4f}, {v_max:.4f}] p.u., "
              f"violation={net['any_voltage_violation']}, "
              f"congestion={net['any_congestion']}")
        if name.endswith('_network'):
            assert not net['any_voltage_violation']
            assert not net['any_congestion']

    print('Total cost:', round(sum(result['costs']), 6), 'EUR')

assert abs(sum(central['costs'])         - sum(local['costs']))          < 1e-4
assert abs(sum(central_network['costs']) - sum(local_network['costs']))  < 1e-4
assert central['charge'].sum() > .1
print('All four examples passed.')