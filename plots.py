"""Visualisations for P2P prosumer results and device FORs.
 
All functions save to root/results/<name>/ in pdf/svg/tiff/png and open the figure.
plot_for() is called once on the data dict; all others take a result dict.
"""
import math, ast, csv
import matplotlib.pyplot as plt
import numpy as np
 
 
# --- shared helpers ---
 
def _rc(size=7):
    return {"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
            "font.size":size,"axes.spines.top":False,"axes.spines.right":False,
            "pdf.fonttype":42,"svg.fonttype":"none"}
 
def _save(fig, out, stem):
    out.mkdir(parents=True, exist_ok=True)
    for ext, kw in [(".pdf",{}),(".svg",{}),(".tiff",{"dpi":600}),(".png",{"dpi":300})]:
        fig.savefig(out / f"{stem}{ext}", bbox_inches="tight", **kw)
    plt.show()
    plt.close(fig)
 
def _polygon_pts(s_max, n):
    """Vertices of the n-sided inner polygon FOR (circumradius = s_max)."""
    a = [math.pi/n + 2*math.pi*k/n for k in range(n+1)]
    return [s_max*math.cos(x) for x in a], [s_max*math.sin(x) for x in a]
 
_COLORS = ["#2F6FA3", "#C65D3B", "#2E6F52", "#B5720A"]
 
 
# --- FOR visualisation ---
 
def plot_for(data, root):
    """Plot device Feasibility Operating Regions in the PQ plane.
 
    BESS / EV  : polygon FOR — flexibility paper Sec A.1
    PV         : triangular FOR — flexibility paper Fig 1b
    HP         : line FOR at fixed pf — flexibility paper Fig 1a
    """
    N     = len(data['demand'])
    theta = np.linspace(0, 2*np.pi, 200)
    plt.rcParams.update(_rc(8))
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
 
    # BESS
    ax = axes[0]
    ax.set_title("BESS")
    for i in range(N):
        s = data['battery_smax'][i]
        px, qx = _polygon_pts(s, data.get('battery_polygon_sides', 8))
        ax.plot(px, qx, color=_COLORS[i%4], label=f"P{i+1}")
        ax.plot(s*np.cos(theta), s*np.sin(theta), color=_COLORS[i%4], lw=0.5, ls='--')
    ax.set_aspect('equal')
    ax.axhline(0, color='k', lw=0.3); ax.axvline(0, color='k', lw=0.3)
    ax.set_xlabel("P net (kW)"); ax.set_ylabel("Q (kVAR)")
    ax.legend(fontsize=6)
    ax.text(0.5, 0.02, "dashed = circle S_max", transform=ax.transAxes, fontsize=5, ha='center')
 
    # PV
    ax = axes[1]
    ax.set_title("PV")
    pf_min = data.get('pf_pv_min', 0.90)                        # REVIEW: EN 50549 / IEEE 1547
    tan_pv = math.tan(math.acos(pf_min))
    plotted = False
    for i in range(N):
        p_max = max(data['generation'][i])
        if p_max > 0:
            ax.fill([0, p_max, p_max, 0], [0, p_max*tan_pv, -p_max*tan_pv, 0],
                    color=_COLORS[i%4], alpha=0.25, label=f"P{i+1}")
            ax.plot([0, p_max, p_max, 0], [0, p_max*tan_pv, -p_max*tan_pv, 0],
                    color=_COLORS[i%4])
            plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "no PV generation", transform=ax.transAxes,
                ha='center', va='center', color='grey')
    ax.axhline(0, color='k', lw=0.3); ax.axvline(0, color='k', lw=0.3)
    ax.set_xlabel("P pv (kW)"); ax.set_ylabel("Q pv (kVAR)")
    ax.legend(fontsize=6)
    ax.text(0.5, 0.02, f"pf_pv_min={pf_min}", transform=ax.transAxes, fontsize=5, ha='center')
 
    # EV charger
    ax = axes[2]
    ax.set_title("EV charger")
    any_ev = False
    for i in range(N):
        if data.get('ev_capacity', [0]*N)[i] > 0:
            s = data.get('ev_smax', [0]*N)[i]
            px, qx = _polygon_pts(s, data.get('ev_polygon_sides', 8))
            ax.plot(px, qx, color=_COLORS[i%4], label=f"P{i+1}")
            ax.plot(s*np.cos(theta), s*np.sin(theta), color=_COLORS[i%4], lw=0.5, ls='--')
            any_ev = True
    if not any_ev:
        ax.text(0.5, 0.5, "disabled", transform=ax.transAxes,
                ha='center', va='center', color='grey')
    ax.set_aspect('equal')
    ax.axhline(0, color='k', lw=0.3); ax.axvline(0, color='k', lw=0.3)
    ax.set_xlabel("P net (kW)"); ax.set_ylabel("Q (kVAR)")
    if any_ev: ax.legend(fontsize=6)
 
    # HP
    ax = axes[3]
    ax.set_title("HP")
    pf_hp  = data.get('pf_hp', 0.97)                            # REVIEW: manufacturer data
    tan_hp = math.tan(math.acos(pf_hp))
    any_hp = False
    for i in range(N):
        p_max = data.get('hp_max_power', [0]*N)[i]
        if p_max > 0:
            ax.plot([0, p_max], [0, -p_max*tan_hp], color=_COLORS[i%4], label=f"P{i+1}")
            any_hp = True
    if not any_hp:
        ax.text(0.5, 0.5, "disabled", transform=ax.transAxes,
                ha='center', va='center', color='grey')
    ax.axhline(0, color='k', lw=0.3); ax.axvline(0, color='k', lw=0.3)
    ax.set_xlabel("P hp (kW)"); ax.set_ylabel("Q hp (kVAR)")
    if any_hp: ax.legend(fontsize=6)
    ax.text(0.5, 0.02, f"pf_hp={pf_hp} (REVIEW)", transform=ax.transAxes, fontsize=5, ha='center')
 
    fig.suptitle("Device Feasibility Operating Regions", fontweight='bold')
    _save(fig, root / "results" / "device_for", "device_for")
 
 
# --- result plots ---
 
def plot_trades(name, result, data, root):
    """Heatmap of bilateral P2P trades."""
    plt.rcParams.update(_rc(7))
    periods = np.arange(len(data["buy_price"]))
    trades  = np.array([[result["trades"][i,j,t] for t in periods]
                        for i,j in data["edges"]])
    fig, ax = plt.subplots(figsize=(3.5, 2.4), constrained_layout=True)
    import matplotlib as mpl; cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "trades", ["#C65D3B","#F7F7F5","#2F6FA3"])
    limit = max(1., float(np.abs(trades).max()))
    img   = ax.imshow(trades, cmap=cmap, vmin=-limit, vmax=limit,
                      aspect="auto", interpolation="nearest")
    for r in range(trades.shape[0]):
        for c in range(trades.shape[1]):
            v = trades[r,c]
            ax.text(c, r, "0" if abs(v)<.005 else f"{v:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if abs(v)>.58*limit else "#222222", fontweight="bold")
    ax.set_xticks(periods, [str(t) for t in periods])
    ax.set_yticks(np.arange(len(data["edges"])), [f"{i+1}→{j+1}" for i,j in data["edges"]])
    ax.set(xlabel="Period", ylabel="Edge (+ follows arrow)",
           title=f"{name.replace('_',' ').title()} trades")
    ax.set_xticks(np.arange(-.5,len(periods),1), minor=True)
    ax.set_yticks(np.arange(-.5,len(data["edges"]),1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for sp in ax.spines.values(): sp.set_visible(False)
    cb = fig.colorbar(img, ax=ax, fraction=.045, pad=.035)
    cb.set_label("Trade (kW)"); cb.outline.set_visible(False)
    out = root / "results" / name
    with (out / f"{name}_p2p_source_data.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["period"] + [f"p2p_{i+1}_to_{j+1}_kw" for i,j in data["edges"]])
        w.writerows(zip(periods, *trades))
    _save(fig, out, f"{name}_p2p")
 
 
def plot_network(name, result, data, root):
    """Feeder voltage profile per period."""
    plt.rcParams.update(_rc(7))
    net = result["network"]
    T   = len(data["buy_price"])
    bus_count = max(bus for bus,_ in (ast.literal_eval(k) for k in net["voltage_pu"])) + 1
    voltage   = np.zeros((bus_count, T))
    for key, val in net["voltage_pu"].items():
        bus, t = ast.literal_eval(key)
        voltage[bus, t] = val
    buses = np.arange(bus_count)
    fig, ax = plt.subplots(figsize=(3.5, 2.4), constrained_layout=True)
    cmap = plt.colormaps["viridis"].resampled(T)
    for t in range(T):
        ax.plot(buses, voltage[:,t], marker=".", markersize=3, linewidth=1.,
                color=cmap(t), label=f"t={t}")
    ax.axhline(.90, color="#C65D3B", lw=.8, ls="--")
    ax.axhline(1.10, color="#C65D3B", lw=.8, ls="--")
    for bus in data["agent_bus"]:
        ax.axvline(bus, color="#999999", lw=.5, ls=":")
    ax.set(xlabel="Bus", ylabel="Voltage (p.u.)",
           title=f"{name.replace('_',' ').title()} voltage")
    ax.legend(frameon=False, fontsize=5, ncol=2, loc="lower left")
    out = root / "results" / name
    with (out / f"{name}_voltage_profile_source_data.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bus"] + [f"t{t}_pu" for t in range(T)])
        w.writerows(zip(buses, *voltage.T))
    _save(fig, out, f"{name}_voltage_profile")
 
 
def plot_balance(name, result, data, root):
    """Per-prosumer net flows (grid + P2P bars) and BESS / EV SOC."""
    plt.rcParams.update(_rc(11))
    T, N    = len(data["buy_price"]), len(data["demand"])
    periods = np.arange(T)
    soc_x   = np.arange(T+1) - .5
    width   = .32
    rows    = []
    fig, axes = plt.subplots(1, N, figsize=(4.*N, 4.4), constrained_layout=True)
    any_ev  = any(data.get('ev_capacity',[0]*N)[i] > 0 for i in range(N))
 
    for i, ax in enumerate(axes):
        net_export = np.array([sum(v for (a,b,t),v in result["trades"].items()
                                   if a==i and t==p) for p in range(T)])
        net_grid = result["buy"][i] - result["sell"][i]
        net_p2p  = -net_export
        soc_pct  = 100. * result["soc"][i] / max(1e-9, data["battery_capacity"][i])
        row = dict(net_grid_kw=net_grid, net_p2p_kw=net_p2p, soc_pct=soc_pct)
 
        ev_soc_pct = None
        if data.get('ev_capacity',[0]*N)[i] > 0 and result.get('ev_soc') is not None:
            cap_eff    = data['ev_capacity'][i] * data.get('ev_cap_factor',[1.]*N)[i]
            ev_soc_pct = 100. * result['ev_soc'][i] / max(1e-9, cap_eff)
            row['ev_soc_pct'] = ev_soc_pct
        rows.append(row)
 
        ax.bar(periods-width/2, net_grid, width, color="#5C7A99")
        ax.bar(periods+width/2, net_p2p,  width, color="#8B6BB1")
        ax.axhline(0, color="#222222", lw=1.)
        ax.set_xticks(periods, [str(t) for t in periods])
        ax.set_xlim(soc_x[0], soc_x[-1])
        ax.set_title(f"Prosumer {i+1}", fontweight="bold", pad=10)
        ax.set_xlabel("Period"); ax.set_ylabel("Net power (kW)")
        ax.margins(y=.15)
        ax2 = ax.twinx()
        ax2.plot(soc_x, soc_pct, color="#2E6F52", lw=2., marker="o", markersize=5)
        if ev_soc_pct is not None:
            ax2.plot(soc_x, ev_soc_pct, color="#B5720A", lw=2., marker="s",
                     markersize=5, ls="--")
        ax2.set_ylim(0, 105); ax2.set_ylabel("SOC (%)")
 
    fig.suptitle(f"{name.replace('_',' ').title()} net flows", fontsize=15, fontweight="bold")
    handles = [plt.Rectangle((0,0),1,1,color="#5C7A99"),
               plt.Rectangle((0,0),1,1,color="#8B6BB1"),
               plt.Line2D([0],[0],color="#2E6F52",marker="o",markersize=5)]
    labels  = ["Grid (net,+=buy)","P2P (net,+=buy)","BESS SOC"]
    if any_ev:
        handles.append(plt.Line2D([0],[0],color="#B5720A",marker="s",markersize=5,ls="--"))
        labels.append("EV SOC")
    fig.legend(handles, labels, frameon=False, ncol=len(handles),
               loc="upper center", bbox_to_anchor=(.5,-.02))
 
    out = root / "results" / name
    with (out / f"{name}_energy_balance_source_data.csv").open("w", newline="") as f:
        w = csv.writer(f)
        hdr = ["prosumer","period","net_grid_kw","net_p2p_kw","soc_start_pct","soc_end_pct"]
        if any_ev: hdr += ["ev_soc_start_pct","ev_soc_end_pct"]
        w.writerow(hdr)
        for i, row in enumerate(rows):
            for t in range(T):
                line = [i+1, t, row["net_grid_kw"][t], row["net_p2p_kw"][t],
                        row["soc_pct"][t], row["soc_pct"][t+1]]
                if any_ev:
                    ep = row.get("ev_soc_pct")
                    line += [ep[t] if ep is not None else 0.,
                             ep[t+1] if ep is not None else 0.]
                w.writerow(line)
    _save(fig, out, f"{name}_energy_balance")
 
 
def plot_devices(name, result, data, root):
    """HP indoor temperature and EV charge/discharge. No-op if both disabled."""
    N = len(data['demand'])
    T = len(data['buy_price'])
    has_hp = [data.get('hp_max_power',[0]*N)[i] > 0 for i in range(N)]
    has_ev = [data.get('ev_capacity', [0]*N)[i] > 0 for i in range(N)]
    if not any(has_hp) and not any(has_ev):
        return
 
    plt.rcParams.update(_rc(9))
    periods = np.arange(T)
    soc_x   = np.arange(T+1) - .5
    fig, axes_grid = plt.subplots(2, N, figsize=(4.*N, 5.5),
                                  constrained_layout=True, sharex=True)
    if N == 1:
        axes_grid = np.array(axes_grid).reshape(2,1)
    rows_data = []
 
    for i in range(N):
        ax_top, ax_bot = axes_grid[0,i], axes_grid[1,i]
        row = {}
        if has_hp[i] and result.get('temp') is not None:
            temp = result['temp'][i]
            ax_top.plot(soc_x, temp, color="#C65D3B", lw=2., marker="o", markersize=4)
            ax_top.axhline(data['hp_T_min'], color="#999999", lw=.8, ls="--",
                           label=f"T_min={data['hp_T_min']}°C")
            ax_top.axhline(data['hp_T_max'], color="#444444", lw=.8, ls="--",
                           label=f"T_max={data['hp_T_max']}°C")
            ax_top.set_ylabel("Temp (°C)"); ax_top.legend(fontsize=6)
            row['temp'] = temp
        else:
            ax_top.set_visible(False)
 
        if has_ev[i] and result.get('ev_charge') is not None:
            ev_c  = result['ev_charge'][i]
            ev_d  = result['ev_discharge'][i]
            avail = np.array(data['ev_availability'][i])
            for t in range(T):
                if avail[t] == 0:
                    ax_bot.axvspan(t-.5, t+.5, color="#EEEEEE", zorder=0)
            ax_bot.bar(periods-.2, ev_c,  .35, color="#2F6FA3", label="charge")
            ax_bot.bar(periods+.2, -ev_d, .35, color="#C65D3B", label="discharge")
            ax_bot.axhline(0, color="#222222", lw=.8)
            ax_bot.set_ylabel("EV (kW)\n+=charge, -=discharge")
            ax_bot.legend(frameon=False, fontsize=7)
            row['ev_charge'] = ev_c; row['ev_discharge'] = ev_d
        else:
            ax_bot.set_visible(False)
 
        ax_bot.set_xlabel("Period")
        ax_top.set_title(f"Prosumer {i+1}", fontweight="bold")
        rows_data.append(row)
 
    fig.suptitle(f"{name.replace('_',' ').title()} device profiles",
                 fontsize=13, fontweight="bold")
    out = root / "results" / name
    with (out / f"{name}_devices_source_data.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prosumer","period","temp_degC","ev_charge_kw","ev_discharge_kw"])
        for i, row in enumerate(rows_data):
            tv = row.get('temp', [None]*(T+1))
            ec = row.get('ev_charge', [None]*T)
            ed = row.get('ev_discharge', [None]*T)
            for t in range(T):
                w.writerow([i+1, t,
                            tv[t] if tv[t] is not None else "",
                            ec[t] if ec[t] is not None else "",
                            ed[t] if ed[t] is not None else ""])
    _save(fig, out, f"{name}_devices")