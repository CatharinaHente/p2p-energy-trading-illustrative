"""Visualisations for P2P prosumer results and device FORs.

All functions save to root/results/<name>/ in pdf/svg/tiff/png and open the figure.
Designed for T=96 (15-min resolution) but works for any T via data['dt'].

plot_for()    — device FORs in PQ plane (call once on data dict)
plot_prices() — electricity prices for the scenario (call once on data dict)
plot_trades() — bilateral P2P trade time series per edge
plot_network()— feeder voltage profile across buses
plot_balance()— per-prosumer power flows + BESS/EV SOC
plot_devices()— HP temperature and EV charge/discharge profiles
"""
import math, ast, csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ── shared helpers ────────────────────────────────────────────────────────────

def _rc(size=8):
    return {"font.family":     "sans-serif",
            "font.sans-serif": ["Arial","Helvetica","DejaVu Sans"],
            "font.size":       size,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype":    42,
            "svg.fonttype":    "none"}

def _save(fig, out, stem):
    out.mkdir(parents=True, exist_ok=True)
    for ext, kw in [(".pdf",{}),(".svg",{}),(".tiff",{"dpi":600}),(".png",{"dpi":300})]:
        fig.savefig(out / f"{stem}{ext}", bbox_inches="tight", **kw)
    plt.show()
    plt.close(fig)

def _t(data):
    """Hour-of-day x-axis for T timesteps."""
    T  = len(data['buy_price'])
    dt = data.get('dt', 1.0)
    return np.arange(T) * dt, np.arange(T+1) * dt, T, dt

def _polygon_pts(s_max, n):
    a = [math.pi/n + 2*math.pi*k/n for k in range(n+1)]
    return [s_max*math.cos(x) for x in a], [s_max*math.sin(x) for x in a]

def _note(ax, txt):
    ax.text(0.5, -0.15, txt, transform=ax.transAxes,
            ha='center', fontsize=7, color='#555555', style='italic')

_C = ["#2F6FA3","#C65D3B","#2E6F52","#B5720A"]


# ── Device FOR visualisation ──────────────────────────────────────────────────

def plot_for(data, root):
    """Device Feasibility Operating Regions in the PQ plane."""
    N     = len(data['demand'])
    theta = np.linspace(0, 2*np.pi, 200)
    plt.rcParams.update(_rc(8))
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)

    # BESS
    ax = axes[0]
    ax.set_title("BESS")
    for i in range(N):
        s = data['battery_smax'][i]
        if s == 0: continue
        px, qx = _polygon_pts(s, data.get('battery_polygon_sides', 8))
        ax.plot(px, qx, color=_C[i%4], label=f"P{i+1}")
        ax.plot(s*np.cos(theta), s*np.sin(theta), color=_C[i%4], lw=0.5, ls='--')
    ax.set_aspect('equal')
    ax.axhline(0, color='k', lw=0.3); ax.axvline(0, color='k', lw=0.3)
    ax.set_xlabel("P net (kW)"); ax.set_ylabel("Q (kVAR)")
    ax.legend(fontsize=6)
    _note(ax, "dashed = S_max circle")

    # PV
    ax = axes[1]
    ax.set_title("PV")
    pf_min = data.get('pf_pv_min', 0.90)
    tan_pv = math.tan(math.acos(pf_min))
    plotted = False
    for i in range(N):
        p_max = max(data['generation'][i])
        if p_max > 0:
            ax.fill([0, p_max, p_max, 0],
                    [0, p_max*tan_pv, -p_max*tan_pv, 0],
                    color=_C[i%4], alpha=0.25, label=f"P{i+1}")
            ax.plot([0, p_max, p_max, 0],
                    [0, p_max*tan_pv, -p_max*tan_pv, 0],
                    color=_C[i%4])
            plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "no PV generation", transform=ax.transAxes,
                ha='center', va='center', color='grey')
    ax.axhline(0, color='k', lw=0.3); ax.axvline(0, color='k', lw=0.3)
    ax.set_xlabel("P pv (kW)"); ax.set_ylabel("Q pv (kVAR)")
    ax.legend(fontsize=6)
    _note(ax, f"pf_pv_min={pf_min}  (REVIEW: EN 50549-1 / IEEE 1547)")

    # EV
    ax = axes[2]
    ax.set_title("EV charger")
    any_ev = False
    for i in range(N):
        if data.get('ev_capacity', [0]*N)[i] > 0:
            s = data.get('ev_smax', [0]*N)[i]
            if s == 0: continue
            px, qx = _polygon_pts(s, data.get('ev_polygon_sides', 8))
            ax.plot(px, qx, color=_C[i%4], label=f"P{i+1}")
            ax.plot(s*np.cos(theta), s*np.sin(theta), color=_C[i%4], lw=0.5, ls='--')
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
    pf_hp  = data.get('pf_hp', 0.97)
    tan_hp = math.tan(math.acos(pf_hp))
    any_hp = False
    for i in range(N):
        p_max = data.get('hp_max_power', [0]*N)[i]
        if p_max > 0:
            ax.plot([0, p_max], [0, -p_max*tan_hp], color=_C[i%4], label=f"P{i+1}")
            any_hp = True
    if not any_hp:
        ax.text(0.5, 0.5, "disabled", transform=ax.transAxes,
                ha='center', va='center', color='grey')
    ax.axhline(0, color='k', lw=0.3); ax.axvline(0, color='k', lw=0.3)
    ax.set_xlabel("P hp (kW)"); ax.set_ylabel("Q hp (kVAR)")
    if any_hp: ax.legend(fontsize=6)
    _note(ax, f"pf_hp={pf_hp}  (REVIEW: manufacturer data)")

    fig.suptitle("Device Feasibility Operating Regions", fontweight='bold')
    _save(fig, root / "results" / "device_for", "device_for")


# ── Price visualisation ───────────────────────────────────────────────────────

def plot_prices(name, data, root):
    """Electricity price schedule for the scenario."""
    t, _, T, dt = _t(data)
    plt.rcParams.update(_rc(8))
    fig, ax = plt.subplots(figsize=(11, 3), constrained_layout=True)
    ax.step(t, data['buy_price'],  color='#C65D3B', lw=2.0,
            label='Buy price', where='post')
    ax.step(t, data['p2p_price'],  color='#2F6FA3', lw=2.0,
            label='P2P price', where='post')
    ax.step(t, data['sell_price'], color='#2E6F52', lw=2.0,
            label='Sell price', where='post')
    ax.fill_between(t, data['sell_price'], data['p2p_price'],
                    alpha=0.10, color='#2F6FA3', step='post')
    ax.fill_between(t, data['p2p_price'],  data['buy_price'],
                    alpha=0.10, color='#C65D3B', step='post')
    ax.set_xlabel('Hour of day'); ax.set_ylabel('EUR/kWh')
    ax.set_xlim(0, T*dt)
    ax.set_title(f"{data.get('season','').title()} electricity prices",
                 fontweight='bold')
    ax.legend(fontsize=8)
    _note(ax, 'Sell < P2P < Buy ensures trading is always beneficial')
    out = root / "results"
    _save(fig, out, f"{name}_prices")


# ── P2P trade time series ─────────────────────────────────────────────────────

def plot_trades(name, result, data, root):
    """Bilateral P2P trade time series — one panel per edge."""
    t, _, T, dt = _t(data)
    edges = data['edges']
    plt.rcParams.update(_rc(8))

    fig, axes = plt.subplots(len(edges), 1,
                             figsize=(12, 2.8*len(edges)),
                             sharex=True, constrained_layout=True)
    if len(edges) == 1: axes = [axes]

    for ax, (i, j), color in zip(axes, edges, _C):
        vals = np.array([result['trades'][i,j,t_] for t_ in range(T)])
        ax.fill_between(t, 0, vals, where=vals >= 0,
                        alpha=0.35, color=color, step='post')
        ax.fill_between(t, 0, vals, where=vals < 0,
                        alpha=0.25, color='#999999', step='post')
        ax.step(t, vals, color=color, lw=1.5, where='post',
                label=f'P{i+1}→P{j+1}  (+= P{i+1} sells)')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_ylabel('kW')
        ax.set_xlim(0, T*dt)
        ax.legend(fontsize=7, loc='upper right')

    axes[-1].set_xlabel('Hour of day')
    fig.suptitle(f"{name.replace('_',' ').title()} — P2P trades",
                 fontweight='bold', fontsize=11)

    out = root / 'results' / name
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{name}_p2p_source_data.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(['hour'] + [f'p2p_{i+1}_to_{j+1}_kw' for i,j in edges])
        for t_idx in range(T):
            w.writerow([round(t_idx*dt, 4)] +
                       [result['trades'][i,j,t_idx] for i,j in edges])
    _save(fig, out, f"{name}_p2p")


# ── Feeder voltage profile ────────────────────────────────────────────────────

def plot_network(name, result, data, root):
    """Feeder voltage profile across buses — envelope for T=96."""
    plt.rcParams.update(_rc(8))
    net = result['network']
    T   = len(data['buy_price'])

    bus_count = max(bus for bus,_ in
                    (ast.literal_eval(k) for k in net['voltage_pu'])) + 1
    voltage = np.zeros((bus_count, T))
    for key, val in net['voltage_pu'].items():
        bus, t_idx = ast.literal_eval(key)
        voltage[bus, t_idx] = val
    buses = np.arange(bus_count)

    plt.rcParams.update(_rc(8))
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)

    # Envelope: min/max/mean across timesteps
    v_min  = voltage.min(axis=1)
    v_max  = voltage.max(axis=1)
    v_mean = voltage.mean(axis=1)

    ax.fill_between(buses, v_min, v_max, alpha=0.20, color='#2F6FA3',
                    label='Min–Max range')
    ax.plot(buses, v_mean, color='#2F6FA3', lw=2.0, label='Mean')
    ax.plot(buses, v_min,  color='#2F6FA3', lw=1.0, ls='--', alpha=0.7)
    ax.plot(buses, v_max,  color='#2F6FA3', lw=1.0, ls='--', alpha=0.7)

    # Also show worst (min) and best (max) single timesteps
    worst_t = int(voltage.min(axis=0).argmin()) if T > 1 else 0
    best_t  = int(voltage.max(axis=0).argmax()) if T > 1 else 0
    ax.plot(buses, voltage[:, worst_t], color='#C65D3B', lw=1.5, ls='-.',
            label=f'Worst timestep (t={worst_t})')

    ax.axhline(0.90, color='#C65D3B', lw=1.0, ls='--', alpha=0.7, label='V limits')
    ax.axhline(1.10, color='#C65D3B', lw=1.0, ls='--', alpha=0.7)
    for bus in data['agent_bus']:
        ax.axvline(bus, color='#999999', lw=0.8, ls=':', alpha=0.7)
        ax.text(bus, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0.88,
                f'P{data["agent_bus"].index(bus)+1}',
                fontsize=6.5, color='#555555', ha='center')

    ax.set_xlabel('Bus index'); ax.set_ylabel('Voltage (p.u.)')
    ax.set_title(f"{name.replace('_',' ').title()} — feeder voltage profile",
                 fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    _note(ax, f'Envelope over T={T} timesteps · prosumer buses marked')

    out = root / 'results' / name
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{name}_voltage_source_data.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(['bus','v_min_pu','v_mean_pu','v_max_pu'])
        for b in buses:
            w.writerow([b, round(v_min[b],5), round(v_mean[b],5),
                        round(v_max[b],5)])
    _save(fig, out, f"{name}_voltage_profile")


# ── Per-prosumer energy balance ───────────────────────────────────────────────

def plot_balance(name, result, data, root):
    """Per-prosumer power flows (area) + BESS/EV SOC."""
    t, t_soc, T, dt = _t(data)
    N   = len(data['demand'])
    plt.rcParams.update(_rc(8))
    any_ev = any(data.get('ev_capacity',[0]*N)[i] > 0 for i in range(N))

    fig, axes = plt.subplots(2, N, figsize=(4.5*N, 8),
                             constrained_layout=True, sharex=False)
    if N == 1: axes = axes.reshape(2,1)

    rows = []
    for i in range(N):
        pname = data.get('prosumer_names', [f'P{k+1}' for k in range(N)])[i]

        # ── top: power flows ──────────────────────────────────────────────
        ax = axes[0, i]
        demand = np.array(data['demand'][i])
        gen    = np.array(data['generation'][i])
        net_grid = result['buy'][i] - result['sell'][i]
        net_p2p  = -np.array([
            sum(v for (a,b,t_),v in result['trades'].items()
                if a==i and t_==t_idx)
            for t_idx in range(T)])
        rows.append(dict(net_grid=net_grid, net_p2p=net_p2p))

        ax.fill_between(t, 0, gen,
                        alpha=0.45, color='#F0C040', step='post',
                        label='PV generation')
        ax.fill_between(t, 0, net_grid,
                        where=net_grid >= 0, alpha=0.45,
                        color='#2F6FA3', step='post', label='Grid buy')
        ax.fill_between(t, 0, net_grid,
                        where=net_grid < 0,  alpha=0.45,
                        color='#6B4C9A', step='post', label='Grid sell')
        ax.fill_between(t, 0, net_p2p,
                        where=net_p2p >= 0,  alpha=0.35,
                        color='#2E6F52', step='post', label='P2P buy')
        ax.fill_between(t, 0, net_p2p,
                        where=net_p2p < 0,   alpha=0.35,
                        color='#B5720A', step='post', label='P2P sell')
        ax.plot(t, demand, color='k', lw=1.5, ls='--', label='Demand')
        ax.axhline(0, color='k', lw=0.5)

        ax.set_title(f"P{i+1}: {pname}", fontweight='bold', fontsize=9)
        ax.set_ylabel('kW')
        ax.set_xlim(0, T*dt)
        ax.legend(fontsize=6, ncol=2, loc='upper right')

        # ── bottom: SOC ───────────────────────────────────────────────────
        ax2 = axes[1, i]
        soc_pct = 100 * result['soc'][i] / max(data['battery_capacity'][i], 1e-9)
        ax2.fill_between(t_soc, 0, soc_pct, alpha=0.3, color='#2E6F52')
        ax2.plot(t_soc, soc_pct, color='#2E6F52', lw=2.0, label='BESS SOC')

        # SOC bounds
        ax2.axhline(100*data['soc_min_fraction'], color='#2E6F52',
                    lw=0.8, ls='--', alpha=0.6)
        ax2.axhline(100*data['soc_max_fraction'], color='#2E6F52',
                    lw=0.8, ls='--', alpha=0.6)

        if data.get('ev_capacity',[0]*N)[i] > 0 and result.get('ev_soc') is not None:
            cap_eff = (data['ev_capacity'][i] *
                       data.get('ev_cap_factor',[1.]*N)[i])
            ev_pct  = 100 * result['ev_soc'][i] / max(cap_eff, 1e-9)
            ax2.fill_between(t_soc, 0, ev_pct, alpha=0.2, color='#B5720A')
            ax2.plot(t_soc, ev_pct, color='#B5720A', lw=2.0,
                     ls='--', label='EV SOC')
            # EV DoD floor
            ax2.axhline(100*(1-data.get('ev_dod_max',0.30)),
                        color='#B5720A', lw=0.8, ls='--', alpha=0.6)

        ax2.set_ylim(0, 108)
        ax2.set_ylabel('SOC (%)')
        ax2.set_xlabel('Hour of day')
        ax2.set_xlim(0, T*dt)
        ax2.legend(fontsize=7)

    fig.suptitle(f"{name.replace('_',' ').title()} — energy balance",
                 fontweight='bold', fontsize=12)

    out = root / 'results' / name
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{name}_energy_balance_source_data.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(['prosumer','hour','net_grid_kw','net_p2p_kw'])
        for i, row in enumerate(rows):
            for t_idx in range(T):
                w.writerow([i+1, round(t_idx*dt,4),
                             round(row['net_grid'][t_idx],4),
                             round(row['net_p2p'][t_idx],4)])
    _save(fig, out, f"{name}_energy_balance")


# ── Device profiles (HP temp + EV charge) ────────────────────────────────────

def plot_devices(name, result, data, root):
    """HP indoor temperature and EV charge/discharge profiles."""
    N      = len(data['demand'])
    t, t_soc, T, dt = _t(data)
    has_hp = [data.get('hp_max_power',[0]*N)[i] > 0 for i in range(N)]
    has_ev = [data.get('ev_capacity', [0]*N)[i] > 0 for i in range(N)]
    if not any(has_hp) and not any(has_ev):
        return

    plt.rcParams.update(_rc(8))
    fig, axes = plt.subplots(2, N, figsize=(4.5*N, 7),
                             constrained_layout=True)
    if N == 1: axes = axes.reshape(2,1)

    for i in range(N):
        ax_top = axes[0, i]
        ax_bot = axes[1, i]
        pname  = data.get('prosumer_names',[f'P{k+1}' for k in range(N)])[i]
        ax_top.set_title(f"P{i+1}: {pname}", fontweight='bold', fontsize=9)

        # HP temperature
        if has_hp[i] and result.get('temp') is not None:
            temp = result['temp'][i]
            ax_top.fill_between(t_soc, data['hp_T_min'], temp,
                                alpha=0.25, color='#C65D3B')
            ax_top.plot(t_soc, temp, color='#C65D3B', lw=2.0,
                        label='Indoor temp')
            ax_top.axhline(data['hp_T_min'], color='#999999',
                           lw=1.0, ls='--',
                           label=f"T_min={data['hp_T_min']}°C")
            ax_top.axhline(data['hp_T_max'], color='#444444',
                           lw=1.0, ls='--',
                           label=f"T_max={data['hp_T_max']}°C")
            ax_top.set_ylabel('Temp (°C)')
            ax_top.set_xlim(0, T*dt)
            ax_top.legend(fontsize=7)

            # HP power on secondary axis
            hp_p = np.array(data['hp_base_power'][i])
            ax_hp = ax_top.twinx()
            ax_hp.step(t, hp_p, color='#C65D3B', lw=1.0,
                       ls=':', alpha=0.6, where='post',
                       label='HP power (kW)')
            ax_hp.set_ylabel('HP power (kW)', fontsize=7)
            ax_hp.spines['top'].set_visible(False)
        else:
            ax_top.text(0.5, 0.5, 'No HP', transform=ax_top.transAxes,
                        ha='center', va='center', color='grey')
            ax_top.set_visible(False)

        # EV charge/discharge
        if has_ev[i] and result.get('ev_charge') is not None:
            ev_c  = result['ev_charge'][i]
            ev_d  = result['ev_discharge'][i]
            avail = np.array(data['ev_availability'][i])

            # Shade away periods
            away_start = None
            for t_idx in range(T):
                if avail[t_idx] == 0 and away_start is None:
                    away_start = t_idx * dt
                elif avail[t_idx] == 1 and away_start is not None:
                    ax_bot.axvspan(away_start, t_idx*dt,
                                   alpha=0.10, color='#888888', zorder=0)
                    away_start = None
            if away_start is not None:
                ax_bot.axvspan(away_start, T*dt,
                               alpha=0.10, color='#888888', zorder=0)

            ax_bot.fill_between(t, 0,  ev_c,  alpha=0.4, color='#2F6FA3',
                                step='post', label='Charge')
            ax_bot.fill_between(t, 0, -ev_d,  alpha=0.4, color='#C65D3B',
                                step='post', label='Discharge')
            ax_bot.step(t,  ev_c,  color='#2F6FA3', lw=1.5, where='post')
            ax_bot.step(t, -ev_d,  color='#C65D3B', lw=1.5, where='post')
            ax_bot.axhline(0, color='k', lw=0.5)
            ax_bot.set_ylabel('EV power (kW)\n+ = charge, − = discharge')
            ax_bot.legend(fontsize=7, loc='upper right')
            _note(ax_bot, 'Grey shading = EV away (driving)')
        else:
            ax_bot.text(0.5, 0.5, 'No EV', transform=ax_bot.transAxes,
                        ha='center', va='center', color='grey')
            ax_bot.set_visible(False)

        ax_bot.set_xlabel('Hour of day')
        ax_bot.set_xlim(0, T*dt)

    fig.suptitle(f"{name.replace('_',' ').title()} — device profiles",
                 fontweight='bold', fontsize=12)
    out = root / 'results' / name
    out.mkdir(parents=True, exist_ok=True)
    _save(fig, out, f"{name}_devices")