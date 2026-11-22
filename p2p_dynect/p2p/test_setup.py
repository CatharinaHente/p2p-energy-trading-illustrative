"""Fast offline checks for the P2P -> DynLQGameTV matrix assembly.

Run `pytest p2p/test_setup.py -q` from the project root. Neither Julia,
Gurobi, pandapower nor DyNECT are required for these assembly tests.
"""
import numpy as np
from .setup import assemble_game, _feeder_maps


def _fixture():
    T=4
    d=dict(
        demand=[[2.]*T,[3.]*T],generation=[[1.]*T,[1.]*T],
        ev_driving_load=[[0.]*T,[0.]*T],ev_availability=[[1]*T,[1]*T],
        hp_base_power=[[1.]*T,[1.]*T],hp_q_heat=[[0.]*T,[0.]*T],
        buy_price=[.2]*T,sell_price=[.1]*T,p2p_price=[.15]*T,
        edges=[(0,1)],dt=.5,T_eval=2,grid_limit=20.,trade_limit=5.,
        battery_capacity=[10.,0.],battery_power=[2.,0.],initial_soc=[5.,0.],
        ev_capacity=[8.,0.],ev_cap_factor=[1.,1.],ev_power=[2.,0.],ev_initial_soc=[7.,0.],
        hp_max_power=[1.,1.],hp_T_init=[20.,20.],hp_T_min=19.,hp_T_max=23.,
        soc_min_fraction=.1,soc_max_fraction=.95,
        ev_soc_min_departure=[.1,0.],season='winter',
        eta_charge=.95,eta_discharge=.95,ev_eta_charge=.95,ev_eta_discharge=.95,
        battery_cost=.005,ev_cost=.005,hp_cost=0.,p_cost=.001,trade_fee=.001,
        ev_terminal_price=.25,cyclic_soc=True,ev_cyclic_soc=False,ev_v2g=True,
        agent_bus=[1,2],line_limit_mva=5.)
    feeder=dict(name='synthetic',root=0,bus_count=3,branches=[(0,1),(1,2)],
                r_ohm=np.array([.1,.1]),p_load_mw=np.array([0,.001,.001]),
                voltage_kv=np.array([12.66]*3),voltage_min=np.array([.9]*3),
                voltage_max=np.array([1.1]*3),root_voltage=1.)
    return d,feeder


def test_assembly_dimensions_and_positive_definiteness():
    data,feeder=_fixture()
    a=assemble_game(data,grid=feeder)
    kw=a['kwargs']
    assert len(kw['A'])==4 and len(kw['Q'])==3
    assert kw['A'][0].shape==(4,4)
    assert len(a['x0'])==4
    assert len(kw['b_x'][-1])==len(kw['b_x'][0])+2
    assert len(kw['b_u'][0])==2+2*2+2*(3-1)
    for t in range(4):
        for i in range(2):
            assert np.linalg.eigvalsh(kw['R'][t][i][i]).min()>0
            assert np.linalg.eigvalsh(kw['Q'][0][i]).min()>0


def test_zero_trading_baseline_physical_feasibility():
    data,feeder=_fixture()
    a=assemble_game(data,grid=feeder)
    kw=a['kwargs']; T=4; N=2
    u=[]
    for i in range(N):
        ni=len(a['controls'][i]);mat=np.zeros((T,ni))
        mat[:,a['controls'][i]['buy']]=np.array(data['demand'][i])-np.array(data['generation'][i])
        u.append(mat)
    x=a['x0'].copy()
    for t in range(T):
        for i in range(N):
            assert np.max(kw['C_loc'][t][i]@u[i][t]-kw['b_loc'][t][i]) <=1e-9
        assert np.max(sum((kw['C_u'][t][i]@u[i][t] for i in range(N)),np.zeros(len(kw['b_u'][t])))-kw['b_u'][t])<=1e-9
        x=kw['A'][t]@x+kw['c'][t]+sum((kw['B'][t][i]@u[i][t] for i in range(N)),np.zeros(len(x)))
        assert np.max(kw['C_x'][t]@x-kw['b_x'][t])<=1e-9
    np.testing.assert_allclose(x,a['x0'])


def test_feeder_injection_sign_and_clearing_epsilon():
    data,feeder=_fixture()
    f0,L,drop=_feeder_maps(feeder,data['agent_bus'])
    np.testing.assert_allclose(f0,[.002,.001])
    net_inj=np.array([-1.,-2.])
    np.testing.assert_allclose(f0+L@net_inj,[.005,.003])
    assert drop[2,0]>0 and drop[2,1]>0
    a=assemble_game(data,grid=feeder,clearing_epsilon_kw=.01)
    for bt in a['kwargs']['b_u']:
        np.testing.assert_allclose(bt[:2],[.01,.01])
