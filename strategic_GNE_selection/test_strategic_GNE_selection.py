"""Dependency-light tests; full Gurobi/DyNECT integration runs locally in notebooks."""
from pathlib import Path
import sys
import unittest
from dataclasses import dataclass
from types import SimpleNamespace,ModuleType
from unittest.mock import patch
import numpy as np
from scipy import sparse

sys.path.insert(0,str(Path(__file__).resolve().parent))
from strategic_GNE_selection import (aggregate_to_hour,clean_local,
    injection_intervals,fairness_metrics,reduction_counts,StrategicGame)


class TestIndependentFunctions(unittest.TestCase):
    def test_aggregate_energy_and_availability(self):
        data={
            'dt':.5,'T_eval':4,'demand':[[1,3,5,7],[2,4,6,8]],
            'generation':[[0,2,0,2],[1,1,1,1]],
            'hp_base_power':[[1,3,1,3],[0,0,0,0]],
            'hp_q_heat':[[0,2,0,2],[1,2,3,4]],
            'ev_driving_load':[[0,.1,.2,.3],[0,0,0,0]],
            'ev_availability':[[1,0,1,1],[1,1,0,1]],
            'buy_price':[1,3,5,7],'sell_price':[0,2,4,6],
            'p2p_price':[5,7,9,11]
        }
        out,mixed=aggregate_to_hour(data,1.)
        np.testing.assert_allclose(out['demand'],[[2,6],[3,7]])
        np.testing.assert_allclose(out['ev_driving_load'],[[.1,.5],[0,0]])
        np.testing.assert_array_equal(out['ev_availability'],[[0,1],[1,0]])
        self.assertEqual(int(mixed.sum()),2)
        self.assertEqual(out['T_eval'],2)
        self.assertEqual(data['dt'],.5)  # source untouched

    def test_duplicate_local_constraints(self):
        A=np.array([[1.,0.],[1.,0.],[0.,1.],[0.,0.]])
        b=np.array([2.,1.,3.,4.])
        ar,br=clean_local(A,b)
        self.assertEqual(ar.shape,(2,2))
        np.testing.assert_allclose(br,[1.,3.])
        with self.assertRaises(ValueError):
            clean_local(np.zeros((1,2)),np.array([-1.]))

    def test_fairness_and_NBS(self):
        mm=fairness_metrics([5.,4.],[7.,5.])
        self.assertAlmostEqual(mm['total_gain'],3.)
        self.assertAlmostEqual(mm['nbs_log'],np.log(2.))
        self.assertTrue(mm['individually_rational'])
        mm2=fairness_metrics([8.,4.],[7.,5.])
        self.assertEqual(mm2['n_losers'],1)
        self.assertTrue(np.isneginf(mm2['nbs_log']))

    def test_pseudogradient_separable_costs_and_violations(self):
        obj=StrategicGame.__new__(StrategicGame)
        obj.N=2;obj.n=2;obj.separable=True
        obj.slices=[slice(0,1),slice(1,2)]
        obj.Hi=[sparse.csr_matrix([[2.]]),sparse.csr_matrix([[4.]])]
        obj.fi=[np.array([-2.]),np.array([-4.])]
        obj.offsets=np.array([5.,10.])
        obj.A=sparse.csr_matrix([[1.,1.],[-1.,0.]])
        obj.b=np.array([1.5,0.])
        np.testing.assert_allclose(obj.model_costs([1.,.5]),[4.,8.5])
        self.assertEqual(obj.feasibility([1.,.5])['max_violation'],0.)
        self.assertAlmostEqual(obj.feasibility([2.,0.])['max_violation'],.5)


class TestReductionStructure(unittest.TestCase):
    def test_single_owner_localization_preserves_edge(self):
        from strategic_GNE_selection import rebuild_constraints,lp_prune_local
        class DummyGame:
            def __init__(self,**kwargs):
                self.__dict__.update(kwargs)
                self.N=2;self.T_hor=1;self.nu=[2,2];self.m_u=len(kwargs['b_u'][0])
        @dataclass
        class FakeCase:
            game:object
            N:int=2
            T:int=1
            shared_labels:tuple=('edge_0_1_pos','flow_x_upper')
        # Shared edge row contains both players; feeder row belongs only to P0.
        kw={key:[] for key in (
            'A','B','Q','R','P','c','q','r','p','C_x','b_x',
            'C_loc','b_loc','C_u','b_u')}
        kw['C_loc']=[[np.array([[1.,0.],[-1.,0.],[2.,0.]]),
                      np.array([[1.,0.],[-1.,0.]])]]
        kw['b_loc']=[[np.array([1.,0.,2.]),np.array([1.,0.])]]
        kw['C_u']=[[np.array([[1.,0.],[0.,1.]]),np.array([[1.,0.],[0.,0.]])]]
        kw['b_u']=[np.array([0.,1.])]
        fake=FakeCase(DummyGame(**kw))
        module=ModuleType('dynect');module.DynLQGameTV=DummyGame
        with patch.dict(sys.modules,{'dynect':module}):
            reduced,moved=rebuild_constraints(fake,clean_locals=True,
                                               localize_single_agent=True)
            self.assertEqual(moved,['flow_x_upper'])
            self.assertEqual(reduced.shared_labels,['edge_0_1_pos'])
            self.assertEqual(reduced.game.C_u[0][0].shape,(1,2))
            self.assertTrue(any(np.array_equal(row,[0.,1.])
                                for row in reduced.game.C_loc[0][0]))
            final,report=lp_prune_local(reduced,tol=1e-9)
            self.assertLessEqual(len(final.game.b_loc[0][0]),
                                 len(reduced.game.b_loc[0][0]))
            self.assertEqual(report[0]['unresolved'],0)


    def test_snapshot_roundtrip_with_mock_adapter(self):
        """Checks the no-pickle snapshot format and exact local-row replay."""
        import tempfile
        from strategic_GNE_selection import save_reduced_model,load_reduced_model,rebuild_constraints
        class DummyGame:
            def __init__(self,**kw):
                self.__dict__.update(kw)
                self.N=2;self.T_hor=1;self.nu=[1,1]
                self.m_u=len(kw['b_u'][0])
        @dataclass
        class Case:
            game:object
            x0:object
            data:dict
            grid:dict
            choices:dict
            shared_labels:list
            controls:list
            N:int=2
            T:int=1
        raw={key:[] for key in (
             'A','B','Q','R','P','c','q','r','p','C_x','b_x',
             'C_loc','b_loc','C_u','b_u')}
        raw['C_u']=[[np.array([[1.],[1.]]),np.array([[1.],[0.]])]]
        raw['b_u']=[np.array([0.,1.])]
        raw['C_loc']=[[np.array([[1.],[-1.]]),np.array([[1.],[-1.]])]]
        raw['b_loc']=[[np.array([2.,1.]),np.array([2.,1.])]]
        grid={'branches':[(0,1)],'root':0,
              'r_ohm':np.array([1.]),'p_load_mw':np.array([0.,0.]),
              'voltage_kv':np.array([1.,1.]),
              'voltage_min':np.array([.9,.9]),'voltage_max':np.array([1.1,1.1])}
        choices={'input_regularization':0.,'state_regularization':0.,
                 'clearing_epsilon_kw':0.}
        def make_full(data,grid,**kw):
            return Case(DummyGame(**raw),np.zeros(0),data,grid,kw,
                        ['edge_0_1_pos','flow_0_1_upper'],
                        [{'trade_1':0},{'trade_0':0}])
        def fake_condense(case):
            g=case.game
            shared=np.hstack([g.C_u[0][0],g.C_u[0][1]])
            local0=np.hstack([g.C_loc[0][0],np.zeros((len(g.b_loc[0][0]),1))])
            local1=np.hstack([np.zeros((len(g.b_loc[0][1]),1)),g.C_loc[0][1]])
            A=np.vstack([shared,local0,local1])
            b=np.r_[g.b_u[0],g.b_loc[0][0],g.b_loc[0][1]]
            return (None,SimpleNamespace(H=np.eye(2),f=np.zeros(2),A=A,b=b),None)
        class FakeMpAVI:
            @staticmethod
            def from_game(game):
                # Snapshot tester only: recover a fake case to reproduce linear matrices.
                return SimpleNamespace(at=lambda x0:fake_condense(SimpleNamespace(game=game))[1])
        dyn=ModuleType('dynect'); dyn.DynLQGameTV=DummyGame; dyn.MpAVI=FakeMpAVI
        p2p=ModuleType('p2p'); p2p.__path__=[]
        setup=ModuleType('p2p.setup')
        setup.to_static_gnep=fake_condense
        setup.build_game=make_full
        src=make_full({'x':np.array([1.,2.])},grid,**choices)
        with patch.dict(sys.modules,{'dynect':dyn,'p2p':p2p,'p2p.setup':setup}):
            reduced,_=rebuild_constraints(src,clean_locals=True,
                                            localize_single_agent=True)
            audit={'test':'snapshot roundtrip'}
            with tempfile.TemporaryDirectory() as td:
                p=Path(td)/'model.npz'
                save_reduced_model(p,reduced,audit)
                self.assertTrue(p.exists())
                got,metadata=load_reduced_model(p,check_matrices=True)
                self.assertEqual(got.shared_labels,['edge_0_1_pos'])
                self.assertEqual(metadata,audit)
                self.assertEqual(got.grid['branches'],[(0,1)])
                np.testing.assert_allclose(got.game.C_loc[0][0],
                                            reduced.game.C_loc[0][0])


class TestReusableDates(unittest.TestCase):
    def test_paths_are_date_specific_and_validate_dates(self):
        from strategic_GNE_selection import scenario_paths
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            a=scenario_paths(directory, '2016-06-14')
            b=scenario_paths(directory, '2016-06-15')
            c=scenario_paths(directory, '2016-06-14', target_dt=0.5)
            self.assertNotEqual(a['model_path'], b['model_path'])
            self.assertNotEqual(a['results_dir'], b['results_dir'])
            self.assertNotEqual(a['model_path'], c['model_path'])
            self.assertEqual(a['model_path'].name, 'reduced_game.npz')
            self.assertEqual(a['model_path'].parents[1].name, '1h')
            self.assertIn('fair_gne', a['model_path'].parts)
            self.assertEqual(a['simulation_dir'].name, '2016-06-14')
            with self.assertRaises(ValueError):
                scenario_paths(directory, 'not-a-date')
            with self.assertRaises(ValueError):
                scenario_paths(directory, '2017-06-14')

    def test_auto_build_cache_and_changed_config(self):
        """A user changes only DATE, or configuration, and never gets stale case."""
        from strategic_GNE_selection import ensure_reduced_scenario
        import tempfile
        built=[]
        store={}
        def build(**params):
            built.append(params.copy())
            case=SimpleNamespace(N=4, T=24, data={'dt':1.0})
            return None,case,{'date':params['date'],'seed':params['seed']}
        def save(path,case,audit):
            path=Path(path)
            path.write_bytes(b'test snapshot placeholder')
            store[str(path)]=(case,audit)
        def load(path,*,check_matrices=True):
            return store[str(path)]
        with tempfile.TemporaryDirectory() as td:
            with patch('strategic_GNE_selection.build_reduced_model',side_effect=build), \
                 patch('strategic_GNE_selection.save_reduced_model',side_effect=save), \
                 patch('strategic_GNE_selection.load_reduced_model',side_effect=load), \
                 patch('strategic_GNE_selection._scenario_signature',side_effect=lambda c:__import__('json').dumps(c,sort_keys=True)):
                _, a, m1=ensure_reduced_scenario('2016-06-14',base_dir=td,verbose=False)
                _, a2,m2=ensure_reduced_scenario('2016-06-14',base_dir=td,verbose=False)
                _, b, m3=ensure_reduced_scenario('2016-06-15',base_dir=td,verbose=False)
                _, b2,m4=ensure_reduced_scenario('2016-06-15',base_dir=td,seed=11,verbose=False)
                _, a3,m5=ensure_reduced_scenario('2016-06-14',base_dir=td,verbose=False)
                self.assertEqual([m['rebuilt'] for m in (m1,m2,m3,m4,m5)],
                                 [True,False,True,True,False])
                self.assertEqual([x['date'] for x in built],
                                 ['2016-06-14','2016-06-15','2016-06-15'])
                self.assertEqual(a3['date'],'2016-06-14')
                self.assertEqual(b2['seed'],11)
                self.assertNotEqual(m1['model_path'],m3['model_path'])
                self.assertTrue(m1['audit_path'].exists())



class TestIntegratedSimulationLayout(unittest.TestCase):
    def test_prior_results_are_read_only(self):
        import json, tempfile
        from strategic_GNE_selection import scenario_paths, discover_prior_simulations
        with tempfile.TemporaryDirectory() as td:
            p=scenario_paths(td,'2016-06-14')
            day=p['simulation_dir']
            day.mkdir(parents=True)
            legacy=day/'no_trade.json'
            legacy.write_text(json.dumps({'status':'solved','some_old_cost':5.0}))
            marker=day/'infeasible.json'
            marker.write_text(json.dumps({'reason':'old version flagged infeasible'}))
            first=discover_prior_simulations(p)
            self.assertEqual([r['kind'] for r in first], ['no_trade','infeasible'])
            self.assertEqual(first[0]['reported_status'],'solved')
            self.assertEqual(legacy.read_text(), json.dumps({'status':'solved','some_old_cost':5.0}))

    def test_solve_cache_requires_exact_saved_model_and_options(self):
        import tempfile,json
        from strategic_GNE_selection import run_or_load_result
        class Dummy:
            n=2
            case=object()
            def feasibility(self,u):
                return {'max_violation':0.0}
            def physical_costs(self,u):
                return np.asarray(u)+[10.0,20.0]
            def model_costs(self,u):
                return self.physical_costs(u)
        def save(path,case,u,*,method,verification=None,extra=None):
            path=Path(path)
            path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,u=u,original_costs=game.physical_costs(u),
                meta_json=np.array(json.dumps(dict(method=method,
                                                  verification=verification,extra=extra))))
        game=Dummy()
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);model=base/'model.npz'
            model.write_bytes(b'first model')
            result=base/'cache'/'baseline.npz'
            count=[0]
            def solve():
                count[0]+=1
                return dict(u=np.array([2.,3.]),method='baseline')
            with patch('strategic_GNE_selection.save_candidate',side_effect=save):
                _,cached=run_or_load_result(game,result,model,
                  method='baseline',solve=solve,options={'weights':[1,1]})
                self.assertFalse(cached)
                _,cached=run_or_load_result(game,result,model,
                  method='baseline',solve=solve,options={'weights':[1,1]})
                self.assertTrue(cached)
                _,cached=run_or_load_result(game,result,model,
                  method='baseline',solve=solve,options={'weights':[2,1]})
                self.assertFalse(cached)
                model.write_bytes(b'changed model')
                _,cached=run_or_load_result(game,result,model,
                  method='baseline',solve=solve,options={'weights':[2,1]})
                self.assertFalse(cached)
            self.assertEqual(count[0],3)

if __name__=='__main__':
    unittest.main()


class TestSaveLocation(unittest.TestCase):
    def test_existing_root_only(self):
        import tempfile
        from strategic_GNE_selection import find_simulation_root, scenario_paths, test_results_write_access
        with tempfile.TemporaryDirectory() as td:
            repo=Path(td)
            existing=repo/'structures'/'ieee33_4_cyclic'/'results'
            existing.mkdir(parents=True)
            found=find_simulation_root(repo,cwd=repo)
            self.assertEqual(found,repo)
            p=scenario_paths(found,'2016-06-15')
            saved=test_results_write_access(p)
            self.assertTrue(saved.is_dir())
            self.assertTrue(saved.is_relative_to(repo))
            self.assertTrue(str(saved).endswith('fair_gne/1h'))

    def test_no_existing_tree_refuses_silent_save(self):
        import tempfile
        from strategic_GNE_selection import find_simulation_root
        with tempfile.TemporaryDirectory() as td:
            repo=Path(td)
            with self.assertRaises(FileNotFoundError):
                find_simulation_root(repo,cwd=repo)

    def test_no_legacy_subproject_search(self):
        import tempfile
        from strategic_GNE_selection import find_simulation_root
        with tempfile.TemporaryDirectory() as td:
            repo=Path(td)
            legacy=repo/'p2p_dynect'
            (legacy/'structures'/'ieee33_4_cyclic'/'results').mkdir(parents=True)
            with self.assertRaises(FileNotFoundError):
                find_simulation_root(repo,cwd=repo)
            with self.assertRaises(ValueError):
                find_simulation_root(repo,override=legacy)

if __name__=='__main__':
    unittest.main()
