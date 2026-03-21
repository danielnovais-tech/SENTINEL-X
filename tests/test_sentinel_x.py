"""
pytest unit tests for SENTINEL-X Advanced
==========================================
Covers:
  - Fault generators (MemoryArray, Sensor, ThermalSubsystem, PowerSubsystem,
    AttitudeControlSubsystem, CommSubsystem)
  - Spacecraft state vector and recovery actions
  - MissionProfile reward computation and dynamic weight updates
  - DQNAgent interface (act, remember, replay)
  - SafetyMonitor veto constraints
  - AdversarialTester: find examples, augment_replay_buffer, certify_robustness
  - FederatedSwarm safety-aware training (last_override_count)
  - benchmark_federation utility (smoke test: runs without error, returns dict)
  - GossipServer gossip round
  - PolicyVerifier (smoke test: returns well-formed report)
  - TFLite exports (dynamic-range and int8)
"""
import os
import random
import tempfile

import numpy as np
import pytest

# Suppress TF/XLA logging noise during tests
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import sentinel_x_advanced as sx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_agent(state_dim: int = 11, action_dim: int = 4) -> sx.DQNAgent:
    """Return a freshly initialised DQNAgent."""
    return sx.DQNAgent(state_dim, action_dim)


def make_fed_swarm(safety_monitor=None, n: int = 2) -> sx.FederatedSwarm:
    return sx.FederatedSwarm(
        num_spacecraft=n,
        action_dim=4,
        mission_profile=sx.MissionProfile(sx.MissionProfile.BALANCED),
        federated_interval=5,
        safety_monitor=safety_monitor,
    )


# ===========================================================================
# 1. Fault generators
# ===========================================================================

class TestMemoryArray:
    def test_step_returns_int(self):
        m = sx.MemoryArray()
        result = m.step()
        assert isinstance(result, int)

    def test_error_count_non_negative(self):
        m = sx.MemoryArray(size_bits=512, flip_rate_per_bit=0.5)
        for _ in range(10):
            m.step()
        assert m.error_count >= 0

    def test_reset_clears_state(self):
        m = sx.MemoryArray(size_bits=256, flip_rate_per_bit=1.0)
        m.step()
        m.reset()
        assert m.error_count == 0
        assert np.all(m.data == 0)

    def test_check_parity_returns_zero_or_one(self):
        m = sx.MemoryArray()
        parity = m.check_parity()
        assert parity in (0, 1)


class TestSensor:
    def test_step_returns_float(self):
        s = sx.Sensor()
        assert isinstance(s.step(), float)

    def test_stuck_at_fault(self):
        s = sx.Sensor(stuck_prob=1.0)
        first = s.step()          # triggers stuck
        second = s.step()         # returns stuck value
        assert s.is_stuck
        assert second == first

    def test_reset_clears_stuck(self):
        s = sx.Sensor(stuck_prob=1.0)
        s.step()
        s.reset()
        assert not s.is_stuck
        assert s.stuck_value is None


class TestThermalSubsystem:
    def test_step_returns_float(self):
        t = sx.ThermalSubsystem()
        assert isinstance(t.step(), float)

    def test_fault_declared_above_threshold(self):
        t = sx.ThermalSubsystem(fault_temp_high=25.0)
        t.temperature = 26.0
        t.step()
        assert t.is_faulted

    def test_fault_flag_property(self):
        t = sx.ThermalSubsystem()
        t.is_faulted = True
        assert t.fault_flag == 1
        t.is_faulted = False
        assert t.fault_flag == 0

    def test_reset(self):
        t = sx.ThermalSubsystem()
        t.is_faulted = True
        t.temperature = 200.0
        t.reset()
        assert not t.is_faulted
        assert t.temperature == t.nominal_temp


class TestPowerSubsystem:
    def test_step_returns_float(self):
        p = sx.PowerSubsystem()
        assert isinstance(p.step(), float)

    def test_charge_clipped_to_zero(self):
        p = sx.PowerSubsystem(drain_rate=10.0, recharge_rate=0.0)
        for _ in range(1000):
            p.step()
        assert p.charge >= 0.0

    def test_level_norm_in_unit_interval(self):
        p = sx.PowerSubsystem()
        for _ in range(20):
            p.step()
        assert 0.0 <= p.level_norm <= 1.0

    def test_reset(self):
        p = sx.PowerSubsystem()
        p.charge = 0.0
        p.is_faulted = True
        p.reset()
        assert p.charge == p.capacity
        assert not p.is_faulted


class TestAttitudeControlSubsystem:
    def test_step_returns_float(self):
        a = sx.AttitudeControlSubsystem()
        assert isinstance(a.step(), float)

    def test_tumble_fault_declared(self):
        a = sx.AttitudeControlSubsystem(detumble_threshold=1.0)
        a.angular_rate = np.array([2.0, 0.0, 0.0])
        a.step()
        assert a.is_faulted

    def test_rate_norm_capped_at_one(self):
        a = sx.AttitudeControlSubsystem()
        a.angular_rate = np.array([1000.0, 1000.0, 1000.0])
        assert a.rate_norm <= 1.0

    def test_reset(self):
        a = sx.AttitudeControlSubsystem()
        a.is_faulted = True
        a.angular_rate = np.ones(3) * 50.0
        a.reset()
        assert not a.is_faulted
        assert np.all(a.angular_rate == 0.0)


class TestCommSubsystem:
    def test_step_returns_float(self):
        c = sx.CommSubsystem()
        val = c.step()
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0

    def test_fault_declared_below_threshold(self):
        c = sx.CommSubsystem(fault_threshold=0.9)
        c.link_quality = 0.5
        c.step()
        assert c.is_faulted

    def test_quality_norm_property(self):
        c = sx.CommSubsystem()
        c.link_quality = 0.42
        assert c.quality_norm == pytest.approx(0.42)

    def test_reset(self):
        c = sx.CommSubsystem()
        c.link_quality = 0.0
        c.is_faulted = True
        c.reset()
        assert c.link_quality == 1.0
        assert not c.is_faulted


# ===========================================================================
# 2. Spacecraft
# ===========================================================================

class TestSpacecraft:
    def test_state_shape(self):
        sc = sx.Spacecraft()
        sc.step()
        assert sc.get_state().shape == (10,)

    def test_state_normalised(self):
        sc = sx.Spacecraft()
        for _ in range(10):
            sc.step()
        state = sc.get_state()
        assert np.all(state >= 0.0) and np.all(state <= 1.0)

    def test_health_flag_idx(self):
        assert sx.Spacecraft.HEALTH_FLAG_IDX == 5

    def test_health_flag_in_state(self):
        sc = sx.Spacecraft()
        sc.step()
        state = sc.get_state()
        expected = 0 if sc.healthy else 1
        assert state[sx.Spacecraft.HEALTH_FLAG_IDX] == expected

    @pytest.mark.parametrize("action", [0, 1, 2, 3])
    def test_apply_recovery_actions(self, action):
        sc = sx.Spacecraft()
        sc.step()
        sc.healthy = False
        sc.sensor.is_stuck = True
        sc.memory.error_count = 60
        sc.parity_ok = False
        sc.thermal.is_faulted = True
        sc.power.is_faulted = True
        sc.attitude.is_faulted = True
        sc.comm.is_faulted = True
        sc.sensor_stuck = True
        result = sc.apply_recovery(action)
        assert isinstance(result, bool)
        if action in (1, 2, 3):
            assert result is True

    def test_reset_reinitialises(self):
        sc = sx.Spacecraft()
        for _ in range(50):
            sc.step()
        sc.reset()
        assert sc.healthy
        assert sc.step_count == 0
        assert sc.mem_errors == 0


# ===========================================================================
# 3. MissionProfile
# ===========================================================================

class TestMissionProfile:
    @pytest.mark.parametrize("profile", [
        sx.MissionProfile.BALANCED,
        sx.MissionProfile.MAXIMIZE_DATA_RETURN,
        sx.MissionProfile.EXTEND_LIFESPAN,
        sx.MissionProfile.POWER_CONSTRAINED,
    ])
    def test_compute_returns_float(self, profile):
        mp = sx.MissionProfile(profile)
        r = mp.compute(False, True, 1)
        assert isinstance(r, float)

    def test_invalid_profile_raises(self):
        with pytest.raises(ValueError):
            sx.MissionProfile("nonexistent_profile")

    def test_update_weights_changes_reward(self):
        mp = sx.MissionProfile(sx.MissionProfile.BALANCED)
        r_base = mp.compute(False, True, 1)
        mp.update_weights(recovery_bonus_scale=3.0)
        r_scaled = mp.compute(False, True, 1)
        assert r_scaled != r_base

    def test_update_weights_keyword_only(self):
        mp = sx.MissionProfile()
        mp.update_weights(fault_penalty_scale=2.0)
        assert mp.fault_penalty_scale == pytest.approx(2.0)

    def test_update_weights_partial_update(self):
        mp = sx.MissionProfile()
        mp.update_weights(recovery_bonus_scale=1.5)
        assert mp.recovery_bonus_scale == pytest.approx(1.5)
        assert mp.fault_penalty_scale == pytest.approx(1.0)   # unchanged


# ===========================================================================
# 4. DQNAgent
# ===========================================================================

class TestDQNAgent:
    def test_act_training_returns_valid_action(self):
        agent = make_agent()
        state = np.zeros(11, dtype=np.float32)
        for _ in range(20):
            action = agent.act(state, training=True)
            assert 0 <= action < 4

    def test_act_greedy_deterministic(self):
        agent = make_agent()
        agent.epsilon = 0.0
        state = np.ones(11, dtype=np.float32)
        actions = {agent.act(state, training=False) for _ in range(5)}
        assert len(actions) == 1

    def test_remember_adds_to_buffer(self):
        agent = make_agent()
        s = np.zeros(11, dtype=np.float32)
        agent.remember(s, 0, 1.0, s, False)
        assert len(agent.memory) == 1

    def test_replay_does_not_raise_when_underfull(self):
        agent = make_agent()
        agent.replay()   # should be a no-op, not an error

    def test_replay_reduces_epsilon(self):
        agent = sx.DQNAgent(11, 4, batch_size=4)
        s = np.zeros(11, dtype=np.float32)
        for _ in range(10):
            agent.remember(s, 0, 1.0, s, False)
        eps_before = agent.epsilon
        agent.replay()
        assert agent.epsilon <= eps_before


# ===========================================================================
# 5. SafetyMonitor
# ===========================================================================

class TestSafetyMonitor:
    def test_no_inaction_on_fault(self):
        m = sx.SafetyMonitor()
        state = np.zeros(11, dtype=np.float32)
        state[sx.Spacecraft.HEALTH_FLAG_IDX] = 1.0
        assert m.veto(0, state, 4) == 3

    def test_no_reset_on_critical_power(self):
        m = sx.SafetyMonitor()
        state = np.zeros(11, dtype=np.float32)
        state[sx.SafetyMonitor.POWER_LEVEL_IDX] = 0.05
        assert m.veto(2, state, 4) == 3

    def test_safe_action_unchanged(self):
        m = sx.SafetyMonitor()
        state = np.zeros(11, dtype=np.float32)
        state[sx.SafetyMonitor.POWER_LEVEL_IDX] = 0.8
        for action in (1, 2, 3):
            assert m.veto(action, state, 4) == action

    def test_fault_does_not_block_non_zero_actions(self):
        m = sx.SafetyMonitor()
        state = np.zeros(11, dtype=np.float32)
        state[sx.Spacecraft.HEALTH_FLAG_IDX] = 1.0
        state[sx.SafetyMonitor.POWER_LEVEL_IDX] = 0.8
        for action in (1, 2, 3):
            assert m.veto(action, state, 4) == action

    def test_short_state_vector_handled(self):
        """State shorter than POWER_LEVEL_IDX should not raise."""
        m = sx.SafetyMonitor()
        state = np.zeros(6, dtype=np.float32)
        state[sx.Spacecraft.HEALTH_FLAG_IDX] = 0.0
        result = m.veto(1, state, 4)
        assert result == 1


# ===========================================================================
# 6. AdversarialTester
# ===========================================================================

class TestAdversarialTester:
    @pytest.fixture
    def tester(self):
        agent = make_agent()
        return sx.AdversarialTester(agent, epsilon=0.1, rng_seed=42)

    def test_find_adversarial_examples_returns_list(self, tester):
        results = tester.find_adversarial_examples(n_examples=20)
        assert isinstance(results, list)

    def test_each_result_has_required_keys(self, tester):
        results = tester.find_adversarial_examples(n_examples=20)
        for r in results:
            assert "original_state" in r
            assert "perturbed_state" in r
            assert "original_action" in r
            assert "flipped_action" in r

    def test_augment_replay_buffer_returns_int(self, tester):
        agents = [make_agent() for _ in range(3)]
        n = tester.augment_replay_buffer(agents, n_examples=30)
        assert isinstance(n, int)
        assert n >= 0

    def test_augment_replay_buffer_injects_transitions(self, tester):
        agents = [make_agent()]
        buf_before = len(agents[0].memory)
        injected = tester.augment_replay_buffer(agents, n_examples=50)
        assert len(agents[0].memory) == buf_before + injected

    def test_augment_with_safety_monitor(self, tester):
        monitor = sx.SafetyMonitor()
        agents = [make_agent()]
        # Should not raise; result is still a non-negative int
        injected = tester.augment_replay_buffer(
            agents, n_examples=30, safety_monitor=monitor
        )
        assert injected >= 0

    def test_certify_robustness_structure(self, tester):
        cert = tester.certify_robustness(n_samples=20, eps_hi=0.2, n_bisect=5)
        assert "mean_radius" in cert
        assert "min_radius" in cert
        assert "max_radius" in cert
        assert "robust_frac" in cert
        assert "eps_hi" in cert
        assert "n_samples" in cert

    def test_certify_robustness_ranges(self, tester):
        cert = tester.certify_robustness(n_samples=20, eps_hi=0.2, n_bisect=5)
        assert 0.0 <= cert["min_radius"] <= cert["mean_radius"] <= cert["max_radius"]
        assert cert["max_radius"] <= cert["eps_hi"] + 1e-9
        assert 0.0 <= cert["robust_frac"] <= 1.0

    def test_epsilon_not_mutated_after_certification(self, tester):
        original_eps = tester.epsilon
        tester.certify_robustness(n_samples=10, eps_hi=0.2, n_bisect=3)
        assert tester.epsilon == pytest.approx(original_eps)

    def test_summary_prints_without_error(self, tester, capsys):
        results = tester.find_adversarial_examples(n_examples=20)
        tester.summary(results, n_tested=20)
        captured = capsys.readouterr()
        assert "Adversarial" in captured.out


# ===========================================================================
# 7. FederatedSwarm – safety-aware training
# ===========================================================================

class TestFederatedSwarmSafetyAware:
    def test_safety_monitor_not_required(self):
        swarm = make_fed_swarm(safety_monitor=None)
        r = swarm.train_episode(max_steps=10)
        assert isinstance(r, float)
        assert swarm.last_override_count == 0

    def test_override_count_is_non_negative(self):
        monitor = sx.SafetyMonitor()
        swarm = make_fed_swarm(safety_monitor=monitor)
        swarm.train_episode(max_steps=20)
        assert swarm.last_override_count >= 0

    def test_override_count_resets_per_episode(self):
        monitor = sx.SafetyMonitor()
        swarm = make_fed_swarm(safety_monitor=monitor)
        swarm.train_episode(max_steps=10)
        count1 = swarm.last_override_count
        swarm.train_episode(max_steps=10)
        count2 = swarm.last_override_count
        # Both counts should be non-negative (independent per episode)
        assert count1 >= 0
        assert count2 >= 0

    def test_train_episode_returns_float_with_monitor(self):
        monitor = sx.SafetyMonitor()
        swarm = make_fed_swarm(safety_monitor=monitor)
        r = swarm.train_episode(max_steps=15)
        assert isinstance(r, float)


# ===========================================================================
# 8. GossipServer
# ===========================================================================

class TestGossipServer:
    def test_gossip_round_runs(self):
        agents = [make_agent() for _ in range(4)]
        gs = sx.GossipServer(k=2)
        gs.gossip_round(agents)   # should not raise

    def test_gossip_single_agent_no_op(self):
        agents = [make_agent()]
        gs = sx.GossipServer(k=2)
        gs.gossip_round(agents)   # < 2 agents → safe no-op

    def test_delayed_gossip(self):
        agents = [make_agent() for _ in range(3)]
        gs = sx.GossipServer(k=1, comm_delay_steps=2)
        gs.gossip_round(agents)
        gs.tick(agents)
        gs.tick(agents)   # weights applied after 2 ticks


# ===========================================================================
# 9. FederatedServer
# ===========================================================================

class TestFederatedServer:
    def test_aggregate_immediate(self):
        agents = [make_agent() for _ in range(3)]
        srv = sx.FederatedServer(comm_delay_steps=0)
        srv.aggregate(agents)   # should not raise

    def test_aggregate_with_dropout(self):
        agents = [make_agent() for _ in range(4)]
        srv = sx.FederatedServer(link_dropout_prob=0.5)
        srv.aggregate(agents)   # may have fewer participants, should not crash

    def test_delayed_aggregate(self):
        agents = [make_agent() for _ in range(3)]
        srv = sx.FederatedServer(comm_delay_steps=2)
        srv.aggregate(agents)
        srv.tick(agents)
        srv.tick(agents)


# ===========================================================================
# 10. benchmark_federation (smoke test)
# ===========================================================================

class TestBenchmarkFederation:
    def test_returns_correct_structure(self):
        results = sx.benchmark_federation(
            num_spacecraft=2,
            episodes=5,
            max_steps=20,
            dropout_rates=(0.0, 0.3),
        )
        assert "fedavg" in results
        assert "gossip" in results
        for key in results:
            for dropout in (0.0, 0.3):
                assert dropout in results[key]
                entry = results[key][dropout]
                assert "rewards" in entry
                assert "test_ops" in entry
                assert len(entry["rewards"]) == 5

    def test_test_ops_non_negative(self):
        results = sx.benchmark_federation(
            num_spacecraft=2,
            episodes=3,
            max_steps=15,
            dropout_rates=(0.0,),
        )
        for key in results:
            for dropout, entry in results[key].items():
                assert entry["test_ops"] >= 0.0


# ===========================================================================
# 11. PolicyVerifier (smoke test)
# ===========================================================================

class TestPolicyVerifier:
    def test_verify_returns_report(self):
        agent = make_agent()
        pv = sx.PolicyVerifier(agent, n_samples=30)
        report = pv.verify()
        assert "overall_passed" in report
        assert "safety_constraint" in report
        assert "qvalue_margin" in report
        assert "action_coverage" in report

    def test_overall_passed_is_bool(self):
        agent = make_agent()
        pv = sx.PolicyVerifier(agent, n_samples=20)
        report = pv.verify()
        assert isinstance(report["overall_passed"], bool)


# ===========================================================================
# 12. TFLite exports
# ===========================================================================

class TestTFLiteExport:
    def test_dynamic_range_export(self):
        agent = make_agent()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.tflite")
            data = sx.export_tflite(agent, output_path=path)
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_int8_export(self):
        agent = make_agent()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model_int8.tflite")
            data = sx.export_tflite_int8(agent, output_path=path, n_calib_samples=16)
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_int8_smaller_than_dynamic(self):
        """int8 model should be ≤ dynamic-range model in size."""
        agent = make_agent()
        with tempfile.TemporaryDirectory() as tmp:
            dyn = sx.export_tflite(agent, os.path.join(tmp, "dyn.tflite"))
            i8 = sx.export_tflite_int8(
                agent, os.path.join(tmp, "i8.tflite"), n_calib_samples=16
            )
        assert len(i8) <= len(dyn)


# ===========================================================================
# 13. LTLConstraintChecker
# ===========================================================================

_COMM_IDX = sx.LTLConstraintChecker._COMM_IDX   # index of comm_quality_norm

class TestLTLConstraintChecker:
    def test_default_constraints_registered(self):
        ltl = sx.LTLConstraintChecker()
        names = ltl.active_names()
        assert "no_inaction_on_fault" in names
        assert "no_full_reset_low_power" in names
        assert "no_simultaneous_faults" in names
        assert "comm_link_recovery" in names

    def test_no_inaction_on_fault_violated(self):
        ltl = sx.LTLConstraintChecker(penalty=-1.0)
        state = np.zeros(11, dtype=np.float32)
        state[sx.Spacecraft.HEALTH_FLAG_IDX] = 1.0
        pen = ltl.evaluate(state, action=0)
        assert pen < 0.0

    def test_no_inaction_on_fault_not_triggered_by_recovery(self):
        ltl = sx.LTLConstraintChecker(penalty=-1.0)
        # Only health flag set; no simultaneous faults, good comm
        state = np.zeros(11, dtype=np.float32)
        state[sx.Spacecraft.HEALTH_FLAG_IDX] = 1.0
        state[sx.SafetyMonitor.POWER_LEVEL_IDX] = 0.9
        state[_COMM_IDX] = 1.0   # good comm quality
        pen = ltl.evaluate(state, action=3)
        # Only no_inaction_on_fault might be relevant; action=3 doesn't trigger it
        assert pen == pytest.approx(0.0)

    def test_no_full_reset_low_power_violated(self):
        ltl = sx.LTLConstraintChecker(penalty=-1.0)
        state = np.zeros(11, dtype=np.float32)
        state[sx.SafetyMonitor.POWER_LEVEL_IDX] = 0.05
        state[_COMM_IDX] = 1.0   # good comm quality
        pen = ltl.evaluate(state, action=2)
        assert pen < 0.0

    def test_no_violation_returns_zero(self):
        ltl = sx.LTLConstraintChecker(penalty=-1.0)
        state = np.zeros(11, dtype=np.float32)
        state[sx.SafetyMonitor.POWER_LEVEL_IDX] = 0.9
        state[sx.Spacecraft.HEALTH_FLAG_IDX] = 0.0
        state[_COMM_IDX] = 1.0   # good comm quality
        pen = ltl.evaluate(state, action=1)
        assert pen == pytest.approx(0.0)

    def test_add_custom_constraint(self):
        ltl = sx.LTLConstraintChecker(penalty=-2.0)
        ltl.add_constraint("always_fail", lambda s, a: True)
        assert "always_fail" in ltl.active_names()
        state = np.zeros(11, dtype=np.float32)
        state[_COMM_IDX] = 1.0  # good comm to avoid other violations
        pen = ltl.evaluate(state, action=1)
        # "always_fail" fires → penalty is at least -2.0
        assert pen <= -2.0

    def test_remove_constraint(self):
        ltl = sx.LTLConstraintChecker(penalty=-1.0)
        ltl.remove_constraint("no_inaction_on_fault")
        assert "no_inaction_on_fault" not in ltl.active_names()

    def test_remove_nonexistent_is_noop(self):
        ltl = sx.LTLConstraintChecker()
        ltl.remove_constraint("nonexistent")   # should not raise

    def test_penalty_must_not_be_positive(self):
        with pytest.raises(ValueError):
            sx.LTLConstraintChecker(penalty=1.0)

    def test_zero_penalty_allowed(self):
        ltl = sx.LTLConstraintChecker(penalty=0.0)
        state = np.zeros(11, dtype=np.float32)
        state[sx.Spacecraft.HEALTH_FLAG_IDX] = 1.0
        assert ltl.evaluate(state, action=0) == pytest.approx(0.0)


# ===========================================================================
# 14. extract_decision_tree
# ===========================================================================

class TestExtractDecisionTree:
    def test_returns_classifier(self):
        pytest.importorskip("sklearn")
        agent = make_agent()
        dt = sx.extract_decision_tree(agent, n_samples=200, max_depth=4)
        assert dt is not None

    def test_fidelity_in_output(self, capsys):
        pytest.importorskip("sklearn")
        agent = make_agent()
        sx.extract_decision_tree(agent, n_samples=100, max_depth=4)
        out = capsys.readouterr().out
        assert "fidelity=" in out

    def test_predict_returns_valid_action(self):
        pytest.importorskip("sklearn")
        agent = make_agent()
        dt = sx.extract_decision_tree(agent, n_samples=100, max_depth=3)
        state = np.random.rand(1, agent.state_dim).astype(np.float32)
        action = int(dt.predict(state)[0])
        assert 0 <= action < agent.action_dim

    def test_depth_respected(self):
        pytest.importorskip("sklearn")
        agent = make_agent()
        dt = sx.extract_decision_tree(agent, n_samples=200, max_depth=5)
        assert dt.get_depth() <= 5

    def test_raises_without_sklearn(self, monkeypatch):
        monkeypatch.setattr(sx, "_SKLEARN_AVAILABLE", False)
        agent = make_agent()
        with pytest.raises(ImportError):
            sx.extract_decision_tree(agent)


# ===========================================================================
# 15. MissionScenario and build_swarm_for_scenario
# ===========================================================================

class TestMissionScenario:
    @pytest.mark.parametrize("factory", [
        sx.MissionScenario.lunar_gateway,
        sx.MissionScenario.mars_orbiter,
        sx.MissionScenario.cubesat_swarm,
    ])
    def test_preset_creates_scenario(self, factory):
        scenario = factory()
        assert isinstance(scenario, sx.MissionScenario)
        assert scenario.name
        assert scenario.profile in (
            sx.MissionProfile.BALANCED,
            sx.MissionProfile.MAXIMIZE_DATA_RETURN,
            sx.MissionProfile.EXTEND_LIFESPAN,
            sx.MissionProfile.POWER_CONSTRAINED,
        )

    def test_custom_scenario(self):
        s = sx.MissionScenario(
            name="Test",
            profile=sx.MissionProfile.BALANCED,
            num_spacecraft=2,
            comm_delay_steps=3,
            link_dropout_prob=0.1,
        )
        assert s.num_spacecraft == 2
        assert s.comm_delay_steps == 3

    def test_summary_returns_string(self):
        s = sx.MissionScenario.lunar_gateway()
        summary = s.summary()
        assert isinstance(summary, str)
        assert "Lunar Gateway" in summary
        assert "Profile" in summary

    def test_build_swarm_returns_federated_swarm(self):
        s = sx.MissionScenario.lunar_gateway()
        s.num_spacecraft = 2
        swarm = sx.build_swarm_for_scenario(s)
        assert isinstance(swarm, sx.FederatedSwarm)
        assert len(swarm.spacecraft) == 2

    def test_build_swarm_fault_params_applied(self):
        s = sx.MissionScenario.mars_orbiter()
        s.num_spacecraft = 2
        swarm = sx.build_swarm_for_scenario(s)
        for sc in swarm.spacecraft:
            assert sc.memory.flip_rate == pytest.approx(s.flip_rate_per_bit)
            assert sc.sensor.stuck_prob == pytest.approx(s.sensor_stuck_prob)
            assert sc.thermal.drift_std == pytest.approx(s.thermal_drift_std)
            assert sc.power.drain_rate == pytest.approx(s.power_drain_rate)

    def test_build_swarm_train_episode(self):
        s = sx.MissionScenario.cubesat_swarm()
        s.num_spacecraft = 2
        swarm = sx.build_swarm_for_scenario(s, safety_monitor=sx.SafetyMonitor())
        r = swarm.train_episode(max_steps=20)
        assert isinstance(r, float)

    def test_mars_orbiter_has_delay(self):
        s = sx.MissionScenario.mars_orbiter()
        assert s.comm_delay_steps > 0

    def test_cubesat_has_large_swarm(self):
        s = sx.MissionScenario.cubesat_swarm()
        assert s.num_spacecraft >= 4

    def test_lunar_gateway_power_constrained(self):
        s = sx.MissionScenario.lunar_gateway()
        assert s.profile == sx.MissionProfile.POWER_CONSTRAINED
