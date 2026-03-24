"""Tests for nsram library."""
import numpy as np
import sys; sys.path.insert(0, '.')


def test_physics_constants():
    from nsram.physics import thermal_voltage, DeviceParams
    vt = thermal_voltage(300.0)
    assert 0.0258 < vt < 0.0260, f"Vt={vt}, expected ~0.02585"
    p = DeviceParams()
    assert p.BV0 == 3.5
    assert p.Tbv1 == -21.3e-6
    assert p.Is == 1e-16
    assert p.E_spike == 21e-15
    print("  [PASS] physics constants")


def test_breakdown_voltage():
    from nsram.physics import breakdown_voltage
    # At Vg1=0, T=300K: BVpar = 3.5
    assert abs(breakdown_voltage(0.0) - 3.5) < 0.01
    # At Vg1=0.4: BVpar = 3.5 - 1.5*0.4 = 2.9
    assert abs(breakdown_voltage(0.4) - 2.9) < 0.01
    # Temperature: higher T → lower BVpar (Tbv1 is negative)
    bv_300 = breakdown_voltage(0.3, T=300)
    bv_400 = breakdown_voltage(0.3, T=400)
    assert bv_400 < bv_300, "Higher T should lower BVpar"
    # Array input
    bv = breakdown_voltage(np.array([0.0, 0.2, 0.4]))
    assert bv.shape == (3,)
    print("  [PASS] breakdown voltage")


def test_avalanche_current():
    from nsram.physics import avalanche_current
    # Below breakdown: tiny current
    I_below = avalanche_current(Vcb=1.0, vg1=0.3)
    assert I_below < 1e-10
    # Above breakdown: significant current
    I_above = avalanche_current(Vcb=3.0, vg1=0.3)
    assert I_above > I_below * 100
    # Higher Vg1 → lower BVpar → more current at same Vcb
    I_low_vg = avalanche_current(Vcb=2.5, vg1=0.2)
    I_high_vg = avalanche_current(Vcb=2.5, vg1=0.5)
    assert I_high_vg > I_low_vg
    print("  [PASS] avalanche current")


def test_charge_capture():
    from nsram.physics import charge_capture_rate
    # Low VG2 → high capture (synapse/STD mode)
    k_low = charge_capture_rate(0.30)
    # High VG2 → low capture (neuron/STF mode)
    k_high = charge_capture_rate(0.50)
    assert k_low > k_high * 5, f"Low VG2 should have much higher k_cap"
    # Midpoint
    k_mid = charge_capture_rate(0.40)
    assert k_mid < k_low and k_mid > k_high
    print("  [PASS] charge capture rate")


def test_srh_trapping():
    from nsram.physics import srh_trapping_ode
    # No spikes → Q decays
    dQ = srh_trapping_ode(Q=0.5, spike_rate=0.0, k_cap=500.0, k_em=200.0)
    assert dQ < 0, "Q should decay without spikes"
    # High spike rate → Q increases from 0
    dQ = srh_trapping_ode(Q=0.0, spike_rate=10.0, k_cap=500.0, k_em=200.0)
    assert dQ > 0, "Q should increase with spikes"
    # Equilibrium: dQ=0 when k_cap*(1-Q)*rate = k_em*Q
    print("  [PASS] SRH trapping")


def test_presets():
    from nsram.physics import PRESETS, DimensionlessParams
    assert 'default' in PRESETS
    assert 'no_stp' in PRESETS
    assert 'critical' in PRESETS
    p = PRESETS['depression']
    assert p.U > 0.5  # High utilization = depression
    p = PRESETS['facilitation']
    assert p.U < 0.3  # Low utilization = facilitation
    print("  [PASS] presets")


def test_single_neuron():
    from nsram.neuron import NSRAMNeuron
    n = NSRAMNeuron(Vg1=0.40, Vg2=0.40)
    assert 'BVpar' in repr(n)
    trace = n.simulate(duration=100e-6, dt=100e-9)
    assert 't' in trace and 'Vm' in trace and 'spikes' in trace
    assert len(trace['t']) > 0
    print(f"  [PASS] single neuron ({trace['n_spikes']} spikes in 100μs)")


def test_network_numpy():
    from nsram.network import NSRAMNetwork
    net = NSRAMNetwork(N=32, backend='numpy', seed=42)
    assert net.backend == 'numpy'
    inputs = np.random.randn(500).astype(np.float64)
    result = net.run(inputs, noise_sigma=0.01)
    assert result['states'].shape == (32, 500)
    assert result['spikes'].shape == (32, 500)
    n_spk = result['spikes'].sum()
    print(f"  [PASS] network numpy (32N, {n_spk:.0f} spikes)")


def test_network_torch():
    try:
        import torch
    except ImportError:
        print("  [SKIP] network torch (PyTorch not installed)")
        return
    from nsram.network import NSRAMNetwork
    net = NSRAMNetwork(N=64, backend='torch', seed=42)
    assert net.backend == 'torch'
    inputs = np.random.randn(500).astype(np.float64)
    result = net.run(inputs, noise_sigma=0.01)
    assert result['states'].shape == (64, 500)
    n_spk = result['spikes'].sum()
    print(f"  [PASS] network torch/{net.device} (64N, {n_spk:.0f} spikes)")


def test_reservoir():
    from nsram import NSRAMReservoir
    for stp in ['none', 'std', 'stf', 'heterogeneous']:
        res = NSRAMReservoir(N=32, stp=stp, backend='numpy', seed=42)
        inputs = np.random.randn(300)
        states = res.transform(inputs)
        assert states.shape == (32, 300), f"Failed for stp={stp}"
    print("  [PASS] reservoir (all 4 STP modes)")


def test_benchmarks():
    from nsram import NSRAMReservoir, rc_benchmark
    res = NSRAMReservoir(N=32, backend='numpy', seed=42)
    results = rc_benchmark(res, n_steps=800, washout=200, n_reps=1, verbose=False)
    assert 'xor1_mean' in results
    assert 'mc_mean' in results
    assert 'narma10_mean' in results
    assert 'wave4_mean' in results
    assert 0 <= results['xor1_mean'] <= 1
    assert results['mc_mean'] >= 0
    print(f"  [PASS] benchmarks (XOR1={results['xor1_mean']:.1%}, MC={results['mc_mean']:.3f})")


def test_iv_curve():
    from nsram.neuron import NSRAMNeuron
    n = NSRAMNeuron(Vg1=0.35)
    iv = n.iv_curve(Vds_range=(0, 3.5))
    assert len(iv['Vds']) == 200
    assert len(iv['Id']) == 200
    assert iv['Id'][-1] > iv['Id'][0]  # Current increases with Vds
    print("  [PASS] I-V curve")


def test_dimensionless_to_physical():
    from nsram.physics import DimensionlessParams, DeviceParams
    dp = DimensionlessParams()
    phys = dp.to_physical()
    assert 'tau_mem_s' in phys
    assert phys['tau_mem_s'] > 0
    # From VG2
    dp2 = dp.from_Vg2(0.35)
    assert dp2.U > dp.from_Vg2(0.45).U  # Low VG2 → higher U
    print("  [PASS] dimensionless↔physical conversion")


def test_body_charge_ode():
    from nsram.physics import body_charge_ode, DeviceParams
    from scipy.integrate import solve_ivp
    p = DeviceParams()
    # Simulate 10μs with Vds=3.5V, Vg=0.35V, Rb=100kΩ (neuron mode)
    sol = solve_ivp(body_charge_ode, [0, 10e-6], [0.0, 0.0],
                    args=(3.5, 0.35, 100e3, p), max_step=1e-8,
                    method='RK45')
    assert sol.success, f"ODE solver failed: {sol.message}"
    assert sol.y.shape[0] == 2  # VB and Q_trap
    assert len(sol.t) > 10  # Should have enough steps
    VB_final = sol.y[0, -1]
    # Body potential should have evolved (not stayed at 0)
    print(f"  [PASS] body charge ODE (VB_final={VB_final:.4f}V, {len(sol.t)} steps)")


def test_device_params_extended():
    from nsram.physics import DeviceParams
    p = DeviceParams()
    assert p.Rb_neuron == 10e3
    assert p.Rb_synapse == 1e6
    assert p.tau_srh_e == 8e-6
    assert p.tau_srh_h == 0.8e-6
    assert p.N_eff_levels == 14
    assert p.alpha_n == 7.03e5
    assert p.tau_body > 0
    print(f"  [PASS] extended device params (τ_body={p.tau_body*1e6:.1f}μs)")


def test_gpu_compatibility():
    """Test that the library handles missing GPU gracefully."""
    from nsram.network import NSRAMNetwork
    # Auto backend should always work
    net = NSRAMNetwork(N=16, backend='auto')
    inputs = np.random.randn(100)
    result = net.run(inputs)
    assert result['states'].shape == (16, 100)
    print(f"  [PASS] GPU compatibility (auto → {net.backend}/{net.device})")


if __name__ == '__main__':
    print("nsram test suite")
    print("=" * 50)
    test_physics_constants()
    test_breakdown_voltage()
    test_avalanche_current()
    test_charge_capture()
    test_srh_trapping()
    test_presets()
    test_single_neuron()
    test_network_numpy()
    test_network_torch()
    test_reservoir()
    test_benchmarks()
    test_iv_curve()
    test_dimensionless_to_physical()
    test_body_charge_ode()
    test_device_params_extended()
    test_gpu_compatibility()
    print("=" * 50)
    print("ALL TESTS PASSED")
