"""Basic tests for LSL components."""
import numpy as np
from lsl import LivingSynapseLayer, DynamicCircuitRouter, Neuromodulator, EpisodicBuffer, LivingSynapseLM
from lsl import SDREncoder, hamming_overlap, pairwise_overlap_matrix, sparsity_ratio, log2_capacity


def test_living_synapse_layer():
    L = LivingSynapseLayer(4, 8, slow_init=0.1, seed=0)
    x = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    y = L.forward(x)
    assert y.shape == (8,)
    L.hebbian_update(modulator=1.0, lr=0.1)
    assert np.linalg.norm(L.W_live) > 0.0
    L.decay_live(rate=0.9)
    L.recover_fatigue(rate=0.9)
    n = L.consolidate(threshold=0.1, fraction=0.1)
    assert n >= 0
    print("LivingSynapseLayer OK")


def test_dynamic_circuit_router():
    R = DynamicCircuitRouter(10, k_ratio=0.3)
    a = np.random.randn(10).astype(np.float32)
    s = np.random.randn(10).astype(np.float32)
    mask = R.gate(a, s)
    assert mask.shape == (10,)
    assert np.isclose(mask.sum(), R.k) or np.isclose(mask.sum(), 10)
    R.reset()
    print("DynamicCircuitRouter OK")


def test_neuromodulator():
    N = Neuromodulator()
    m = N.compute(surprise=2.0, novelty=0.5, reward=0.0)
    assert -1.0 <= m <= 2.0
    N.reset()
    print("Neuromodulator OK")


def test_episodic_buffer():
    B = EpisodicBuffer(capacity=10)
    B.add((1, 2))
    B.add((3, 4))
    sample = B.sample(n=2)
    assert len(sample) == 2
    B.clear()
    assert len(B) == 0
    print("EpisodicBuffer OK")


def test_living_synapse_lm():
    M = LivingSynapseLM(vocab_size=5, hidden_dim=8, k_ratio=0.4, seed=0)
    probs = M.predict(0)
    assert probs.shape == (5,)
    assert np.isclose(probs.sum(), 1.0)
    info = M.observe(0, 1, reward=0.0)
    assert "prediction_error" in info
    assert "modulator" in info
    M.consolidate(threshold=0.2, fraction=0.05)
    M.replay(n=4)
    M.reset_live()
    assert M.live_norm() == 0.0
    print("LivingSynapseLM OK")


def test_sdr_encoder():
    enc = SDREncoder(dim=16, sparsity=0.25, seed=42)
    x = np.random.randn(16).astype(np.float32)
    code = enc.encode(x)
    assert code.shape == (16,)
    assert np.all((code == 0) | (code == 1)), "SDR should be binary"
    active = code.sum()
    assert active == enc.k, f"Expected {enc.k} active bits, got {active}"
    print("SDREncoder OK")


def test_sdr_metrics():
    enc = SDREncoder(dim=32, sparsity=0.2, seed=42)
    codes = []
    for _ in range(10):
        x = np.random.randn(32).astype(np.float32)
        codes.append(enc.encode(x))
    codes = np.stack(codes)
    overlap = pairwise_overlap_matrix(codes)
    assert overlap.shape == (10, 10)
    # Diagonal should equal k (self-overlap)
    assert np.allclose(np.diag(overlap), enc.k)
    # Off-diagonal should be less than k (different codes)
    assert np.all(overlap - np.diag(np.diag(overlap)) < enc.k)
    cap_log2 = log2_capacity(32, enc.k)
    assert cap_log2 > 0
    print("SDR metrics OK")


def test_living_synapse_lm_sdr():
    M = LivingSynapseLM(vocab_size=5, hidden_dim=8, k_ratio=0.4, seed=0, use_sdr=True, sdr_sparsity=0.25)
    probs = M.predict(0)
    assert probs.shape == (5,)
    assert np.isclose(probs.sum(), 1.0)
    info = M.observe(0, 1, reward=0.0)
    assert "prediction_error" in info
    # Check SDR metrics exist
    metrics = M.metrics()
    assert "sdr_sparsity_target" in metrics
    assert "sdr_k" in metrics
    assert "sdr_capacity_log2" in metrics
    assert "sdr_actual_sparsity_embed" in metrics
    M.consolidate(threshold=0.2, fraction=0.05)
    M.replay(n=4)
    M.reset_live()
    assert M.live_norm() == 0.0
    print("LivingSynapseLM with SDR OK")


def test_predictive_coding():
    M = LivingSynapseLM(vocab_size=5, hidden_dim=8, k_ratio=0.4, seed=0,
                       use_predictive_coding=True, theta=0.0)
    assert M.theta == 0.0
    probs = M.predict(0)
    # Check that predictions and prediction errors were cached
    assert M._last_e_emb is not None
    assert M._last_e_ssm is not None
    assert M._last_e_rec is not None

    # Observe and check that top-down weights are modified
    info = M.observe(0, 1)
    info2 = M.observe(1, 2)

    pred_norms_after = {
        "emb": np.linalg.norm(M.W_emb_pred.W_live),
        "ssm": np.linalg.norm(M.W_ssm_pred.W_live),
        "rec": np.linalg.norm(M.W_rec_pred.W_live),
    }
    # Norms should be greater than zero due to updates
    assert pred_norms_after["emb"] > 0.0
    assert pred_norms_after["ssm"] > 0.0
    assert pred_norms_after["rec"] > 0.0

    metrics = M.metrics()
    assert "e_emb_norm" in metrics
    print("PredictiveCoding OK")


if __name__ == "__main__":
    test_living_synapse_layer()
    test_dynamic_circuit_router()
    test_neuromodulator()
    test_episodic_buffer()
    test_living_synapse_lm()
    test_sdr_encoder()
    test_sdr_metrics()
    test_living_synapse_lm_sdr()
    test_predictive_coding()
    print("\nAll tests passed.")
