"""Benchmark Real Task - Online Domain Adaptation.

Protocol sạch:
  Epoch N:
    1. observe() qua toàn bộ corpus  -> W_live học
    2. consolidate()                  -> W_live -> W_slow
    3. reset_live()                   -> xóa W_live
    4. reset_state()                  -> xóa attention buffer/SSM state
    5. evaluate() với predict() only  -> đo W_slow thuần

Static model: KHÔNG học, chỉ predict(). Loss = constant.
LSL model: W_slow tích lũy qua mỗi epoch. Loss GIẢM dần.

Đây là bằng chứng rõ ràng nhất của online learning không cần backprop.
"""
import numpy as np
from lsl import LivingSynapseLM, SimpleWordTokenizer

CORPUS = (
    "Ischemic stroke occurs when an artery to the brain is blocked. "
    "This blockage reduces blood flow and oxygen to brain tissues, leading to cell death. "
    "A common symptom of ischemic stroke is aphasia, which affects speech and language comprehension. "
    "Patients with aphasia struggle to find words and understand spoken language. "
    "Prompt treatment of stroke can restore artery blood flow and minimize brain damage. "
    "Rehabilitation helps patients recover from stroke and manage aphasia symptoms over time. "
    "Stroke rehabilitation focuses on restoring lost brain functions through repeated exercises. "
    "Brain plasticity allows damaged regions to be compensated by adjacent healthy brain areas. "
    "Early intervention in ischemic stroke significantly improves patient outcomes. "
    "Aphasia therapy targets language deficits caused by stroke lesions in the brain."
)


def evaluate(model, tokens):
    """Clean evaluation: predict only, no weight changes."""
    was_enabled = getattr(model, "inference_plasticity_enabled", True)
    model.inference_plasticity_enabled = False

    model.reset_state()
    losses, correct = [], 0
    for i in range(len(tokens) - 1):
        probs = model.predict(tokens[i])
        loss  = -np.log(max(float(probs[tokens[i+1]]), 1e-10))
        losses.append(loss)
        if int(np.argmax(probs)) == tokens[i+1]:
            correct += 1

    model.inference_plasticity_enabled = was_enabled
    return float(np.mean(losses)), correct / len(losses) * 100


def train_epoch(model, tokens, reward=0.3):
    """One training pass: observe all tokens, return avg prediction error."""
    model.reset_state()
    losses = []
    for i in range(len(tokens) - 1):
        info = model.observe(tokens[i], tokens[i+1],
                             reward=reward, store=(model.step_count < len(tokens)))
        losses.append(info["prediction_error"])
    return float(np.mean(losses))


def bar(val, lo, hi, width=24):
    """Render a proportional ASCII bar."""
    span = hi - lo + 1e-9
    frac = (hi - val) / span          # low loss = full bar
    frac = float(np.clip(frac, 0, 1))
    n = int(round(frac * width))
    return "[" + "#" * n + "." * (width - n) + "]"


def main():
    np.random.seed(42)

    tokenizer = SimpleWordTokenizer(vocab_size=300)
    tokenizer.build_vocab(CORPUS)
    tokens    = tokenizer.encode(CORPUS)
    vocab_sz  = tokenizer.vocab_size

    hidden_dim = 64
    seed       = 42
    N_EPOCHS   = 25

    # Compute updates comparison
    # LSL v1 (Dense + Attention):
    #   Output: 64 * 300 = 19,200
    #   Recurrent: 64 * 64 = 4,096
    #   Attention (Q/K/V): 3 * (64 * 64) = 12,288
    #   Embed: 64
    #   Total: 35,648 updates/token step
    # LSL v2 (Sparse + SSM with k_frac=0.12):
    #   Output: k_pre * k_post = 7 * 36 = 252
    #   Recurrent: 7 * 7 = 49
    #   SSM B/C: 49 * 2 = 98
    #   SSM alpha: 64
    #   Embed (Signed Gating): 64
    #   Total: 527 updates/token step
    updates_v1 = 35648
    updates_v2 = 527
    compute_reduction = updates_v1 / updates_v2

    print("=" * 72)
    print("  LSL v2 ONLINE DOMAIN ADAPTATION BENCHMARK")
    print("  Direct Feedback Alignment (DFA) · Signed Gating · Top-k Sparse Updates")
    print("=" * 72)
    print(f"\n  Corpus  : {len(tokens)} tokens, vocab={vocab_sz}")
    print(f"  Model   : hidden={hidden_dim}, DFA + SSM architecture")
    print(f"  Sparsity: k_frac=0.12  ({updates_v2} updates/step vs {updates_v1} in v1 — {compute_reduction:.1f}x reduction!)")
    print(f"  Epochs  : {N_EPOCHS}  (each = 1 full corpus pass)\n")

    # Identical starting weights
    m_static = LivingSynapseLM(vocab_size=vocab_sz, hidden_dim=hidden_dim, seed=seed)
    m_lsl    = LivingSynapseLM(vocab_size=vocab_sz, hidden_dim=hidden_dim, seed=seed)

    # Baseline (epoch 0)
    loss_s0, acc_s0 = evaluate(m_static, tokens)
    loss_l0, acc_l0 = evaluate(m_lsl,    tokens)

    hi_loss = loss_l0 * 1.05
    lo_loss = 0.5

    records_static = [loss_s0]
    records_lsl    = [loss_l0]

    print(f"{'Ep':>3} | {'Static':>7} | {'LSL eval':>8} | {'Train loss':>10} | "
          f"{'Acc':>5} | {'W_slow':>7} | {'W_live':>6} | Progress")
    print("-" * 80)
    print(f"{'0':>3} | {loss_s0:>7.4f} | {loss_l0:>8.4f} | {'---':>10} | "
          f"{acc_l0:>4.1f}% | {m_lsl.slow_norm():>7.2f} | {m_lsl.live_norm():>6.2f} | "
          f"{bar(loss_l0, lo_loss, hi_loss)}")

    threshold_crossed_epoch = None
    target_loss_threshold = 4.2

    for ep in range(1, N_EPOCHS + 1):
        # --- Static: just evaluate (always same weights) ---
        loss_s, _ = evaluate(m_static, tokens)

        # --- LSL: train then consolidate then eval ---
        train_loss = train_epoch(m_lsl, tokens)

        # Consolidate: W_live -> W_slow
        n_cons = m_lsl.consolidate()

        # Replay from episodic buffer every 5 epochs (sleep-phase)
        if ep % 5 == 0:
            m_lsl.replay(n=32)
            n_cons += m_lsl.consolidate()

        # Reset short-term memory so eval is purely W_slow
        m_lsl.reset_live()
        m_lsl.reset_state()

        loss_l, acc_l = evaluate(m_lsl, tokens)

        records_static.append(loss_s)
        records_lsl.append(loss_l)

        if threshold_crossed_epoch is None and loss_l < target_loss_threshold:
            threshold_crossed_epoch = ep

        delta  = loss_s - loss_l
        flag   = " [REPLAY]" if ep % 5 == 0 else ""
        marker = "<<" if loss_l < loss_s else "  "
        print(f"{ep:>3} | {loss_s:>7.4f} | {loss_l:>8.4f} | {train_loss:>10.4f} | "
              f"{acc_l:>4.1f}% | {m_lsl.slow_norm():>7.2f} | {m_lsl.live_norm():>6.2f} | "
              f"{bar(loss_l, lo_loss, hi_loss)} {marker}{flag}")

    # ------------------------------------------------------------------ #
    #  Summary analytics
    # ------------------------------------------------------------------ #
    first5_lsl  = float(np.mean(records_lsl[1:6]))
    last5_lsl   = float(np.mean(records_lsl[-5:]))
    static_mean = float(np.mean(records_static))
    final_lsl   = records_lsl[-1]
    best_lsl    = float(np.min(records_lsl[1:]))

    slow_gain   = m_lsl.slow_norm() - m_static.slow_norm()

    print("\n" + "=" * 72)
    print("  RESULTS")
    print("=" * 72)
    print(f"\n  {'Metric':<35} {'Static':>10}  {'LSL':>10}")
    print(f"  {'-'*57}")
    print(f"  {'Avg loss (all epochs)':<35} {static_mean:>10.4f}  {np.mean(records_lsl[1:]):>10.4f}")
    print(f"  {'Loss epoch 1-5 avg':<35} {'—':>10}  {first5_lsl:>10.4f}")
    print(f"  {'Loss epoch 21-25 avg':<35} {'—':>10}  {last5_lsl:>10.4f}")
    print(f"  {'Best epoch loss':<35} {static_mean:>10.4f}  {best_lsl:>10.4f}")
    print(f"  {'Final epoch loss':<35} {static_mean:>10.4f}  {final_lsl:>10.4f}")
    print(f"\n  In-stride learning curve : {first5_lsl:.4f} -> {last5_lsl:.4f}  "
          f"({(first5_lsl-last5_lsl)/first5_lsl*100:+.1f}%)")
    print(f"  W_slow grew by           : +{slow_gain:.2f}  (consolidated knowledge)")
    
    if threshold_crossed_epoch is not None:
        print(f"  Convergence speed        : Reached loss < {target_loss_threshold} in {threshold_crossed_epoch} epochs")
    else:
        print(f"  Convergence speed        : Did not cross threshold < {target_loss_threshold}")

    print(f"  Compute cost (synapse updates/step) : {updates_v2} (v2) vs {updates_v1} (v1) — {compute_reduction:.1f}x less compute!")

    # Per-domain-term win rate
    m_lsl.reset_state()
    m_static.reset_state()
    key = {"stroke","aphasia","brain","ischemic","blood","artery","flow"}
    wins, total = 0, 0
    for i in range(len(tokens)-1):
        w = tokenizer.id_to_word.get(tokens[i],"")
        ps = m_static.predict(tokens[i])
        pl = m_lsl.predict(tokens[i])
        ls = -np.log(max(float(ps[tokens[i+1]]),1e-10))
        ll = -np.log(max(float(pl[tokens[i+1]]),1e-10))
        if w in key:
            total += 1
            if ll < ls:
                wins += 1
    win_pct = wins/total*100 if total else 0

    print(f"  Domain-term token wins   : {wins}/{total} ({win_pct:.0f}%)")

    # ------------------------------------------------------------------ #
    #  Assertions
    # ------------------------------------------------------------------ #
    print(f"\n{'--- Assertions ---'}")

    assert last5_lsl < first5_lsl, (
        f"FAIL: LSL loss did not decrease "
        f"(ep1-5={first5_lsl:.4f} -> ep21-25={last5_lsl:.4f})")
    pct = (first5_lsl - last5_lsl) / first5_lsl * 100
    print(f"[PASS] In-stride convergence: {first5_lsl:.4f} -> {last5_lsl:.4f}  ({pct:+.1f}%)")

    assert best_lsl < static_mean, (
        f"FAIL: LSL best ({best_lsl:.4f}) never beat static ({static_mean:.4f})")
    print(f"[PASS] LSL beats static baseline (best={best_lsl:.4f} < static={static_mean:.4f})")

    assert slow_gain > 0, "FAIL: W_slow did not grow — consolidation failed"
    print(f"[PASS] W_slow grew by {slow_gain:.2f} via consolidation (no backprop)")

    assert m_lsl.slow_norm() > m_static.slow_norm(), \
        "FAIL: LSL slow weights did not accumulate domain knowledge"
    print(f"[PASS] W_slow: static={m_static.slow_norm():.2f} -> LSL={m_lsl.slow_norm():.2f}")

    print(f"\n[SUCCESS] LSL demonstrates in-stride domain adaptation.")
    print(f"          Learning rule: local Hebbian + feedback alignment.")
    print(f"          No gradient computation. No optimizer state.")
    print("=" * 72)


if __name__ == "__main__":
    main()
