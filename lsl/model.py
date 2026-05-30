"""LivingSynapseLM - local online autoregressive model.

Strict path properties:
  - no global backward pass
  - no optimizer state
  - no deep learning framework
  - hidden predictive-coding updates use only local previous state, current
    state, and local prediction error
"""
import numpy as np

from .synapse import LivingSynapseLayer
from .router import DynamicCircuitRouter
from .neuromod import Neuromodulator
from .memory import EpisodicBuffer
from .long_context import LongContextMemory
from .reasoning import RelationMemory, RoleBindingMemory
from .ssm import LivingSSM
from .utils import softmax, one_hot
from .sdr import SDREncoder, sparsity_ratio, log2_capacity
from .semantic_sdr import SemanticSDREncoder


def sparsify(x, k_ratio=0.02):
    """Keep only top-k activations to maintain sparsity."""
    k = max(1, int(len(x) * k_ratio))
    if k >= len(x):
        return x
    threshold = np.partition(np.abs(x), -k)[-k]
    mask = np.abs(x) >= threshold
    result = x.copy()
    result[~mask] = 0.0
    return result


class _LocalTransitionPredictor:
    def __init__(self, dim, key_size=8, lr=0.55):
        self.dim = int(dim)
        self.key_size = int(key_size)
        self.lr = float(lr)
        self.table = {}

    def _key(self, state):
        state = np.asarray(state, dtype=np.float32)
        if not np.any(np.abs(state) > 1e-8):
            return ("zero",)
        k = min(self.key_size, len(state))
        idx = np.argpartition(np.abs(state), -k)[-k:]
        return tuple(sorted(int(i) for i in idx))

    def predict(self, prev_state):
        pred = self.table.get(self._key(prev_state))
        if pred is None:
            return np.zeros(self.dim, dtype=np.float32)
        return pred.copy()

    def update(self, prev_state, current_state, error):
        key = self._key(prev_state)
        pred = self.table.get(key)
        if pred is None:
            pred = np.zeros(self.dim, dtype=np.float32)
        pred = pred + self.lr * np.asarray(error, dtype=np.float32)
        norm = float(np.linalg.norm(pred))
        if norm > 12.0:
            pred = pred * (12.0 / norm)
        self.table[key] = pred.astype(np.float32)

    def reset(self):
        self.table.clear()


class LivingSynapseLM:
    def __init__(self, vocab_size, hidden_dim, k_ratio=0.4, seed=0,
                 slow_init=0.1, attn_window=4, use_sdr=False, sdr_sparsity=0.2,
                 use_predictive_coding=False, theta=0.0,
                 use_semantic_sdr=False, semantic_hidden_dim=1000,
                 embedding_dim=300, use_pretrained=False,
                 use_sparse_computation=False,
                 use_sparse_memory=False,
                 use_role_binding=False,
                 use_hierarchical_routing=False,
                 memory_candidate_cap=64,
                 use_long_context_memory=False,
                 long_context_capacity=131072,
                 long_context_strength=10.0,
                 long_context_confidence_threshold=0.55):
        self.vocab_size = int(vocab_size)
        self.hidden_dim = int(semantic_hidden_dim if use_semantic_sdr else hidden_dim)
        self.use_sdr = bool(use_sdr)
        self.sdr_sparsity = float(sdr_sparsity)
        self.use_predictive_coding = bool(use_predictive_coding)
        self.theta = float(theta)
        self.use_semantic_sdr = bool(use_semantic_sdr)
        self.semantic_hidden_dim = int(semantic_hidden_dim)
        self.embedding_dim = int(embedding_dim)
        self.use_sparse_computation = bool(use_sparse_computation)
        self.use_sparse_memory = bool(use_sparse_memory)
        self.use_role_binding = bool(use_role_binding)
        self.use_hierarchical_routing = bool(use_hierarchical_routing)
        self.memory_candidate_cap = int(memory_candidate_cap)
        self.use_long_context_memory = bool(use_long_context_memory)
        self.long_context_strength = float(long_context_strength)
        self.long_context_confidence_threshold = float(long_context_confidence_threshold)

        self.embed = LivingSynapseLayer(vocab_size, self.hidden_dim,
                                        slow_init=slow_init, seed=seed)
        self.ssm = LivingSSM(self.hidden_dim, slow_init=slow_init, seed=seed + 3)
        self.recurrent = LivingSynapseLayer(self.hidden_dim, self.hidden_dim,
                                            slow_init=slow_init, seed=seed + 1)
        self.output = LivingSynapseLayer(self.hidden_dim, vocab_size,
                                         slow_init=slow_init, seed=seed + 2)
        self.W_emb_pred = LivingSynapseLayer(self.hidden_dim, self.hidden_dim,
                                             slow_init=slow_init, seed=seed + 5)
        self.W_ssm_pred = LivingSynapseLayer(self.hidden_dim, self.hidden_dim,
                                             slow_init=slow_init, seed=seed + 6)
        self.W_rec_pred = LivingSynapseLayer(self.hidden_dim, self.hidden_dim,
                                             slow_init=slow_init, seed=seed + 7)

        self.router = DynamicCircuitRouter(self.hidden_dim, k_ratio=k_ratio)
        self.modulator = Neuromodulator()
        self.episodic = EpisodicBuffer(capacity=256, candidate_cap=self.memory_candidate_cap)
        self.long_context = (
            LongContextMemory(
                capacity=long_context_capacity,
                vocab_size=vocab_size,
                candidate_cap=self.memory_candidate_cap,
                seed=seed + 31,
            )
            if self.use_long_context_memory
            else None
        )
        self.relation_memory = RelationMemory()
        self.role_binding_memory = RoleBindingMemory() if self.use_role_binding else None
        self.global_state = np.zeros(self.hidden_dim, dtype=np.float32)
        self.prev_h_ssm = np.zeros(self.hidden_dim, dtype=np.float32)
        self.prev_h_rec = np.zeros(self.hidden_dim, dtype=np.float32)
        self.step_count = 0
        self.inference_plasticity_enabled = True

        self._pc_emb = _LocalTransitionPredictor(self.hidden_dim)
        self._pc_ssm = _LocalTransitionPredictor(self.hidden_dim)
        self._pc_rec = _LocalTransitionPredictor(self.hidden_dim)

        self.next_token_assoc = np.zeros((self.vocab_size, self.vocab_size), dtype=np.float32)
        self.relation_assoc = np.zeros((self.vocab_size, self.vocab_size), dtype=np.float32)
        self.recent_tokens = []
        self.assoc_window = 20
        self.assoc_strength = 9.0
        self.relation_strength = 7.0

        if self.use_sdr:
            if self.use_semantic_sdr:
                self.sdr_encoder = SemanticSDREncoder(
                    vocab_size=vocab_size,
                    sdr_dim=self.semantic_hidden_dim,
                    sparsity=self.sdr_sparsity,
                    embed_dim=self.embedding_dim,
                    seed=seed + 10,
                    use_pretrained=use_pretrained,
                )
            else:
                self.sdr_encoder = SDREncoder(self.hidden_dim, sparsity=self.sdr_sparsity,
                                              seed=seed + 10)
        else:
            self.sdr_encoder = None

        self._last_h_embed = None
        self._last_h_embed_pre = None
        self._last_h_gated = None
        self._last_h_attn = None
        self._last_h_attn_pre = None
        self._last_h_recurrent = None
        self._last_h_recurrent_pre = None
        self._last_e_emb = None
        self._last_e_ssm = None
        self._last_e_rec = None
        self._last_e_emb_supp = None
        self._last_e_ssm_supp = None
        self._last_h_emb_pred = None
        self._last_h_ssm_pred = None
        self._last_h_rec_pred = None
        self._current_h_ssm = None
        self._current_h_rec = None

    def _hard_suppress(self, error):
        out = np.asarray(error, dtype=np.float32).copy()
        if self.theta > 0:
            out[np.abs(out) < self.theta] = 0.0
        return out

    def _normalised_column(self, matrix, token_id):
        col = matrix[:, int(token_id)]
        peak = float(np.max(col))
        if peak <= 1e-8:
            return np.zeros_like(col)
        return col / peak

    def _association_logits(self, token_id):
        direct = self._normalised_column(self.next_token_assoc, token_id)
        relation = self._normalised_column(self.relation_assoc, token_id)
        return self.assoc_strength * direct + self.relation_strength * relation

    def _update_associations(self, token_id, target_id):
        token_id = int(token_id)
        target_id = int(target_id)
        self.next_token_assoc[target_id, token_id] += 1.0
        self.relation_assoc[target_id, token_id] += 0.5
        for source in self.recent_tokens[-self.assoc_window:]:
            self.relation_assoc[target_id, int(source)] += 1.0
        self.recent_tokens.append(token_id)
        if len(self.recent_tokens) > self.assoc_window:
            self.recent_tokens.pop(0)

    def forward(self, token_id, target_id=None):
        x = one_hot(token_id, self.vocab_size)
        h_pre = self.embed.forward(x)
        h = np.tanh(h_pre)

        if self.use_sdr and self.sdr_encoder is not None:
            self._last_h_embed_pre = h.copy()
            h = self.sdr_encoder.encode(token_id) if self.use_semantic_sdr else self.sdr_encoder.encode(h)
        else:
            self._last_h_embed_pre = None
        self._last_h_embed = h.copy()

        mask = self.router.gate(h, self.global_state)
        h_gated = h * mask
        if self.use_sparse_computation:
            h_gated = sparsify(h_gated, k_ratio=0.02)
        self._last_h_gated = h_gated.copy()

        if self.use_predictive_coding:
            self.W_emb_pred.forward(self.prev_h_ssm, use_sparse=self.use_sparse_computation)
            h_emb_pred = self._pc_emb.predict(self.prev_h_ssm)
            e_emb = h_gated - h_emb_pred
            e_emb_supp = self._hard_suppress(e_emb)
        else:
            h_emb_pred = np.zeros_like(h_gated)
            e_emb = np.zeros_like(h_gated)
            e_emb_supp = h_gated

        h_ssm = self.ssm.forward(h_gated, use_sparse=self.use_sparse_computation)
        if self.use_sdr and self.sdr_encoder is not None and not self.use_semantic_sdr:
            self._last_h_attn_pre = h_ssm.copy()
            h_ssm = self.sdr_encoder.encode(h_ssm)
        else:
            self._last_h_attn_pre = None
        self._last_h_attn = h_ssm.copy()

        if self.use_predictive_coding:
            self.W_ssm_pred.forward(self.prev_h_ssm, use_sparse=self.use_sparse_computation)
            h_ssm_pred = self._pc_ssm.predict(self.prev_h_ssm)
            e_ssm = h_ssm - h_ssm_pred
            e_ssm_supp = self._hard_suppress(e_ssm)
        else:
            h_ssm_pred = np.zeros_like(h_ssm)
            e_ssm = np.zeros_like(h_ssm)
            e_ssm_supp = h_ssm

        ctx = 0.5 * h_ssm + 0.5 * self.global_state
        h2_pre = self.recurrent.forward(ctx, use_sparse=self.use_sparse_computation)
        h2 = np.tanh(h2_pre)
        if self.use_sdr and self.sdr_encoder is not None and not self.use_semantic_sdr:
            self._last_h_recurrent_pre = h2.copy()
            h2 = self.sdr_encoder.encode(h2)
        else:
            self._last_h_recurrent_pre = None
        self._last_h_recurrent = h2.copy()

        if self.use_predictive_coding:
            self.W_rec_pred.forward(self.prev_h_rec, use_sparse=self.use_sparse_computation)
            h_rec_pred = self._pc_rec.predict(self.prev_h_rec)
            e_rec = h2 - h_rec_pred
        else:
            h_rec_pred = np.zeros_like(h2)
            e_rec = np.zeros_like(h2)

        self._last_e_emb = e_emb.copy()
        self._last_e_ssm = e_ssm.copy()
        self._last_e_rec = e_rec.copy()
        self._last_e_emb_supp = e_emb_supp.copy()
        self._last_e_ssm_supp = e_ssm_supp.copy()
        self._last_h_emb_pred = h_emb_pred.copy()
        self._last_h_ssm_pred = h_ssm_pred.copy()
        self._last_h_rec_pred = h_rec_pred.copy()
        self._current_h_ssm = h_ssm.copy()
        self._current_h_rec = h2.copy()

        if target_id is None:
            self.prev_h_ssm = self._current_h_ssm.copy()
            self.prev_h_rec = self._current_h_rec.copy()

        self.global_state = 0.9 * self.global_state + 0.1 * h2
        logits = self.output.forward(h2, use_sparse=self.use_sparse_computation)
        if self.use_predictive_coding:
            logits = logits + self._association_logits(token_id)
        if self.long_context is not None:
            remembered, confidence = self.long_context.predict_next(
                token_id,
                vocab_size=self.vocab_size,
                return_confidence=True,
                update_context=(target_id is None),
            )
            if remembered is not None and 0 <= int(remembered) < self.vocab_size:
                if confidence >= self.long_context_confidence_threshold:
                    logits[int(remembered)] += self.long_context_strength * min(1.0, float(confidence))
        return logits

    def relation_probability(self, source_id, effect_id, candidate_ids=None, top_k=3):
        scores = self.relation_assoc[:, int(source_id)].astype(np.float32)
        if candidate_ids is not None:
            mask = np.zeros(self.vocab_size, dtype=np.float32)
            mask[[int(i) for i in candidate_ids]] = 1.0
            scores = scores * mask
        if np.max(scores) <= 1e-8:
            return 0.0
        if top_k is not None and top_k > 0:
            keep = np.argsort(scores)[-int(top_k):]
            mask = np.zeros_like(scores)
            mask[keep] = 1.0
            scores = scores * mask
        total = float(np.sum(scores))
        if total <= 1e-8:
            return 0.0
        return float(scores[int(effect_id)] / total)

    def predict(self, token_id):
        logits = self.forward(token_id)
        if getattr(self, "inference_plasticity_enabled", True):
            for layer in (self.embed, self.recurrent, self.output,
                          self.W_emb_pred, self.W_ssm_pred, self.W_rec_pred):
                layer.inference_plasticity(lr=0.003)
            self.ssm.inference_plasticity(lr=0.003)
        return softmax(logits)

    def observe(self, token_id, target_id, reward=0.0, store=True):
        logits = self.forward(token_id, target_id=target_id)
        probs = softmax(logits)
        err_out = (one_hot(target_id, self.vocab_size) - probs).astype(np.float32)
        prediction_err = float(-np.log(max(float(probs[int(target_id)]), 1e-10)))

        novelty = self.modulator.novelty(token_id)
        mod = self.modulator.compute(prediction_err, novelty=novelty, reward=reward)
        gain = float(np.clip(mod, 0.02, 2.0))

        self.output.top_k_supervised_update(err_out * gain, lr=2.0,
                                            k_frac=0.12, max_norm=12.0)

        if self.use_predictive_coding:
            self._pc_emb.update(self.prev_h_ssm, self._last_h_gated,
                                self._last_e_emb_supp)
            self._pc_ssm.update(self.prev_h_ssm, self._current_h_ssm,
                                self._last_e_ssm_supp)
            self._pc_rec.update(self.prev_h_rec, self._current_h_rec,
                                self._last_e_rec)
            self.W_emb_pred.top_k_supervised_update(
                self._last_e_emb_supp * gain, lr=0.3, k_frac=0.12, max_norm=12.0
            )
            self.W_ssm_pred.top_k_supervised_update(
                self._last_e_ssm_supp * gain, lr=0.3, k_frac=0.12, max_norm=12.0
            )
            self.W_rec_pred.top_k_supervised_update(
                self._last_e_rec * gain, lr=0.3, k_frac=0.12, max_norm=12.0
            )
            self._update_associations(token_id, target_id)
            self.prev_h_ssm = self._current_h_ssm.copy()
            self.prev_h_rec = self._current_h_rec.copy()
        else:
            c = int(token_id)
            self.recurrent.top_k_hebbian_update(gain, lr=0.02,
                                                k_frac=0.12, max_norm=12.0)
            self.ssm.hebbian_update(gain, lr=0.02, k_frac=0.12,
                                    max_norm=12.0, use_sparse=True)
            self.embed.W_live[:, c] += 0.02 * self._last_h_embed * gain
            self.embed._clip_norm(12.0)

        for layer in (self.embed, self.recurrent, self.output,
                      self.W_emb_pred, self.W_ssm_pred, self.W_rec_pred):
            layer.recover_fatigue(rate=0.98)
            layer.decay_live(rate=0.999)
        self.ssm.recover_fatigue(rate=0.98)
        self.ssm.decay_live(rate=0.999)

        if store:
            self.episodic.add((int(token_id), int(target_id)))
            if self.long_context is not None:
                self.long_context.observe_transition(token_id, target_id, vocab_size=self.vocab_size)
        self.step_count += 1
        return {
            "prediction_error": float(prediction_err),
            "modulator": float(mod),
            "novelty": float(novelty),
            "top1": int(np.argmax(probs)),
            "p_target": float(probs[int(target_id)]),
        }

    def consolidate(self, threshold=None, fraction=None):
        if threshold is None:
            threshold = 0.005 / (0.5 + 0.5 * self.modulator.surprise_baseline)
        if fraction is None:
            fraction = 0.3 / (1.0 + 0.005 * self.slow_norm())
        n = 0
        for layer in (self.embed, self.recurrent, self.output,
                      self.W_emb_pred, self.W_ssm_pred, self.W_rec_pred):
            n += layer.consolidate(threshold=threshold, fraction=fraction)
        n += self.ssm.consolidate(threshold=threshold, fraction=fraction)
        return n

    def replay(self, n=16, lr_factor=0.5, rng=None):
        for inp, tgt in self.episodic.sample(n=n, rng=rng):
            self.observe(inp, tgt, reward=0.0, store=False)

    def reset_state(self):
        self.global_state[:] = 0.0
        self.router.reset()
        self.ssm.reset_state()
        self.prev_h_ssm[:] = 0.0
        self.prev_h_rec[:] = 0.0
        self.recent_tokens.clear()
        if self.long_context is not None:
            self.long_context.reset_state()

    def reset_live(self):
        for layer in (self.embed, self.recurrent, self.output,
                      self.W_emb_pred, self.W_ssm_pred, self.W_rec_pred):
            layer.W_live[:] = 0.0
            layer.fatigue[:] = 0.0
        self.ssm.reset_live()
        self.reset_state()

    def live_norm(self):
        return float(sum(np.linalg.norm(L.W_live)
                         for L in (self.embed, self.recurrent, self.output,
                                   self.W_emb_pred, self.W_ssm_pred, self.W_rec_pred))
                     + self.ssm.live_norm())

    def slow_norm(self):
        return float(sum(np.linalg.norm(L.W_slow)
                         for L in (self.embed, self.recurrent, self.output,
                                   self.W_emb_pred, self.W_ssm_pred, self.W_rec_pred))
                     + self.ssm.slow_norm())

    def metrics(self):
        result = {
            "live_norm": self.live_norm(),
            "slow_norm": self.slow_norm(),
            "fatigue_means": [
                float(np.mean(self.embed.fatigue)),
                float(np.mean(self.recurrent.fatigue)),
                float(np.mean(self.output.fatigue)),
                float(np.mean(self.ssm.B_proj.fatigue)),
                float(np.mean(self.ssm.C_proj.fatigue)),
            ],
            "router_usage_mean": float(np.mean(self.router.usage_history)),
            "global_state_norm": float(np.linalg.norm(self.global_state)),
            "step_count": self.step_count,
        }
        for name, err in (("e_emb", self._last_e_emb),
                          ("e_ssm", self._last_e_ssm),
                          ("e_rec", self._last_e_rec)):
            if err is not None:
                result[f"{name}_norm"] = float(np.linalg.norm(err))
                if self.theta > 0:
                    result[f"{name}_suppression_pct"] = float(np.mean(np.abs(err) < self.theta))
        if self.use_sdr and self.sdr_encoder is not None:
            result["sdr_sparsity_target"] = self.sdr_sparsity
            result["sdr_k"] = self.sdr_encoder.k
            result["sdr_capacity_log2"] = float(log2_capacity(self.hidden_dim, self.sdr_encoder.k))
            if self._last_h_embed is not None:
                result["sdr_actual_sparsity_embed"] = float(sparsity_ratio(self._last_h_embed))
            if self._last_h_attn is not None:
                result["sdr_actual_sparsity_attn"] = float(sparsity_ratio(self._last_h_attn))
            if self._last_h_recurrent is not None:
                result["sdr_actual_sparsity_recurrent"] = float(sparsity_ratio(self._last_h_recurrent))
        return result

    def load_semantic_embeddings(self, vocab: dict):
        if self.use_semantic_sdr and self.sdr_encoder is not None:
            return self.sdr_encoder.load_builtin_embeddings(vocab)
        return 0

    def effective_weight_norms(self):
        norms = {
            "embed": self.embed.effective_weight_norm(),
            "recurrent": self.recurrent.effective_weight_norm(),
            "output": self.output.effective_weight_norm(),
            "W_emb_pred": self.W_emb_pred.effective_weight_norm(),
            "W_ssm_pred": self.W_ssm_pred.effective_weight_norm(),
            "W_rec_pred": self.W_rec_pred.effective_weight_norm(),
        }
        norms.update({f"ssm_{k}": v for k, v in self.ssm.effective_weight_norms().items()})
        return norms

    def observe_with_trace(self, token_id, target_id, reward=0.0, store=True):
        m_before = self.metrics()
        w_before = self.effective_weight_norms()
        info = self.observe(token_id, target_id, reward=reward, store=store)
        m_after = self.metrics()
        w_after = self.effective_weight_norms()
        return {
            "before": m_before,
            "after": m_after,
            "w_eff_before": w_before,
            "w_eff_after": w_after,
            "w_eff_diff": {k: w_after[k] - w_before[k] for k in w_before},
            "step_info": info,
        }
