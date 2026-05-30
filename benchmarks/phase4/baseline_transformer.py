"""Tiny Transformer Baseline for Phase 4 Comparison.

Simple numpy-only Transformer implementation for fair comparison
with the sparse architecture on the same CPU/RAM budget.
"""
import numpy as np
from typing import List, Tuple


class TinyTransformer:
    """Minimal Transformer with self-attention and feed-forward layers."""
    
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, 
                 d_ff: int, n_layers: int, max_seq_len: int = 512, seed: int = 42):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        rng = np.random.default_rng(seed)
        
        # Embedding
        self.W_embed = rng.standard_normal((vocab_size, d_model)) * 0.02
        
        # Positional encoding (simple sinusoidal)
        self.pos_encoding = self._create_positional_encoding(max_seq_len, d_model)
        
        # Transformer layers
        self.layers = []
        for _ in range(n_layers):
            self.layers.append({
                'W_q': rng.standard_normal((d_model, d_model)) * 0.02,
                'W_k': rng.standard_normal((d_model, d_model)) * 0.02,
                'W_v': rng.standard_normal((d_model, d_model)) * 0.02,
                'W_o': rng.standard_normal((d_model, d_model)) * 0.02,
                'W_ff1': rng.standard_normal((d_model, d_ff)) * 0.02,
                'W_ff2': rng.standard_normal((d_ff, d_model)) * 0.02,
            })
        
        # Output projection
        self.W_out = rng.standard_normal((d_model, vocab_size)) * 0.02
        
        # Layer norm parameters
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> np.ndarray:
        """Create sinusoidal positional encoding."""
        pos = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(pos * div_term)
        pe[:, 1::2] = np.cos(pos * div_term)
        
        return pe
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def _layer_norm(self, x: np.ndarray) -> np.ndarray:
        """Layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + 1e-5) + self.beta
    
    def _self_attention(self, x: np.ndarray, layer: dict) -> np.ndarray:
        """Multi-head self-attention."""
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections
        Q = x @ layer['W_q']  # (batch, seq, d_model)
        K = x @ layer['W_k']
        V = x @ layer['W_v']
        
        # Reshape for multi-head
        head_dim = d_model // self.n_heads
        Q = Q.reshape(batch_size, seq_len, self.n_heads, head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.n_heads, head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.n_heads, head_dim).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(head_dim)
        attn_weights = self._softmax(scores, axis=-1)
        attn_output = attn_weights @ V
        
        # Reshape back
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        
        # Output projection
        output = attn_output @ layer['W_o']
        
        return output
    
    def _feed_forward(self, x: np.ndarray, layer: dict) -> np.ndarray:
        """Position-wise feed-forward network."""
        return (x @ layer['W_ff1']) @ layer['W_ff2']
    
    def forward(self, token_ids: List[int]) -> np.ndarray:
        """Forward pass through the transformer."""
        seq_len = len(token_ids)
        x = self.W_embed[token_ids] + self.pos_encoding[:seq_len]
        x = x[np.newaxis, :, :]  # Add batch dimension
        
        # Transformer layers
        for layer in self.layers:
            # Self-attention with residual
            attn_out = self._self_attention(x, layer)
            x = self._layer_norm(x + attn_out)
            
            # Feed-forward with residual
            ff_out = self._feed_forward(x, layer)
            x = self._layer_norm(x + ff_out)
        
        # Output projection (use last token)
        logits = x[0, -1] @ self.W_out
        
        return logits
    
    def get_num_params(self) -> int:
        """Count total parameters."""
        params = self.W_embed.size + self.W_out.size
        params += 2 * self.d_model  # layer norm
        for layer in self.layers:
            params += layer['W_q'].size + layer['W_k'].size + layer['W_v'].size
            params += layer['W_o'].size
            params += layer['W_ff1'].size + layer['W_ff2'].size
        return params


def compare_with_sparse(vocab_size: int, d_model: int, num_tokens: int, 
                       seed: int = 42) -> dict:
    """Compare tiny Transformer with sparse architecture."""
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    from lsl.model import LivingSynapseLM
    import time
    
    print("=" * 80)
    print("Baseline Comparison: Tiny Transformer vs Sparse Architecture")
    print("=" * 80)
    print(f"Vocab: {vocab_size}, Model dim: {d_model}, Tokens: {num_tokens}")
    print()
    
    # Initialize models
    transformer = TinyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=4,
        d_ff=d_model * 4,
        n_layers=2,
        max_seq_len=128,
        seed=seed,
    )
    
    sparse_model = LivingSynapseLM(
        vocab_size=vocab_size,
        hidden_dim=d_model,
        use_sparse_computation=True,
        seed=seed,
    )
    
    # Count parameters
    transformer_params = transformer.get_num_params()
    sparse_params = sum(
        layer.W_slow.size + layer.W_live.size + layer.fatigue.size
        for layer in [sparse_model.embed, sparse_model.recurrent, 
                     sparse_model.output, sparse_model.ssm.B_proj, 
                     sparse_model.ssm.C_proj]
    )
    
    print(f"Transformer parameters: {transformer_params:,}")
    print(f"Sparse parameters: {sparse_params:,}")
    print(f"Parameter ratio: {sparse_params / transformer_params:.2f}x")
    print()
    
    # Generate test tokens
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, vocab_size, size=num_tokens).tolist()
    
    # Measure Transformer latency
    print("Measuring Transformer latency...")
    times_tf = []
    for i in range(min(20, num_tokens)):
        t0 = time.perf_counter_ns()
        transformer.forward(tokens[i:i+min(10, num_tokens-i)])
        dt = time.perf_counter_ns() - t0
        times_tf.append(float(dt) / 1000.0)
    
    tf_latency = np.mean(times_tf)
    print(f"Transformer mean latency: {tf_latency:.2f}us")
    
    # Measure Sparse latency
    print("Measuring Sparse latency...")
    times_sparse = []
    for i in range(min(20, num_tokens)):
        t0 = time.perf_counter_ns()
        sparse_model.forward(tokens[i])
        dt = time.perf_counter_ns() - t0
        times_sparse.append(float(dt) / 1000.0)
    
    sparse_latency = np.mean(times_sparse)
    print(f"Sparse mean latency: {sparse_latency:.2f}us")
    
    # Calculate speedup
    speedup = tf_latency / sparse_latency if sparse_latency > 0 else 0
    print()
    print(f"Latency speedup (Sparse vs Transformer): {speedup:.2f}x")
    print()
    
    return {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "transformer_params": transformer_params,
        "sparse_params": sparse_params,
        "param_ratio": sparse_params / transformer_params,
        "transformer_latency_us": tf_latency,
        "sparse_latency_us": sparse_latency,
        "speedup": speedup,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-tokens", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    result = compare_with_sparse(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_tokens=args.num_tokens,
        seed=args.seed,
    )
    
    print("=" * 80)
    print("Comparison Complete")
    print("=" * 80)
