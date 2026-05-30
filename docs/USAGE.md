# Hướng dẫn Sử dụng LSL

## Bắt đầu nhanh

### Ví dụ cơ bản

```python
from lsl import LivingSynapseLM, SimpleWordTokenizer

# Tạo tokenizer
tokenizer = SimpleWordTokenizer()
text = "the cat sat on the mat"
tokens = tokenizer.encode(text)

# Tạo mô hình
model = LivingSynapseLM(
    vocab_size=tokenizer.vocab_size,
    hidden_dim=256,
    seed=42
)

# Huấn luyện online
for i in range(len(tokens) - 1):
    model.observe(tokens[i], tokens[i+1])

# Dự đoán token tiếp theo
last_token = tokens[-1]
probs = model.predict(last_token)
next_token = int(probs.argmax())
print(f"Next token: {tokenizer.decode([next_token])}")
```

## Các chế độ hoạt động

### 1. Basic Mode (Hebbian Learning)

```python
model = LivingSynapseLM(
    vocab_size=1000,
    hidden_dim=256,
    use_predictive_coding=False,  # Tắt predictive coding
    use_sdr=False,               # Tắt SDR
    use_sparse_computation=False # Tắt sparse computation
)
```

### 2. Predictive Coding Mode

```python
model = LivingSynapseLM(
    vocab_size=1000,
    hidden_dim=256,
    use_predictive_coding=True,  # Bật predictive coding
    theta=0.02,                  # Suppression threshold
    use_sdr=False,
    use_sparse_computation=False
)
```

### 3. SDR Mode

```python
model = LivingSynapseLM(
    vocab_size=1000,
    hidden_dim=256,
    use_sdr=True,
    sdr_sparsity=0.2,            # 20% active bits
    use_predictive_coding=False,
    use_sparse_computation=False
)
```

### 4. Semantic SDR Mode

```python
model = LivingSynapseLM(
    vocab_size=10000,
    hidden_dim=256,
    use_semantic_sdr=True,
    semantic_hidden_dim=1000,     # SDR dimension
    embedding_dim=300,           # Embedding dimension
    use_pretrained=True,         # Sử dụng semantic priors
    use_predictive_coding=True,
    theta=0.02
)

# Load semantic embeddings
vocab = tokenizer.get_vocab()
model.load_semantic_embeddings(vocab)
```

### 5. Sparse Computation Mode

```python
model = LivingSynapseLM(
    vocab_size=1000,
    hidden_dim=256,
    use_sparse_computation=True, # Bật sparse forward
    use_predictive_coding=True,
    theta=0.02,
    use_sdr=True,
    sdr_sparsity=0.2
)
```

### 6. Long Context Mode

```python
model = LivingSynapseLM(
    vocab_size=1000,
    hidden_dim=256,
    use_long_context_memory=True,
    long_context_capacity=131072,  # 128k chunks
    long_context_strength=10.0,
    long_context_confidence_threshold=0.55,
    use_predictive_coding=True
)
```

### 7. Full Mode (Tất cả features)

```python
model = LivingSynapseLM(
    vocab_size=10000,
    hidden_dim=256,
    use_predictive_coding=True,
    theta=0.02,
    use_sdr=True,
    sdr_sparsity=0.2,
    use_semantic_sdr=True,
    semantic_hidden_dim=1000,
    embedding_dim=300,
    use_pretrained=True,
    use_sparse_computation=True,
    use_sparse_memory=True,
    use_role_binding=True,
    use_hierarchical_routing=True,
    use_long_context_memory=True,
    long_context_capacity=131072,
    memory_candidate_cap=64
)
```

## Tokenization

### Word Tokenizer

```python
from lsl import SimpleWordTokenizer

tokenizer = SimpleWordTokenizer()
text = "hello world"
tokens = tokenizer.encode(text)
print(tokens)  # [token_ids]
decoded = tokenizer.decode(tokens)
print(decoded)  # "hello world"
```

### Subword Tokenizer

```python
from lsl import SimpleSubwordTokenizer

tokenizer = SimpleSubwordTokenizer(vocab_size=1000)
tokenizer.train(["hello world", "hello there"])
tokens = tokenizer.encode("hello world")
decoded = tokenizer.decode(tokens)
```

## Huấn luyện

### Online Learning

```python
# Huấn luyện từng token
for i in range(len(tokens) - 1):
    info = model.observe(tokens[i], tokens[i+1])
    print(f"Prediction error: {info['prediction_error']:.4f}")
```

### Với Reward Signal

```python
# Positive reward cho correct prediction
for i in range(len(tokens) - 1):
    reward = 1.0 if tokens[i+1] == expected else 0.0
    model.observe(tokens[i], tokens[i+1], reward=reward)
```

### Consolidation

```python
# Consolidate live weights vào slow weights
n_consolidated = model.consolidate(threshold=0.005, fraction=0.3)
print(f"Consolidated {n_consolidated} weights")
```

### Replay

```python
# Replay từ episodic buffer
model.replay(n=16, lr_factor=0.5)
```

## Inference

### Single Token Prediction

```python
probs = model.predict(token_id)
next_token = int(probs.argmax())
top_k_tokens = probs.argsort()[-5:][::-1]
```

### Generation

```python
from lsl import GenerationController

controller = GenerationController(model, tokenizer)

# Generate text
generated = controller.generate(
    prompt="the cat",
    max_tokens=50,
    temperature=0.8,
    top_p=0.9
)
print(generated)
```

### Sampling Strategies

```python
# Temperature sampling
import numpy as np
probs = model.predict(token_id)
logits = np.log(probs) / temperature
probs = np.exp(logits) / np.exp(logits).sum()
next_token = np.random.choice(len(probs), p=probs)

# Top-k sampling
k = 10
top_k_indices = probs.argsort()[-k:]
top_k_probs = probs[top_k_indices]
top_k_probs = top_k_probs / top_k_probs.sum()
next_token = np.random.choice(top_k_indices, p=top_k_probs)

# Nucleus (top-p) sampling
p = 0.9
sorted_probs = probs.argsort()[::-1]
cumsum = np.cumsum(probs[sorted_probs])
cutoff = np.searchsorted(cumsum, p)
candidates = sorted_probs[:cutoff+1]
candidate_probs = probs[candidates] / probs[candidates].sum()
next_token = np.random.choice(candidates, p=candidate_probs)
```

## Reasoning

### Relation Memory

```python
# Tìm relation probability
prob = model.relation_probability(
    source_id=tokenizer.encode("stroke")[0],
    effect_id=tokenizer.encode("aphasia")[0],
    top_k=3
)
print(f"P(aphasia | stroke) = {prob:.3f}")
```

### Role Binding

```python
from lsl import RoleBindingMemory

role_memory = RoleBindingMemory()
role_memory.bind("subject", "cat", token_id_cat)
role_memory.bind("verb", "sat", token_id_sat)
role_memory.bind("object", "mat", token_id_mat)

# Query
subject = role_memory.query("subject")
```

## Long Context Memory

### Store Facts

```python
if model.long_context is not None:
    # Store fact
    fact_tokens = tokenizer.encode("Paris is the capital of France")
    for i in range(len(fact_tokens) - 1):
        model.long_context.observe_transition(
            fact_tokens[i],
            fact_tokens[i+1],
            vocab_size=tokenizer.vocab_size
        )
```

### Retrieve Context

```python
if model.long_context is not None:
    remembered, confidence = model.long_context.predict_next(
        token_id=tokenizer.encode("Paris")[0],
        vocab_size=tokenizer.vocab_size,
        return_confidence=True
    )
    if confidence > 0.5:
        print(f"Retrieved: {tokenizer.decode([remembered])}")
```

## Metrics và Monitoring

### Get Metrics

```python
metrics = model.metrics()
print(f"Live norm: {metrics['live_norm']:.4f}")
print(f"Slow norm: {metrics['slow_norm']:.4f}")
print(f"Step count: {metrics['step_count']}")
```

### Prediction Error Tracking

```python
info = model.observe(token_id, target_id)
print(f"Prediction error: {info['prediction_error']:.4f}")
print(f"Modulator: {info['modulator']:.4f}")
print(f"Novelty: {info['novelty']:.4f}")
```

### SDR Metrics

```python
if model.use_sdr:
    print(f"SDR sparsity target: {model.sdr_sparsity}")
    print(f"SDR k: {model.sdr_encoder.k}")
    print(f"SDR capacity (log2): {model.sdr_encoder.log2_capacity}")
```

## Reset và Cleanup

### Reset State

```python
model.reset_state()  # Reset hidden states
```

### Reset Live Weights

```python
model.reset_live()  # Reset live weights và states
```

### Reset Predictors

```python
model._pc_emb.reset()
model._pc_ssm.reset()
model._pc_rec.reset()
```

## Benchmarks

### Chạy Strict Benchmark

```bash
python benchmark_goal_strict.py
```

### Chạy Phase Benchmarks

```bash
# Phase 1: SDR
python benchmark_sdr_phase1.py

# Phase 2: Predictive Coding
python benchmark_pc_phase2.py

# Phase 3: Cortical Column
python benchmark_cortical_column_sequence.py

# Phase 4: Scaling
python benchmark_semantic_sdr_scaling.py
python benchmark_sparse_physical_compute.py

# Phase 5: Moonshot
python benchmark_moonshot.py --profile quick
python benchmark_moonshot.py --profile full

# Phase 6: Competitive
python benchmark_phase6.py --profile quick
python benchmark_phase6.py --profile full
```

### Chạy Tất cả

```bash
python run_all.py
```

## Examples

### Demo Scripts

```bash
# Association demo
python demo_association.py

# Consolidation demo
python demo_consolidation.py

# Stability demo
python demo_stability.py

# Scaled LSL demo
python demo_scaled_lsl.py

# Mini tokens demo
python demo_mini_tokens.py

# Attention living demo
python demo_attention_living.py
```

### Interactive Chat

```bash
python interactive_chat.py
```

## Best Practices

### 1. Seed Control

```python
# Luôn sử dụng seed cho reproducibility
model = LivingSynapseLM(vocab_size=1000, hidden_dim=256, seed=42)
```

### 2. Batch Processing

```python
# Xử lý batch tokens
def process_batch(model, tokens):
    for i in range(len(tokens) - 1):
        model.observe(tokens[i], tokens[i+1])
```

### 3. Memory Management

```python
# Consolidate định kỳ
if model.step_count % 1000 == 0:
    model.consolidate()
    model.replay(n=16)
```

### 4. Sparsity Tuning

```python
# Điều chỉnh sparsity cho trade-off capacity/compute
model = LivingSynapseLM(
    vocab_size=1000,
    hidden_dim=256,
    use_sdr=True,
    sdr_sparsity=0.1  # 10% sparsity = higher capacity
)
```

### 5. Predictive Coding Tuning

```python
# Điều chỉnh theta cho suppression
model = LivingSynapseLM(
    vocab_size=1000,
    hidden_dim=256,
    use_predictive_coding=True,
    theta=0.01  # Lower = more suppression
)
```

## Troubleshooting

### Prediction Error quá cao

- Giảm learning rate
- Tăng số epoch
- Kiểm tra tokenization
- Thử với semantic SDR

### Memory overflow

- Giảm long_context_capacity
- Gi giảm episodic buffer size
- Tắt các features không cần thiết

### Generation kém

- Tăng temperature
- Sử dụng top-p sampling
- Tăng training data
- Bật long context memory

### Compute quá chậm

- Bật use_sparse_computation
- Giảm hidden_dim
- Giảm sdr_sparsity
- Sử dụng SDR mode
