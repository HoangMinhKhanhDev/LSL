# Tài liệu API LSL

## LivingSynapseLM

### Constructor

```python
LivingSynapseLM(
    vocab_size: int,
    hidden_dim: int,
    k_ratio: float = 0.4,
    seed: int = 0,
    slow_init: float = 0.1,
    attn_window: int = 4,
    use_sdr: bool = False,
    sdr_sparsity: float = 0.2,
    use_predictive_coding: bool = False,
    theta: float = 0.0,
    use_semantic_sdr: bool = False,
    semantic_hidden_dim: int = 1000,
    embedding_dim: int = 300,
    use_pretrained: bool = False,
    use_sparse_computation: bool = False,
    use_sparse_memory: bool = False,
    use_role_binding: bool = False,
    use_hierarchical_routing: bool = False,
    memory_candidate_cap: int = 64,
    use_long_context_memory: bool = False,
    long_context_capacity: int = 131072,
    long_context_strength: float = 10.0,
    long_context_confidence_threshold: float = 0.55
)
```

**Parameters:**
- `vocab_size`: Kích thước vocabulary
- `hidden_dim`: Kích thước hidden layer
- `k_ratio`: Tỷ lệ sparsity cho router (0.0-1.0)
- `seed`: Random seed cho reproducibility
- `slow_init`: Khởi tạo trọng số chậm
- `attn_window`: Kích thước window cho attention
- `use_sdr`: Bật SDR encoding
- `sdr_sparsity`: Tỷ lệ sparsity SDR (0.0-1.0)
- `use_predictive_coding`: Bật predictive coding
- `theta`: Threshold cho suppression
- `use_semantic_sdr`: Bật semantic SDR
- `semantic_hidden_dim`: Kích thước SDR dimension
- `embedding_dim`: Kích thước embedding dimension
- `use_pretrained`: Sử dụng pretrained embeddings
- `use_sparse_computation`: Bật sparse forward pass
- `use_sparse_memory`: Bật sparse memory
- `use_role_binding`: Bật role binding memory
- `use_hierarchical_routing`: Bật hierarchical routing
- `memory_candidate_cap`: Số candidates cho memory lookup
- `use_long_context_memory`: Bật long context memory
- `long_context_capacity`: Dung lượng long context memory
- `long_context_strength`: Strength cho long context retrieval
- `long_context_confidence_threshold`: Threshold confidence cho retrieval

### Methods

#### forward

```python
forward(token_id: int, target_id: Optional[int] = None) -> np.ndarray
```

Forward pass qua mô hình.

**Parameters:**
- `token_id`: ID token input
- `target_id`: ID token target (optional, cho training)

**Returns:**
- `logits`: Logits distribution over vocabulary

**Example:**
```python
logits = model.forward(token_id)
```

#### predict

```python
predict(token_id: int) -> np.ndarray
```

Dự đoán token tiếp theo với inference plasticity.

**Parameters:**
- `token_id`: ID token input

**Returns:**
- `probs`: Probability distribution over vocabulary

**Example:**
```python
probs = model.predict(token_id)
next_token = int(probs.argmax())
```

#### observe

```python
observe(
    token_id: int,
    target_id: int,
    reward: float = 0.0,
    store: bool = True
) -> Dict[str, float]
```

Observe token pair và cập nhật mô hình (online learning).

**Parameters:**
- `token_id`: ID token input
- `target_id`: ID token target
- `reward`: Reward signal (default: 0.0)
- `store`: Có lưu vào memory không (default: True)

**Returns:**
- Dictionary với keys:
  - `prediction_error`: Cross-entropy loss
  - `modulator`: Neuromodulator value
  - `novelty`: Novelty score
  - `top1`: Top-1 prediction
  - `p_target`: Probability của target token

**Example:**
```python
info = model.observe(token_id, target_id, reward=1.0)
print(f"Error: {info['prediction_error']:.4f}")
```

#### relation_probability

```python
relation_probability(
    source_id: int,
    effect_id: int,
    candidate_ids: Optional[List[int]] = None,
    top_k: Optional[int] = 3
) -> float
```

Tính probability của relation source -> effect.

**Parameters:**
- `source_id`: ID token source
- `effect_id`: ID token effect
- `candidate_ids`: List candidate IDs (optional)
- `top_k`: Chỉ giữ top-k candidates (optional)

**Returns:**
- `probability`: P(effect | source)

**Example:**
```python
prob = model.relation_probability(
    source_id=stroke_id,
    effect_id=aphasia_id,
    top_k=3
)
```

#### consolidate

```python
consolidate(
    threshold: Optional[float] = None,
    fraction: Optional[float] = None
) -> int
```

Consolidate live weights vào slow weights.

**Parameters:**
- `threshold`: Threshold cho consolidation (optional)
- `fraction`: Fraction của weights để consolidate (optional)

**Returns:**
- `n`: Số weights đã consolidate

**Example:**
```python
n = model.consolidate(threshold=0.005, fraction=0.3)
```

#### replay

```python
replay(n: int = 16, lr_factor: float = 0.5, rng: Optional[np.random.Generator] = None)
```

Replay từ episodic buffer.

**Parameters:**
- `n`: Số samples để replay
- `lr_factor`: Learning rate factor
- `rng`: Random generator (optional)

**Example:**
```python
model.replay(n=16, lr_factor=0.5)
```

#### reset_state

```python
reset_state()
```

Reset hidden states và buffers.

**Example:**
```python
model.reset_state()
```

#### reset_live

```python
reset_live()
```

Reset live weights và states.

**Example:**
```python
model.reset_live()
```

#### metrics

```python
metrics() -> Dict[str, float]
```

Lấy metrics hiện tại của mô hình.

**Returns:**
- Dictionary với các metrics:
  - `live_norm`: Norm của live weights
  - `slow_norm`: Norm của slow weights
  - `fatigue_means`: Mean fatigue per layer
  - `router_usage_mean`: Mean router usage
  - `global_state_norm`: Norm của global state
  - `step_count`: Số steps đã thực hiện
  - SDR metrics (nếu bật SDR)
  - Prediction error metrics (nếu bật predictive coding)

**Example:**
```python
metrics = model.metrics()
print(f"Live norm: {metrics['live_norm']:.4f}")
```

#### load_semantic_embeddings

```python
load_semantic_embeddings(vocab: Dict[str, int]) -> int
```

Load semantic embeddings cho semantic SDR.

**Parameters:**
- `vocab`: Dictionary mapping word -> token_id

**Returns:**
- `n`: Số embeddings đã load

**Example:**
```python
vocab = tokenizer.get_vocab()
n = model.load_semantic_embeddings(vocab)
```

---

## SDREncoder

### Constructor

```python
SDREncoder(dim: int, sparsity: float = 0.2, seed: Optional[int] = None)
```

**Parameters:**
- `dim`: Dimension của SDR
- `sparsity`: Tỷ lệ active bits (0.0-1.0)
- `seed`: Random seed

### Methods

#### encode

```python
encode(x: np.ndarray) -> np.ndarray
```

Encode dense vector thành sparse binary code.

**Parameters:**
- `x`: Dense vector shape (dim,)

**Returns:**
- Binary vector shape (dim,) với k active bits

**Example:**
```python
encoder = SDREncoder(dim=1024, sparsity=0.2)
sdr = encoder.encode(dense_vector)
```

#### encode_batch

```python
encode_batch(X: np.ndarray) -> np.ndarray
```

Encode batch của vectors.

**Parameters:**
- `X`: Dense vectors shape (batch, dim)

**Returns:**
- Binary codes shape (batch, dim)

**Example:**
```python
codes = encoder.encode_batch(dense_vectors)
```

---

## SemanticSDREncoder

### Constructor

```python
SemanticSDREncoder(
    vocab_size: int,
    sdr_dim: int,
    sparsity: float,
    embed_dim: int,
    seed: int,
    use_pretrained: bool = False
)
```

**Parameters:**
- `vocab_size`: Kích thước vocabulary
- `sdr_dim`: Dimension của SDR
- `sparsity`: Tỷ lệ active bits
- `embed_dim`: Dimension của embedding
- `seed`: Random seed
- `use_pretrained`: Sử dụng pretrained embeddings

### Methods

#### encode

```python
encode(token_id: int) -> np.ndarray
```

Encode token thành semantic SDR.

**Parameters:**
- `token_id`: ID token

**Returns:**
- Semantic SDR vector

**Example:**
```python
encoder = SemanticSDREncoder(vocab_size=10000, sdr_dim=1000, sparsity=0.2, embed_dim=300, seed=42)
sdr = encoder.encode(token_id)
```

#### load_builtin_embeddings

```python
load_builtin_embeddings(vocab: Dict[str, int]) -> int
```

Load builtin semantic embeddings.

**Parameters:**
- `vocab`: Dictionary mapping word -> token_id

**Returns:**
- Số embeddings đã load

**Example:**
```python
n = encoder.load_builtin_embeddings(vocab)
```

---

## LivingSynapseLayer

### Constructor

```python
LivingSynapseLayer(
    in_dim: int,
    out_dim: int,
    slow_init: float = 0.1,
    seed: int = 0
)
```

**Parameters:**
- `in_dim`: Input dimension
- `out_dim`: Output dimension
- `slow_init`: Khởi tạo trọng số chậm
- `seed`: Random seed

### Methods

#### forward

```python
forward(x: np.ndarray, use_sparse: bool = False) -> np.ndarray
```

Forward pass qua layer.

**Parameters:**
- `x`: Input vector
- `use_sparse`: Sử dụng sparse computation

**Returns:**
- Output vector

**Example:**
```python
layer = LivingSynapseLayer(in_dim=256, out_dim=256)
output = layer.forward(input_vector, use_sparse=True)
```

#### top_k_supervised_update

```python
top_k_supervised_update(
    error: np.ndarray,
    lr: float,
    k_frac: float,
    max_norm: float
)
```

Update với supervised error (top-k).

**Parameters:**
- `error`: Error vector
- `lr`: Learning rate
- `k_frac`: Fraction của top-k weights để update
- `max_norm`: Maximum norm clipping

**Example:**
```python
layer.top_k_supervised_update(error, lr=2.0, k_frac=0.12, max_norm=12.0)
```

#### top_k_hebbian_update

```python
top_k_hebbian_update(
    gain: float,
    lr: float,
    k_frac: float,
    max_norm: float
)
```

Update với Hebbian rule (top-k).

**Parameters:**
- `gain`: Gain factor
- `lr`: Learning rate
- `k_frac`: Fraction của top-k weights
- `max_norm`: Maximum norm clipping

**Example:**
```python
layer.top_k_hebbian_update(gain=1.0, lr=0.02, k_frac=0.12, max_norm=12.0)
```

#### consolidate

```python
consolidate(threshold: float, fraction: float) -> int
```

Consolidate live weights.

**Parameters:**
- `threshold`: Threshold
- `fraction`: Fraction

**Returns:**
- Số weights đã consolidate

**Example:**
```python
n = layer.consolidate(threshold=0.005, fraction=0.3)
```

#### inference_plasticity

```python
inference_plasticity(lr: float)
```

Bật plasticity trong inference.

**Parameters:**
- `lr`: Learning rate

**Example:**
```python
layer.inference_plasticity(lr=0.003)
```

#### recover_fatigue

```python
recover_fatigue(rate: float)
```

Phục hồi fatigue.

**Parameters:**
- `rate`: Recovery rate

**Example:**
```python
layer.recover_fatigue(rate=0.98)
```

#### decay_live

```python
decay_live(rate: float)
```

Decay live weights.

**Parameters:**
- `rate`: Decay rate

**Example:**
```python
layer.decay_live(rate=0.999)
```

---

## CorticalColumnSequenceMemory

### Constructor

```python
CorticalColumnSequenceMemory(
    vocab_size: int,
    hidden_dim: int,
    num_cells: int = 32,
    seed: int = 0
)
```

**Parameters:**
- `vocab_size`: Kích thước vocabulary
- `hidden_dim`: Hidden dimension
- `num_cells`: Số cells per column
- `seed`: Random seed

### Methods

#### forward

```python
forward(token_id: int) -> np.ndarray
```

Forward pass cho sequence prediction.

**Parameters:**
- `token_id`: ID token

**Returns:**
- Hidden state

**Example:**
```python
memory = CorticalColumnSequenceMemory(vocab_size=1000, hidden_dim=256)
state = memory.forward(token_id)
```

#### observe

```python
observe(token_id: int, target_id: int)
```

Observe transition.

**Parameters:**
- `token_id`: ID token input
- `target_id`: ID token target

**Example:**
```python
memory.observe(token_id, target_id)
```

#### predict

```python
predict(token_id: int) -> np.ndarray
```

Dự đoán token tiếp theo.

**Parameters:**
- `token_id`: ID token

**Returns:**
- Probability distribution

**Example:**
```python
probs = memory.predict(token_id)
```

---

## SparseKeyValueMemory

### Constructor

```python
SparseKeyValueMemory(
    capacity: int,
    key_dim: int,
    value_dim: int,
    candidate_cap: int = 64,
    seed: int = 0
)
```

**Parameters:**
- `capacity`: Dung lượng memory
- `key_dim`: Dimension của key
- `value_dim`: Dimension của value
- `candidate_cap`: Số candidates cho lookup
- `seed`: Random seed

### Methods

#### store

```python
store(key: np.ndarray, value: np.ndarray)
```

Lưu key-value pair.

**Parameters:**
- `key`: Key vector
- `value`: Value vector

**Example:**
```python
memory = SparseKeyValueMemory(capacity=1000, key_dim=256, value_dim=256)
memory.store(key_vector, value_vector)
```

#### retrieve

```python
retrieve(query: np.ndarray, k: int = 10) -> np.ndarray
```

Retrieve value cho query.

**Parameters:**
- `query`: Query vector
- `k`: Số candidates

**Returns:**
- Retrieved value

**Example:**
```python
value = memory.retrieve(query_vector, k=10)
```

---

## LongContextMemory

### Constructor

```python
LongContextMemory(
    capacity: int,
    vocab_size: int,
    candidate_cap: int = 64,
    seed: int = 0
)
```

**Parameters:**
- `capacity`: Dung lượng memory (chunks)
- `vocab_size`: Kích thước vocabulary
- `candidate_cap`: Số candidates
- `seed`: Random seed

### Methods

#### observe_transition

```python
observe_transition(token_id: int, target_id: int, vocab_size: int)
```

Observe token transition.

**Parameters:**
- `token_id`: ID token input
- `target_id`: ID token target
- `vocab_size`: Kích thước vocabulary

**Example:**
```python
memory = LongContextMemory(capacity=131072, vocab_size=1000)
memory.observe_transition(token_id, target_id, vocab_size=1000)
```

#### predict_next

```python
predict_next(
    token_id: int,
    vocab_size: int,
    return_confidence: bool = False,
    update_context: bool = True
) -> Tuple[Optional[int], Optional[float]]
```

Dự đoán token tiếp theo từ long context.

**Parameters:**
- `token_id`: ID token
- `vocab_size`: Kích thước vocabulary
- `return_confidence`: Trả về confidence
- `update_context`: Cập nhật context

**Returns:**
- Tuple (predicted_token_id, confidence)

**Example:**
```python
token, conf = memory.predict_next(token_id, vocab_size=1000, return_confidence=True)
```

#### reset_state

```python
reset_state()
```

Reset memory state.

**Example:**
```python
memory.reset_state()
```

---

## GenerationController

### Constructor

```python
GenerationController(
    model: LivingSynapseLM,
    tokenizer: Any,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9
)
```

**Parameters:**
- `model`: LivingSynapseLM instance
- `tokenizer`: Tokenizer instance
- `max_tokens`: Số tokens tối đa
- `temperature`: Sampling temperature
- `top_p`: Nucleus sampling threshold

### Methods

#### generate

```python
generate(
    prompt: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None
) -> str
```

Generate text từ prompt.

**Parameters:**
- `prompt`: Prompt string
- `max_tokens`: Số tokens tối đa (override)
- `temperature`: Temperature (override)
- `top_p`: Top-p threshold (override)

**Returns:**
- Generated text

**Example:**
```python
controller = GenerationController(model, tokenizer)
text = controller.generate("The cat", max_tokens=50)
```

---

## SDR Utility Functions

### hamming_overlap

```python
hamming_overlap(a: np.ndarray, b: np.ndarray) -> float
```

Tính Hamming overlap giữa hai binary vectors.

**Parameters:**
- `a`, `b`: Binary vectors

**Returns:**
- Số positions nơi cả hai đều là 1

**Example:**
```python
overlap = hamming_overlap(sdr1, sdr2)
```

### pairwise_overlap_matrix

```python
pairwise_overlap_matrix(codes: np.ndarray) -> np.ndarray
```

Tính pairwise overlap matrix.

**Parameters:**
- `codes`: Binary codes shape (n, dim)

**Returns:**
- Overlap matrix shape (n, n)

**Example:**
```python
matrix = pairwise_overlap_matrix(codes)
```

### combinatorial_capacity

```python
combinatorial_capacity(dim: int, k: int) -> int
```

Tính combinatorial capacity C(dim, k).

**Parameters:**
- `dim`: Total dimension
- `k`: Số active bits

**Returns:**
- Số distinct codes

**Example:**
```python
cap = combinatorial_capacity(1024, 20)  # 2^130+
```

### log2_capacity

```python
log2_capacity(dim: int, k: int) -> float
```

Tính log2 của combinatorial capacity.

**Parameters:**
- `dim`: Total dimension
- `k`: Số active bits

**Returns:**
- log2(C(dim, k))

**Example:**
```python
log_cap = log2_capacity(1024, 20)  # ~130
```

### sparsity_ratio

```python
sparsity_ratio(code: np.ndarray) -> float
```

Tính actual sparsity ratio.

**Parameters:**
- `code`: Binary vector

**Returns:**
- Fraction của active bits

**Example:**
```python
ratio = sparsity_ratio(sdr)  # 0.2 cho 20% sparsity
```

### active_indices

```python
active_indices(code: np.ndarray) -> List[int]
```

Lấy indices của active bits.

**Parameters:**
- `code`: Binary vector

**Returns:**
- List indices nơi code == 1

**Example:**
```python
indices = active_indices(sdr)
```

---

## Tokenizers

### SimpleWordTokenizer

```python
tokenizer = SimpleWordTokenizer()
tokens = tokenizer.encode("hello world")
text = tokenizer.decode(tokens)
vocab = tokenizer.get_vocab()
vocab_size = tokenizer.vocab_size
```

### SimpleSubwordTokenizer

```python
tokenizer = SimpleSubwordTokenizer(vocab_size=1000)
tokenizer.train(["hello world", "hello there"])
tokens = tokenizer.encode("hello world")
text = tokenizer.decode(tokens)
```

---

## Memory Components

### EpisodicBuffer

```python
from lsl import EpisodicBuffer

buffer = EpisodicBuffer(capacity=256, candidate_cap=64)
buffer.add((token_id, target_id))
samples = buffer.sample(n=16)
```

### RelationMemory

```python
from lsl import RelationMemory

memory = RelationMemory()
memory.update_relation(source_id, effect_id, strength=1.0)
prob = memory.relation_probability(source_id, effect_id)
```

### RoleBindingMemory

```python
from lsl import RoleBindingMemory

memory = RoleBindingMemory()
memory.bind("subject", "cat", token_id_cat)
subject = memory.query("subject")
```

### WorldMemory

```python
from lsl import WorldMemory

memory = WorldMemory(capacity=1000)
memory.add_evidence(fact, evidence)
answer = memory.query(question)
```
