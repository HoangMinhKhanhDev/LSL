# Kiến trúc LSL (Living Synapse Language Model)

## Tổng quan

LSL là một mô hình ngôn ngữ bio-inspired sử dụng học cục bộ (local learning) online mà không cần backpropagation, optimizer state, GPU, hay các deep learning frameworks. Mô hình được thiết kế để hoạt động trên CPU với NumPy, tập trung vào tính toán thưa (sparse computation) và các cơ chế học tập sinh học.

## Các nguyên tắc thiết kế cốt lõi

### 1. Không có backpropagation toàn cục
- Không có `.backward()` hay gradient propagation
- Mọi cập nhật đều dựa trên local error signals
- Học tập diễn ra online theo thời gian thực

### 2. Không có optimizer state
- Không sử dụng Adam, SGD, momentum
- Không có tracking momenta
- Cập nhật weights trực tiếp với local rules

### 3. Tính toán thưa (Sparse Computation)
- Chỉ xử lý các active neurons
- Sử dụng Sparse Distributed Representations (SDR)
- Tối ưu hóa hiệu suất CPU

### 4. Predictive Coding
- Dự đoán local state từ previous state
- Tính prediction error cục bộ
- Suppression signal để tiết kiệm năng lượng

## Các thành phần chính

### 1. LivingSynapseLayer (`lsl/synapse.py`)

Lớp synapse sống với cơ chế:
- **W_slow**: Trọng số dài hạn (consolidated)
- **W_live**: Trọng số ngắn hạn (plastic)
- **fatigue**: Theo dõi sự mệt mỏi của synapse
- **inference_plasticity**: Học tập trong quá trình inference

```python
class LivingSynapseLayer:
    def __init__(self, in_dim, out_dim, slow_init=0.1, seed=0)
    def forward(self, x, use_sparse=False)
    def top_k_supervised_update(self, error, lr, k_frac, max_norm)
    def top_k_hebbian_update(self, gain, lr, k_frac, max_norm)
    def consolidate(self, threshold, fraction)
    def inference_plasticity(self, lr)
```

### 2. SDREncoder (`lsl/sdr.py`)

Encoder cho Sparse Distributed Representations:
- Chuyển đổi dense vectors thành sparse binary codes
- Sử dụng top-k selection theo magnitude
- Cung cấp capacity rất lớn (2^130 cho d=1024, k=20)

```python
class SDREncoder:
    def __init__(self, dim, sparsity=0.2, seed=None)
    def encode(self, x)  # Trả về binary vector với k active bits
    def encode_batch(self, X)
```

### 3. SemanticSDREncoder (`lsl/semantic_sdr.py`)

Encoder SDR với semantic priors:
- Sử dụng offline semantic embeddings
- Fixed random projection
- Top-k binarization
- Scale lên đến 1M vocabulary

```python
class SemanticSDREncoder:
    def __init__(self, vocab_size, sdr_dim, sparsity, embed_dim, seed, use_pretrained)
    def encode(self, token_id)
    def load_builtin_embeddings(self, vocab)
```

### 4. LivingSynapseLM (`lsl/model.py`)

Mô hình ngôn ngữ chính với các thành phần:

#### Embedding Layer
- `embed`: Token embedding với LivingSynapseLayer
- Có thể sử dụng SDR encoding

#### State Space Model (SSM)
- `ssm`: LivingSSM cho sequence modeling
- Không sử dụng attention matrix
- O(n) compute per token

#### Recurrent Layer
- `recurrent`: Recurrent layer với LivingSynapseLayer
- Kết hợp với global state

#### Predictive Coding
- `_pc_emb`, `_pc_ssm`, `_pc_rec`: Local transition predictors
- Tính prediction error cục bộ
- Suppression với threshold theta

#### Memory Systems
- `episodic`: EpisodicBuffer cho replay
- `long_context`: LongContextMemory cho retrieval 128k tokens
- `relation_memory`: RelationMemory cho reasoning
- `role_binding_memory`: RoleBindingMemory cho compositional reasoning

#### Association Memory
- `next_token_assoc`: Next-token association matrix
- `relation_assoc`: Relation association matrix
- Window-based updates

### 5. CorticalColumnSequenceMemory (`lsl/cortical_column.py`)

Sequence memory với burst/silent dynamics:
- Mỗi token có một column với multiple cells
- Expected tokens activate predicted cells
- Surprising tokens trigger burst
- Context segments cho prediction

### 6. SparseKeyValueMemory (`lsl/memory.py`)

Memory key-value thưa:
- Bounded candidate lookup
- Không full scan
- O(k) lookup với k candidates

### 7. LongContextMemory (`lsl/long_context.py`)

Long-context memory cho:
- Facts storage
- Instructions
- Next-token transitions
- Bounded retrieval đến 128k chunks

### 8. WorldMemory (`lsl/world_memory.py`)

World/evidence memory cho QA:
- Bounded storage
- Citation-style retrieval
- Evidence tracking

### 9. Reasoning Components (`lsl/reasoning.py`)

- `RelationMemory`: Lưu trữ quan hệ cause-effect
- `RoleBindingMemory`: Binding roles cho compositional reasoning
- `TraceReasoningMemory`: Multi-hop reasoning

### 10. Hierarchy (`lsl/hierarchy.py`)

Learned hierarchy/routing:
- Token-to-phrase routing
- Phrase-to-topic routing
- Dynamic circuit routing

### 11. Generation Controller (`lsl/generation.py`)

Open generation controller:
- Discourse state management
- Repetition fatigue
- Local discourse planning

## Luồng xử lý chính

### Forward Pass

```
token_id → one_hot → embed → tanh → [SDR encode] → gate → sparsify
    ↓
predictive coding (local prediction error)
    ↓
SSM → [SDR encode] → global_state mix → recurrent → tanh → [SDR encode]
    ↓
predictive coding (local prediction error)
    ↓
output → logits + association + long_context
```

### Learning (Observe)

```
token_id, target_id → forward → prediction error
    ↓
neuromodulation (novelty, reward)
    ↓
output update (supervised)
    ↓
predictive coding updates (local)
    ↓
association updates
    ↓
fatigue recovery, live decay
    ↓
store in episodic/long_context
```

## Các chế độ hoạt động

### 1. Basic Mode
- Hebbian learning
- Không predictive coding
- Dense computation

### 2. Predictive Coding Mode
- Local transition predictors
- Error suppression
- Energy savings

### 3. SDR Mode
- Sparse binary encoding
- Combinatorial capacity
- Sparse computation

### 4. Semantic SDR Mode
- Offline semantic priors
- Large vocabulary support
- Semantic structure preservation

### 5. Sparse Computation Mode
- Top-k sparse forward
- Active-index only
- 40x+ speedup

### 6. Long Context Mode
- Bounded retrieval
- Fact/instruction/transition memory
- 128k horizon

## Các metrics quan trọng

### 1. Prediction Error
- Cross-entropy loss
- Local prediction errors (e_emb, e_ssm, e_rec)
- Suppression percentage

### 2. Sparsity Metrics
- Target sparsity
- Actual sparsity per layer
- SDR capacity (log2)

### 3. Weight Norms
- Live norm (plastic weights)
- Slow norm (consolidated weights)
- Effective weight norms

### 4. Fatigue Metrics
- Mean fatigue per layer
- Recovery rate
- Usage statistics

### 5. Memory Metrics
- Episodic buffer size
- Long-context capacity
- Retrieval confidence

## Các constraints nghiêm ngặt

### Forbidden Constructs
- ❌ No backpropagation calls
- ❌ No optimizer state (Adam, SGD, momentum)
- ❌ No GPU requirement
- ❌ No deep learning frameworks (PyTorch, TensorFlow, JAX)
- ❌ No attention matrix (Q/K/V, all-pairs interaction)
- ❌ No DFA feedback matrices
- ❌ No global hidden error signal

### Required Properties
- ✅ Online local updates only
- ✅ CPU-only NumPy implementation
- ✅ Sparse computation with 40x+ speedup
- ✅ SDR capacity >= 2^130
- ✅ Semantic overlap >= 3x random
- ✅ Prediction error drop >= 50%
- ✅ Signal suppression >= 60%

## Scaling Laws

### SDR Scaling
- 1k vocabulary: d=256, k=20
- 10k vocabulary: d=512, k=40
- 100k vocabulary: d=1024, k=80
- 1M vocabulary: d=2048, k=160

### Memory Scaling
- Episodic: 256 samples
- Long context: 128k chunks
- Candidate cap: 64 per lookup

### Compute Scaling
- O(n) per token (constant with context)
- Sparse: O(k) with k << n
- Bounded retrieval: O(candidate_cap)

## Verification

### Strict Benchmark
```bash
python benchmark_goal_strict.py
```

Kiểm tra 18 goals:
- Phase 1: SDR (6 goals)
- Phase 2: Predictive Coding (6 goals)
- Phase 3: Cortical Column (6 goals)

### Moonshot Benchmark
```bash
python benchmark_moonshot.py --profile full
```

Mở rộng với:
- Semantic SDR scaling
- Physical sparse compute
- Long-context retrieval
- Real-corpus evaluation
- Exact-answer tasks
- Continual learning
- Hierarchy/routing
- Baseline competition

### Competitive Benchmark
```bash
python benchmark_phase6.py --profile full
```

Open generation, world memory, public reasoning, CPU efficiency.
