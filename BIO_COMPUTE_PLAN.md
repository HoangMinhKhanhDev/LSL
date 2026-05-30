# Bio-Compute Completion Plan

Plan này không xem repo như trang trắng. Repo đã chứng minh được khá nhiều cơ
chế bằng thực nghiệm, nhưng đa số mới ở mức partial proof hoặc chưa đạt đúng
mức/đúng cách mong muốn. Mục tiêu bây giờ là hoàn thiện 4 cơ chế cốt lõi vận
hành LSL như một LM strict-path, đồng thời giữ neuromodulation và dendritic
computation như các cơ chế hỗ trợ sinh học làm tăng chọn lọc học và mật độ
tính toán.

Không kết thúc bằng cảm giác "có vẻ đúng". Chỉ kết thúc khi goal có benchmark
executable, benchmark fail/pass rõ ràng, JSON `success=true`, process exit
code `0`, và strict scanner sạch.

## Final Product - LSL as a Language Model

Đích cuối không phải là một bộ benchmark rời rạc. Đích cuối là một kiến trúc LM:
LSL phải là model có thể học online, thay đổi trọng số realtime, học liên tục,
không quên, train nhanh/chi phí thấp trên CPU, và tiến tới reasoning/generation
ngang ngửa LLM hiện đại trong các benchmark đã định nghĩa.

Vì vậy cần tách rõ 3 lớp:

- Mechanism proofs: chứng minh từng cơ chế sinh học hoạt động riêng lẻ.
- Integrated architecture: tổng hợp các cơ chế vào một inference/learning loop
  thống nhất, không phải gọi module rời để lấy điểm benchmark.
- Model-level claim: đo LSL như một LM thật bằng generation, reasoning,
  continual learning, adaptation latency, compute/energy proxy, và ablation.

`LivingSynapseLM` là core model gốc cho W_live/W_slow, local update,
predictive coding, SDR, sparse computation. `IntegratedLSLAgent` là lớp agent
cho memory/reasoning/generation. `BioComputeAgent` phải trở thành model chính
claim-bearing: nơi 4 cơ chế cốt lõi và các cơ chế hỗ trợ được tổng hợp vào một
kiến trúc LM thống nhất.

## Four Core Mechanisms

4 cơ chế cốt lõi là xương sống trực tiếp thay các thành phần chính của
Transformer:

- SDR thay dense embeddings/vectors: mọi biểu diễn ngữ nghĩa ưu tiên sparse
  binary active indices, pattern completion, capacity tổ hợp và sparse physical
  compute trên CPU.
- Hierarchical Predictive Coding thay backprop/Adam/SGD: mỗi tầng học bằng
  local prediction error, hard suppression triệt tín hiệu quen thuộc, và update
  xảy ra realtime tại synapse.
- Cortical Column Sequence Memory thay self-attention matrix: sequence context
  được học bằng cột/temporal segments, predicted state, burst-on-surprise, chi
  phí O(1) proxy mỗi token thay vì O(N^2).
- Two-Speed Memory thay batch retraining/centralized storage: W_live học nhanh,
  W_slow giữ bền, hippocampal sparse memory và replay/consolidation chống quên.

Neuromodulation là cơ chế điều phối học/chú ý/ổn định cho 4 core mechanisms.
Dendritic computation là compute substrate tăng sức mạnh phi tuyến trong từng
neuron/branch. Chúng rất quan trọng, nhưng không thay đổi việc 4 core trên là
trục kiến trúc LM chính.

## Current Maturity Map

- Cơ chế 1 - Predictive Coding: đã có thực nghiệm và kết quả ban đầu; cần nâng
  từ local predictive proof lên cơ chế truyền prediction error đúng nghĩa hơn,
  suppression sâu hơn, zero redundant update rõ hơn, và real-time adaptation.
- Cơ chế 2 - SDR: đã có SDR/capacity/completion/semantic overlap; cần nâng mức
  semantic structure, online incremental fit, cross-lingual overlap, noise
  robustness 70% mask, và kiểm soát dense allocation trá hình.
- Cơ chế 3 - Cortical Column: đã có sequence memory/burst/suppression proof;
  cần nâng grammar after one pass, long-range coherence, zero-shot grammar
  transfer, anti-attention latency, và energy proxy.
- Cơ chế 4 - Two-Speed Memory/Hippocampus: LSL đã có W_live/W_slow và memory;
  cần biến hippocampus thành sparse auto-associative memory đúng hơn, bounded
  partial-cue recall, surprise-gated encoding, sparse replay <= 10%.
- Hỗ trợ 1 - Neuromodulation: đã có neuromodulator/gating; cần align rõ hơn
  với dopamine/acetylcholine/serotonin, perfect attention gating, emotional
  context, homeostasis, curiosity drive.
- Hỗ trợ 2 - Dendritic Computation: đã có branch/tree proof mới; cần tiếp tục
  đưa sâu hơn vào compute path chính của LM, không để chỉ là benchmark
  context-pattern phụ.

## Non-Negotiable Constraints

- CPU-only strict path.
- Không backpropagation, `.backward()`, optimizer state, GPU dependency,
  PyTorch/TensorFlow/JAX trong strict path, Q/K/V attention, attention matrix,
  all-pairs token interaction, global hidden error signal, external API, hoặc
  hardcoded eval answer.
- Mọi learning update phải online, local, sparse, và đo được bằng diagnostics.
- Benchmark failure phải trả non-zero process code.
- Không hạ threshold để pass; nếu fail thì sửa cơ chế hoặc benchmark design.
- Claim boundary: pass các cơ chế sinh học không đồng nghĩa GPT-4/frontier LLM
  parity.

## Workstream 1 - Harden 4 Core Mechanisms + Neuromodulation

Predictive Coding:

- Nâng benchmark để đo error drop per layer >= 90% trong 10 epochs, deep
  suppression >= 95%, zero-update tokens >= 80%, one-pass causal reasoning
  >= 10x random, 3-hop causal chain >= 70%, và loss giảm ngay token kế tiếp
  sau fact mới.
- Implementation phải giữ local layer predictors; không dùng global hidden
  error hoặc backward pass.

SDR:

- Bảo đảm virtual sparse SDR chạy `d=100000,k=20` không dense allocation và
  report `log2(C(d,k))`.
- Thêm/harden tests cho 70% masked completion >= 80%, subword semantic
  `"unhappy"` / `"unhappiness"`, online incremental semantic update, và
  `"não"` / `"brain"` overlap >= 30% nếu claim cross-lingual.

Cortical Column:

- Nâng sequence memory để seen sequences recall 100%, grammar >= 95% sau 1
  pass corpus, topic coherence 200-token span >= 0.5, latency seq len 1000
  không tăng đáng kể so với len 10, và energy proxy >= 100x dense baseline.
- Cơ chế phải giữ predicted-token silence và burst-on-surprise learning.

Hippocampus:

- Chuyển từ memory proof sang sparse auto-associative memory: 10,000 facts,
  exact recall 100%, bounded partial-cue retrieval, no full scan, encode only
  `surprise > threshold`, consolidation replay <= 10%.
- W_live học nhanh 1 lần; W_slow chỉ nhận qua consolidation.

Neuromodulation:

- Gate update bằng dopamine/acetylcholine/serotonin-style signals: >= 95%
  updates trên novel/surprising tokens, formal/casual tone switch, weight norms
  stable within +/-10%, curiosity chọn uncertain predictions tốt hơn random.
- Diagnostics phải tách novelty, surprise, reward/plasticity, stability.
- Benchmark canonical cho toàn bộ workstream này là
  `benchmarks/phase9/benchmark_bio_mechanisms_1_5_targets.py`; nó nằm trong
  `run_phase9.py` để `claim/full` fail nếu bất kỳ target nào của 4 core
  mechanisms hoặc neuromodulation tụt.

## Workstream 2 - Deepen Dendritic Compute Substrate

- Implement dendritic segments như sparse nonlinear mini-processors: mỗi
  segment có active-bit pattern, threshold/nonlinear spike, local strength, và
  output vote/gating.
- Support role/context disambiguation: cùng subject/object bits nhưng context
  khác phải ra output khác.
- Support noisy/partial cues: không chỉ exact 3-bit lookup; benchmark phải che
  nhiễu/mask và vẫn đạt target.
- Track sparse ops thực sự: `last_ops`, segment count, dense baseline proxy,
  ops gain >= 50x.
- Target pass: nonlinear/context accuracy >= 95%, role/context accuracy >=
  90%, quality degradation <= 5%, sparse ops >= 50x better than dense proxy.
- G6 strict pass: branch nonlinearity khác flat sum, branch overlap <= 10%,
  AND/OR coincidence >= 90%, XOR bằng 1 dendritic neuron, active branches <=
  5%.
- Moonshot pass: 1,000 branches per neuron, branch-level SDR unique receptive
  fields, 100% branch-local Hebbian learning with zero global error updates,
  zero-update branches >= 90%, compute-density gain >= 100x.

## Workstream 3 - Integrated LSL/BioComputeAgent

- Tích hợp 4 core mechanisms + support mechanisms vào một strict-path LM/agent,
  không chỉ benchmark riêng lẻ.
- Public surface tối thiểu: `observe_text`, `observe_fact`, `recall_fact`,
  `observe_context_pattern`, `predict_context_pattern`, `consolidate`,
  `generate`, `answer`, `diagnostics`.
- Learning loop thống nhất: token -> SDR/subword representation -> predictive
  stack/columns -> dendritic/context gating -> hippocampal/slow memory ->
  neuromodulated local update -> generation/reasoning output.
- Mỗi cơ chế có constructor flag `use_<mechanism>` để ablation được.
- Integrated benchmark phải chứng minh: fact recall, chain reasoning, dendrite
  accuracy, generation coherence, retention, pc zero-update, neuromod novel
  update, replay budget, no full scan.
- Ablation rule: tắt từng cơ chế phải làm metric liên quan giảm >= 20%; nếu
  không giảm thì cơ chế chưa thật sự đóng góp.

## Workstream 4 - Model-Level LM Evidence

- Thêm benchmark đo LSL như model chính, không chỉ subsystem: online language
  learning, next-token adaptation sau 1 token mới, open generation quality,
  multi-hop reasoning, long-context recall, và continual learning trên cùng
  instance model.
- Benchmark canonical cho lớp này là
  `benchmarks/phase9/benchmark_lsl_model_level.py`; nó phải nằm trong
  `run_phase9.py` để `claim/full` fail nếu model-level fail.
- So sánh cost với baseline nhỏ: parameter count < 1M khi claim, training data
  target cỡ MB, CPU training/inference, sparse ops/energy proxy, latency per
  token.
- Đo catastrophic forgetting sau 10,000 facts; target cuối là ~0% trong strict
  synthetic suite và < 1% trong expanded suite nếu benchmark thực tế có noise.
- Tách claim "đạt benchmark LSL-defined" khỏi claim "ngang ngửa LLM hiện đại";
  muốn claim ngang ngửa phải có benchmark public/heldout tương ứng và baseline
  cùng điều kiện.

## Final Gates

Trước khi kết thúc, chạy lại toàn bộ từ đầu:

```powershell
python run_all.py
python benchmark_goal_strict.py
python benchmarks/phase5/run_moonshot.py --profile claim
python benchmarks/phase6/run_phase6.py --profile claim
python benchmarks/phase7/run_phase7.py --profile claim
python benchmarks/phase8/run_phase8.py --profile claim
python benchmarks/phase9/benchmark_bio_mechanisms_1_5_targets.py
python benchmarks/phase9/run_phase9.py --profile claim
python benchmarks/phase9/run_phase9.py --profile full
```

Stop ngay và sửa nếu:

- Bất kỳ command nào exit khác `0`.
- Bất kỳ benchmark JSON nào thiếu `success=true`.
- Strict scanner fail.
- Một target trong yêu cầu chưa có executable benchmark.
- Benchmark pass nhờ hardcoded answer, full scan, dense allocation trá hình,
  external API, attention, backprop, optimizer, hoặc framework bị cấm.

Kết luận hợp lệ khi pass toàn bộ: repo có executable evidence cho 4 core
mechanisms được nâng lên target mong muốn, neuromodulation và dendritic
computation đóng vai trò hỗ trợ sinh học có benchmark riêng, và
LSL/BioComputeAgent được đo như một LM thống nhất thay vì một tập proof rời
rạc.
