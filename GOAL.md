


LSL — Master Research Plan: 3 Phases, 3 Mechanisms
Mục tiêu tối thượng
Chứng minh bằng code + số liệu rằng một kiến trúc lấy cảm hứng từ não bộ có thể:

Học liên tục real-time mà không cần backprop toàn cục
Hiểu ngữ pháp và sinh văn bản mạch lạc mà không cần attention O(n²)
Suy luận logic cơ bản từ pattern sequence
Chi phí tính toán thấp đến mức khó tin so với Transformer
Thứ tự các Phase
IMPORTANT

Thứ tự này không phải ngẫu nhiên. SDR phải có trước vì Predictive Coding và Cortical Column đều cần biểu diễn thưa mới hoạt động đúng. Đây là phụ thuộc kiến trúc cứng, không thể đảo.


Phase 1: SDR        → Biểu diễn đúng
                        ↓
Phase 2: Predictive Coding → Học đúng (dùng SDR làm signal carrier)
                        ↓
Phase 3: Cortical Column  → Nhớ đúng (dùng PC + SDR làm nền tảng)
                        ↓
         Kết quả:  Sinh văn bản + Suy luận + Học real-time
PHASE 1 — Sparse Distributed Representations (SDR)
Câu hỏi cần chứng minh
"Liệu biểu diễn thưa nhị phân có thể mang đủ thông tin ngữ nghĩa, lưu trữ không xung đột, và học từ ít ví dụ hơn dense vector không?"

Mục tiêu (Goals)
G1.1 — Biểu diễn Ngữ nghĩa (Semantic Encoding)
Tạo ra SDR sao cho các khái niệm liên quan có Hamming overlap cao và khái niệm không liên quan có overlap thấp
Mục tiêu số: overlap(related) / overlap(random) ≥ 3.0x
Ví dụ cụ thể: overlap("stroke", "brain") phải cao hơn overlap("stroke", "table") ít nhất 3 lần
G1.2 — Sức chứa tăng mũ (Exponential Capacity)
Chứng minh bằng toán học và đo đếm: với d=1024, k=20, không gian biểu diễn là C(1024,20)≈10 
41
  patterns
So sánh với dense: với d=64, không gian biểu diễn chỉ là R 
64
  (liên tục nhưng dễ bị nhiễu)
Mục tiêu số: capacity_log2 ≥ 130 bits (tức 2 
130
  patterns)
G1.3 — Không xung đột (Interference-Free Storage)
Học lần lượt 80 token/khái niệm khác nhau
Kiểm tra: pattern của token đầu tiên có bị corrupted sau khi học token thứ 80 không
Mục tiêu số: Pattern retention ≥ 90% (Hamming distance giữa learned pattern và original ≤ 2 bits)
G1.4 — Học từ ít ví dụ (1-shot / few-shot)
Sau khi học 1 lần xuất hiện của từ mới, mô hình phải nhận ra lần sau với xác suất cao
Mục tiêu số: Recognition accuracy ≥ 80% sau 1 lần học
G1.5 — Khôi phục Pattern (Pattern Completion)
Khi nhận input bị che 50% bit, hệ thống phải khôi phục ≥ 70% pattern gốc
Mục tiêu số: Completion accuracy ≥ 70% với 50% mask
G1.6 — Tính toán Thưa thực sự (Sparse Computation)
forward() chỉ cần O(k × d) phép cộng, không dùng dense matmul O(d²)
Mục tiêu số: Compute reduction ≥ 40x so với dense matmul (d=1024, k=20)
Ràng buộc (Constraints)
Ràng buộc	Lý do
SDR PHẢI là binary {0,1}, không phải float	Float làm mất tính interference-free
Sparsity cố định: k ≤ 2% × d, mọi lúc	Sparsity cao hơn → capacity giảm theo hàm mũ
Semantic encoding PHẢI dùng pre-trained info	Corpus nhỏ (132 từ) không đủ để tự học semantic
Random Projection Matrix PHẢI cố định	Thay đổi projection = mất semantic structure
Không dùng dense matmul khi input là SDR	Vi phạm mục tiêu sparse computation
Mọi update PHẢI chỉ tác động tại active indices	Dense update phá vỡ interference-free
Không được phép (Anti-goals)
❌ Dùng dropout (SDR không cần — đã có sparsity tự nhiên)
❌ Normalize vector (SDR binary không cần norm)
❌ Softmax trong hidden layers (chỉ dùng ở output)
❌ Thay đổi sparsity k theo thời gian (phá vỡ capacity bounds)
PHASE 2 — Predictive Coding (Thay backprop)
Câu hỏi cần chứng minh
"Liệu một mạng neuron thuần cục bộ, không có backprop toàn cục, có thể học ngữ cảnh và giảm loss dự đoán từ tiếp theo không?"

Mục tiêu (Goals)
G2.1 — Hierarchical Prediction (Dự đoán Phân cấp)
Mỗi tầng phải tạo ra top-down prediction của tầng dưới dựa trên trạng thái của nó ở bước trước
Mục tiêu số: Prediction error norm tại mỗi tầng phải giảm ít nhất 50% sau 25 epochs
G2.2 — Signal Suppression (Triệt tiêu Tín hiệu)
Khi dự đoán đúng, tín hiệu phải bị triệt tiêu (suppress về 0)
Chỉ truyền sai số dự đoán lên tầng cao hơn
Mục tiêu số: Suppression ratio ≥ 60% tại ngưỡng θ = 0.02
G2.3 — Local Learning Only (Học Cục bộ Hoàn toàn)
Không có bất kỳ ma trận feedback toàn cục (DFA matrices B_rec, B_ssm, B_emb)
Mỗi synapse chỉ dùng thông tin cục bộ: pre-synaptic, post-synaptic, và local prediction error
Mục tiêu số: Zero global backward passes, zero cross-layer gradient flow
G2.4 — Next-token Prediction Convergence
Với cơ chế học cục bộ, mô hình vẫn phải hội tụ dự đoán từ tiếp theo
Mục tiêu số: Eval loss ≤ 4.0 sau 25 epochs (so với random baseline ~4.45)
G2.5 — Energy Savings (Tiết kiệm Năng lượng)
Signal suppression phải chứng minh được tiết kiệm tính toán thực sự
Mục tiêu số: Với θ = 0.02, lượng synapse được update giảm ≥ 60%
G2.6 — Simple Reasoning Proof (Suy luận Đơn giản)
Mô hình phải học được quan hệ A→B→C từ corpus
Mục tiêu số: Sau khi học "stroke causes aphasia", predict "stroke → aphasia" với p ≥ 0.3
Ràng buộc (Constraints)
Ràng buộc	Lý do
Prediction weights PHẢI độc lập với feedforward weights	Dùng chung = circular dependency → không hội tụ
Prediction PHẢI dùng trạng thái t-1, không phải t	Dùng t = thông tin tương lai = gian lận
Không dùng global error signal để update hidden layers	Vi phạm local learning constraint
Suppression PHẢI là hard threshold (không phải soft)	Soft suppression = vẫn truyền full signal, chỉ scaled
Error amplification × scale PHẢI cố định	Adaptive scale → hidden hyperparameter optimization
Không được phép (Anti-goals)
❌ Backpropagation qua bất kỳ layer nào
❌ Global loss gradient (chỉ dùng local prediction error)
❌ DFA random matrices từ output đến hidden layers
❌ Dùng output error để update embedding layer
PHASE 3 — Cortical Column Sequence Memory
Câu hỏi cần chứng minh
"Liệu các đơn vị xử lý chuỗi cục bộ (cortical columns) có thể sinh ra văn bản mạch lạc và học ngữ pháp mà không cần attention matrix O(n²) không?"

Mục tiêu (Goals)
G3.1 — Temporal Sequence Learning (Học Chuỗi Thời gian)
Mỗi cortical column học một tập sequence patterns: nếu đã thấy [A, B, C], nó biết D sẽ đến
Mục tiêu số: Column next-step prediction accuracy ≥ 60% trên corpus training
G3.2 — Burst Firing on Surprise (Kích hoạt Khi Bất ngờ)
Khi column dự đoán đúng → im lặng (0 output, 0 compute)
Khi column dự đoán sai → burst (full activation, trigger learning)
Mục tiêu số: ≥ 80% của tokens sau epoch 10 phải được xử lý bởi "silent" columns
G3.3 — Grammar Emergence (Ngữ pháp Xuất hiện Tự nhiên)
Không hardcode bất kỳ quy tắc ngữ pháp nào
Mô hình phải tự học: "noun phrase → verb phrase → object phrase"
Mục tiêu số: Sinh 10 câu, ít nhất 7 câu phải có cấu trúc Subject-Verb-Object hợp lệ
G3.4 — Coherent Text Generation (Sinh văn bản Mạch lạc)
Sinh văn bản 20-50 từ từ một prompt
Văn bản phải giữ chủ đề nhất quán trong suốt chuỗi
Mục tiêu số: Topic coherence score ≥ 0.6 (dùng SDR overlap giữa các tokens liên tiếp)
G3.5 — O(n) Compute per Token (Không cần Attention)
Mỗi column xử lý token mới trong O(1) (kiểm tra sequence, predict, cập nhật)
Toàn bộ n columns xử lý song song → tổng O(n_columns), không phụ thuộc context length
Mục tiêu số: Processing time per token phải O(1) — không tăng khi sequence dài hơn
G3.6 — Continual Learning Without Forgetting (Học Mà Không Quên)
Học văn bản domain mới mà không làm hỏng patterns domain cũ
Mục tiêu số: Pattern retention của domain cũ ≥ 85% sau khi học domain mới
Ràng buộc (Constraints)
Ràng buộc	Lý do
Mỗi column PHẢI độc lập — không có cross-column communication trong forward pass	Cho phép xử lý song song, tránh attention bottleneck
Column chỉ được dùng SDR input từ Phase 1	Dense input → phá vỡ interference-free storage
Sequence learning PHẢI là local Hebbian — không backprop	Phase 2 ràng buộc, kế thừa sang Phase 3
Burst firing PHẢI là binary — không phải activation scaling	Biological fidelity + compute efficiency
Column count PHẢI được set trước, không thay đổi khi chạy	Dynamic topology → unstable training
Không dùng attention mechanism trong bất kỳ hình thức nào	Đây là mục tiêu chứng minh chính
Không được phép (Anti-goals)
❌ Self-attention, cross-attention, flash-attention
❌ Positional encoding (columns tự học vị trí qua sequence memory)
❌ Layer normalization (SDR đã binary, không cần norm)
❌ Giao tiếp cross-column trong forward pass (chỉ trong consolidation)
Ràng buộc Xuyên suốt Cả 3 Phase
Ràng buộc Sinh học (Biological Constraints)
Ràng buộc	Vi phạm sẽ dẫn đến
Không có global backward pass	Mất tính "real-time learning"
Không có optimizer state (Adam, SGD, momentum)	Không phải "living" weights
Mọi update phải là online (sau mỗi token, không batch)	Mất tính real-time continual learning
Phải có forgetting mechanism (fatigue, decay)	Không phản ánh biological memory
Học chỉ xảy ra tại synapse cục bộ	Mất tính scalability lên hardware neural
Ràng buộc Kỹ thuật (Engineering Constraints)
Ràng buộc	Lý do
Không dùng GPU — toàn bộ phải chạy trên CPU	Chứng minh compute-efficient, không phải GPU-dependent
Không dùng framework deep learning (PyTorch, TF, JAX)	Ràng buộc "no backprop" không thể enforce nếu dùng autograd
Bộ nhớ trọng số ≤ 500MB	Phải chạy được trên edge device
Inference time per token ≤ 10ms (trên CPU thông thường)	Thực tế sử dụng
Không dùng external API trong forward/observe	Offline, privacy-preserving
Ràng buộc Chứng minh (Scientific Constraints)
Ràng buộc	Lý do
Mọi claim phải có benchmark đo được	Tránh "theoretical" mà không verify
Benchmark phải so sánh với baseline rõ ràng (static model, dense model)	Tránh cherry-picking
Phải chứng minh được trên corpus mới (không chỉ corpus training)	Tránh overfitting
Số liệu phải reproducible (seed cố định)	Scientific validity
Tóm tắt Goals & Metrics
Phase	Cơ chế	Goals	Metric Chính
1	SDR	G1.1–G1.6	Semantic overlap ≥3x, Interference ≤10%, Compute ≥40x faster
2	Predictive Coding	G2.1–G2.6	Local error ↓50%, Loss ≤4.0, Suppression ≥60%
3	Cortical Column	G3.1–G3.6	Coherent text, Grammar emergence, O(n) compute
Kết quả cuối cùng nếu cả 3 phase thành công:

Một mô hình ngôn ngữ chạy CPU
Học real-time từ dữ liệu mới (không cần retrain)
Sinh văn bản mạch lạc (không cần attention)
Chi phí tính toán ≤ 1% so với Transformer cùng độ phức tạp ngữ nghĩa