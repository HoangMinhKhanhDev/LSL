"""Living Synapse Language Model - bio-inspired real-time adaptive LM PoC.

This is NOT a LoRA / adapter variant. It is a small autoregressive model
whose internal weights and circuit routing change in real time during
inference using local biological rules - no optimizer, no backprop.
"""
from .synapse import LivingSynapseLayer
from .router import DynamicCircuitRouter
from .neuromod import Neuromodulator
from .memory import EpisodicBuffer
from .ssm import LivingSSM
from .model import LivingSynapseLM
from .tokenizer import SimpleWordTokenizer
from .sdr import (
    SDREncoder,
    hamming_overlap,
    pairwise_overlap_matrix,
    combinatorial_capacity,
    log2_capacity,
    sparsity_ratio,
    active_indices,
    capacity_stats,
)
from .semantic_sdr import (
    SemanticSDREncoder,
    semantic_overlap,
    semantic_overlap_ratio,
)
from .associative_memory import (
    SparseAssociativeMemory,
    AssociativeMemory,         # legacy alias
)
from .cortical_column import CorticalColumnSequenceMemory

__all__ = [
    # Core layers
    "LivingSynapseLayer",
    "DynamicCircuitRouter",
    "Neuromodulator",
    "EpisodicBuffer",
    "LivingSSM",
    # Top-level model
    "LivingSynapseLM",
    "SimpleWordTokenizer",
    # Phase 1 — SDR primitives
    "SDREncoder",
    "hamming_overlap",
    "pairwise_overlap_matrix",
    "combinatorial_capacity",
    "log2_capacity",
    "sparsity_ratio",
    "active_indices",
    "capacity_stats",
    # Phase 1 — Semantic SDR
    "SemanticSDREncoder",
    "semantic_overlap",
    "semantic_overlap_ratio",
    # Phase 1 — Associative Memory / Pattern Completion
    "SparseAssociativeMemory",
    "AssociativeMemory",
    # Phase 3 — Cortical Column
    "CorticalColumnSequenceMemory",
]
