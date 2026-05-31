"""Living Synapse Language Model - bio-inspired online local learning."""
from .synapse import LivingSynapseLayer
from .sparse_native import NATIVE_AVAILABLE, require_native
from .router import DynamicCircuitRouter
from .neuromod import Neuromodulator
from .memory import EpisodicBuffer, SparseKeyValueMemory
from .long_context import LongContextMemory
from .generation import DiscoursePlan, DiscourseState, GenerationController
from .world_memory import EvidenceAnswer, EvidenceRecord, WorldMemory
from .reasoning import RelationMemory, RoleBindingMemory, TraceReasoningMemory
from .homeostasis import HomeostaticController, HomeostaticState
from .workspace import EntityEventGraph, ReasoningWorkspace, WorkspaceStep
from .event_ssm import EventDrivenSSM
from .prior import OfflinePriorSDR
from .agent import IntegratedLSLAgent
from .core import LSLCoreConfig, LSLCoreModel
from .bio import (
    BioComputeAgent,
    BioNeuromodulator,
    DendriticLayer,
    DendriticSegment,
    HippocampalMemory,
    LocalPredictiveStack,
    OnePassCausalMemory,
    VirtualSparseSDR,
)
from .ssm import LivingSSM
from .model import LivingSynapseLM
from .tokenizer import SimpleWordTokenizer
from .subword_tokenizer import SimpleSubwordTokenizer
from .dataset_loader import DatasetConfig, DatasetLoader, DatasetSource, DatasetStats, DatasetTextSplits
from .results import run_metadata, write_result
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
    AssociativeMemory,
)
from .cortical_column import CorticalColumnSequenceMemory

__all__ = [
    "LivingSynapseLayer",
    "NATIVE_AVAILABLE",
    "require_native",
    "DynamicCircuitRouter",
    "Neuromodulator",
    "EpisodicBuffer",
    "SparseKeyValueMemory",
    "LongContextMemory",
    "DiscoursePlan",
    "DiscourseState",
    "GenerationController",
    "EvidenceAnswer",
    "EvidenceRecord",
    "WorldMemory",
    "RelationMemory",
    "RoleBindingMemory",
    "TraceReasoningMemory",
    "HomeostaticController",
    "HomeostaticState",
    "ReasoningWorkspace",
    "WorkspaceStep",
    "EntityEventGraph",
    "EventDrivenSSM",
    "OfflinePriorSDR",
    "IntegratedLSLAgent",
    "LSLCoreConfig",
    "LSLCoreModel",
    "BioComputeAgent",
    "BioNeuromodulator",
    "DendriticLayer",
    "DendriticSegment",
    "HippocampalMemory",
    "LocalPredictiveStack",
    "OnePassCausalMemory",
    "VirtualSparseSDR",
    "LivingSSM",
    "LivingSynapseLM",
    "SimpleWordTokenizer",
    "SimpleSubwordTokenizer",
    "DatasetConfig",
    "DatasetLoader",
    "DatasetSource",
    "DatasetStats",
    "DatasetTextSplits",
    "run_metadata",
    "write_result",
    "SDREncoder",
    "hamming_overlap",
    "pairwise_overlap_matrix",
    "combinatorial_capacity",
    "log2_capacity",
    "sparsity_ratio",
    "active_indices",
    "capacity_stats",
    "SemanticSDREncoder",
    "semantic_overlap",
    "semantic_overlap_ratio",
    "SparseAssociativeMemory",
    "AssociativeMemory",
    "CorticalColumnSequenceMemory",
]
