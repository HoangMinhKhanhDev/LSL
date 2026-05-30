"""Integrated strict-path agent combining memory, reasoning, and generation."""
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .generation import DiscoursePlan, GenerationController
from .homeostasis import HomeostaticController
from .long_context import LongContextMemory
from .reasoning import TraceReasoningMemory
from .subword_tokenizer import SimpleSubwordTokenizer
from .tokenizer import SimpleWordTokenizer
from .workspace import EntityEventGraph, ReasoningWorkspace
from .world_memory import EvidenceAnswer, WorldMemory


class SymbolTable:
    def __init__(self):
        self.to_id: Dict[str, int] = {}
        self.to_name: Dict[int, str] = {}

    def id(self, value) -> int:
        key = str(value).strip().lower()
        if key not in self.to_id:
            idx = len(self.to_id) + 1
            self.to_id[key] = idx
            self.to_name[idx] = key
        return self.to_id[key]

    def name(self, idx: Optional[int]) -> Optional[str]:
        if idx is None:
            return None
        return self.to_name.get(int(idx))


class IntegratedLSLAgent:
    """One strict local/online pipeline for Phase 8 external-style checks."""

    def __init__(
        self,
        vocab_size: int = 4000,
        tokenizer: str = "subword",
        candidate_cap: int = 128,
        seed: int = 0,
    ):
        self.vocab_size = int(vocab_size)
        self.seed = int(seed)
        self.symbols = SymbolTable()
        self.world = WorldMemory(capacity=262144, candidate_cap=candidate_cap, seed=seed)
        self.events = EntityEventGraph(candidate_cap=candidate_cap)
        self.workspace = ReasoningWorkspace()
        self.traces = TraceReasoningMemory()
        self.homeostasis = HomeostaticController()
        self.long_context = LongContextMemory(
            capacity=65536,
            vocab_size=vocab_size,
            candidate_cap=min(candidate_cap, 64),
            store_transition_index=False,
            seed=seed,
        )
        if tokenizer == "subword":
            self.tokenizer = SimpleSubwordTokenizer(vocab_size=vocab_size, max_merges=600)
        else:
            self.tokenizer = SimpleWordTokenizer(vocab_size=vocab_size)
        self.generator: Optional[GenerationController] = None
        self.last_diagnostics: Dict[str, float] = {}

    def build_tokenizer(self, text: str) -> None:
        self.tokenizer.build_vocab(text)
        self.vocab_size = self.tokenizer.vocab_size
        self.long_context.vocab_size = self.vocab_size

    def observe_text(self, text: str, source: str = "text", learn_transitions: bool = True) -> None:
        self.world.observe_chunk(text, source=source)
        if not learn_transitions:
            return
        tokens = self.tokenizer.encode(text)
        for i in range(len(tokens) - 1):
            self.long_context.observe_transition(tokens[i], tokens[i + 1], vocab_size=self.vocab_size)
            self.homeostasis.observe(active_count=1, total_count=max(1, self.vocab_size), local_error=0.10)
        self.generator = None

    def observe_texts(self, texts: Iterable[str], source: str = "text") -> None:
        for idx, text in enumerate(texts):
            self.observe_text(text, source=f"{source}:{idx}")

    def observe_event(self, subject: str, relation: str, obj: str, episode_id: int = 0, evidence_id: int = 0) -> None:
        s = self.symbols.id(subject)
        r = self.symbols.id(relation)
        o = self.symbols.id(obj)
        self.events.observe_event(s, r, o, episode_id=episode_id, evidence_id=evidence_id)
        self.workspace.bind_pair(s, r, o)

    def answer(self, question: str):
        math = self.traces.execute_math(question)
        if math is not None:
            self.last_diagnostics = {"mode": 1.0, "full_scan": 0.0}
            return str(math)
        stack = self.traces.execute_stack(question)
        if stack is not None:
            self.last_diagnostics = {"mode": 2.0, "full_scan": 0.0}
            return str(stack)

        chain = self._answer_chain(question)
        if chain is not None:
            self.last_diagnostics = {**self.events.diagnostics(), "mode": 3.0}
            return chain

        world_answer = self.world.answer(question)
        if world_answer.answer is not None:
            self.last_diagnostics = {**world_answer.diagnostics, "mode": 4.0}
            return world_answer.answer

        event = self._answer_event(question)
        if event is not None:
            self.last_diagnostics = {**self.events.diagnostics(), "mode": 5.0}
            return event
        self.last_diagnostics = {"mode": 0.0, "full_scan": 0.0}
        return None

    def answer_with_evidence(self, question: str) -> EvidenceAnswer:
        return self.world.answer(question)

    def _answer_event(self, question: str) -> Optional[str]:
        q = str(question).strip().lower()
        location = re.search(r"where is ([a-z0-9_\-.]+)\?", q)
        if location:
            value = self.events.query(self.symbols.id(location.group(1)), self.symbols.id("location"))
            return self.symbols.name(value)
        holder = re.search(r"where is the ([a-z0-9_\-.]+)\?", q)
        if holder:
            value = self.events.query(self.symbols.id(holder.group(1)), self.symbols.id("holder"))
            return self.symbols.name(value)
        owner = re.search(r"who has the ([a-z0-9_\-.]+)\?", q)
        if owner:
            value = self.events.query(self.symbols.id(owner.group(1)), self.symbols.id("holder"))
            return self.symbols.name(value)
        patterns = [
            r"what does ([a-z0-9_\-.]+) ([a-z0-9_\-.]+)\?",
            r"where does ([a-z0-9_\-.]+) ([a-z0-9_\-.]+)\?",
            r"who does ([a-z0-9_\-.]+) ([a-z0-9_\-.]+)\?",
        ]
        for pattern in patterns:
            match = re.search(pattern, q)
            if not match:
                continue
            subject, relation = match.groups()
            value = self.events.query(self.symbols.id(subject), self.symbols.id(relation))
            return self.symbols.name(value)
        return None

    def _answer_chain(self, question: str) -> Optional[str]:
        q = str(question).strip().lower()
        match = re.search(r"starting from ([a-z0-9_\-.]+), follow (.+?)\?", q)
        if not match:
            return None
        start, raw_relations = match.groups()
        relations = [self.symbols.id(part.strip()) for part in raw_relations.split(" then ") if part.strip()]
        value = self.events.query_chain(self.symbols.id(start), relations)
        return self.symbols.name(value)

    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        if self.generator is None:
            self.generator = GenerationController(
                memory=self.long_context,
                vocab_size=self.vocab_size,
                candidate_limit=16,
                unk_id=getattr(self.tokenizer, "word_to_id", getattr(self.tokenizer, "token_to_id", {})).get("<UNK>", 1),
                seed=self.seed,
            )
        prompt_tokens = self.tokenizer.encode(prompt)
        plan = DiscoursePlan(target_length=max_new_tokens)
        generated = self.generator.generate(prompt_tokens, max_new_tokens=max_new_tokens, plan=plan)
        return self.tokenizer.decode(generated)

    def generation_metrics(self, text: str) -> Dict[str, float]:
        if self.generator is None:
            self.generate(text, 1)
        tokens = self.tokenizer.encode(text)
        unk = getattr(self.tokenizer, "word_to_id", getattr(self.tokenizer, "token_to_id", {})).get("<UNK>", 1)
        return GenerationController.generation_metrics(tokens, unk_id=unk)

    def diagnostics(self) -> Dict[str, float]:
        return {
            **{f"world_{k}": v for k, v in self.world.diagnostics().items()},
            **{f"event_{k}": v for k, v in self.events.diagnostics().items()},
            **{f"workspace_{k}": v for k, v in self.workspace.diagnostics().items()},
            **{f"homeostasis_{k}": v for k, v in self.homeostasis.diagnostics().items()},
            **{f"last_{k}": v for k, v in self.last_diagnostics.items()},
        }
