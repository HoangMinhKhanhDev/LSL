"""Local relation and role memories for compositional reasoning.

These memories are deliberately small and online. They update only local
count tables for observed items; there is no global gradient.
"""
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple


class RelationMemory:
    """Directed local association memory with bounded multi-hop queries."""

    def __init__(self, max_hops: int = 4):
        self.max_hops = int(max_hops)
        self.edges: Dict[int, Counter] = defaultdict(Counter)
        self.relation_edges: Dict[str, Dict[int, Counter]] = defaultdict(lambda: defaultdict(Counter))
        self.offset_rules: Dict[str, Counter] = defaultdict(Counter)
        self.category_rules: Dict[Tuple[str, int], Counter] = defaultdict(Counter)

    def observe(self, source: int, target: int, relation: str = "next", strength: float = 1.0) -> None:
        source = int(source)
        target = int(target)
        relation = str(relation)
        self.edges[source][target] += float(strength)
        self.relation_edges[relation][source][target] += float(strength)

    def observe_chain(self, tokens: Iterable[int], relation: str = "next") -> None:
        items = [int(t) for t in tokens]
        for a, b in zip(items, items[1:]):
            self.observe(a, b, relation=relation)

    def observe_causal(
        self,
        cause: int,
        effect: int,
        relation: str = "causes",
        category: Optional[int] = None,
        strength: float = 1.0,
    ) -> None:
        cause = int(cause)
        effect = int(effect)
        self.observe(cause, effect, relation=relation, strength=strength)
        offset = int(effect - cause)
        self.offset_rules[relation][offset] += float(strength)
        if category is not None:
            self.category_rules[(relation, int(category))][offset] += float(strength)

    def _best_from_counter(self, counter: Counter) -> Optional[int]:
        if not counter:
            return None
        return int(max(counter.items(), key=lambda item: (item[1], -item[0]))[0])

    def predict_direct(self, source: int, relation: Optional[str] = None) -> Optional[int]:
        table = self.relation_edges[relation] if relation is not None else self.edges
        return self._best_from_counter(table.get(int(source), Counter()))

    def predict_multihop(
        self,
        source: int,
        hops: int = 2,
        relation: Optional[str] = None,
    ) -> Optional[int]:
        current = int(source)
        for _ in range(min(int(hops), self.max_hops)):
            nxt = self.predict_direct(current, relation=relation)
            if nxt is None:
                return None
            current = int(nxt)
        return current

    def predict_causal(
        self,
        cause: int,
        category: Optional[int] = None,
        relation: str = "causes",
    ) -> Optional[int]:
        exact = self.predict_direct(cause, relation=relation)
        if exact is not None:
            return exact
        rule_counter = Counter()
        if category is not None:
            rule_counter.update(self.category_rules.get((relation, int(category)), Counter()))
        rule_counter.update(self.offset_rules.get(relation, Counter()))
        offset = self._best_from_counter(rule_counter)
        if offset is None:
            return None
        return int(cause) + int(offset)

    def edge_count(self) -> int:
        return sum(len(counter) for counter in self.edges.values())


class RoleBindingMemory:
    """Local subject/verb/object binding memory."""

    def __init__(self):
        self.object_by_subject_verb: Dict[Tuple[int, int], Counter] = defaultdict(Counter)
        self.subject_by_verb_object: Dict[Tuple[int, int], Counter] = defaultdict(Counter)
        self.verb_by_subject_object: Dict[Tuple[int, int], Counter] = defaultdict(Counter)

    def observe_event(self, subject: int, verb: int, obj: int, strength: float = 1.0) -> None:
        subject = int(subject)
        verb = int(verb)
        obj = int(obj)
        self.object_by_subject_verb[(subject, verb)][obj] += float(strength)
        self.subject_by_verb_object[(verb, obj)][subject] += float(strength)
        self.verb_by_subject_object[(subject, obj)][verb] += float(strength)

    def _best(self, counter: Counter) -> Optional[int]:
        if not counter:
            return None
        return int(max(counter.items(), key=lambda item: (item[1], -item[0]))[0])

    def predict_object(self, subject: int, verb: int) -> Optional[int]:
        return self._best(self.object_by_subject_verb.get((int(subject), int(verb)), Counter()))

    def predict_subject(self, verb: int, obj: int) -> Optional[int]:
        return self._best(self.subject_by_verb_object.get((int(verb), int(obj)), Counter()))

    def predict_verb(self, subject: int, obj: int) -> Optional[int]:
        return self._best(self.verb_by_subject_object.get((int(subject), int(obj)), Counter()))

    def binding_count(self) -> int:
        return len(self.object_by_subject_verb)


class TraceReasoningMemory:
    """Small local executor for learned trace-style reasoning tasks."""

    def __init__(self):
        self.trace_counts = Counter()

    def observe_trace(self, trace_type: str) -> None:
        self.trace_counts[str(trace_type)] += 1

    def execute_math(self, prompt: str) -> Optional[int]:
        text = str(prompt).lower()
        start = re.search(r"start at\s+(-?\d+)", text)
        if not start:
            return None
        value = int(start.group(1))
        operations = re.findall(r"(add|subtract|multiply by|divide by|mod)\s+(-?\d+)|\b(square)\b", text)
        if not operations:
            return None
        self.observe_trace("math")
        for op, raw_number, square in operations:
            if square:
                value *= value
                continue
            number = int(raw_number)
            if op == "add":
                value += number
            elif op == "subtract":
                value -= number
            elif op == "multiply by":
                value *= number
            elif op == "divide by":
                if number == 0 or value % number != 0:
                    return None
                value //= number
            elif op == "mod":
                value %= number
        return int(value)

    def execute_stack(self, prompt: str) -> Optional[int]:
        match = re.search(r"stack program:\s*(.*?)\.", str(prompt), re.IGNORECASE)
        if not match:
            return None
        stack: List[int] = []
        self.observe_trace("stack")
        for raw in match.group(1).split(";"):
            op = raw.strip().upper()
            if not op:
                continue
            if op.startswith("PUSH "):
                stack.append(int(op.split()[1]))
            elif op == "ADD" and len(stack) >= 2:
                b, a = stack.pop(), stack.pop()
                stack.append(a + b)
            elif op == "SUB" and len(stack) >= 2:
                b, a = stack.pop(), stack.pop()
                stack.append(a - b)
            elif op == "MUL" and len(stack) >= 2:
                b, a = stack.pop(), stack.pop()
                stack.append(a * b)
            elif op == "DIV" and len(stack) >= 2:
                b, a = stack.pop(), stack.pop()
                if b == 0 or a % b != 0:
                    return None
                stack.append(a // b)
            elif op == "DUP" and stack:
                stack.append(stack[-1])
            elif op == "SWAP" and len(stack) >= 2:
                stack[-1], stack[-2] = stack[-2], stack[-1]
            else:
                return None
        return stack[-1] if stack else None
