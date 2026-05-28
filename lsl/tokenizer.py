"""SimpleWordTokenizer - Lightweight word-level tokenizer for scaling LSL.

Splits text into words, builds a vocabulary based on word frequencies,
and provides clean encode/decode methods.
"""
import re
from collections import Counter


class SimpleWordTokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = int(vocab_size)
        self.word_to_id = {}
        self.id_to_word = {}
        self.special_tokens = ["<PAD>", "<UNK>"]
        
    def build_vocab(self, text):
        """Build vocabulary from a corpus of text based on word frequency."""
        # Normalize and split text into lowercase words, keeping punctuation separate
        words = self._tokenize_raw(text)
        
        # Count word frequencies
        counts = Counter(words)
        
        # Get most common words
        available_slots = self.vocab_size - len(self.special_tokens)
        most_common = counts.most_common(available_slots)
        
        # Build mapping
        self.word_to_id = {}
        for idx, token in enumerate(self.special_tokens):
            self.word_to_id[token] = idx
            
        for idx, (word, _) in enumerate(most_common):
            self.word_to_id[word] = idx + len(self.special_tokens)
            
        # Build reverse mapping
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.vocab_size = len(self.word_to_id)
        
    def _tokenize_raw(self, text):
        """Split text into lowercase words and punctuation tokens."""
        text = text.lower()
        # Find all words and punctuation marks as separate tokens
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        return tokens

    def encode(self, text):
        """Encode string text into a list of token IDs."""
        tokens = self._tokenize_raw(text)
        unk_id = self.word_to_id.get("<UNK>", 1)
        return [self.word_to_id.get(tok, unk_id) for tok in tokens]

    def decode(self, token_ids):
        """Decode a list of token IDs back into string text."""
        words = []
        for tid in token_ids:
            word = self.id_to_word.get(int(tid), "<UNK>")
            # Simple spacing rule: don't space before common punctuation
            if words and word in [".", ",", "!", "?", ";", ":", ")", "]"]:
                words[-1] = words[-1] + word
            elif words and words[-1] in ["(", "["]:
                words[-1] = words[-1] + word
            else:
                words.append(word)
        return " ".join(words)
