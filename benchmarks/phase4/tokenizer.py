"""Simple tokenizer for TinyStories benchmark."""
import re
from typing import List, Dict
from collections import Counter


class SimpleTokenizer:
    """Simple character-level tokenizer for benchmarking."""
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_built = False
    
    def build_vocab(self, text: str):
        """Build vocabulary from text."""
        # Count character frequencies
        char_counts = Counter(text)
        
        # Take most common characters
        most_common = char_counts.most_common(self.vocab_size - 2)
        
        # Build vocabulary
        self.char_to_id = {'<pad>': 0, '<unk>': 1}
        self.id_to_char = {0: '<pad>', 1: '<unk>'}
        
        for idx, (char, _) in enumerate(most_common, start=2):
            self.char_to_id[char] = idx
            self.id_to_char[idx] = char
        
        self.vocab_built = True
        print(f"Built vocabulary: {len(self.char_to_id)} characters")
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if not self.vocab_built:
            self.build_vocab(text)
        
        token_ids = []
        for char in text:
            token_id = self.char_to_id.get(char, 1)  # <unk> if not in vocab
            token_ids.append(token_id)
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        chars = []
        for token_id in token_ids:
            char = self.id_to_char.get(token_id, '<unk>')
            chars.append(char)
        return ''.join(chars)


def load_tinystories_tokens(file_path: str, vocab_size: int = 1000) -> List[int]:
    """Load TinyStories and tokenize."""
    print(f"Loading TinyStories from {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Loaded {len(text):,} characters")
    
    # Tokenize
    tokenizer = SimpleTokenizer(vocab_size=vocab_size)
    token_ids = tokenizer.encode(text)
    
    print(f"Tokenized to {len(token_ids):,} tokens")
    print(f"Vocab size: {len(tokenizer.char_to_id)}")
    
    return token_ids
