"""Download TinyStories subset for Phase 4 benchmark.

Downloads a subset of TinyStories dataset (100k tokens) from HuggingFace
and saves it as a simple tokenized file for benchmarking.
"""
import sys
import os
from huggingface_hub import hf_hub_download
import json
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


def download_tinystories_subset(num_tokens: int = 100000, output_file: str = "tinystories_subset.txt"):
    """Download TinyStories subset and save as text file."""
    print(f"Downloading TinyStories subset ({num_tokens} tokens)...")
    
    # Download TinyStories from HuggingFace
    try:
        # Download the TinyStories dataset (small subset)
        file_path = hf_hub_download(
            repo_id="roneneldan/TinyStories",
            filename="TinyStoriesV2-GPT4-valid.txt",
            repo_type="dataset"
        )
        
        print(f"Downloaded to: {file_path}")
        
        # Read and sample tokens
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"Total characters in dataset: {len(text):,}")
        
        # Sample subset
        if len(text) > num_tokens:
            # Take first num_tokens characters for simplicity
            subset = text[:num_tokens]
        else:
            subset = text
        
        # Save subset
        output_path = os.path.join(os.path.dirname(__file__), output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(subset)
        
        print(f"Saved subset to: {output_path}")
        print(f"Subset size: {len(subset):,} characters")
        
        return output_path
        
    except Exception as e:
        print(f"Error downloading TinyStories: {e}")
        print("Falling back to synthetic data...")
        return None


def create_synthetic_tinystories(num_tokens: int = 100000, output_file: str = "tinystories_subset.txt"):
    """Create synthetic TinyStories-like data as fallback."""
    print(f"Creating synthetic TinyStories-like data ({num_tokens} tokens)...")
    
    # Simple synthetic stories
    stories = [
        "Once upon a time there was a little girl named Lily. She loved to play in the garden. One day she found a magic flower. The flower could talk! It told her stories about faraway lands. Lily was amazed. She visited the flower every day. They became best friends. ",
        "Tom was a brave knight. He protected the kingdom from dragons. One day a big dragon attacked. Tom fought bravely. He defeated the dragon. The king was proud. Tom became a hero. Everyone celebrated. ",
        "The cat sat on the mat. The dog ran around. The bird flew in the sky. The fish swam in the water. They were all friends. They played together every day. ",
        "In a small village lived a wise old man. He knew many secrets. People came from far away to ask him questions. He always gave good advice. The village prospered. ",
        "The moon was bright tonight. The stars were shining. The night was peaceful. A little boy looked out his window. He saw a shooting star. He made a wish. His wish was to be an astronaut. ",
    ]
    
    # Repeat stories to reach desired length
    text = ""
    while len(text) < num_tokens:
        text += random.choice(stories)
    
    text = text[:num_tokens]
    
    # Save
    output_path = os.path.join(os.path.dirname(__file__), output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"Saved synthetic data to: {output_path}")
    print(f"Data size: {len(text):,} characters")
    
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=100000)
    parser.add_argument("--output", type=str, default="tinystories_subset.txt")
    args = parser.parse_args()
    
    # Try to download real data
    result = download_tinystories_subset(args.num_tokens, args.output)
    
    # Fallback to synthetic if download fails
    if result is None:
        result = create_synthetic_tinystories(args.num_tokens, args.output)
    
    print("\nDone!")
