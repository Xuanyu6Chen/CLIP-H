# Portions adapted from HypotheSAEs v0.2.0 (https://github.com/rmovva/HypotheSAEs),
# Apache License 2.0. See THIRD_PARTY_NOTICES.md at the repository root.
"""Embed profile texts with a local SentenceTransformer model, with a chunked cache on disk.

    get_local_embeddings   {text: vector} for a list of texts; texts already in the cache are not embedded again
    check_embeddings       stop when a matrix cannot be the embeddings of the given number of profiles

The cache is the folder ``<EMB_CACHE_DIR>/<cache_name>/`` of ``chunk_NNN.npy`` files, each a list
of (text, vector) pairs. ``EMB_CACHE_DIR`` is read from the environment when this module is
imported (the runner notebook sets it in Setup).
"""

import numpy as np
from typing import List, Optional, Dict
from tqdm.auto import tqdm
import os
import time
from pathlib import Path
import glob
import torch

# Use environment variable for cache dir if set, otherwise use default
CACHE_DIR = os.getenv('EMB_CACHE_DIR') or os.path.join(Path(__file__).parent.parent.parent, 'emb_cache')

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def filter_invalid_texts(texts: List[str]) -> List[str]:
    """Filter out None values and empty strings from a list of texts.
    
    Args:
        texts: List of text strings, potentially containing None or empty strings
        
    Returns:
        Filtered list with None values and empty strings removed
    """
    original_count = len(texts)
    filtered_texts = [text for text in texts if text is not None and len(str(text).strip()) > 0]
    filtered_count = original_count - len(filtered_texts)
    
    if filtered_count > 0:
        print(f"Warning: ignoring {filtered_count} items which are None or empty strings")
    
    return filtered_texts


def load_embedding_cache(cache_name: str) -> dict:
    """Load cached embeddings from chunked files."""
    if not cache_name:
        return {}
    
    cache_dir = f"{CACHE_DIR}/{cache_name}"
    if not os.path.exists(cache_dir):
        return {}
    
    text2embedding = {}
    chunk_files = sorted(glob.glob(f"{cache_dir}/chunk_*.npy"))
    
    start_time = time.time()
    for chunk_file in tqdm(chunk_files, desc="Loading embedding chunks"):
        # Each chunk file contains a list of (text, embedding) tuples
        chunk_data = np.load(chunk_file, allow_pickle=True)
        for text, emb in chunk_data:
            text2embedding[text] = emb
            
    load_time = time.time() - start_time
    print(f"Loaded {len(text2embedding)} embeddings in {load_time:.1f}s")
            
    return text2embedding


def _get_next_chunk_index(cache_name: str) -> int:
    """Determine the next available chunk index for a cache."""
    if not cache_name:
        return 0
        
    cache_dir = f"{CACHE_DIR}/{cache_name}"
    if not os.path.exists(cache_dir):
        return 0
        
    chunk_files = glob.glob(f"{cache_dir}/chunk_*.npy")
    if not chunk_files:
        return 0
        
    indices = [int(os.path.basename(f).split("_")[1].split(".")[0]) for f in chunk_files]
    return max(indices) + 1


def _save_embedding_chunk(cache_name: str, chunk_embeddings: dict, chunk_idx: int) -> int:
    """Save a chunk of embeddings to disk."""
    if not cache_name or not chunk_embeddings:
        return chunk_idx
        
    cache_dir = f"{CACHE_DIR}/{cache_name}"
    os.makedirs(cache_dir, exist_ok=True)
    
    chunk_path = f"{cache_dir}/chunk_{chunk_idx:03d}.npy"
    chunk_items = list(chunk_embeddings.items())
    np.save(chunk_path, np.array(chunk_items, dtype=object))
    print(f"Saved {len(chunk_items)} embeddings to {chunk_path}")
    
    return chunk_idx + 1


def check_embeddings(emb, n_expected: int, name: str = "embeddings") -> None:
    """Raise ValueError unless ``emb`` has one row per profile (``n_expected`` rows) and only finite values.

    A saved file with another number of rows was made for another set of profiles.
    """
    emb = np.asarray(emb)
    if emb.ndim != 2 or len(emb) != n_expected:
        raise ValueError(f"{name}: shape {emb.shape}, expected {n_expected} rows; "
                         "the file was made for another set of profiles")
    if not np.isfinite(emb).all():
        raise ValueError(f"{name}: {int((~np.isfinite(emb)).sum())} values are NaN or infinite")


def get_local_embeddings(
    texts: List[str],
    model: str = "nomic-ai/modernbert-embed-base",
    batch_size: int = 128,
    show_progress: bool = True,
    cache_name: Optional[str] = None,
    chunk_size: int = 50000,
) -> Dict[str, np.ndarray]:
    """Embed ``texts`` with a local SentenceTransformer model. Returns ``{text: vector}``.

    The key is always the text as given, whatever prefix the model needs. With ``cache_name``
    the vectors are saved in chunks of ``chunk_size`` texts and read back on the next call.
    """
    from sentence_transformers import SentenceTransformer

    # Filter out None values and empty strings
    texts = filter_invalid_texts(texts)
    
    # Setup cache
    text2embedding = load_embedding_cache(cache_name)
    texts_to_embed = [text for text in texts if text not in text2embedding]
    
    if not texts_to_embed:
        return text2embedding
    
    # Load model
    transformer_model = SentenceTransformer(model, device=device)
    print(f"Loaded model {model} to {device}")
    
    # Process in chunks
    next_chunk_idx = _get_next_chunk_index(cache_name)
    
    # Create chunk ranges
    chunk_ranges = [(i, min(i+chunk_size, len(texts_to_embed))) 
                   for i in range(0, len(texts_to_embed), chunk_size)]
    
    # Outer progress bar for chunks
    chunk_iterator = chunk_ranges
    if show_progress:
        chunk_iterator = tqdm(chunk_iterator, desc="Processing chunks", total=len(chunk_ranges))
    
    for chunk_start, chunk_end in chunk_iterator:
        chunk_texts = texts_to_embed[chunk_start:chunk_end]
        chunk_embeddings = {}
        
        # Process chunk in batches
        batch_iterator = range(0, len(chunk_texts), batch_size)
        if show_progress:
            batch_iterator = tqdm(batch_iterator, desc=f"Chunk {next_chunk_idx}")
            
        for i in batch_iterator:
            batch = chunk_texts[i:i+batch_size]
            if "nomic-ai" in model:      # these two model families expect an instruction with each text
                inputs = ["search_document: " + text for text in batch]
            elif "instructor" in model:
                inputs = [["Represent the text for classification: ", text] for text in batch]
            else:
                inputs = batch
            batch_embs = transformer_model.encode(inputs, batch_size=batch_size)

            for text, embedding in zip(batch, batch_embs):
                chunk_embeddings[text] = embedding
                text2embedding[text] = embedding
        
        # Save completed chunk
        next_chunk_idx = _save_embedding_chunk(cache_name, chunk_embeddings, next_chunk_idx)
    
    return text2embedding