# Portions adapted from HypotheSAEs v0.2.0 (https://github.com/rmovva/HypotheSAEs),
# Apache License 2.0. See THIRD_PARTY_NOTICES.md at the repository root.
"""CLIP-H's own SAE code, on top of the unchanged HypotheSAEs files in ``hypothesaes/``.

    train_sae               train or load the SAE (HypotheSAEs quickstart.train_sae, plus a seed)
    load_sae_pickle         load an SAE saved whole with pickle, onto the CPU
    sae_activations         activations of every neuron on every profile
    select_neurons          pick neurons: 'diverse' (ours) or any original method
    select_neurons_diverse  predictive and mutually de-correlated neurons (MMR)
    split_weak_neurons      set aside neurons that fire on too few profiles
"""

import os
import pickle
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from hypothesaes.sae import SparseAutoencoder, load_model, get_multiple_sae_activations, get_sae_checkpoint_name
from hypothesaes.select_neurons import select_neurons as _select_neurons_original


def train_sae(
    embeddings: Union[List, np.ndarray],
    M: int,
    K: int,
    *,
    matryoshka_prefix_lengths: Optional[List[int]] = None,
    checkpoint_dir: Optional[str] = None,
    overwrite_checkpoint: bool = False,
    val_embeddings: Optional[Union[List, np.ndarray]] = None,
    aux_k: Optional[int] = None,
    multi_k: Optional[int] = None,
    dead_neuron_threshold_steps: int = 256,
    batch_size: int = 512,
    learning_rate: float = 5e-4,
    n_epochs: int = 100,
    aux_coef: float = 1/32,
    multi_coef: float = 0.0,
    patience: int = 3,
    clip_grad: float = 1.0,
    show_progress: bool = True,
    seed: Optional[int] = None,
) -> SparseAutoencoder:
    """Train a Sparse Autoencoder or load an existing one.
    
    Args:
        embeddings: Pre-computed embeddings for training (list or numpy array)
        M: Number of neurons in SAE
        K: Number of top-activating neurons to keep per forward pass
        matryoshka_prefix_lengths: List of prefix lengths for Matryoshka loss (None for vanilla SAE)
        checkpoint_dir: Optional directory for storing/loading SAE checkpoints
        overwrite_checkpoint: Whether to overwrite existing checkpoints
        val_embeddings: Optional validation embeddings for early stopping during SAE training
        aux_k: Number of neurons to consider for dead neuron revival
        multi_k: Number of neurons for secondary reconstruction
        dead_neuron_threshold_steps: Number of non-firing steps after which a neuron is considered dead
        batch_size: Batch size for training
        learning_rate: Learning rate for training
        n_epochs: Maximum number of training epochs
        aux_coef: Coefficient for auxiliary loss
        multi_coef: Coefficient for multi-k loss
        patience: Early stopping patience
        clip_grad: Gradient clipping value
        show_progress: Whether to show training progress bar
        seed: Seed for the initial weights and the batch order (None leaves training unseeded)
        
    Returns:
        Trained SparseAutoencoder model
    """
    embeddings = np.array(embeddings)
    input_dim = embeddings.shape[1]
    
    X = torch.tensor(embeddings, dtype=torch.float)
    X_val = torch.tensor(val_embeddings, dtype=torch.float) if val_embeddings is not None else None
    
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_name = get_sae_checkpoint_name(M, K, matryoshka_prefix_lengths)
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        if os.path.exists(checkpoint_path) and not overwrite_checkpoint:
            return load_model(checkpoint_path)
    
    if seed is not None:
        torch.manual_seed(seed)

    sae = SparseAutoencoder(
        input_dim=input_dim,
        m_total_neurons=M,
        k_active_neurons=K,
        aux_k=aux_k,
        multi_k=multi_k,
        dead_neuron_threshold_steps=dead_neuron_threshold_steps,
        prefix_lengths=matryoshka_prefix_lengths,
    )
    
    sae.fit(
        X_train=X,
        X_val=X_val,
        save_dir=checkpoint_dir,
        batch_size=batch_size,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        aux_coef=aux_coef,
        multi_coef=multi_coef,
        patience=patience,
        clip_grad=clip_grad,
        show_progress=show_progress,
    )

    return sae


def load_sae_pickle(path) -> SparseAutoencoder:
    """Load an SAE that was saved whole with pickle (the runner's sae.pkl), onto the CPU.

    Works for a model that was saved on a GPU machine.
    """
    original_load = torch.load
    torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, "map_location": "cpu"})
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    finally:
        torch.load = original_load


def sae_activations(sae, embeddings) -> Tuple[np.ndarray, list]:
    """Activations of every SAE neuron on every profile.

    Returns:
        activations: array of shape (n_profiles, n_neurons)
        neuron_source_sae_info: for each neuron, the (M, K) of the SAE it comes from
    """
    if not isinstance(embeddings, torch.Tensor):
        X = torch.tensor(embeddings, dtype=torch.float)
    else:
        X = embeddings
    return get_multiple_sae_activations(sae, X, return_neuron_source_info=True)


def select_neurons(
    activations: np.ndarray,
    target: np.ndarray,
    n_select: int,
    method: str = "diverse",
    classification: bool = False,
    mmr_lambda: float = 0.7,
    **kwargs
) -> Tuple[List[int], List[float]]:
    """Select neurons. ``method='diverse'`` is CLIP-H's; 'lasso', 'correlation',
    'separation_score' and 'custom' are passed to the original HypotheSAEs function.

    Returns the indices of the selected neurons and their scores.
    """
    if method == "diverse":
        return select_neurons_diverse(
            activations=activations,
            target=target,
            n_select=n_select,
            mmr_lambda=mmr_lambda,
            classification=classification,
            **kwargs
        )
    return _select_neurons_original(
        activations=activations, target=target, n_select=n_select, method=method,
        classification=classification, **kwargs)


def select_neurons_diverse(
    activations: np.ndarray,
    target: np.ndarray,
    n_select: int,
    mmr_lambda: float = 0.7,
    candidate_multiple: int = 10,
    classification: bool = True,
    verbose: bool = False,
    **kwargs,
) -> Tuple[List[int], List[float]]:
    """Select predictive AND mutually de-correlated neurons via Maximal Marginal
    Relevance (MMR) -- the 'diverse' method.

    Relevance = |Pearson corr(neuron activation, target)|. Starting from the most
    relevant neuron, greedily add the neuron that maximizes
        (1 - mmr_lambda) * relevance - mmr_lambda * max|corr(candidate, selected)|
    so ``mmr_lambda`` is the DIVERSITY weight: higher -> more diverse / less
    redundant. Fully vectorized (no iterative model refitting), so it is far
    faster than the lasso path while still avoiding redundant near-duplicate
    neurons.
    """
    X = np.asarray(activations, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    n, M = X.shape
    n_select = int(min(n_select, M))

    # Relevance: |correlation| of each neuron with the target.
    Xc = X - X.mean(axis=0)
    x_norm = np.sqrt((Xc ** 2).sum(axis=0))
    yc = y - y.mean()
    y_norm = np.sqrt((yc ** 2).sum())
    rel = np.zeros(M)
    valid = (x_norm > 0) & (y_norm > 0)
    rel[valid] = np.abs((Xc[:, valid] * yc[:, None]).sum(axis=0) / (x_norm[valid] * y_norm))

    # Restrict to a candidate pool of the most relevant live neurons (efficiency).
    pool_size = int(min(M, max(n_select, candidate_multiple * n_select)))
    cand = np.argsort(rel)[::-1][:pool_size]
    cand = cand[rel[cand] > 0]
    if len(cand) <= n_select:
        idx = cand[:n_select]
        return [int(i) for i in idx], [float(rel[i]) for i in idx]

    # |correlation| among candidate neurons.
    Xcand = X[:, cand]
    Xz = Xcand - Xcand.mean(axis=0)
    norms = np.sqrt((Xz ** 2).sum(axis=0)); norms[norms == 0] = 1.0
    Xn = Xz / norms
    sim = np.abs(Xn.T @ Xn)
    np.fill_diagonal(sim, 0.0)

    relc = rel[cand]
    relc_n = (relc - relc.min()) / (relc.max() - relc.min() + 1e-12)

    selected = [int(np.argmax(relc_n))]
    while len(selected) < n_select:
        redundancy = sim[:, selected].max(axis=1)
        mmr = (1.0 - mmr_lambda) * relc_n - mmr_lambda * redundancy
        mmr[selected] = -np.inf
        selected.append(int(np.argmax(mmr)))
        if verbose and len(selected) % 20 == 0:
            print(f"  MMR diverse: selected {len(selected)}/{n_select}")

    chosen = cand[np.array(selected)]
    if verbose:
        print(f"Diverse (MMR, lambda={mmr_lambda}) selected {len(chosen)} neurons from a pool of {len(cand)}.")
    return [int(i) for i in chosen], [float(rel[i]) for i in chosen]


def split_weak_neurons(
    activations: np.ndarray,
    neurons: List[int],
    scores: List[float],
    min_profiles: int,
) -> Tuple[List[int], List[float], List[Tuple[int, int]]]:
    """Split selected neurons by how many profiles they fire on.

    A neuron that fires on fewer than ``min_profiles`` profiles has too few examples to be
    interpreted on its own; those are returned separately so they can be pooled into groups.

    Returns:
        strong_neurons: neurons firing on at least ``min_profiles`` profiles, in selection order
        strong_scores: their selection scores
        weak_neurons: (neuron index, number of profiles it fires on) for the rest
    """
    acts = np.asarray(activations)
    counts = (acts[:, list(neurons)] > 0).sum(axis=0)
    strong_neurons, strong_scores, weak_neurons = [], [], []
    for neuron, score, count in zip(neurons, scores, counts):
        if int(count) >= min_profiles:
            strong_neurons.append(int(neuron))
            strong_scores.append(score)
        else:
            weak_neurons.append((int(neuron), int(count)))
    return strong_neurons, strong_scores, weak_neurons
