# Modified for CLIP-H from HypotheSAEs (https://github.com/rmovva/HypotheSAEs),
# Apache License 2.0. See ../README.md for the changes.
"""Methods for selecting relevant neurons based on target variables."""

import time
import numpy as np
from typing import List, Optional, Callable, Tuple
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

def select_neurons_lasso(
    activations: np.ndarray,
    target: np.ndarray,
    n_select: int,
    classification: bool = False,
    alpha: Optional[float] = None,
    max_iter: int = 1000,
    verbose: bool = False,
    random_state: int = 0
) -> Tuple[List[int], List[float]]:
    """
    Select neurons using an L1-regularized linear model (LASSO), which produces sparse coefficient vectors.
    Returns (indices, coefficients) tuple.

    Args:
        activations: Neuron activation matrix (n_samples, n_neurons)
        target: Target variable (n_samples,)
        n_select: Number of neurons to select
        classification: Whether this is a classification task
        alpha: LASSO alpha (if None, searches for alpha yielding n_select features)
        max_iter: Maximum iterations for LASSO
        verbose: Whether to print progress of alpha search
        random_state: Seed for the liblinear/saga solver so selection is reproducible

    Returns:
        Indices of selected neurons and corresponding coefficients
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(activations)

    if alpha is not None:
        if verbose:
            print(f"Using provided alpha: {alpha:.2e}")

        start_time = time.time()
        if classification:
            model = LogisticRegression(
                penalty='l1',
                solver='liblinear',
                C=1/alpha,
                max_iter=max_iter,
                random_state=random_state
            )
        else:
            model = Lasso(alpha=alpha, max_iter=max_iter)
        
        model.fit(X_scaled, target)
        coef = model.coef_.flatten()
        if verbose:
            print(f"Fitting took {time.time() - start_time:.2f}s")
        
    else:
        # Search for alpha that gives desired number of features
        alpha_low, alpha_high = 1e-6, 1e4
        
        if verbose:
            print(f"{'LASSO iteration':>8} {'L1 Alpha':>10} {'# Features':>10} {'Time (s)':>10}")
            print("-" * 40)
        
        total_start_time = time.time()
        for iteration in range(20):  # Maximum binary search iterations
            iter_start_time = time.time()
            alpha = np.sqrt(alpha_low * alpha_high)
            
            if classification:
                solver = 'liblinear' if max(target) <= 1 else 'saga'
                model = LogisticRegression(
                    penalty='l1',
                    solver=solver,
                    C=1/alpha,
                    max_iter=10*max_iter,
                    random_state=random_state
                )
            else:
                model = Lasso(alpha=alpha, max_iter=max_iter)
            
            model.fit(X_scaled, target)
            if classification and model.coef_.ndim == 2:
                # Multi-class: one row of coefficients per class (n_classes, n_neurons).
                # Aggregate them into one importance vector (the L1 norm across classes).
                # A binary LogisticRegression also has a 2-D coef_ of shape (1, n_neurons),
                # so for binary targets this yields absolute coefficient values.
                coef = np.sum(np.abs(model.coef_), axis=0)
            else:
                coef = model.coef_.flatten()
            n_nonzero = np.sum(coef != 0)
            iter_time = time.time() - iter_start_time
            
            if verbose:
                print(f"{iteration:8d} {alpha:10.2e} {n_nonzero:10d} {iter_time:10.2f}")
            
            if n_nonzero == n_select:
                break
            elif n_nonzero < n_select:
                alpha_high = alpha
            else:
                alpha_low = alpha
        
        total_time = time.time() - total_start_time
        if verbose and n_nonzero == n_select:
            print(f"\nFound alpha={alpha:.2e} yielding exactly {n_select} features")
            print(f"Total search time: {total_time:.2f}s")
        if n_nonzero != n_select:
            print(f"Warning: Search ended with {n_nonzero} features (target: {n_select})")
    
    # Get top neurons by absolute coefficient value
    sorted_indices = np.argsort(-np.abs(coef))[:n_select]
    selected_coefs = coef[sorted_indices]  # Return coefficients (signed unless aggregated above)
    
    return sorted_indices.tolist(), selected_coefs.tolist()

def select_neurons_diverse(
    activations: np.ndarray,
    target: np.ndarray,
    n_select: int,
    classification: bool = False,
    candidate_mult: int = 3,
    mmr_lambda: float = 0.5,
    seed: int = 0,
    max_iter: int = 1000,
    verbose: bool = False,
    **kwargs
) -> Tuple[List[int], List[float]]:
    """
    Select predictive *and* mutually de-correlated neurons via Maximal Marginal
    Relevance (MMR). This counteracts the failure mode where LASSO picks many
    near-duplicate neurons (e.g. several "emergency admission" variants), which
    produces redundant hypotheses and collinear downstream features.

    Procedure:
      1. Build a candidate pool of `candidate_mult * n_select` neurons ranked by
         |coefficient| using the existing LASSO selector.
      2. Greedily pick `n_select` neurons. At each step choose the candidate that
         maximises:
             mmr_lambda * relevance  -  (1 - mmr_lambda) * max_sim_to_selected
         where relevance is the min-max-normalised |coef| and similarity is the
         absolute cosine similarity between (z-scored) neuron activation columns.

    Deterministic given the candidate pool (the only RNG is inside LASSO's alpha
    search, which is itself deterministic for fixed data).

    Returns (indices, coefficients) tuple, ordered by selection.
    """
    n_neurons = activations.shape[1]
    n_candidates = min(candidate_mult * n_select, n_neurons)

    # 1. Candidate pool by |coef| (reuse the LASSO selector).
    cand_idx, cand_coef = select_neurons_lasso(
        activations=activations,
        target=target,
        n_select=n_candidates,
        classification=classification,
        max_iter=max_iter,
        verbose=verbose,
        random_state=seed,
    )
    cand_idx = list(cand_idx)
    coef_by_idx = {i: c for i, c in zip(cand_idx, cand_coef)}

    if n_candidates <= n_select:
        sel = cand_idx[:n_select]
        return sel, [coef_by_idx[i] for i in sel]

    # Relevance = min-max normalised |coef| over the candidate pool.
    abs_coef = np.array([abs(coef_by_idx[i]) for i in cand_idx], dtype=float)
    rng = abs_coef.max() - abs_coef.min()
    relevance = (abs_coef - abs_coef.min()) / rng if rng > 0 else np.ones_like(abs_coef)
    relevance_by_idx = {i: r for i, r in zip(cand_idx, relevance)}

    # Pairwise |cosine| similarity between z-scored activation columns of candidates.
    cols = activations[:, cand_idx].astype(float)
    mu = cols.mean(axis=0, keepdims=True)
    sd = cols.std(axis=0, keepdims=True)
    cols_z = (cols - mu) / np.where(sd > 0, sd, 1.0)
    norms = np.linalg.norm(cols_z, axis=0, keepdims=True)
    cols_unit = cols_z / np.where(norms > 0, norms, 1.0)
    sim = np.abs(cols_unit.T @ cols_unit)  # (n_candidates, n_candidates)
    pos_in_pool = {idx: p for p, idx in enumerate(cand_idx)}

    # 2. Greedy MMR. Seed with the most relevant candidate.
    selected = [max(cand_idx, key=lambda i: relevance_by_idx[i])]
    remaining = [i for i in cand_idx if i not in selected]

    while len(selected) < n_select and remaining:
        best_i, best_score = None, -np.inf
        for i in remaining:
            pi = pos_in_pool[i]
            max_sim = max(sim[pi, pos_in_pool[j]] for j in selected)
            score = mmr_lambda * relevance_by_idx[i] - (1.0 - mmr_lambda) * max_sim
            if score > best_score:
                best_score, best_i = score, i
        selected.append(best_i)
        remaining.remove(best_i)

    return selected, [coef_by_idx[i] for i in selected]

def select_neurons_correlation(
    activations: np.ndarray,
    target: np.ndarray,
    n_select: int,
    **kwargs
) -> Tuple[List[int], List[float]]:
    """
    Select neurons with highest correlation with target.
    Returns (indices, correlations) tuple.
    
    Args:
        activations: Neuron activation matrix (n_samples, n_neurons)
        target: Target variable (n_samples,)
        n_select: Number of neurons to select
        abs_correlation: Whether to use absolute correlation values
    
    Returns:
        Indices of selected neurons and corresponding correlations
    """
    correlations = np.array([
        pearsonr(activations[:, i], target)[0]
        for i in range(activations.shape[1])
    ])
    
    # Sort by absolute correlation but return raw correlations
    sorted_indices = np.argsort(-np.abs(correlations))[:n_select]
    selected_correlations = correlations[sorted_indices]
    
    return sorted_indices.tolist(), selected_correlations.tolist()

def select_neurons_separation_score(
    activations: np.ndarray,
    target: np.ndarray,
    n_select: int,
    n_top_activating: int = 100,
    n_zero_activating: Optional[int] = None,
    seed: Optional[int] = 0,
    **kwargs
) -> Tuple[List[int], List[float]]:
    """
    Select neurons based on separation between top activations and zero activations.
    Returns (indices, scores) tuple.

    Args:
        activations: Neuron activation matrix (n_samples, n_neurons)
        target: Target variable (n_samples,)
        n_select: Number of neurons to select
        n_top: Number of top activations to use for high activation mean
        seed: Seed for the local RNG used when subsampling zero-activation examples

    Returns:
        Indices of selected neurons and corresponding separation scores
    """
    rng = np.random.default_rng(seed)
    scores = []
    for i in range(activations.shape[1]):
        neuron_acts = activations[:, i]
        sorted_indices = np.argsort(-neuron_acts)

        # Get mean target value for top n activations
        top_mean = np.mean(target[sorted_indices[:n_top_activating]])

        # Get mean target value for zero activations
        zero_indices = neuron_acts == 0
        if n_zero_activating is not None:
            zero_idx = np.where(zero_indices)[0]
            rand_zero_idx = rng.choice(zero_idx, size=n_zero_activating, replace=False)
            zero_mean = np.mean(target[rand_zero_idx])
        else:
            zero_mean = np.mean(target[zero_indices])

        scores.append(top_mean - zero_mean)
    
    scores = np.array(scores)
    sorted_indices = np.argsort(-np.abs(scores))[:n_select]
    selected_scores = scores[sorted_indices]
    
    return sorted_indices.tolist(), selected_scores.tolist()

def select_neurons_custom(
    activations: np.ndarray,
    target: np.ndarray,
    n_select: int,
    metric_fn: Callable[[np.ndarray, np.ndarray], float]
) -> Tuple[List[int], List[float]]:
    """
    Select neurons using a custom metric function.
    Returns (indices, scores) tuple.
    
    Args:
        activations: Neuron activation matrix (n_samples, n_neurons)
        target: Target variable (n_samples,)
        n_select: Number of neurons to select
        metric_fn: Function that takes (neuron_activations, target) and returns score
    
    Returns:
        Indices of selected neurons and corresponding scores
    """
    scores = np.array([
        metric_fn(activations[:, i], target)
        for i in range(activations.shape[1])
    ])
    
    sorted_indices = np.argsort(scores)[-n_select:]
    selected_scores = scores[sorted_indices]
    
    return sorted_indices.tolist(), selected_scores.tolist()

def select_neurons(
    activations: np.ndarray,
    target: np.ndarray,
    n_select: int,
    method: str = "lasso",
    classification: bool = False,
    **kwargs
) -> Tuple[List[int], List[float]]:
    """
    Select neurons using specified method.
    Returns (indices, scores) tuple where scores are the raw values
    (coefficients for lasso, correlations for correlation method, etc.)
    
    Args:
        activations: Neuron activation matrix (n_samples, n_neurons)
        target: Target variable (n_samples,)
        n_select: Number of neurons to select
        method: One of "lasso", "diverse", "correlation", "separation_score", or "custom"
        classification: Whether this is a classification task
        **kwargs: Additional arguments passed to specific method
    
    Returns:
        Indices of selected neurons and corresponding scores
    """
    # Multi-class classification targets are not rejected here; select_neurons_lasso
    # aggregates per-class coefficients into one importance vector.

    if method == "lasso":
        return select_neurons_lasso(
            activations=activations,
            target=target,
            n_select=n_select,
            classification=classification,
            **kwargs
        )
    elif method == "diverse":
        return select_neurons_diverse(
            activations=activations,
            target=target,
            n_select=n_select,
            classification=classification,
            **kwargs
        )
    elif method == "correlation":
        return select_neurons_correlation(
            activations=activations,
            target=target,
            n_select=n_select,
            **kwargs
        )
    elif method == "separation_score":
        return select_neurons_separation_score(
            activations=activations,
            target=target,
            n_select=n_select,
            **kwargs
        )
    elif method == "custom":
        if "metric_fn" not in kwargs:
            raise ValueError("Must provide metric_fn for custom method")
        return select_neurons_custom(
            activations=activations,
            target=target,
            n_select=n_select,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown selection method: {method}")