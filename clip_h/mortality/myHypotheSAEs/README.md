# HypotheSAEs, modified for CLIP-H

`hypothesaes/` is a modified copy of [HypotheSAEs](https://github.com/rmovva/HypotheSAEs)
by Rajiv Movva, Kenny Peng, Nikhil Garg, Jon Kleinberg and Emma Pierson, distributed under the
Apache License 2.0 (see `LICENSE` in this folder). It is based on upstream commit `05a5ef8`
(11 June 2025). The CLIP-H runners import it from this folder; it is not installed as a package.

## Changes made for CLIP-H

Modified files: `annotate.py`, `embedding.py`, `evaluation.py`, `interpret_neurons.py`,
`llm_api.py`, `quickstart.py`, `sae.py`, `select_neurons.py` and the three prompt files in
`prompts/`. `__init__.py` and `utils.py` are unchanged.

- **Model providers** (`llm_api.py`): `gpt-5` is pinned to `gpt-5-2025-08-07`, the snapshot the other
  CLIP-H tasks use. Requests are routed by model name to OpenAI, Anthropic,
  DeepSeek, Gemini or an OpenAI-compatible local server. DeepSeek thinking mode is set
  explicitly, and temperature is not sent to reasoning models. Transient network and server
  errors are retried with backoff, and a client-side token-rate limit spreads parallel requests.
- **Annotation** (`annotate.py`): the cache is keyed by model, concept and the SHA-256 of the full
  text, and is saved in chunks. Annotations that fail in the parallel pass are retried in
  sequential rounds; those that still fail are recorded as absent, counted and not cached. Large
  annotation jobs can optionally go through the OpenAI Batch API.
- **Interpretation** (`interpret_neurons.py`, `prompts/interpret-neuron-binary.txt`): the
  interpreter prompt takes a naming policy (`'faithful'`, which names a neuron at the granularity
  of its firing, or `'broad'`) and task-specific instructions; sampling of examples is seeded.
- **Neuron selection** (`select_neurons.py`): an additional `'diverse'` method selects neurons by
  maximal marginal relevance between outcome relevance and redundancy; selection is seeded, and
  the lasso method accepts multi-class targets.
- **SAE training** (`sae.py`, `quickstart.py`): early stopping on a validation set restores the
  weights with the best validation loss and accepts a minimum improvement `min_delta`.
- **Embeddings** (`embedding.py`): unreadable cache chunks are skipped and re-embedded.
- **Evaluation** (`evaluation.py`): accuracy, F1, precision and recall at a 0.5 threshold are
  added to the reported metrics.
- **Prompts** (`prompts/annotate.txt`, `prompts/surface-similarity.txt`): reworded for strict,
  one-word answers.

## Citation

```bibtex
@misc{movva_sparse_2025,
      title={Sparse Autoencoders for Hypothesis Generation},
      author={Rajiv Movva and Kenny Peng and Nikhil Garg and Jon Kleinberg and Emma Pierson},
      year={2025},
      eprint={2502.04382},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.04382},
}
```
