# Third-party notices

The CLIP-H code in this repository is released under the MIT License (see [`LICENSE`](LICENSE)).
It includes code from the project below, which remains under its own license.

## HypotheSAEs

- Source: <https://github.com/rmovva/HypotheSAEs>
- Authors: Rajiv Movva, Kenny Peng, Nikhil Garg, Jon Kleinberg and Emma Pierson
- License: Apache License 2.0, as stated in the upstream `LICENSE` file. A copy of the license is
  included next to each copy of the code: in `clip_h/readmission`, `clip_h/hosp_los` and
  `clip_h/icu_los`, `code/sae/hypothesaes/LICENSE`; in `clip_h/mortality` and `clip_h/discharge`,
  `myHypotheSAEs/LICENSE`.

| Location | What it contains |
|---|---|
| `clip_h/readmission/code/sae/hypothesaes/` (same in `clip_h/hosp_los`, `clip_h/icu_los`) | `sae.py` and `select_neurons.py` from HypotheSAEs v0.2.0, unmodified |
| `clip_h/readmission/code/` (same in `clip_h/hosp_los`, `clip_h/icu_los`) | `sae/embed_profiles.py`, `train_sae` in `sae/clip_h_sae.py`, and parts of `hypotheses/interpret/interpret_neurons.py`, `common/annotation.py` and `prediction/evaluation.py`, adapted from HypotheSAEs v0.2.0 |
| `direct_llm_hypotheses/code/` | `common/annotation.py` and `prediction/evaluation.py`, copies of the CLIP-H files above (with the same adapted parts) |
| `clip_h/mortality/myHypotheSAEs/hypothesaes/` (same in `clip_h/discharge`) | the HypotheSAEs package as of upstream commit `05a5ef8`, modified for CLIP-H; the changes are listed in `myHypotheSAEs/README.md` |

Modified and adapted files carry a notice at the top. The upstream `sae.py` credits Bussmann et al.
(2025) and O'Neill et al. (2024) for parts of the sparse autoencoder; those credits are kept in the
file.

If you use this code, please also cite HypotheSAEs:

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

## Data and models used but not included

- **MIMIC-IV** is available from PhysioNet (<https://physionet.org/content/mimiciv/>) to
  credentialed users under the PhysioNet Credentialed Health Data License 1.5.0 and its data use
  agreement. No MIMIC-IV data, and nothing derived from it, is included in this repository.
- **Qwen3-Embedding-0.6B** (Apache License 2.0) is downloaded from Hugging Face when the pipeline
  first runs.
- The hosted language models are accessed through the OpenAI, Anthropic and DeepSeek APIs under
  each provider's terms of use.
