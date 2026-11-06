# CYTAP model weights license

Applies to the CYTAP model weights (e.g. `cytap_state_comp_v1.pt`, ablation and cross-validation models)
and to predictions made with them. The source code of the `cytap` package is MIT-licensed (`LICENSE`).

## 1. CYTAP weights: CC BY-NC 4.0

Copyright (c) 2026 Yujin L. Jeong and Jong Kyoung Kim. The CYTAP model weights are licensed under the Creative Commons
Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0):
https://creativecommons.org/licenses/by-nc/4.0/legalcode

## 2. State model terms also apply

The CYTAP classifier was trained on, and takes as input, cell embeddings produced by the State SE-600M
model of the Arc Institute. These embeddings are Output of the State Model and are governed by the
**Arc Research Institute State Model Non-Commercial License** and the **State Model Acceptable Use Policy**
(https://github.com/ArcInstitute/state, files `MODEL_LICENSE.md` and `MODEL_ACCEPTABLE_USE_POLICY.md`).
Use of the CYTAP weights and of embeddings or predictions derived from State must therefore also comply
with those terms; in particular, use is limited to non-commercial purposes, and commercial use requires
a separate license from the Arc Institute (legal@arcinstitute.org). The State model weights are not
distributed with CYTAP; obtain SE-600M from the Arc Institute under its license.

Model used: State SE-600M (Hugging Face `arcinstitute/SE-600M`; folder with `config.yaml`,
`model.safetensors`, `protein_embeddings.pt`, `se600m_epoch16.ckpt`), applied with
`state emb transform --model-folder SE-600M`.

## 3. Citation

When using or redistributing the CYTAP weights or outputs, cite CYTAP (Jeong, Y. L. & Kim, J. K. *CYTAP: inferring cytokine responses in single cells from adapted foundation model embeddings* (2026; journal and DOI to be added)) and State:

Adduri, A. K. et al. Predicting cellular responses to perturbation across diverse contexts with State.
Cell 189, 5914–5931.e20 (2026). https://doi.org/10.1016/j.cell.2026.07.052

## 4. Disclaimer

The weights are provided "as is", without warranty of any kind. They are research tools and are not
intended for clinical or diagnostic use.
