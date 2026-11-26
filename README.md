# CYTAP

**CYTAP** (CYTokine response prediction by Adapting Pretrained embeddings) scores, for every cell of a query
scRNA-seq dataset, how closely its change relative to control cells resembles the response to each of 90
cytokines. A classifier trained on the Parse 10M PBMC cytokine screen takes State SE-600M cell embeddings
centred on matched control cells and returns a softmax over the 90 cytokines; conditions are compared with
the logits.

## Installation

```bash
pip install git+https://github.com/CB-postech/CYTAP.git
```

Python ≥ 3.9, PyTorch ≥ 2.0; a GPU is optional. The model weights (41 MB, GitHub release) are downloaded on
first use of `cytap.load_model()`.

## Usage

Input: an AnnData with State SE-600M embeddings of log-normalized counts in `obsm['X_state']`
([State](https://github.com/ArcInstitute/state): `state emb transform --model-folder SE-600M ...`) and
`obs` columns for condition, donor and cell type.

```python
import cytap

model = cytap.load_model()
out = cytap.predict(adata, model, design="paired",            # or "unpaired" (reference="pooled" / "counterfactual")
                    control_key="condition", control_label="pre", donor_key="donor", celltype_key="celltype")
res = cytap.stats.compare_conditions(out, "condition", "pre", "post", donor_key="donor")
cytap.plotting.lollipop(res)
```

`out` is a new AnnData with `obsm['pred_probs']`, `obsm['pred_logits']` and `uns['pred_classes']`.
Step-by-step example: [`tutorials/cytap_tutorial.ipynb`](tutorials/cytap_tutorial.ipynb). Command line:
`cytap predict -h`.

| module | content |
|---|---|
| `cytap.predict` | per-cell cytokine probabilities and logits |
| `cytap.stats` | condition comparison, severity trend, pseudobulk, within-donor change |
| `cytap.plotting` | lollipop, response, delta and embedding plots |
| `cytap.counterfactual` | CellOT counterfactual controls for unpaired data |

Tests: `pip install -e ".[dev]" && pytest tests/` in a clone. Analysis code of the paper:
[CYTAP-paper](https://github.com/CB-postech/CYTAP-paper).

## License

Code: MIT (`LICENSE`); `src/cytap/_cellot.py` is adapted from [CellOT](https://github.com/bunnech/cellot)
(BSD 3-Clause, `LICENSE-CELLOT`). Model weights and predictions: CC BY-NC 4.0 and the Arc Institute State
Model Non-Commercial License, as they derive from State embeddings (`MODEL_LICENSE.md`).

## Citation

Jeong, Y. L. & Kim, J. K. *Task-aligned foundation model representations enable single-cell cytokine response 
inference* (2026; journal and DOI to be added).

Adduri, A. K. et al. *Predicting cellular responses to perturbation across diverse contexts with State.*
Cell 189, 5914–5931.e20 (2026). 
For counterfactual controls also cite Bunne, Stark et al., *Learning single-cell perturbation responses using 
neural optimal transport*, Nat. Methods 20, 1759–1768 (2023).
