"""Command line: ``cytap predict``."""
import argparse

import anndata as ad

from .model import load_model
from .predict import predict


def main(argv=None):
    p = argparse.ArgumentParser(prog="cytap")
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("predict", help="predict cytokine responses for a query .h5ad")
    q.add_argument("input")
    q.add_argument("output")
    q.add_argument("--model", default="cytap_state_comp_v1",
                   help="checkpoint file or pretrained model name (default: cytap_state_comp_v1, downloaded on first use)")
    q.add_argument("--control-key", required=True)
    q.add_argument("--control-label", required=True)
    q.add_argument("--donor-key")
    q.add_argument("--celltype-key")
    q.add_argument("--design", required=True, choices=["paired", "unpaired"],
                   help="paired: each donor has own control cells; unpaired: controls from other donors")
    q.add_argument("--reference", choices=["pooled", "counterfactual"],
                   help="unpaired control reference (default: pooled)")
    q.add_argument("--counterfactual", nargs="+",
                   help="with --reference counterfactual: 'cellot' (default; train CellOT maps), a saved CellOT .pt "
                        "file or pretrained map name (combat_cellot), or predicted control .h5ad file(s)")
    q.add_argument("--cellot-save", help="with --counterfactual cellot: save the fitted maps here")
    q.add_argument("--cellot-iters", type=int, help="with --counterfactual cellot: training iterations")
    q.add_argument("--cf-embed-key")
    q.add_argument("--exclude-celltypes", nargs="+", metavar="CELLTYPE", help="cell types to leave out")
    q.add_argument("--embed-key")
    q.add_argument("--batch-size", type=int, default=4096)
    q.add_argument("--device")
    a = p.parse_args(argv)

    model = load_model(a.model, device=a.device)
    adata = ad.read_h5ad(a.input)
    cf, cellot_kwargs = None, None
    if a.reference == "counterfactual" and not a.counterfactual:
        a.counterfactual = ["cellot"]
    if a.counterfactual == ["cellot"]:
        cf = "cellot"
        cellot_kwargs = {k: v for k, v in [("save", a.cellot_save), ("n_iters", a.cellot_iters)] if v is not None}
        if a.device:
            cellot_kwargs["device"] = a.device
    elif a.counterfactual and len(a.counterfactual) == 1 and (a.counterfactual[0].endswith(".pt")
                                                              or a.counterfactual[0] == "combat_cellot"):
        from .counterfactual import CellOT
        cf = CellOT.load(a.counterfactual[0], device=a.device)
    elif a.counterfactual:
        cf = [ad.read_h5ad(f) for f in a.counterfactual]
    out = predict(
        adata, model,
        control_key=a.control_key, control_label=a.control_label,
        design=a.design, reference=a.reference, donor_key=a.donor_key, celltype_key=a.celltype_key,
        counterfactual=cf, cf_embed_key=a.cf_embed_key, cellot_kwargs=cellot_kwargs, exclude_celltypes=a.exclude_celltypes,
        embed_key=a.embed_key,
        batch_size=a.batch_size,
    )
    print(f"design={out.uns['cytap']['design']} reference={out.uns['cytap']['reference']} | {out.n_obs}/{adata.n_obs} cells | "
          f"centroids {out.uns['cytap']['centroid_source']}")
    out.write_h5ad(a.output)


if __name__ == "__main__":
    main()
