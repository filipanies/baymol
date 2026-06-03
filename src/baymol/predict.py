"""
Predict molecular properties for a product library and store them.

This module is the *orchestration* only: it reads product SMILES from the
database, hands batches to a `predictor`, and writes the results into the
`predictions` table — keyed by (product_id, model), resumable by absence.

It deliberately knows nothing about the model. A `predictor` is just:

    Callable[[list[str]], Sequence[Sequence[float]]]

i.e. given a list of SMILES, return one (homo_ev, lumo_ev, gap_ev) triple per
SMILES (same order). The Chemprop-backed predictor (`load_chemprop_predictor`,
which pulls in PyTorch) is the only part that needs the optional `[ml]` extra,
and it imports chemprop/torch lazily — so `predict_properties` and importing
this module stay torch-free and fully testable with a trivial fake predictor.

CLI:
    python -m baymol.predict products.db --model path/to/chemprop_model
"""

import logging
import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path

from baymol.db import init_predictions_table

logger = logging.getLogger(__name__)

Predictor = Callable[[list[str]], Sequence[Sequence[float]]]


def predict_properties(
    products_db: str,
    predictor: Predictor,
    *,
    model_name: str,
    batch_size: int = 10_000,
) -> None:
    """Predict HOMO/LUMO/gap for every product not yet predicted by `model_name`.

    Resume-by-absence: only products lacking a predictions row for this model are
    processed, and results are committed per batch, so an interrupted run can be
    re-run. Running with a different `model_name` adds a second set of rows
    rather than overwriting.

    Args:
        products_db: Path to the products SQLite database.
        predictor:   list[str] of SMILES -> sequence of (homo_ev, lumo_ev, gap_ev).
        model_name:  Provenance label stored in the `model` column.
        batch_size:  Number of products passed to the predictor per call.
    """
    init_predictions_table(products_db)

    read_conn = sqlite3.connect(products_db)
    read_conn.execute("PRAGMA journal_mode=WAL")  # let the read cursor and writer coexist
    write_conn = sqlite3.connect(products_db)

    n = 0
    logger.info("Starting prediction for %s (model=%s)", products_db, model_name)
    try:
        cursor = read_conn.cursor()
        cursor.execute(
            """
            SELECT p.id, p.product_smiles
            FROM products p
            LEFT JOIN predictions pr
                   ON pr.product_id = p.id AND pr.model = ?
            WHERE pr.product_id IS NULL
            """,
            (model_name,),
        )
        while True:
            batch = cursor.fetchmany(batch_size)
            if not batch:
                break
            ids = [row[0] for row in batch]
            smiles = [row[1] for row in batch]

            preds = predictor(smiles)
            if len(preds) != len(ids):
                raise ValueError(
                    f"predictor returned {len(preds)} rows for {len(ids)} SMILES"
                )

            rows = [
                (pid, model_name, float(homo), float(lumo), float(gap))
                for pid, (homo, lumo, gap) in zip(ids, preds)
            ]
            write_conn.executemany(
                "INSERT INTO predictions (product_id, model, homo_ev, lumo_ev, gap_ev)"
                " VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            write_conn.commit()
            n += len(rows)
            logger.info("Predicted %d products (model=%s)", n, model_name)
    finally:
        read_conn.close()
        write_conn.close()

    logger.info("Prediction complete: %d rows (model=%s, %s)", n, model_name, products_db)


# ── Chemprop adapter (optional: requires `pip install baymol[ml]`) ─────────────

def load_chemprop_predictor(model_dir: str, *, batch_size: int = 512) -> Predictor:
    """Load a trained Chemprop model (or ensemble) and return a `Predictor`.

    Looks for `model_*/best.pt` checkpoints under `model_dir`; if several are
    present, their predictions are averaged (ensemble). The model is loaded
    **once** here and reused for every batch the returned predictor is called
    with — so this is cheap to pass to `predict_properties`.

    Requires the optional `[ml]` extra (chemprop + torch), imported lazily so the
    rest of this module stays dependency-light.
    """
    model_paths = sorted(Path(model_dir).glob("model_*/best.pt"))
    if not model_paths:
        raise FileNotFoundError(f"No model_*/best.pt checkpoints found under {model_dir!r}")

    try:
        import numpy as np
        import torch
        from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
        from chemprop.models.utils import load_model, load_output_columns
        from lightning import Trainer
    except ImportError as e:
        raise ImportError(
            "Chemprop prediction needs the optional dependencies — "
            "install them with: pip install 'baymol[ml]'"
        ) from e

    models = [load_model(p) for p in model_paths]
    for m in models:
        m.eval()

    # Reorder the model's output columns into (homo_ev, lumo_ev, gap_ev).
    output_cols = load_output_columns(model_paths[0]) or ["homo_ev", "lumo_ev", "gap_ev"]
    order = [output_cols.index(c) for c in ("homo_ev", "lumo_ev", "gap_ev")]
    trainer = Trainer(accelerator="auto", logger=False, enable_progress_bar=False)

    def predictor(smiles: list[str]) -> list[tuple[float, float, float]]:
        dataset = MoleculeDataset([MoleculeDatapoint.from_smi(s) for s in smiles])
        loader = build_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        per_model = [torch.cat(trainer.predict(m, loader), dim=0).numpy() for m in models]
        preds = np.mean(per_model, axis=0)  # (n_molecules, n_targets)
        return [(float(r[order[0]]), float(r[order[1]]), float(r[order[2]])) for r in preds]

    logger.info("Loaded %d Chemprop model(s) from %s", len(models), model_dir)
    return predictor


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Predict HOMO/LUMO/gap for a products database with a trained Chemprop model."
    )
    parser.add_argument("products_db", help="Path to the products SQLite database.")
    parser.add_argument(
        "--model", required=True,
        help="Chemprop model directory containing model_*/best.pt checkpoints.",
    )
    parser.add_argument(
        "--model-name", default=None,
        help="Provenance label stored in predictions.model (default: the model directory name).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=10_000,
        help="Products per database batch (default: 10000).",
    )
    args = parser.parse_args()

    model_name = args.model_name or Path(args.model).name
    predictor = load_chemprop_predictor(args.model)
    predict_properties(args.products_db, predictor, model_name=model_name, batch_size=args.batch_size)


if __name__ == "__main__":
    main()