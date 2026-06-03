"""
Predict molecular properties for a product library and store them.

This module is the *orchestration* only: it reads product SMILES from the
database, hands batches to a `predictor`, and writes the results into the
`predictions` table — keyed by (product_id, model), resumable by absence.

It deliberately knows nothing about the model. A `predictor` is just:

    Callable[[list[str]], Sequence[Sequence[float]]]

i.e. given a list of SMILES, return one (homo_ev, lumo_ev, gap_ev) triple per
SMILES (same order). The Chemprop-backed predictor (which pulls in PyTorch)
lives in a separate, optional adapter; tests use a trivial fake. This keeps the
core torch-free and the orchestration fully testable.

CLI: provided by the Chemprop adapter (optional `[ml]` extra), not here.
"""

import logging
import sqlite3
from collections.abc import Callable, Sequence

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