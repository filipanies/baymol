"""
Quickstart — generate, deduplicate, featurise and predict a molecular library,
end to end, from a frozen set of 100 real precursors.

Runs with just the core install (`pip install -e .` → RDKit + NumPy), no
external data:

    python examples/quickstart.py

The precursors live in ``examples/example_precursors.csv`` (a small catalogue
slice: SMILES, CAS, product number and supplier — supplier names and product
numbers are masked), sampled from a supplier catalogue and product database so
that this short run exercises the whole toolkit:

  1. detect reactive sites and drop self-polymerisable precursors,
  2. enumerate the cross-coupling / condensation products (all four reactions),
  3. deduplicate — the same product is often reachable by several routes (e.g.
     a boronic acid via Suzuki *and* a stannane via Stille); those routes are
     merged into one row,
  4. compute molecular features (descriptors + Morgan fingerprints),
  5. predict HOMO/LUMO/gap.

Step 5 replays real predictions frozen in ``examples/example_predictions.csv`` —
genuine outputs of an OE62+CEPDB10k-trained Chemprop model, precomputed for these
products so the demo stays torch-free. To predict live instead, install the ML
extra and swap in the trained model:

    pip install -e ".[ml]"
    from baymol.predict import load_chemprop_predictor
    predict_properties(products_db,
                       load_chemprop_predictor("path/to/chemprop_model"),
                       model_name="oe62_cepdb10k")
"""

import csv
import sqlite3
import tempfile
from pathlib import Path

import numpy as np

from baymol.db import init_precursors_database, save_precursors
from baymol.featurise import populate
from baymol.predict import predict_properties
from baymol.reactions import deduplicate_products, generate_products
from baymol.reactive_sites import count_reactive_sites, is_self_polymerisable

PRECURSORS_CSV = Path(__file__).with_name("example_precursors.csv")
PREDICTIONS_CSV = Path(__file__).with_name("example_predictions.csv")
MAX_WORKERS = 4


def short(smiles: str, width: int = 48) -> str:
    """Truncate a SMILES string for slim terminal output."""
    return smiles if len(smiles) <= width else smiles[: width - 3] + "..."


def load_example_predictor():
    """A predictor that replays the real model values frozen in PREDICTIONS_CSV.

    Returns the standard predictor interface (list[str] of SMILES -> list of
    (homo_ev, lumo_ev, gap_ev) triples), so it drops straight into
    predict_properties exactly where a live Chemprop model would.
    """
    with open(PREDICTIONS_CSV, encoding="utf-8") as fh:
        table = {
            r["smiles"]: (float(r["homo_ev"]), float(r["lumo_ev"]), float(r["gap_ev"]))
            for r in csv.DictReader(fh)
        }
    return lambda smiles: [table[s] for s in smiles]


def main():
    work = Path(tempfile.mkdtemp(prefix="baymol_quickstart_"))
    precursors_db = work / "precursors.db"
    products_db = work / "products.db"
    dedup_db = work / "products_dedup.db"

    # 1. Reactive-site detection — keep precursors with usable sites, drop the
    #    self-polymerisable ones (they carry both a halide and a nucleophile).
    print("== 1. precursors (reactive-site detection) ==")
    init_precursors_database(str(precursors_db))
    rows, skipped = [], []
    with open(PRECURSORS_CSV, encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            sites = count_reactive_sites(rec["smiles"])
            row = {
                "smiles": rec["smiles"],
                "cas": rec["cas"] or None,
                "product_no": rec["product_no"],
                "supplier": rec["supplier"],
                **sites,
            }
            if is_self_polymerisable(sites):
                skipped.append(row)
                continue
            rows.append(row)
    save_precursors(rows, str(precursors_db))
    print(f"   kept {len(rows)} precursors, "
          f"dropped {len(skipped)} self-polymerisable:")
    for row in skipped:
        print(f"      skip  {row['product_no']:7} {row['supplier']:12} {short(row['smiles'], 36)}")

    # 2. Enumerate cross-coupling / condensation products.
    print("\n== 2. generate products ==")
    generate_products(str(precursors_db), str(products_db), max_workers=MAX_WORKERS)
    conn = sqlite3.connect(products_db)
    n_raw = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    print(f"   {n_raw} products (before dedup), by reaction:")
    for reaction, count in conn.execute(
        "SELECT reaction_name, COUNT(*) c FROM products GROUP BY reaction_name ORDER BY c DESC"
    ):
        example = conn.execute(
            "SELECT product_smiles FROM products WHERE reaction_name = ? LIMIT 1",
            (reaction,),
        ).fetchone()[0]
        print(f"      {reaction:12} {count:4}   e.g. {short(example, 36)}")
    conn.close()

    # 3. Deduplicate — collapse identical products, merging their routes.
    print("\n== 3. deduplicate ==")
    deduplicate_products(str(products_db), str(dedup_db))
    conn = sqlite3.connect(dedup_db)
    n_dedup = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    print(f"   {n_raw} -> {n_dedup} products ({n_raw - n_dedup} duplicates merged)")
    print("   products reachable by more than one reaction (routes merged):")
    for product_smiles, reaction_name in conn.execute(
        "SELECT product_smiles, reaction_name FROM products "
        "WHERE reaction_name LIKE '%,%' LIMIT 3"
    ):
        print(f"      [{reaction_name}] {short(product_smiles, 40)}")
    conn.close()

    # 4. Compute molecular features (descriptors + Morgan fingerprints).
    print("\n== 4. featurise ==")
    populate(str(dedup_db), fingerprints=True)
    conn = sqlite3.connect(dedup_db)
    conn.row_factory = sqlite3.Row
    n_feat = conn.execute("SELECT COUNT(*) FROM molecular_features").fetchone()[0]
    row = conn.execute(
        "SELECT p.id, p.product_smiles, mf.heavy_atom_count, mf.molecular_weight, "
        "mf.aromatic_ring_count FROM molecular_features mf "
        "JOIN products p ON p.id = mf.product_id LIMIT 1"
    ).fetchone()
    fp_blob = conn.execute(
        "SELECT morgan_fp FROM fingerprints WHERE product_id = ?", (row["id"],)
    ).fetchone()[0]
    conn.close()
    bits_set = int(np.unpackbits(np.frombuffer(fp_blob, dtype=np.uint8)).sum())
    print(f"   featurised {n_feat} products; e.g. {short(row['product_smiles'], 36)}")
    print(f"      heavy_atoms={row['heavy_atom_count']}, "
          f"MW={row['molecular_weight']:.1f}, "
          f"aromatic_rings={row['aromatic_ring_count']}, "
          f"Morgan fp {bits_set}/2048 bits set")

    # 5. Predict HOMO/LUMO/gap (real OE62+CEPDB10k model values, replayed from
    #    example_predictions.csv — see module docstring).
    print("\n== 5. predict (OE62+CEPDB10k Chemprop model) ==")
    predict_properties(str(dedup_db), load_example_predictor(), model_name="oe62_cepdb10k")
    conn = sqlite3.connect(dedup_db)
    for smiles, homo, lumo, gap in conn.execute(
        "SELECT p.product_smiles, pr.homo_ev, pr.lumo_ev, pr.gap_ev "
        "FROM predictions pr JOIN products p ON p.id = pr.product_id LIMIT 3"
    ):
        print(f"      {short(smiles, 36)}  HOMO={homo:+.2f} LUMO={lumo:+.2f} gap={gap:.2f}")
    conn.close()

    print(f"\nDone. Demo database: {dedup_db}")


if __name__ == "__main__":
    main()