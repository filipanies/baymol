"""
Bulk featurisation: apply features.compute_features across a whole products
database and store the results.

`molecular_features` (scalar descriptors + element composition + substructure
flags) is always populated. The Morgan `fingerprints` table (packed BLOBs) is
populated only when requested — it's slow and large, so it's opt-in.

Resume is by absence: each run processes only products that don't yet have a row
in the target table, so an interrupted run can simply be re-run. To refresh after
changing the feature definitions, drop the table and re-run.

CLI:
    python -m baymol.featurise output/products.db
    python -m baymol.featurise output/products.db --fingerprints --workers 8 --batch-size 1000
"""

import itertools
import json
import logging
import sqlite3
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np

from baymol.db import init_fingerprints_table, init_molecular_features_table
from baymol.features import compute_features, morgan_count_fingerprint, morgan_fingerprint

logger = logging.getLogger(__name__)

# Fingerprints are fixed at these parameters (radius is not recoverable from the
# stored BLOB, so changing it would silently mix incompatible fingerprints).
RADIUS = 2
NBITS = 2048


# ── Fingerprint packing ───────────────────────────────────────────────────────

def pack_bits(bits: list[int]) -> bytes:
    """Pack a 0/1 bit list into bytes (nbits/8 bytes)."""
    return np.packbits(np.array(bits, dtype=np.uint8)).tobytes()


def pack_counts(counts: list[int]) -> bytes:
    """Pack a count list into uint16 bytes (nbits*2 bytes)."""
    return np.array(counts, dtype=np.uint16).tobytes()


# ── Worker (pure: computes from SMILES, no DB access) ─────────────────────────

def _feature_row(product_id: int, f: dict) -> dict:
    """Flatten a compute_features dict into a molecular_features row dict."""
    row = {
        "product_id": product_id,
        "total_atom_count": f["total_atom_count"],
        "heavy_atom_count": f["heavy_atom_count"],
        "molecular_weight": f["molecular_weight"],
        "aromatic_ring_count": f["aromatic_ring_count"],
        "fused_ring_count": f["fused_ring_count"],
        "aromatic_atom_count": f["aromatic_atom_count"],
        "fraction_sp2": f["fraction_sp2"],
        "fraction_sp": f["fraction_sp"],
        "h_bond_donor_count": f["h_bond_donor_count"],
        "h_bond_acceptor_count": f["h_bond_acceptor_count"],
        "rotatable_bond_count": f["rotatable_bond_count"],
        "unique_elements": json.dumps(f["unique_elements"]),
        "unique_elements_with_counts": json.dumps(f["unique_elements_with_counts"]),
    }
    for name, present in f["substructure_flags"].items():
        row[name] = int(present)
    return row


def _featurise_chunk(chunk: list[tuple]) -> tuple[list[dict], list[tuple]]:
    """Compute features (and fingerprints, where requested) for a chunk of products.

    chunk items: (product_id, smiles, need_features, need_fingerprints).
    Runs in a worker process; computes only from the SMILES, no DB access.
    """
    feature_rows: list[dict] = []
    fingerprint_rows: list[tuple] = []
    for product_id, smiles, need_features, need_fingerprints in chunk:
        if need_features:
            feature_rows.append(_feature_row(product_id, compute_features(smiles)))
        if need_fingerprints:
            fingerprint_rows.append((
                product_id,
                pack_bits(morgan_fingerprint(smiles, RADIUS, NBITS)),
                pack_counts(morgan_count_fingerprint(smiles, RADIUS, NBITS)),
            ))
    return feature_rows, fingerprint_rows


# ── Main-process helpers (read pending, write results) ────────────────────────

def _pending_cursor(conn: sqlite3.Connection, fingerprints: bool) -> sqlite3.Cursor:
    """Cursor over products missing from the target table(s).

    Yields (product_id, smiles, need_features, need_fingerprints).
    """
    cur = conn.cursor()
    if fingerprints:
        cur.execute("""
            SELECT p.id, p.product_smiles,
                   mf.product_id IS NULL,
                   fp.product_id IS NULL
            FROM products p
            LEFT JOIN molecular_features mf ON mf.product_id = p.id
            LEFT JOIN fingerprints fp ON fp.product_id = p.id
            WHERE mf.product_id IS NULL OR fp.product_id IS NULL
        """)
    else:
        cur.execute("""
            SELECT p.id, p.product_smiles, 1, 0
            FROM products p
            LEFT JOIN molecular_features mf ON mf.product_id = p.id
            WHERE mf.product_id IS NULL
        """)
    return cur


def _chunked(cursor: sqlite3.Cursor, size: int):
    """Yield lists of up to `size` rows (as tuples) from a cursor."""
    while True:
        rows = cursor.fetchmany(size)
        if not rows:
            return
        yield [tuple(row) for row in rows]


def _insert_features(conn: sqlite3.Connection, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    placeholders = ", ".join(f":{c}" for c in cols)
    conn.executemany(
        f"INSERT INTO molecular_features ({', '.join(cols)}) VALUES ({placeholders})",
        rows,
    )


def _insert_fingerprints(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    if not rows:
        return
    conn.executemany(
        "INSERT INTO fingerprints (product_id, morgan_fp, morgan_count_fp) VALUES (?, ?, ?)",
        rows,
    )


def populate(
    products_db: str,
    *,
    fingerprints: bool = False,
    max_workers: int = 8,
    batch_size: int = 1000,
) -> None:
    """Featurise every product in products_db that hasn't been featurised yet.

    molecular_features is always populated; the fingerprints table only when
    `fingerprints=True`. Results are committed per chunk, so the run is resumable.
    """
    init_molecular_features_table(products_db)
    if fingerprints:
        init_fingerprints_table(products_db)

    # WAL lets the read cursor and the writer coexist on the same DB file.
    read_conn = sqlite3.connect(products_db)
    read_conn.execute("PRAGMA journal_mode=WAL")
    write_conn = sqlite3.connect(products_db)

    n_features = n_fingerprints = 0
    logger.info("Starting featurisation of %s (fingerprints=%s)", products_db, fingerprints)
    try:
        chunks = _chunked(_pending_cursor(read_conn, fingerprints), batch_size)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_featurise_chunk, chunk)
                for chunk in itertools.islice(chunks, max_workers)
            }
            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    try:
                        feature_rows, fingerprint_rows = future.result()
                    except Exception as e:
                        logger.warning("Featurisation chunk failed: %s", e)
                        feature_rows, fingerprint_rows = [], []
                    _insert_features(write_conn, feature_rows)
                    _insert_fingerprints(write_conn, fingerprint_rows)
                    write_conn.commit()
                    n_features += len(feature_rows)
                    n_fingerprints += len(fingerprint_rows)
                # Refill the pool to keep max_workers chunks in flight.
                for chunk in itertools.islice(chunks, len(done)):
                    futures.add(executor.submit(_featurise_chunk, chunk))
    finally:
        read_conn.close()
        write_conn.close()

    logger.info(
        "Featurisation complete: %d feature rows, %d fingerprint rows (%s)",
        n_features, n_fingerprints, products_db,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Featurise a products database (molecular_features, optional fingerprints)."
    )
    parser.add_argument("products_db", help="Path to the products SQLite database.")
    parser.add_argument(
        "--fingerprints", action="store_true",
        help="Also compute and store Morgan fingerprints (slow, large).",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of parallel worker processes (default: 8).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1000,
        help="Products per worker task (default: 1000).",
    )
    args = parser.parse_args()

    populate(
        args.products_db,
        fingerprints=args.fingerprints,
        max_workers=args.workers,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
