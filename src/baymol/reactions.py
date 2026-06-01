"""
Reaction engine: generates cross-coupled products from a precursor database.

Supported reaction types
------------------------
  Suzuki     — aryl/alkene halide  +  boronic acid/ester
  Stille     — aryl/alkene halide  +  organotin
  Sonogashira — aryl/alkene halide  +  terminal alkyne
  Knoevenagel — aryl aldehyde       +  active methylene acceptor

CLI usage
---------
  python -m baymol.reactions generate --precursors-db output/merged.db --products-db output/products.db
  python -m baymol.reactions generate --start-index 500 --max-workers 4
  python -m baymol.reactions dedup --products-db output/products.db --output-db output/dedup.db
"""

import logging
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from typing import Optional

from rdkit import Chem
from rdkit.Chem import rdChemReactions

from baymol.db import init_products_database

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _get_reaction(smarts: str) -> rdChemReactions.ChemicalReaction:
    """Compile a reaction SMARTS, caching the result (per process).

    The same handful of SMARTS strings are reused across the entire precursor
    cross-product, so compiling once and reusing is a large speedup. The cache
    is per-process, which is exactly what the ProcessPoolExecutor workers want.
    """
    return rdChemReactions.ReactionFromSmarts(smarts)


def mol_from_smiles(smiles: str) -> Chem.Mol:
    """Convert a SMILES string to an RDKit Mol, raising ValueError on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    return mol


# ── Reaction SMARTS ───────────────────────────────────────────────────────────

SUZUKI            = r"[c:1][Br,I] . [c:2][B](O)O >> [c,C:1][c,C:2]"
SUZUKI_ALKENE_HAL = r"[c,C;^2:1]=[C:2][Br,I] . [c:3][B](O)O >> [c,C;^2:1]=[C:2][c:3]"
SUZUKI_ALKENE_BO2 = r"[c:1][Br,I] . [c,C;^2:2]=[C:3][B](O)O >> [c:1][C:3]=[c,C;^2:2]"
SUZUKI_ALKENES    = r"[c,C;^2:1]=[C:2][Br,I] . [c,C;^2:3]=[C:4][B](O)O >> [c,C;^2:1]=[C:2][C:4]=[c,C;^2:3]"

STILLE            = r"[c:1][Br,I] . [c:2][Sn](C)(C)C >> [c,C:1][c,C:2]"
STILLE_ALKENE_HAL = r"[c,C;^2:1]=[C:2][Br,I] . [c:3][Sn](C)(C)C >> [c,C;^2:1]=[C:2][c:3]"
STILLE_ALKENE_SNR3 = r"[c:1][Br,I] . [c,C;^2:2]=[C:3][Sn](C)(C)C >> [c:1][C:3]=[c,C;^2:2]"
STILLE_ALKENES    = r"[c,C;^2:1]=[C:2][Br,I] . [c,C;^2:3]=[C:4][Sn](C)(C)C >> [c,C;^2:1]=[C:2][C:4]=[c,C;^2:3]"

SONOGASHIRA       = r"[c:1][Br,I] . [C;H1:2]#[C:3] >> [c,C:1][C:2]#[C:3]"
SONOGASHIRA_ALKENE_HAL = r"[c,C;^2:1]=[C:2][Br,I] . [C;H1:3]#[C:4] >> [c,C;^2:1]=[C:2][C:3]#[C:4]"

KNOEVENAGEL_MALONONITRILE = r"""
[c:1][CH:2](=[O]) . [C:3](=[O:4])[CH2:5][C:6]=[C:7]([C]#[N])[C]#[N]
>> [c:1][CH:2]=[C:5]([C:3](=[O:4]))[C:6]=[C:7]([C]#[N])[C]#[N]
"""
KNOEVENAGEL_KETONE = r"""
[c:1][CH:2](=[O]) . [c:3][C:4](=[O:5])[CH2:6][C:7](=[O:8])[c:9]
>> [c:1][CH:2]=[C:6]([C:4](=[O:5])[c:3])[C:7](=[O:8])[c:9]
"""


# ── Reaction execution ────────────────────────────────────────────────────────

_SANITIZE_FLAG_NAMES = (
    (Chem.SanitizeFlags.SANITIZE_PROPERTIES, "PROPERTIES"),
    (Chem.SanitizeFlags.SANITIZE_SYMMRINGS, "SYMMRINGS"),
    (Chem.SanitizeFlags.SANITIZE_KEKULIZE, "KEKULIZE"),
    (Chem.SanitizeFlags.SANITIZE_SETAROMATICITY, "AROMATICITY"),
)


def sanitize(
    products: list,
    precursor_a_smiles: Optional[str] = None,
    precursor_b_smiles: Optional[str] = None,
) -> list[str]:
    """Sanitize RDKit product molecules and return valid SMILES strings.

    Molecules that fail sanitization are skipped (not returned); a warning is
    logged for each, with context about the precursors that produced it.
    """
    smiles_list = []
    for mol in products:
        try:
            result = Chem.SanitizeMol(mol, catchErrors=True)
        except Exception as e:
            logger.warning(
                "Sanitization raised %s: %s (A: %s, B: %s)",
                type(e).__name__, e, precursor_a_smiles, precursor_b_smiles,
            )
            continue

        if result != 0:
            error_types = [name for flag, name in _SANITIZE_FLAG_NAMES if result & flag]
            logger.warning(
                "Skipping product that failed sanitization (code %s%s) (A: %s, B: %s)",
                result,
                f", {', '.join(error_types)}" if error_types else "",
                precursor_a_smiles, precursor_b_smiles,
            )
            continue

        try:
            smiles_list.append(Chem.MolToSmiles(mol))
        except Exception as e:
            logger.warning(
                "Could not generate SMILES for sanitized product: %s: %s (A: %s, B: %s)",
                type(e).__name__, e, precursor_a_smiles, precursor_b_smiles,
            )
    return smiles_list


def chemical_reaction(
    reaction: str,
    precursor_a_smiles: str,
    precursor_b_smiles: str,
) -> list[str]:
    """Execute a single reaction between two precursors.

    Args:
        reaction:           SMARTS reaction string.
        precursor_a_smiles: SMILES of the first reactant (matches left side of SMARTS).
        precursor_b_smiles: SMILES of the second reactant.

    Returns:
        List of product SMILES. Empty list if precursors cannot be parsed or
        the reaction produces no products.
    """
    rxn = _get_reaction(reaction)
    try:
        mol_a = mol_from_smiles(precursor_a_smiles)
        mol_b = mol_from_smiles(precursor_b_smiles)
    except ValueError as e:
        logger.warning("Precursor parse error: %s", e)
        return []

    products = []
    for prod_set in rxn.RunReactants((mol_a, mol_b)):
        products.extend(prod_set)
    return sanitize(products, precursor_a_smiles, precursor_b_smiles)


# ── Product generation ────────────────────────────────────────────────────────

def _couple_n(
    reaction: str,
    threading: str,
    constant: str,
    n: int,
    threading_is_first: bool = True,
) -> Optional[str]:
    """Apply a reaction n times, threading one SMILES through as the growing molecule.

    Each step's output becomes the next step's input for the threading argument.
    The constant argument is the same reagent every step.

    Args:
        reaction:           SMARTS reaction string.
        threading:          Starting SMILES for the molecule being built up.
        constant:           SMILES of the constant reagent.
        n:                  Number of times to apply the reaction.
        threading_is_first: If True, threading is reactant 1; otherwise reactant 2.

    Returns:
        Final SMILES after n successful couplings, or None if any step fails.
    """
    current = threading
    for _ in range(n):
        if threading_is_first:
            products = chemical_reaction(reaction, current, constant)
        else:
            products = chemical_reaction(reaction, constant, current)
        if not products:
            return None
        current = products[0]
    return current


# Each worker process opens one read-only connection and reuses it across all
# the rows it handles, rather than reopening per row. Workers only ever read
# the precursors DB, and concurrent readers on a SQLite file are safe.
_worker_conn: Optional[sqlite3.Connection] = None


def _init_worker(precursors_db: str) -> None:
    """ProcessPoolExecutor initializer: open one connection per worker process."""
    global _worker_conn
    _worker_conn = sqlite3.connect(precursors_db)
    _worker_conn.row_factory = sqlite3.Row


def process_row(row_a: dict, precursors_db: Optional[str] = None) -> list[dict]:
    """Generate all cross-coupling products for precursor row_a against all
    compatible partners in the database.

    This function runs in a subprocess (via ProcessPoolExecutor), so row_a must
    be a plain dict rather than a sqlite3.Row (which is not picklable).

    Inside the pool it reuses the per-worker connection opened by _init_worker.
    If called standalone (e.g. in tests) with an explicit precursors_db, it
    opens and closes its own connection.

    Returns:
        List of dicts with keys: product_smiles, precursor_a_smiles, precursor_b_smiles.
    """
    a = row_a
    row_products = []

    if _worker_conn is not None:
        conn = _worker_conn
        owns_conn = False
    elif precursors_db is not None:
        conn = sqlite3.connect(precursors_db)
        conn.row_factory = sqlite3.Row
        owns_conn = True
    else:
        raise ValueError("process_row needs a worker connection or a precursors_db path")
    cursor = conn.cursor()

    def _add(product_smiles, partner):
        row_products.append({
            "product_smiles":     product_smiles,
            "precursor_a_smiles": a["smiles"],
            "precursor_b_smiles": partner["smiles"],
        })

    try:
        # ── Suzuki coupling (boronic acid/ester) ──────────────────────────────
        n_bo2 = a["aryl_bo2"] + a["alkene_bo2"]

        if n_bo2 == 1:
            # A has one bo2 → couple with every B that has any halide
            cursor.execute(
                "SELECT * FROM precursors WHERE id != ? AND (aryl_hal + alkene_hal) > 0",
                (a["id"],),
            )
            for row_b in cursor:
                b = dict(row_b)
                x, y = b["aryl_hal"], b["alkene_hal"]
                out = b["smiles"]

                if a["aryl_bo2"] == 1:
                    if x > 0:
                        out = _couple_n(SUZUKI, out, a["smiles"], x)
                        if out is None:
                            continue
                    if y > 0:
                        out = _couple_n(SUZUKI_ALKENE_HAL, out, a["smiles"], y)
                        if out is None:
                            continue
                else:  # a["alkene_bo2"] == 1
                    if x > 0:
                        out = _couple_n(SUZUKI_ALKENE_BO2, out, a["smiles"], x)
                        if out is None:
                            continue
                    if y > 0:
                        out = _couple_n(SUZUKI_ALKENES, out, a["smiles"], y)
                        if out is None:
                            continue

                _add(out, b)

        elif n_bo2 > 1:
            # A has multiple bo2 → couple with every B that has exactly one halide
            x, y = a["aryl_bo2"], a["alkene_bo2"]
            cursor.execute(
                "SELECT * FROM precursors WHERE id != ? AND (aryl_hal + alkene_hal) = 1",
                (a["id"],),
            )
            for row_b in cursor:
                b = dict(row_b)
                out = a["smiles"]

                if b["aryl_hal"] == 1:
                    if x > 0:
                        out = _couple_n(SUZUKI, out, b["smiles"], x, threading_is_first=False)
                        if out is None:
                            continue
                    if y > 0:
                        out = _couple_n(SUZUKI_ALKENE_BO2, out, b["smiles"], y, threading_is_first=False)
                        if out is None:
                            continue
                else:  # b["alkene_hal"] == 1
                    if x > 0:
                        out = _couple_n(SUZUKI_ALKENE_HAL, out, b["smiles"], x, threading_is_first=False)
                        if out is None:
                            continue
                    if y > 0:
                        out = _couple_n(SUZUKI_ALKENES, out, b["smiles"], y, threading_is_first=False)
                        if out is None:
                            continue

                _add(out, b)

        # ── Stille coupling (organotin) ───────────────────────────────────────
        n_sn = a["aryl_snr3"] + a["alkene_snr3"]

        if n_sn == 1:
            cursor.execute(
                "SELECT * FROM precursors WHERE id != ? AND (aryl_hal + alkene_hal) > 0",
                (a["id"],),
            )
            for row_b in cursor:
                b = dict(row_b)
                x, y = b["aryl_hal"], b["alkene_hal"]
                out = b["smiles"]

                if a["aryl_snr3"] == 1:
                    if x > 0:
                        out = _couple_n(STILLE, out, a["smiles"], x)
                        if out is None:
                            continue
                    if y > 0:
                        out = _couple_n(STILLE_ALKENE_HAL, out, a["smiles"], y)
                        if out is None:
                            continue
                else:  # a["alkene_snr3"] == 1
                    if x > 0:
                        out = _couple_n(STILLE_ALKENE_SNR3, out, a["smiles"], x)
                        if out is None:
                            continue
                    if y > 0:
                        out = _couple_n(STILLE_ALKENES, out, a["smiles"], y)
                        if out is None:
                            continue

                _add(out, b)

        elif n_sn > 1:
            x, y = a["aryl_snr3"], a["alkene_snr3"]
            cursor.execute(
                "SELECT * FROM precursors WHERE id != ? AND (aryl_hal + alkene_hal) = 1",
                (a["id"],),
            )
            for row_b in cursor:
                b = dict(row_b)
                out = a["smiles"]

                if b["aryl_hal"] == 1:
                    if x > 0:
                        out = _couple_n(STILLE, out, b["smiles"], x, threading_is_first=False)
                        if out is None:
                            continue
                    if y > 0:
                        out = _couple_n(STILLE_ALKENE_SNR3, out, b["smiles"], y, threading_is_first=False)
                        if out is None:
                            continue
                else:  # b["alkene_hal"] == 1
                    if x > 0:
                        out = _couple_n(STILLE_ALKENE_HAL, out, b["smiles"], x, threading_is_first=False)
                        if out is None:
                            continue
                    if y > 0:
                        out = _couple_n(STILLE_ALKENES, out, b["smiles"], y, threading_is_first=False)
                        if out is None:
                            continue

                _add(out, b)

        # ── Sonogashira coupling (terminal alkyne) ────────────────────────────
        n_alk = a["terminal_alkyne"]
        a_has_hal = (a["aryl_hal"] + a["alkene_hal"]) > 0

        if n_alk == 1 and not a_has_hal:
            cursor.execute(
                "SELECT * FROM precursors WHERE id != ? AND (aryl_hal + alkene_hal) > 0",
                (a["id"],),
            )
            for row_b in cursor:
                b = dict(row_b)
                x, y = b["aryl_hal"], b["alkene_hal"]
                out = b["smiles"]

                if x > 0:
                    out = _couple_n(SONOGASHIRA, out, a["smiles"], x)
                    if out is None:
                        continue
                if y > 0:
                    out = _couple_n(SONOGASHIRA_ALKENE_HAL, out, a["smiles"], y)
                    if out is None:
                        continue

                _add(out, b)

        elif n_alk > 1 and not a_has_hal:
            cursor.execute(
                "SELECT * FROM precursors WHERE id != ? AND (aryl_hal + alkene_hal) = 1",
                (a["id"],),
            )
            for row_b in cursor:
                b = dict(row_b)
                out = a["smiles"]

                if b["aryl_hal"] == 1:
                    out = _couple_n(SONOGASHIRA, out, b["smiles"], n_alk, threading_is_first=False)
                    if out is None:
                        continue
                else:  # b["alkene_hal"] == 1
                    out = _couple_n(SONOGASHIRA_ALKENE_HAL, out, b["smiles"], n_alk, threading_is_first=False)
                    if out is None:
                        continue

                _add(out, b)

        # ── Knoevenagel condensation (aryl aldehyde) ──────────────────────────
        n_ald = a["aryl_aldehyde"]

        if n_ald == 1:
            # A has one aldehyde → couple with B that has any active methylene site
            cursor.execute(
                "SELECT * FROM precursors WHERE id != ? AND (malononitrile_ketone + diketone) > 0",
                (a["id"],),
            )
            for row_b in cursor:
                b = dict(row_b)
                x, y = b["malononitrile_ketone"], b["diketone"]
                out = b["smiles"]

                # Threading arg (out) is the methylene compound (2nd in SMARTS);
                # a["smiles"] (the aldehyde) is constant 1st.
                if x > 0:
                    out = _couple_n(KNOEVENAGEL_MALONONITRILE, out, a["smiles"], x, threading_is_first=False)
                    if out is None:
                        continue
                if y > 0:
                    out = _couple_n(KNOEVENAGEL_KETONE, out, a["smiles"], y, threading_is_first=False)
                    if out is None:
                        continue

                _add(out, b)

        elif n_ald > 1:
            # A has multiple aldehydes → couple with B that has exactly one methylene site
            cursor.execute(
                "SELECT * FROM precursors WHERE id != ? AND (malononitrile_ketone + diketone) = 1",
                (a["id"],),
            )
            for row_b in cursor:
                b = dict(row_b)
                out = a["smiles"]

                # Threading arg (out) is the multi-aldehyde A (1st in SMARTS);
                # b (the mono methylene) is constant 2nd.
                if b["malononitrile_ketone"] == 1:
                    out = _couple_n(KNOEVENAGEL_MALONONITRILE, out, b["smiles"], n_ald)
                    if out is None:
                        continue
                else:  # b["diketone"] == 1
                    out = _couple_n(KNOEVENAGEL_KETONE, out, b["smiles"], n_ald)
                    if out is None:
                        continue

                _add(out, b)

    finally:
        if owns_conn:
            conn.close()

    return row_products


def generate_products(
    precursors_db: str,
    products_db: str,
    start_index: int = 0,
    max_workers: int = 8,
) -> None:
    """Generate all possible cross-coupling products from the precursor database.

    Uses a sliding window of subprocess workers so exactly max_workers jobs are
    in flight at any time. Results are committed to products_db as each job
    completes, making the run resumable via --start-index.

    Args:
        precursors_db: Path to the precursors SQLite database.
        products_db:   Path to the output products SQLite database.
        start_index:   0-based row index to resume from.
        max_workers:   Number of parallel worker processes.
    """
    logger.info("Starting product generation...")
    if start_index > 0:
        logger.info("Resuming from row index %d...", start_index)

    init_products_database(products_db)

    conn_pre = sqlite3.connect(precursors_db)
    conn_pre.row_factory = sqlite3.Row
    cur_a = conn_pre.cursor()
    cur_a.execute("SELECT COUNT(*) FROM precursors")
    total_rows = cur_a.fetchone()[0]
    # ORDER BY id so the OFFSET (start_index) is deterministic across runs.
    cur_a.execute("SELECT * FROM precursors ORDER BY id LIMIT -1 OFFSET ?", (start_index,))
    logger.info("Rows to process: %d", total_rows - start_index)

    conn_prod = sqlite3.connect(products_db)
    cur_prod = conn_prod.cursor()
    total_products = 0

    try:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
            initargs=(precursors_db,),
        ) as executor:
            futures = {}
            rows_iter = enumerate(cur_a, start=start_index)

            # Fill the pool to max_workers initially
            for idx, row_a in rows_iter:
                futures[executor.submit(process_row, dict(row_a))] = idx
                if len(futures) >= max_workers:
                    break

            # As each job completes, save results and submit the next row
            while futures:
                for future in as_completed(futures):
                    idx = futures.pop(future)
                    try:
                        row_products = future.result()
                    except Exception as e:
                        logger.warning("Row %d failed: %s", idx, e)
                        row_products = []

                    if row_products:
                        cur_prod.executemany(
                            """INSERT INTO products
                               (product_smiles, precursor_a_smiles, precursor_b_smiles)
                               VALUES (:product_smiles, :precursor_a_smiles, :precursor_b_smiles)""",
                            row_products,
                        )
                        conn_prod.commit()
                        total_products += len(row_products)
                        logger.info("Row %d — saved %d products", idx, len(row_products))

                    for next_idx, next_row in rows_iter:
                        futures[executor.submit(process_row, dict(next_row))] = next_idx
                        break
                    break

    finally:
        conn_pre.close()
        conn_prod.close()

    logger.info("Product generation complete. Total: %d (%s)", total_products, products_db)


def deduplicate_products(db_path: str, output_path: str) -> None:
    """Deduplicate a products database by product_smiles.

    For duplicate rows the precursor SMILES lists are merged into the surviving
    (lowest-id) row. Done entirely in SQL to avoid loading data into RAM.

    Args:
        db_path:     Input products SQLite database.
        output_path: Path for the deduplicated output (file is copied then modified).
    """
    import shutil
    shutil.copy2(db_path, output_path)
    logger.info("Copied %r -> %r", db_path, output_path)

    conn = sqlite3.connect(output_path)
    cur = conn.cursor()

    cur.executescript("""
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous  = OFF;
        PRAGMA cache_size   = -262144;
    """)

    cur.execute("""
        SELECT COUNT(*) FROM (
            SELECT product_smiles FROM products
            GROUP BY product_smiles HAVING COUNT(*) > 1
        )
    """)
    n_dupes = cur.fetchone()[0]
    if n_dupes == 0:
        logger.info("No duplicates found.")
        conn.close()
        return

    logger.info("Found %d SMILES with duplicates. Deduplicating...", n_dupes)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_product_smiles ON products(product_smiles)")
    conn.commit()

    # For each duplicate group, merge ALL precursors into the surviving
    # (lowest-id) row. The precursor_a and precursor_b lists are treated as
    # independent sets: collect the distinct values across the whole group and
    # comma-join them. Using a DISTINCT subquery (rather than substring tests)
    # avoids both the "only one duplicate merged" bug and the false-positive
    # substring matches of the previous instr()-based approach.
    cur.execute("""
        UPDATE products
        SET
            precursor_a_smiles = (
                SELECT group_concat(val, ',') FROM (
                    SELECT DISTINCT g.precursor_a_smiles AS val
                    FROM products g
                    WHERE g.product_smiles = products.product_smiles
                    ORDER BY g.id
                )
            ),
            precursor_b_smiles = (
                SELECT group_concat(val, ',') FROM (
                    SELECT DISTINCT g.precursor_b_smiles AS val
                    FROM products g
                    WHERE g.product_smiles = products.product_smiles
                    ORDER BY g.id
                )
            )
        WHERE id IN (SELECT MIN(id) FROM products GROUP BY product_smiles)
    """)
    conn.commit()

    cur.execute("""
        DELETE FROM products
        WHERE id NOT IN (SELECT MIN(id) FROM products GROUP BY product_smiles)
    """)
    deleted = cur.rowcount
    conn.commit()
    logger.info("Deleted %d duplicate rows.", deleted)

    cur.execute("VACUUM")
    conn.close()
    logger.info("Done. Deduplicated database saved to %r", output_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate and deduplicate cross-coupled products."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate products from a precursor database.")
    gen.add_argument(
        "--precursors-db", default="output/merged_dedup_precursors.db",
        help="Path to the precursors SQLite database.",
    )
    gen.add_argument(
        "--products-db", default="output/generated_products.db",
        help="Path to the output products SQLite database.",
    )
    gen.add_argument(
        "--start-index", default=0, type=int,
        help="0-based row index to resume from (default: 0).",
    )
    gen.add_argument(
        "--max-workers", default=8, type=int,
        help="Number of parallel worker processes (default: 8).",
    )

    dedup = sub.add_parser("dedup", help="Deduplicate a products database by product SMILES.")
    dedup.add_argument(
        "--products-db", required=True,
        help="Path to the products SQLite database to deduplicate.",
    )
    dedup.add_argument(
        "--output-db", required=True,
        help="Path for the deduplicated output database.",
    )

    args = parser.parse_args()

    if args.command == "generate":
        generate_products(
            precursors_db=args.precursors_db,
            products_db=args.products_db,
            start_index=args.start_index,
            max_workers=args.max_workers,
        )
    elif args.command == "dedup":
        deduplicate_products(args.products_db, args.output_db)
