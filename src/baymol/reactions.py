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
  python -m baymol.reactions
  python -m baymol.reactions --precursors-db output/merged.db --products-db output/products.db
  python -m baymol.reactions --start-index 500 --max-workers 4
"""

import os
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

from rdkit import Chem
from rdkit.Chem import rdChemReactions


def mol_from_smiles(smiles: str) -> Chem.Mol:
    """Convert a SMILES string to an RDKit Mol, raising ValueError on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    return mol


# ── Products database ───────────────────────────────────────────────────────

def init_products_database(db_path: str) -> None:
    """Create (or verify) the products SQLite database."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            product_smiles     TEXT NOT NULL,
            precursor_a_smiles TEXT NOT NULL,
            precursor_b_smiles TEXT NOT NULL,
            created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_product_smiles ON products(product_smiles)"
    )
    conn.commit()
    conn.close()
    print(f"Products database initialised: {db_path}")


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

def sanitize(
    products: list,
    precursor_a_smiles: str = None,
    precursor_b_smiles: str = None,
) -> list[str]:
    """Sanitize RDKit product molecules and return valid SMILES strings.

    Logs a warning for each molecule that fails sanitization, with context
    about the precursors that produced it.
    """
    smiles_list = []
    for mol in products:
        try:
            product_smiles = Chem.MolToSmiles(mol, kekuleSmiles=False)
        except Exception:
            product_smiles = "[Could not generate SMILES]"

        try:
            result = Chem.SanitizeMol(mol, catchErrors=True)
            if result != 0:
                print(f"    ⚠ Sanitization warning (code {result})")
                if precursor_a_smiles:
                    print(f"       A: {precursor_a_smiles}")
                if precursor_b_smiles:
                    print(f"       B: {precursor_b_smiles}")
                error_types = []
                if result & Chem.SanitizeFlags.SANITIZE_PROPERTIES:
                    error_types.append("PROPERTIES")
                if result & Chem.SanitizeFlags.SANITIZE_SYMMRINGS:
                    error_types.append("SYMMRINGS")
                if result & Chem.SanitizeFlags.SANITIZE_KEKULIZE:
                    error_types.append("KEKULIZE")
                if result & Chem.SanitizeFlags.SANITIZE_SETAROMATICITY:
                    error_types.append("AROMATICITY")
                if error_types:
                    print(f"       Error types: {', '.join(error_types)}")
            smiles_list.append(Chem.MolToSmiles(mol))
        except Exception as e:
            print(f"    ⚠ Sanitization error: {type(e).__name__}: {e}")
            print(f"       Product: {product_smiles}")
            if precursor_a_smiles:
                print(f"       A: {precursor_a_smiles}")
            if precursor_b_smiles:
                print(f"       B: {precursor_b_smiles}")
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
    rxn = rdChemReactions.ReactionFromSmarts(reaction)
    try:
        mol_a = mol_from_smiles(precursor_a_smiles)
        mol_b = mol_from_smiles(precursor_b_smiles)
    except ValueError as e:
        print(f"    ⚠ Precursor parse error: {e}")
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


def process_row(row_a: dict, precursors_db: str) -> list[dict]:
    """Generate all cross-coupling products for precursor row_a against all
    compatible partners in the database.

    This function runs in a subprocess (via ProcessPoolExecutor), so row_a must
    be a plain dict rather than a sqlite3.Row (which is not picklable).

    Returns:
        List of dicts with keys: product_smiles, precursor_a_smiles, precursor_b_smiles.
    """
    a = row_a
    row_products = []

    conn = sqlite3.connect(precursors_db)
    conn.row_factory = sqlite3.Row
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
                        if out is None: continue
                    if y > 0:
                        out = _couple_n(SUZUKI_ALKENE_HAL, out, a["smiles"], y)
                        if out is None: continue
                else:  # a["alkene_bo2"] == 1
                    if x > 0:
                        out = _couple_n(SUZUKI_ALKENE_BO2, out, a["smiles"], x)
                        if out is None: continue
                    if y > 0:
                        out = _couple_n(SUZUKI_ALKENES, out, a["smiles"], y)
                        if out is None: continue

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
                        if out is None: continue
                    if y > 0:
                        out = _couple_n(SUZUKI_ALKENE_BO2, out, b["smiles"], y, threading_is_first=False)
                        if out is None: continue
                else:  # b["alkene_hal"] == 1
                    if x > 0:
                        out = _couple_n(SUZUKI_ALKENE_HAL, out, b["smiles"], x, threading_is_first=False)
                        if out is None: continue
                    if y > 0:
                        out = _couple_n(SUZUKI_ALKENES, out, b["smiles"], y, threading_is_first=False)
                        if out is None: continue

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
                        if out is None: continue
                    if y > 0:
                        out = _couple_n(STILLE_ALKENE_HAL, out, a["smiles"], y)
                        if out is None: continue
                else:  # a["alkene_snr3"] == 1
                    if x > 0:
                        out = _couple_n(STILLE_ALKENE_SNR3, out, a["smiles"], x)
                        if out is None: continue
                    if y > 0:
                        out = _couple_n(STILLE_ALKENES, out, a["smiles"], y)
                        if out is None: continue

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
                        if out is None: continue
                    if y > 0:
                        out = _couple_n(STILLE_ALKENE_SNR3, out, b["smiles"], y, threading_is_first=False)
                        if out is None: continue
                else:  # b["alkene_hal"] == 1
                    if x > 0:
                        out = _couple_n(STILLE_ALKENE_HAL, out, b["smiles"], x, threading_is_first=False)
                        if out is None: continue
                    if y > 0:
                        out = _couple_n(STILLE_ALKENES, out, b["smiles"], y, threading_is_first=False)
                        if out is None: continue

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
                    if out is None: continue
                if y > 0:
                    out = _couple_n(SONOGASHIRA_ALKENE_HAL, out, a["smiles"], y)
                    if out is None: continue

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
                    if out is None: continue
                else:  # b["alkene_hal"] == 1
                    out = _couple_n(SONOGASHIRA_ALKENE_HAL, out, b["smiles"], n_alk, threading_is_first=False)
                    if out is None: continue

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
                    if out is None: continue
                if y > 0:
                    out = _couple_n(KNOEVENAGEL_KETONE, out, a["smiles"], y, threading_is_first=False)
                    if out is None: continue

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
                    if out is None: continue
                else:  # b["diketone"] == 1
                    out = _couple_n(KNOEVENAGEL_KETONE, out, b["smiles"], n_ald)
                    if out is None: continue

                _add(out, b)

    finally:
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
    print("Starting product generation...")
    if start_index > 0:
        print(f"Resuming from row index {start_index}...")
    print("=" * 60)

    init_products_database(products_db)

    conn_pre = sqlite3.connect(precursors_db)
    conn_pre.row_factory = sqlite3.Row
    cur_a = conn_pre.cursor()
    cur_a.execute("SELECT COUNT(*) FROM precursors")
    total_rows = cur_a.fetchone()[0]
    cur_a.execute("SELECT * FROM precursors LIMIT -1 OFFSET ?", (start_index,))
    print(f"Rows to process: {total_rows - start_index}")

    conn_prod = sqlite3.connect(products_db)
    cur_prod = conn_prod.cursor()
    total_products = 0

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            rows_iter = enumerate(cur_a, start=start_index)

            # Fill the pool to max_workers initially
            for idx, row_a in rows_iter:
                futures[executor.submit(process_row, dict(row_a), precursors_db)] = idx
                if len(futures) >= max_workers:
                    break

            # As each job completes, save results and submit the next row
            while futures:
                for future in as_completed(futures):
                    idx = futures.pop(future)
                    try:
                        row_products = future.result()
                    except Exception as e:
                        print(f"  ⚠ Row {idx} failed: {e}")
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
                        print(f"  Row {idx} — saved {len(row_products)} products")

                    for next_idx, next_row in rows_iter:
                        futures[executor.submit(process_row, dict(next_row), precursors_db)] = next_idx
                        break
                    break

    finally:
        conn_pre.close()
        conn_prod.close()

    print(f"\n{'='*60}")
    print(f"Product generation complete. Total: {total_products}")
    print(f"Database: {products_db}")
    print("=" * 60)


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
    print(f"Copied {db_path!r} → {output_path!r}")

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
        print("No duplicates found.")
        conn.close()
        return

    print(f"Found {n_dupes} SMILES with duplicates. Deduplicating...")

    cur.execute("CREATE INDEX IF NOT EXISTS idx_product_smiles ON products(product_smiles)")
    conn.commit()

    # Merge precursor lists into the surviving row for each duplicate group
    cur.execute("""
        UPDATE products
        SET
            precursor_a_smiles = CASE
                WHEN instr(p.precursor_a_smiles, dup.precursor_a_smiles) = 0
                THEN p.precursor_a_smiles || ',' || dup.precursor_a_smiles
                ELSE p.precursor_a_smiles
            END,
            precursor_b_smiles = CASE
                WHEN instr(p.precursor_b_smiles, dup.precursor_b_smiles) = 0
                THEN p.precursor_b_smiles || ',' || dup.precursor_b_smiles
                ELSE p.precursor_b_smiles
            END
        FROM (
            SELECT a.product_smiles,
                   a.precursor_a_smiles,
                   a.precursor_b_smiles,
                   MIN(b.id) AS survivor_id
            FROM products a
            JOIN products b ON a.product_smiles = b.product_smiles
            WHERE a.id != b.id
            GROUP BY a.id
        ) dup
        JOIN products p ON p.id = dup.survivor_id
        WHERE products.id = dup.survivor_id
    """)
    conn.commit()

    cur.execute("""
        DELETE FROM products
        WHERE id NOT IN (SELECT MIN(id) FROM products GROUP BY product_smiles)
    """)
    deleted = cur.rowcount
    conn.commit()
    print(f"Deleted {deleted} duplicate rows.")

    cur.execute("VACUUM")
    conn.close()
    print(f"Done. Deduplicated database saved to {output_path!r}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate cross-coupled products from a precursor database."
    )
    parser.add_argument(
        "--precursors-db", default="output/merged_dedup_precursors.db",
        help="Path to the precursors SQLite database.",
    )
    parser.add_argument(
        "--products-db", default="output/generated_products.db",
        help="Path to the output products SQLite database.",
    )
    parser.add_argument(
        "--start-index", default=0, type=int,
        help="0-based row index to resume from (default: 0).",
    )
    parser.add_argument(
        "--max-workers", default=8, type=int,
        help="Number of parallel worker processes (default: 8).",
    )
    args = parser.parse_args()

    generate_products(
        precursors_db=args.precursors_db,
        products_db=args.products_db,
        start_index=args.start_index,
        max_workers=args.max_workers,
    )
