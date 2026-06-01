"""
SQLite database helpers for both the precursors and products tables.

All functions accept an explicit db_path argument — there are no module-level
path defaults here. Each scraper / script passes its own path.
"""

import logging
import os
import sqlite3

from baymol.reactive_sites import criteria_smarts

logger = logging.getLogger(__name__)


# ── Precursor database ────────────────────────────────────────────────────────

def init_precursors_database(db_path: str) -> None:
    """Create (or verify) the precursors SQLite database.

    Pre-creates one INTEGER column per reactive site defined in criteria_smarts,
    so the schema is consistent regardless of which scrapers have run.
    """
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

    site_columns = ",\n            ".join(
        f"{name} INTEGER DEFAULT 0" for name in criteria_smarts
    )
    create_sql = f"""
        CREATE TABLE IF NOT EXISTS precursors (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            smiles     TEXT NOT NULL,
            cas        TEXT,
            product_no TEXT,
            supplier   TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            {site_columns}
        )
    """
    conn = sqlite3.connect(db_path)
    conn.execute(create_sql)
    conn.commit()
    conn.close()
    logger.info("Database initialised: %s", db_path)


def save_precursors(precursors: list[dict], db_path: str) -> None:
    """Append a list of precursor dicts to the precursors table.

    The whole list is inserted in a single transaction: if any row fails (e.g. a
    dict with an unknown column or a missing NOT NULL field), the batch is rolled
    back and the error propagates. The connection is always closed.
    """
    if not precursors:
        return

    conn = sqlite3.connect(db_path)
    try:
        with conn:  # commits on success, rolls back if any insert raises
            for p in precursors:
                columns      = ", ".join(p.keys())
                placeholders = ", ".join("?" for _ in p)
                conn.execute(
                    f"INSERT INTO precursors ({columns}) VALUES ({placeholders})",
                    list(p.values()),
                )
    finally:
        conn.close()
    logger.info("Saved %d precursors.", len(precursors))


def merge_precursors(file_names: list[str], output_path: str) -> None:
    """Merge multiple precursor databases into one (without deduplicating).

    Any existing file at output_path is overwritten. Call deduplicate_precursors
    on the result afterwards to collapse duplicate SMILES across sources.

    Args:
        file_names:  Paths to source SQLite database files.
        output_path: Path for the merged output database.
    """
    reactive_cols = list(criteria_smarts.keys())

    if os.path.exists(output_path):
        os.remove(output_path)

    conn = sqlite3.connect(output_path)
    cur = conn.cursor()

    cur.execute(f"""
        CREATE TABLE precursors (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            smiles     TEXT NOT NULL,
            cas        TEXT,
            product_no TEXT,
            supplier   TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            {", ".join(f"{col} INTEGER DEFAULT 0" for col in reactive_cols)}
        )
    """)
    conn.commit()

    for i, path in enumerate(file_names):
        alias = f"src{i}"
        cur.execute(f"ATTACH DATABASE ? AS {alias}", (path,))
        cur.execute(f"""
            INSERT INTO precursors (smiles, cas, product_no, supplier, created_at,
                                    {", ".join(reactive_cols)})
            SELECT smiles, cas, product_no, supplier, created_at,
                   {", ".join(reactive_cols)}
            FROM   {alias}.precursors
        """)
        conn.commit()
        cur.execute(f"DETACH DATABASE {alias}")
        logger.info("Appended %r", path)

    cur.execute("SELECT COUNT(*) FROM precursors")
    logger.info("Merged %d rows into %s", cur.fetchone()[0], output_path)
    conn.close()


def deduplicate_precursors(db_path: str) -> None:
    """Deduplicate the precursors table in-place by SMILES.

    For duplicate rows the supplier lists are merged; the row with the lowest id
    survives. The table is rebuilt in-place (rename → recreate → reinsert → drop backup).
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM precursors ORDER BY id ASC")
    rows = cursor.fetchall()
    logger.info("Rows before deduplication: %d", len(rows))

    seen: dict[str, dict] = {}
    suppliers: dict[str, set] = {}
    for row in rows:
        smiles = row["smiles"]
        row_suppliers = {s.strip() for s in row["supplier"].split(",") if s.strip()}
        if smiles not in seen:
            seen[smiles] = dict(row)
            suppliers[smiles] = row_suppliers
        else:
            suppliers[smiles].update(row_suppliers)

    for smiles, survivor in seen.items():
        survivor["supplier"] = ",".join(sorted(suppliers[smiles]))

    removed = len(rows) - len(seen)
    logger.info("Duplicate rows removed: %d", removed)
    if removed == 0:
        logger.info("Nothing to do — already deduplicated.")
        conn.close()
        return

    cursor.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='precursors'"
    )
    create_sql = cursor.fetchone()[0]

    cursor.execute("ALTER TABLE precursors RENAME TO precursors_backup")
    cursor.execute(create_sql)

    cursor.execute("PRAGMA table_info(precursors_backup)")
    columns = [col["name"] for col in cursor.fetchall()]
    insert_cols = [c for c in columns if c != "id"]
    placeholders = ", ".join("?" for _ in insert_cols)
    insert_sql = (
        f"INSERT INTO precursors ({', '.join(insert_cols)}) VALUES ({placeholders})"
    )
    for survivor in seen.values():
        cursor.execute(insert_sql, [survivor[c] for c in insert_cols])

    cursor.execute("DROP TABLE precursors_backup")
    conn.commit()
    conn.close()
    logger.info("Done. %d duplicate(s) removed from %r.", removed, db_path)


# ── Product database ──────────────────────────────────────────────────────────

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
    logger.info("Products database initialised: %s", db_path)