import sqlite3

import pytest

from baymol.db import (
    deduplicate_precursors,
    init_fingerprints_table,
    init_molecular_features_table,
    init_precursors_database,
    init_products_database,
    merge_precursors,
    save_precursors,
)
from baymol.features import SUBSTRUCTURE_SMARTS
from baymol.reactive_sites import criteria_smarts


def table_columns(db_path, table):
    conn = sqlite3.connect(db_path)
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    conn.close()
    return cols


def table_names(db_path):
    conn = sqlite3.connect(db_path)
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    return names


# ── precursors schema ─────────────────────────────────────────────────────────

class TestInitPrecursors:
    def test_creates_table_with_site_columns(self, tmp_path):
        db = tmp_path / "pre.db"
        init_precursors_database(str(db))
        cols = table_columns(db, "precursors")
        assert {"id", "smiles", "cas", "product_no", "supplier", "created_at"} <= cols
        # one column per reactive site
        assert set(criteria_smarts) <= cols

    def test_idempotent(self, tmp_path):
        db = tmp_path / "pre.db"
        init_precursors_database(str(db))
        init_precursors_database(str(db))  # IF NOT EXISTS — should not raise
        assert "precursors" in table_names(db)


# ── save_precursors ─────────────────────────────────────────────────────────

class TestSavePrecursors:
    def test_insert_and_read_back(self, tmp_path):
        db = tmp_path / "pre.db"
        init_precursors_database(str(db))
        save_precursors(
            [
                {"smiles": "c1ccccc1", "supplier": "S1", "aryl_hal": 0},
                {"smiles": "Brc1ccccc1", "supplier": "S2", "aryl_hal": 1},
            ],
            str(db),
        )
        conn = sqlite3.connect(db)
        rows = conn.execute("SELECT smiles, supplier, aryl_hal FROM precursors ORDER BY id").fetchall()
        conn.close()
        assert rows == [("c1ccccc1", "S1", 0), ("Brc1ccccc1", "S2", 1)]

    def test_empty_is_noop(self, tmp_path):
        db = tmp_path / "pre.db"
        init_precursors_database(str(db))
        save_precursors([], str(db))
        conn = sqlite3.connect(db)
        n = conn.execute("SELECT COUNT(*) FROM precursors").fetchone()[0]
        conn.close()
        assert n == 0

    def test_bad_row_rolls_back_whole_batch(self, tmp_path):
        db = tmp_path / "pre.db"
        init_precursors_database(str(db))
        with pytest.raises(sqlite3.OperationalError):
            save_precursors(
                [
                    {"smiles": "A", "supplier": "S1"},               # valid
                    {"smiles": "B", "supplier": "S2", "nope": 1},    # unknown column → fails
                ],
                str(db),
            )
        # atomic: the valid first row must not have been committed
        conn = sqlite3.connect(db)
        n = conn.execute("SELECT COUNT(*) FROM precursors").fetchone()[0]
        conn.close()
        assert n == 0


# ── deduplicate_precursors ────────────────────────────────────────────────────

class TestDeduplicatePrecursors:
    def test_merges_suppliers_and_dedups(self, tmp_path):
        db = tmp_path / "pre.db"
        init_precursors_database(str(db))
        save_precursors(
            [
                {"smiles": "A", "supplier": "S1"},
                {"smiles": "A", "supplier": "S2"},
                {"smiles": "B", "supplier": "S3"},
            ],
            str(db),
        )
        deduplicate_precursors(str(db))
        conn = sqlite3.connect(db)
        rows = conn.execute("SELECT smiles, supplier FROM precursors ORDER BY smiles").fetchall()
        conn.close()
        assert rows == [("A", "S1,S2"), ("B", "S3")]

    def test_no_duplicates_noop(self, tmp_path):
        db = tmp_path / "pre.db"
        init_precursors_database(str(db))
        save_precursors([{"smiles": "A", "supplier": "S1"}], str(db))
        deduplicate_precursors(str(db))
        conn = sqlite3.connect(db)
        n = conn.execute("SELECT COUNT(*) FROM precursors").fetchone()[0]
        conn.close()
        assert n == 1


# ── merge_precursors ──────────────────────────────────────────────────────────

class TestMergePrecursors:
    def test_merge_keeps_duplicates_then_dedup_collapses(self, tmp_path):
        db1, db2, out = tmp_path / "a.db", tmp_path / "b.db", tmp_path / "out.db"
        for db, supplier in ((db1, "S1"), (db2, "S2")):
            init_precursors_database(str(db))
            save_precursors(
                [
                    {"smiles": "shared", "supplier": supplier},
                    {"smiles": f"only_{supplier}", "supplier": supplier},
                ],
                str(db),
            )
        # merge alone does NOT deduplicate — both copies of "shared" survive
        merge_precursors([str(db1), str(db2)], str(out))
        conn = sqlite3.connect(out)
        assert conn.execute(
            "SELECT COUNT(*) FROM precursors WHERE smiles = 'shared'"
        ).fetchone()[0] == 2
        conn.close()

        # dedup is a separate step that collapses and merges suppliers
        deduplicate_precursors(str(out))
        conn = sqlite3.connect(out)
        rows = dict(conn.execute("SELECT smiles, supplier FROM precursors").fetchall())
        conn.close()
        assert rows["shared"] == "S1,S2"
        assert set(rows) == {"shared", "only_S1", "only_S2"}


# ── products tables ───────────────────────────────────────────────────────────

class TestProductsTables:
    def test_init_products(self, tmp_path):
        db = tmp_path / "prod.db"
        init_products_database(str(db))
        assert {"product_smiles", "precursor_a_smiles", "precursor_b_smiles", "reaction_name"} <= table_columns(db, "products")


class TestFeatureTables:
    def test_molecular_features_columns(self, tmp_path):
        db = tmp_path / "prod.db"
        init_products_database(str(db))
        init_molecular_features_table(str(db))
        cols = table_columns(db, "molecular_features")
        assert {"product_id", "heavy_atom_count", "molecular_weight",
                "unique_elements", "unique_elements_with_counts"} <= cols
        # one column per substructure pattern
        assert set(SUBSTRUCTURE_SMARTS) <= cols

    def test_fingerprints_columns(self, tmp_path):
        db = tmp_path / "prod.db"
        init_products_database(str(db))
        init_fingerprints_table(str(db))
        assert {"product_id", "morgan_fp", "morgan_count_fp"} <= table_columns(db, "fingerprints")
