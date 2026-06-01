import sqlite3

import pytest
from rdkit import Chem

from baymol.reactions import (
    SONOGASHIRA,
    STILLE,
    SUZUKI,
    chemical_reaction,
    deduplicate_products,
    init_products_database,
    mol_from_smiles,
    process_row,
    sanitize,
    _couple_n,
)

# Feature columns a precursor row carries (produced by reactive_sites).
FEATURE_COLUMNS = [
    "aryl_hal", "alkene_hal", "aryl_bo2", "alkene_bo2",
    "aryl_snr3", "alkene_snr3", "terminal_alkyne", "aryl_aldehyde",
    "malononitrile_ketone", "diketone",
]


def canon(smiles: str) -> str:
    return Chem.CanonSmiles(smiles)


def make_precursors_db(path, rows):
    """Create a precursors DB with the feature schema and insert `rows`.

    Each row is a dict with at least 'smiles'; missing feature columns default 0.
    """
    cols = ", ".join(f"{c} INTEGER DEFAULT 0" for c in FEATURE_COLUMNS)
    conn = sqlite3.connect(path)
    conn.execute(f"""
        CREATE TABLE precursors (
            id     INTEGER PRIMARY KEY AUTOINCREMENT,
            smiles TEXT NOT NULL,
            {cols}
        )
    """)
    for row in rows:
        keys = ["smiles"] + [c for c in FEATURE_COLUMNS if c in row]
        placeholders = ", ".join("?" for _ in keys)
        conn.execute(
            f"INSERT INTO precursors ({', '.join(keys)}) VALUES ({placeholders})",
            [row[k] for k in keys],
        )
    conn.commit()
    conn.close()
    return path


# ── mol_from_smiles ───────────────────────────────────────────────────────────

class TestMolFromSmiles:
    def test_valid(self):
        assert mol_from_smiles("c1ccccc1") is not None

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            mol_from_smiles("not_a_smiles((")


# ── chemical_reaction ───────────────────────────────────────────────────────

class TestChemicalReaction:
    def test_suzuki_makes_biphenyl(self):
        products = chemical_reaction(SUZUKI, "Brc1ccccc1", "OB(O)c1ccccc1")
        assert canon("c1ccc(-c2ccccc2)cc1") in {canon(p) for p in products}

    def test_stille_makes_biphenyl(self):
        products = chemical_reaction(STILLE, "Brc1ccccc1", "C[Sn](C)(C)c1ccccc1")
        assert canon("c1ccc(-c2ccccc2)cc1") in {canon(p) for p in products}

    def test_sonogashira_makes_alkyne(self):
        products = chemical_reaction(SONOGASHIRA, "Brc1ccccc1", "C#Cc1ccccc1")
        assert canon("C(#Cc1ccccc1)c1ccccc1") in {canon(p) for p in products}

    def test_bad_precursor_returns_empty(self):
        assert chemical_reaction(SUZUKI, "garbage((", "OB(O)c1ccccc1") == []

    def test_incompatible_pair_returns_empty(self):
        # bromobenzene has no boronic acid partner site here
        assert chemical_reaction(SUZUKI, "c1ccccc1", "c1ccccc1") == []


# ── sanitize ──────────────────────────────────────────────────────────────────

class TestSanitize:
    def test_valid_mol_returned(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        assert sanitize([mol]) == ["c1ccccc1"]

    def test_failed_sanitization_skipped(self):
        # Pentavalent carbon: parses only with sanitize=False, then fails sanitize.
        bad = Chem.MolFromSmiles("C(C)(C)(C)(C)C", sanitize=False)
        assert bad is not None
        assert sanitize([bad]) == []

    def test_mix_keeps_only_valid(self):
        good = Chem.MolFromSmiles("c1ccccc1")
        bad = Chem.MolFromSmiles("C(C)(C)(C)(C)C", sanitize=False)
        assert sanitize([good, bad]) == ["c1ccccc1"]


# ── _couple_n ─────────────────────────────────────────────────────────────────

class TestCoupleN:
    def test_double_suzuki_makes_terphenyl(self):
        # 1,4-dibromobenzene + 2x phenylboronic acid → p-terphenyl
        out = _couple_n(SUZUKI, "Brc1ccc(Br)cc1", "OB(O)c1ccccc1", 2)
        assert out is not None
        assert canon(out) == canon("c1ccc(-c2ccc(-c3ccccc3)cc2)cc1")

    def test_failed_step_returns_none(self):
        # constant has no boronic site, so the first coupling fails
        assert _couple_n(SUZUKI, "Brc1ccccc1", "c1ccccc1", 1) is None


# ── process_row ───────────────────────────────────────────────────────────────

class TestProcessRow:
    def test_suzuki_pairing(self, tmp_path):
        db = make_precursors_db(tmp_path / "pre.db", [
            {"smiles": "OB(O)c1ccccc1", "aryl_bo2": 1},   # id 1: boronic acid
            {"smiles": "Brc1ccccc1", "aryl_hal": 1},       # id 2: aryl halide
        ])
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        row_a = dict(conn.execute("SELECT * FROM precursors WHERE id = 1").fetchone())
        conn.close()

        results = process_row(row_a, str(db))
        assert len(results) == 1
        assert canon(results[0]["product_smiles"]) == canon("c1ccc(-c2ccccc2)cc1")
        assert results[0]["precursor_a_smiles"] == "OB(O)c1ccccc1"
        assert results[0]["precursor_b_smiles"] == "Brc1ccccc1"
        assert results[0]["reaction_name"] == "Suzuki"

    def test_stille_pairing_tags_reaction(self, tmp_path):
        db = make_precursors_db(tmp_path / "pre.db", [
            {"smiles": "C[Sn](C)(C)c1ccccc1", "aryl_snr3": 1},
            {"smiles": "Brc1ccccc1", "aryl_hal": 1},
        ])
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        row_a = dict(conn.execute("SELECT * FROM precursors WHERE id = 1").fetchone())
        conn.close()
        results = process_row(row_a, str(db))
        assert len(results) == 1
        assert results[0]["reaction_name"] == "Stille"
        assert canon(results[0]["product_smiles"]) == canon("c1ccc(-c2ccccc2)cc1")

    def test_sonogashira_pairing_tags_reaction(self, tmp_path):
        db = make_precursors_db(tmp_path / "pre.db", [
            {"smiles": "C#Cc1ccccc1", "terminal_alkyne": 1},
            {"smiles": "Brc1ccccc1", "aryl_hal": 1},
        ])
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        row_a = dict(conn.execute("SELECT * FROM precursors WHERE id = 1").fetchone())
        conn.close()
        results = process_row(row_a, str(db))
        assert len(results) == 1
        assert results[0]["reaction_name"] == "Sonogashira"
        assert canon(results[0]["product_smiles"]) == canon("C(#Cc1ccccc1)c1ccccc1")

    def test_knoevenagel_pairing_tags_reaction(self, tmp_path):
        db = make_precursors_db(tmp_path / "pre.db", [
            {"smiles": "O=Cc1ccccc1", "aryl_aldehyde": 1},
            {"smiles": "O=C(CC(=O)c1ccccc1)c1ccccc1", "diketone": 1},
        ])
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        row_a = dict(conn.execute("SELECT * FROM precursors WHERE id = 1").fetchone())
        conn.close()
        results = process_row(row_a, str(db))
        assert len(results) == 1
        assert results[0]["reaction_name"] == "Knoevenagel"

    def test_no_partner_no_products(self, tmp_path):
        db = make_precursors_db(tmp_path / "pre.db", [
            {"smiles": "OB(O)c1ccccc1", "aryl_bo2": 1},
            {"smiles": "c1ccccc1"},  # no reactive sites
        ])
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        row_a = dict(conn.execute("SELECT * FROM precursors WHERE id = 1").fetchone())
        conn.close()
        assert process_row(row_a, str(db)) == []

    def test_requires_connection_or_path(self):
        with pytest.raises(ValueError):
            process_row({"smiles": "c1ccccc1"})


# ── init_products_database / deduplicate_products ─────────────────────────────

class TestProductsDatabase:
    def test_init_creates_table(self, tmp_path):
        db = tmp_path / "out.db"
        init_products_database(str(db))
        conn = sqlite3.connect(db)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(products)")}
        conn.close()
        assert {"product_smiles", "precursor_a_smiles", "precursor_b_smiles", "reaction_name"} <= cols

    def _seed(self, path, rows):
        init_products_database(str(path))
        conn = sqlite3.connect(path)
        conn.executemany(
            "INSERT INTO products"
            " (product_smiles, precursor_a_smiles, precursor_b_smiles, reaction_name)"
            " VALUES (?,?,?,?)",
            rows,
        )
        conn.commit()
        conn.close()

    def test_dedup_merges_all_precursors(self, tmp_path):
        src, out = tmp_path / "p.db", tmp_path / "p_dedup.db"
        self._seed(src, [
            ("P", "X1", "Y1", "Suzuki"), ("P", "X2", "Y2", "Stille"),
            ("P", "X3", "Y3", "Suzuki"),
            ("Q", "Z1", "W1", "Sonogashira"),
        ])
        deduplicate_products(str(src), str(out))
        conn = sqlite3.connect(out)
        rows = conn.execute(
            "SELECT product_smiles, precursor_a_smiles, precursor_b_smiles, reaction_name"
            " FROM products ORDER BY product_smiles"
        ).fetchall()
        conn.close()
        # reaction_name merges as a DISTINCT set: the two Suzukis collapse to one
        assert rows == [
            ("P", "X1,X2,X3", "Y1,Y2,Y3", "Suzuki,Stille"),
            ("Q", "Z1", "W1", "Sonogashira"),
        ]

    def test_dedup_deduplicates_identical_precursors(self, tmp_path):
        src, out = tmp_path / "p.db", tmp_path / "p_dedup.db"
        self._seed(src, [("P", "X1", "Y1", "Suzuki"), ("P", "X1", "Y1", "Suzuki")])
        deduplicate_products(str(src), str(out))
        conn = sqlite3.connect(out)
        rows = conn.execute(
            "SELECT product_smiles, precursor_a_smiles, precursor_b_smiles, reaction_name FROM products"
        ).fetchall()
        conn.close()
        assert rows == [("P", "X1", "Y1", "Suzuki")]

    def test_dedup_no_duplicates(self, tmp_path):
        src, out = tmp_path / "p.db", tmp_path / "p_dedup.db"
        self._seed(src, [("P", "X1", "Y1", "Suzuki"), ("Q", "X2", "Y2", "Stille")])
        deduplicate_products(str(src), str(out))
        conn = sqlite3.connect(out)
        n = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        conn.close()
        assert n == 2
