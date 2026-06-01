import json
import sqlite3

import numpy as np

from baymol.db import init_products_database
from baymol.featurise import (
    NBITS,
    RADIUS,
    _featurise_chunk,
    pack_bits,
    pack_counts,
    populate,
)
from baymol.features import compute_features, morgan_count_fingerprint, morgan_fingerprint

PRODUCTS = [
    ("c1ccccc1", "Suzuki"),
    ("c1ccncc1C#N", "Stille"),
    ("Oc1ccccc1", "Suzuki"),
]


def make_products_db(path, products=PRODUCTS):
    init_products_database(str(path))
    conn = sqlite3.connect(path)
    conn.executemany(
        "INSERT INTO products (product_smiles, precursor_a_smiles, precursor_b_smiles, reaction_name)"
        " VALUES (?, ?, ?, ?)",
        [(smi, "A", "B", rxn) for smi, rxn in products],
    )
    conn.commit()
    conn.close()
    return path


def count(db, table):
    conn = sqlite3.connect(db)
    n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    conn.close()
    return n


# ── fingerprint packing ───────────────────────────────────────────────────────

class TestPacking:
    def test_bits_roundtrip(self):
        bits = morgan_fingerprint("c1ccccc1", RADIUS, NBITS)
        blob = pack_bits(bits)
        assert len(blob) == NBITS // 8
        assert np.unpackbits(np.frombuffer(blob, np.uint8)).tolist() == bits

    def test_counts_roundtrip(self):
        counts = morgan_count_fingerprint("c1ccccc1", RADIUS, NBITS)
        blob = pack_counts(counts)
        assert len(blob) == NBITS * 2
        assert np.frombuffer(blob, np.uint16).tolist() == counts


# ── worker (no pool) ──────────────────────────────────────────────────────────

class TestFeaturiseChunk:
    def test_features_only(self):
        feats, fps = _featurise_chunk([(1, "c1ccccc1", True, False)])
        assert fps == []
        assert len(feats) == 1
        assert feats[0]["product_id"] == 1
        assert feats[0]["heavy_atom_count"] == 6
        assert feats[0]["aromatic_ring_count"] == 1
        assert feats[0]["phenol"] in (0, 1)   # flag flattened to int

    def test_fingerprints_only(self):
        feats, fps = _featurise_chunk([(1, "c1ccccc1", False, True)])
        assert feats == []
        product_id, fp_blob, count_blob = fps[0]
        assert product_id == 1
        assert len(fp_blob) == NBITS // 8
        assert len(count_blob) == NBITS * 2


# ── populate (end-to-end, uses the process pool) ──────────────────────────────

class TestPopulate:
    def test_populates_molecular_features(self, tmp_path):
        db = make_products_db(tmp_path / "p.db")
        populate(str(db), max_workers=2, batch_size=2)
        assert count(db, "molecular_features") == len(PRODUCTS)
        conn = sqlite3.connect(db)
        hac = conn.execute(
            "SELECT mf.heavy_atom_count FROM molecular_features mf"
            " JOIN products p ON p.id = mf.product_id WHERE p.product_smiles = 'c1ccccc1'"
        ).fetchone()[0]
        conn.close()
        assert hac == 6

    def test_molecular_features_values_roundtrip(self, tmp_path):
        db = make_products_db(tmp_path / "p.db", products=[("c1ccccc1C#N", "Suzuki")])
        populate(str(db), max_workers=1, batch_size=1)
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM molecular_features").fetchone()
        conn.close()

        expected = compute_features("c1ccccc1C#N")
        assert row["heavy_atom_count"] == expected["heavy_atom_count"]
        assert json.loads(row["unique_elements"]) == expected["unique_elements"]
        assert row["nitrile"] == 1   # benzonitrile → nitrile flag stored as int

    def test_resume_is_idempotent(self, tmp_path):
        db = make_products_db(tmp_path / "p.db")
        populate(str(db), max_workers=2, batch_size=2)
        populate(str(db), max_workers=2, batch_size=2)   # second run is a no-op
        assert count(db, "molecular_features") == len(PRODUCTS)

    def test_fingerprints_opt_in(self, tmp_path):
        db = make_products_db(tmp_path / "p.db")
        populate(str(db), max_workers=2, batch_size=2)   # features only

        conn = sqlite3.connect(db)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        conn.close()
        assert "fingerprints" not in tables   # not created unless requested

        populate(str(db), fingerprints=True, max_workers=2, batch_size=2)
        assert count(db, "fingerprints") == len(PRODUCTS)
        assert count(db, "molecular_features") == len(PRODUCTS)   # features untouched

    def test_stored_fingerprint_matches_direct(self, tmp_path):
        db = make_products_db(tmp_path / "p.db", products=[("c1ccccc1", "Suzuki")])
        populate(str(db), fingerprints=True, max_workers=1, batch_size=1)
        conn = sqlite3.connect(db)
        fp_blob, count_blob = conn.execute(
            "SELECT morgan_fp, morgan_count_fp FROM fingerprints"
        ).fetchone()
        conn.close()
        assert np.unpackbits(np.frombuffer(fp_blob, np.uint8)).tolist() == \
            morgan_fingerprint("c1ccccc1", RADIUS, NBITS)
        assert np.frombuffer(count_blob, np.uint16).tolist() == \
            morgan_count_fingerprint("c1ccccc1", RADIUS, NBITS)
