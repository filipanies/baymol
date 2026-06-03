import sqlite3

import pytest

from baymol.db import init_products_database
from baymol.predict import predict_properties

PRODUCTS = [
    ("c1ccccc1", "Suzuki"),     # id 1, len 8
    ("Oc1ccccc1", "Suzuki"),    # id 2, len 9
    ("c1ccncc1C#N", "Stille"),  # id 3, len 11
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


def fake_predictor(smiles):
    """Deterministic stand-in for a model: derive numbers from SMILES length."""
    return [(float(len(s)), -float(len(s)), 2.0 * len(s)) for s in smiles]


def _short_predictor(smiles):
    """Returns the wrong number of rows, to exercise the length check."""
    return [(1.0, -1.0, 2.0)]


def rows(db, model=None):
    conn = sqlite3.connect(db)
    if model is None:
        out = conn.execute(
            "SELECT product_id, model FROM predictions ORDER BY product_id, model"
        ).fetchall()
    else:
        out = conn.execute(
            "SELECT product_id, homo_ev, lumo_ev, gap_ev FROM predictions"
            " WHERE model = ? ORDER BY product_id",
            (model,),
        ).fetchall()
    conn.close()
    return out


class TestPredictProperties:
    def test_predicts_all_products(self, tmp_path):
        db = make_products_db(tmp_path / "p.db")
        predict_properties(str(db), fake_predictor, model_name="m1", batch_size=2)
        r = rows(db, "m1")
        assert len(r) == len(PRODUCTS)
        # product 1 is "c1ccccc1" (len 8) → fake predictor yields (8, -8, 16)
        assert r[0] == (1, 8.0, -8.0, 16.0)

    def test_resume_is_idempotent(self, tmp_path):
        db = make_products_db(tmp_path / "p.db")
        predict_properties(str(db), fake_predictor, model_name="m1", batch_size=2)
        predict_properties(str(db), fake_predictor, model_name="m1", batch_size=2)  # no-op
        assert len(rows(db, "m1")) == len(PRODUCTS)

    def test_multiple_models_coexist(self, tmp_path):
        db = make_products_db(tmp_path / "p.db")
        predict_properties(str(db), fake_predictor, model_name="m1", batch_size=10)
        predict_properties(str(db), fake_predictor, model_name="m2", batch_size=10)
        assert len(rows(db, "m1")) == len(PRODUCTS)
        assert len(rows(db, "m2")) == len(PRODUCTS)
        assert len(rows(db)) == len(PRODUCTS) * 2   # one set per model

    def test_only_missing_products_predicted_on_rerun(self, tmp_path):
        db = make_products_db(tmp_path / "p.db", products=[("c1ccccc1", "Suzuki")])
        predict_properties(str(db), fake_predictor, model_name="m1", batch_size=10)
        # add a new product, then re-run: only the new one should be predicted
        conn = sqlite3.connect(db)
        conn.execute(
            "INSERT INTO products (product_smiles, precursor_a_smiles, precursor_b_smiles, reaction_name)"
            " VALUES (?, ?, ?, ?)",
            ("Oc1ccccc1", "A", "B", "Suzuki"),
        )
        conn.commit()
        conn.close()
        predict_properties(str(db), fake_predictor, model_name="m1", batch_size=10)
        assert len(rows(db, "m1")) == 2

    def test_predictor_length_mismatch_raises(self, tmp_path):
        db = make_products_db(tmp_path / "p.db")
        with pytest.raises(ValueError):
            predict_properties(str(db), _short_predictor, model_name="m1", batch_size=10)
