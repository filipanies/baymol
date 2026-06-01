import pytest
from rdkit import Chem

from baymol.features import (
    SUBSTRUCTURE_SMARTS,
    compute_features,
    compute_substructure_flags,
    count_aromatic_rings,
    count_fused_rings,
    mol_from_smiles,
    morgan_count_fingerprint,
    morgan_fingerprint,
    ordered_elements,
)


def canon(smiles: str) -> str:
    return Chem.CanonSmiles(smiles)


# ── mol_from_smiles ───────────────────────────────────────────────────────────

class TestMolFromSmiles:
    def test_valid(self):
        assert mol_from_smiles("c1ccccc1") is not None

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            mol_from_smiles("not_a_smiles((")


# ── compute_features ──────────────────────────────────────────────────────────

class TestComputeFeatures:
    def test_benzonitrile_scalars(self):
        f = compute_features("c1ccccc1C#N")  # benzonitrile, C7H5N
        assert canon(f["canonical_smiles"]) == canon("N#Cc1ccccc1")
        assert f["heavy_atom_count"] == 8
        assert f["total_atom_count"] == 13            # includes explicit H
        assert f["aromatic_ring_count"] == 1
        assert f["fused_ring_count"] == 0
        assert abs(f["molecular_weight"] - 103.12) < 0.5

    def test_element_composition(self):
        f = compute_features("c1ccccc1C#N")
        assert f["unique_elements"] == ["C", "H", "N"]  # C, H first, then alpha
        counts = {d["element"]: d["count"] for d in f["unique_elements_with_counts"]}
        assert counts == {"C": 7, "H": 5, "N": 1}

    def test_flags_present_for_every_pattern(self):
        f = compute_features("c1ccccc1")
        assert set(f["substructure_flags"]) == set(SUBSTRUCTURE_SMARTS)
        assert all(isinstance(v, bool) for v in f["substructure_flags"].values())

    def test_does_not_include_fingerprints(self):
        # fingerprints are deliberately separate from the tabular feature set
        f = compute_features("c1ccccc1")
        assert "morgan_fingerprint" not in f
        assert "morgan_count_fingerprint" not in f

    def test_invalid_smiles_raises(self):
        with pytest.raises(ValueError):
            compute_features("not_valid((")


# ── substructure flags ────────────────────────────────────────────────────────

class TestSubstructureFlags:
    def test_detects_present_group(self):
        flags = compute_substructure_flags(mol_from_smiles("c1ccccc1C#N"))
        assert flags["nitrile"] is True

    def test_absent_group_is_false(self):
        flags = compute_substructure_flags(mol_from_smiles("c1ccccc1"))
        assert flags["nitrile"] is False

    def test_phenol_detected(self):
        flags = compute_substructure_flags(mol_from_smiles("Oc1ccccc1"))
        assert flags["phenol"] is True


# ── ring counting ─────────────────────────────────────────────────────────────

class TestRingCounting:
    def test_benzene(self):
        mol = mol_from_smiles("c1ccccc1")
        assert count_aromatic_rings(mol) == 1
        assert count_fused_rings(mol) == 0

    def test_naphthalene_fused(self):
        mol = mol_from_smiles("c1ccc2ccccc2c1")
        assert count_aromatic_rings(mol) == 2
        assert count_fused_rings(mol) == 2


# ── element ordering ──────────────────────────────────────────────────────────

def test_ordered_elements_c_h_first():
    from collections import Counter
    counts = Counter({"O": 1, "N": 2, "H": 4, "C": 3})
    assert ordered_elements(counts) == ["C", "H", "N", "O"]


# ── Morgan fingerprints ───────────────────────────────────────────────────────

class TestMorganFingerprints:
    def test_bit_fingerprint_shape_and_values(self):
        fp = morgan_fingerprint("c1ccccc1C#N")
        assert len(fp) == 2048
        assert set(fp) <= {0, 1}
        assert sum(fp) > 0

    def test_custom_nbits(self):
        assert len(morgan_fingerprint("c1ccccc1", nbits=512)) == 512

    def test_count_fingerprint(self):
        cfp = morgan_count_fingerprint("c1ccccc1C#N")
        assert len(cfp) == 2048
        assert max(cfp) >= 1

    def test_deterministic(self):
        assert morgan_fingerprint("c1ccccc1C#N") == morgan_fingerprint("c1ccccc1C#N")

    def test_invalid_smiles_raises(self):
        with pytest.raises(ValueError):
            morgan_fingerprint("not_valid((")
