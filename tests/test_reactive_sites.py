import pytest

from baymol.reactive_sites import (
    count_reactive_sites,
    criteria_smarts,
    has_reactive_sites,
    is_self_polymerisable,
)


# ── count_reactive_sites ──────────────────────────────────────────────────────

class TestCountReactiveSites:

    def test_returns_all_keys(self):
        result = count_reactive_sites("c1ccccc1")
        assert set(result.keys()) == set(criteria_smarts.keys())

    def test_invalid_smiles_raises(self):
        with pytest.raises(ValueError, match="Invalid SMILES"):
            count_reactive_sites("not_a_smiles")

    # aryl halide  ─────────────────────────────────────────────────────────────

    def test_aryl_hal_single(self):
        assert count_reactive_sites("Brc1ccccc1")["aryl_hal"] == 1

    def test_aryl_hal_double(self):
        assert count_reactive_sites("Brc1ccc(Br)cc1")["aryl_hal"] == 2

    def test_aryl_hal_zero_on_vinyl_halide(self):
        # Br on vinyl carbon should not count as aryl_hal
        result = count_reactive_sites("Br/C=C/c1ccccc1")
        assert result["aryl_hal"] == 0
        assert result["alkene_hal"] == 1

    # alkene halide  ───────────────────────────────────────────────────────────

    def test_alkene_hal_single(self):
        assert count_reactive_sites("Br/C=C/c1ccccc1")["alkene_hal"] == 1

    # boronic acid/ester (aryl)  ───────────────────────────────────────────────

    def test_aryl_bo2_single(self):
        assert count_reactive_sites("OB(O)c1ccccc1")["aryl_bo2"] == 1

    def test_aryl_bo2_zero_on_vinyl_boronate(self):
        result = count_reactive_sites("OB(O)/C=C/c1ccccc1")
        assert result["aryl_bo2"] == 0
        assert result["alkene_bo2"] == 1

    # boronic acid/ester (alkene)  ─────────────────────────────────────────────

    def test_alkene_bo2_single(self):
        assert count_reactive_sites("OB(O)/C=C/c1ccccc1")["alkene_bo2"] == 1

    # organotin  ───────────────────────────────────────────────────────────────

    def test_aryl_snr3_single(self):
        assert count_reactive_sites("[Sn](C)(C)(C)c1ccccc1")["aryl_snr3"] == 1

    def test_alkene_snr3_single(self):
        assert count_reactive_sites("[Sn](C)(C)(C)/C=C/c1ccccc1")["alkene_snr3"] == 1

    # boronate ester  ──────────────────────────────────────────────────────────

    def test_aryl_bo2_boronate_ester(self):
        # phenylboronic acid pinacol ester
        assert count_reactive_sites("B1(OC(C)(C)C(C)(C)O1)c1ccccc1")["aryl_bo2"] == 1

    # terminal alkyne  ─────────────────────────────────────────────────────────

    def test_terminal_alkyne_single(self):
        assert count_reactive_sites("C#Cc1ccccc1")["terminal_alkyne"] == 1

    def test_internal_alkyne_not_counted(self):
        # Internal alkyne has no terminal CH — should not match
        assert count_reactive_sites("C(#CC)c1ccccc1")["terminal_alkyne"] == 0

    # aryl aldehyde  ───────────────────────────────────────────────────────────

    def test_aryl_aldehyde_single(self):
        assert count_reactive_sites("O=Cc1ccccc1")["aryl_aldehyde"] == 1

    # Knoevenagel acceptors  ───────────────────────────────────────────────────

    def test_diketone_single(self):
        # 1,3-diphenyl-1,3-propanedione (dibenzoylmethane)
        assert count_reactive_sites("O=C(CC(=O)c1ccccc1)c1ccccc1")["diketone"] == 1

    def test_malononitrile_ketone_single(self):
        # Ph-CO-CH2-C(=C(CN)2)-Ph
        smi = "N#CC(=C(CC(=O)c1ccccc1)c1ccccc1)C#N"
        assert count_reactive_sites(smi)["malononitrile_ketone"] == 1

    # no reactive sites  ───────────────────────────────────────────────────────

    def test_no_sites_all_zero(self):
        result = count_reactive_sites("c1ccccc1")
        assert all(v == 0 for v in result.values())

    # custom patterns  ─────────────────────────────────────────────────────────

    def test_custom_patterns(self):
        result = count_reactive_sites("N#Cc1ccccc1", categories={"nitrile": "[CX2]#N"})
        assert result == {"nitrile": 1}

    def test_custom_patterns_invalid_smarts_raises(self):
        with pytest.raises(ValueError, match="Invalid SMARTS"):
            count_reactive_sites("c1ccccc1", categories={"bad": "!!!"})


# ── has_reactive_sites ────────────────────────────────────────────────────────

class TestHasReactiveSites:

    def test_true_when_one_site(self):
        assert has_reactive_sites({"aryl_hal": 1, "aryl_bo2": 0}) is True

    def test_false_when_all_zero(self):
        assert has_reactive_sites({"aryl_hal": 0, "aryl_bo2": 0}) is False

    def test_true_on_real_molecule(self):
        assert has_reactive_sites(count_reactive_sites("Brc1ccccc1")) is True

    def test_false_on_benzene(self):
        assert has_reactive_sites(count_reactive_sites("c1ccccc1")) is False


# ── is_self_polymerisable ─────────────────────────────────────────────────────

class TestIsSelfPolymerisable:

    def test_halide_only_false(self):
        assert is_self_polymerisable(count_reactive_sites("Brc1ccccc1")) is False

    def test_boronic_acid_only_false(self):
        assert is_self_polymerisable(count_reactive_sites("OB(O)c1ccccc1")) is False

    def test_organotin_only_false(self):
        assert is_self_polymerisable(count_reactive_sites("[Sn](C)(C)(C)c1ccccc1")) is False

    def test_neither_false(self):
        assert is_self_polymerisable(count_reactive_sites("c1ccccc1")) is False

    def test_halide_and_boronic_acid_true(self):
        # 4-bromophenylboronic acid
        assert is_self_polymerisable(count_reactive_sites("OB(O)c1ccc(Br)cc1")) is True

    def test_halide_and_organotin_true(self):
        # (4-bromophenyl)trimethylstannane
        assert is_self_polymerisable(count_reactive_sites("[Sn](C)(C)(C)c1ccc(Br)cc1")) is True

    def test_halide_and_alkyne_false(self):
        # terminal alkyne is not a cross-coupling nucleophile in this scheme,
        # so alkyne + halide must not trigger the self-polymerisation filter
        assert is_self_polymerisable(count_reactive_sites("C#Cc1ccc(Br)cc1")) is False