"""
Reactive site detection: SMARTS patterns, site counting, and precursor filters.
"""

from rdkit import Chem


# ── Reactive site detection ───────────────────────────────────────────────────

def _compile_smarts(patterns: dict[str, str]) -> dict[str, Chem.rdchem.Mol]:
    """Compile a {name: SMARTS} dict into {name: RDKit Mol pattern}."""
    compiled = {}
    for name, smarts in patterns.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            raise ValueError(f"Invalid SMARTS: {name!r} → {smarts!r}")
        compiled[name] = pattern
    return compiled


criteria_smarts: dict[str, str] = {
    "aryl_hal":             "[c][Br,I]",
    "alkene_hal":           "[c,C;^2]=[C][Br,I]",
    "aryl_bo2":             "[c][B](O)O",
    "alkene_bo2":           "[c,C;^2]=[C][B](O)O",
    "aryl_snr3":            "[c][Sn](C)(C)C",
    "alkene_snr3":          "[c,C;^2]=[C][Sn](C)(C)C",
    "terminal_alkyne":      "[C]#[CH]",
    "aryl_aldehyde":        "[c][CH](=[O])",
    "malononitrile_ketone": "[c][C](=[O])[CH2][C](=[C]([C]#[N])[C]#[N])[c]",
    "diketone":             "[c][C](=[O])[CH2][C](=[O])[c]",
}

_criteria_mol: dict[str, Chem.rdchem.Mol] = _compile_smarts(criteria_smarts)


def count_reactive_sites(
    smiles: str,
    categories: dict[str, str] | None = None,
) -> dict[str, int]:
    """Count occurrences of each reactive site pattern in a molecule.

    Args:
        smiles:     Input SMILES string.
        categories: SMARTS pattern dict to use. Defaults to criteria_smarts.
                    Pass a custom dict to count sites for a different reaction set.

    Returns:
        Dict mapping site name → integer count.

    Raises:
        ValueError: If the SMILES string cannot be parsed.
        ValueError: If any SMARTS pattern in a custom categories dict is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    compiled = _criteria_mol if categories is None else _compile_smarts(categories)
    return {
        name: len(mol.GetSubstructMatches(pattern, uniquify=True))
        for name, pattern in compiled.items()
    }


def has_reactive_sites(reactive_sites: dict[str, int]) -> bool:
    """Return True if any reactive site count is greater than zero."""
    return any(v > 0 for v in reactive_sites.values())


def is_self_polymerisable(reactive_sites: dict[str, int]) -> bool:
    """Return True if the molecule carries both a halide and a nucleophilic
    cross-coupling group, meaning it could react with itself.

    Such compounds are excluded from the precursor library to prevent
    uncontrolled self-polymerisation during product generation.
    """
    has_hal = (
        reactive_sites.get("aryl_hal", 0) + reactive_sites.get("alkene_hal", 0)
    ) > 0
    has_nucleophile = (
        reactive_sites.get("aryl_bo2", 0)
        + reactive_sites.get("alkene_bo2", 0)
        + reactive_sites.get("aryl_snr3", 0)
        + reactive_sites.get("alkene_snr3", 0)
    ) > 0
    return has_hal and has_nucleophile