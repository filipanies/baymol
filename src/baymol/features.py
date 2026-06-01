"""
Compute molecular features from a SMILES string using RDKit.

`compute_features` returns the tabular feature set — scalar descriptors, element
composition, and boolean substructure flags. Morgan fingerprints are produced
separately (`morgan_fingerprint` / `morgan_count_fingerprint`) because they have
different storage (packed BLOBs) and downstream use (similarity / ML vectors).

CLI:
    python -m baymol.features "c1ccncc1C#N"
    python -m baymol.features "c1ccncc1C#N" --fingerprints --pretty
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.rdchem import HybridizationType


# Substructure patterns flagged on each molecule. Edit to match the chemistry
# you care about. These are general medicinal-chemistry groups, independent of
# the cross-coupling patterns in reactive_sites.
SUBSTRUCTURE_SMARTS: dict[str, str] = {
    # Nitrogen
    "nitrile":                        "[CX2]#N",
    "primary_amine":                  "[NX3H2][#6]",
    "secondary_amine":                "[NX3H1]([#6])[#6]",
    "amine_tertiary":                 "[NX3]([#6])([#6])[#6]",
    "amide":                          "[NX3][CX3](=[OX1])[#6]",
    "imide":                          "[CX3](=[OX1])[NX3][CX3](=[OX1])",
    "carbamate":                      "[NX3][CX3](=[OX1])[OX2]",
    "sulfonamide":                    "[NX3][SX4](=[OX1])(=[OX1])",
    "lactam":                         "[NX3;r][CX3;r](=[OX1])",
    "pyridine_like_n":                "[nH0]",
    "pyrimidine":                     "c1ccncn1",
    "pyrazole":                       "c1cn[nH]c1",
    "indole":                         "c1ccc2[nH]ccc2c1",
    # Oxygen
    "aldehyde":                       "[CX3H1](=O)",
    "ketone":                         "[CX3](=[OX1])([#6])[#6]",
    "carboxylic_acid":                "[CX3](=O)[OX2H1]",
    "ester":                          "[CX3](=O)[OX2][#6]",
    "ether":                          "[OD2]([#6])[#6]",
    "alcohol":                        "[OX2H][CX4]",
    "phenol":                         "[OX2H][c]",
    "furan":                          "c1ccoc1",
    # Sulfur
    "thioether":                      "[#16X2]([#6])[#6]",
    "thiophene":                      "c1ccsc1",
    # Halogens
    "fluoro_aryl":                    "[c][F]",
    "aryl_halide":                    "[c][Cl,Br,I]",
    "alkyl_halide":                   "[CX4][F,Cl,Br,I]",
    # Boron
    "boronic_acid":                   "[B]([OX2H])[OX2H]",
    "boronate_ester":                 "[B]([OX2])[OX2]",
    # Unsaturated / aromatic systems
    "alkyne":                         "[CX2]#C",
    "styrene_like":                   "[c][CX3]=[CX3]",
    "allyl_system":                   "[CX3]=[CX3][CX4]",
    "five_membered_heteroaromatic":   "[a;r5;!#6]",
    "six_membered_heteroaromatic":    "[a;r6;!#6]",
    # Saturated N-heterocycles
    "piperidine":                     "[C;r6][N;r6;X3][C;r6]",
    "pyrrolidine":                    "[C;r5][N;r5;X3][C;r5]",
    "morpholine":                     "[N;r6][C;r6][C;r6][O;r6]",
    # Sulfur (oxidised)
    "sulfone":                        "[#16X4](=[OX1])(=[OX1])",
    # Bicyclic N-heteroaromatics
    "quinoline":                      "c1ccc2ncccc2c1",
    "imidazole":                      "c1cnc[nH]1",
}


def _compile_smarts(patterns: dict[str, str]) -> dict[str, Chem.Mol]:
    """Compile a {name: SMARTS} dict into {name: RDKit pattern}, validating each."""
    compiled = {}
    for name, smarts in patterns.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            raise ValueError(f"Invalid SMARTS for {name!r}: {smarts!r}")
        compiled[name] = pattern
    return compiled


_SUBSTRUCTURE_PATTERNS: dict[str, Chem.Mol] = _compile_smarts(SUBSTRUCTURE_SMARTS)


def mol_from_smiles(smiles: str) -> Chem.Mol:
    """Convert a SMILES string to an RDKit Mol, raising ValueError on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return mol


def with_explicit_hydrogens(mol: Chem.Mol) -> Chem.Mol:
    return Chem.AddHs(mol)


# ── Element composition ───────────────────────────────────────────────────────

def ordered_elements(counts: Counter[str]) -> list[str]:
    """Order element symbols as C, H, then the rest alphabetically."""
    ordered: list[str] = []
    if "C" in counts:
        ordered.append("C")
    if "H" in counts:
        ordered.append("H")
    for symbol in sorted(counts):
        if symbol not in {"C", "H"}:
            ordered.append(symbol)
    return ordered


def unique_elements_and_counts(mol_with_h: Chem.Mol) -> tuple[list[str], list[dict[str, int]]]:
    counts: Counter[str] = Counter(atom.GetSymbol() for atom in mol_with_h.GetAtoms())
    ordered = ordered_elements(counts)
    counted = [{"element": symbol, "count": counts[symbol]} for symbol in ordered]
    return ordered, counted


# ── Scalar descriptors ────────────────────────────────────────────────────────

def count_aromatic_rings(mol: Chem.Mol) -> int:
    ring_info = mol.GetRingInfo()
    return sum(
        all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)
        for ring in ring_info.AtomRings()
    )


def count_fused_rings(mol: Chem.Mol) -> int:
    """Count rings fused to at least one other ring (ring-level, not ring-system)."""
    atom_rings = list(mol.GetRingInfo().AtomRings())
    fused: set[int] = set()
    for i in range(len(atom_rings)):
        set_i = set(atom_rings[i])
        for j in range(i + 1, len(atom_rings)):
            if len(set_i.intersection(atom_rings[j])) >= 2:
                fused.add(i)
                fused.add(j)
    return len(fused)


def count_aromatic_atoms(mol: Chem.Mol) -> int:
    return sum(atom.GetIsAromatic() for atom in mol.GetAtoms())


def fraction_hybridization(mol: Chem.Mol, hybridization: HybridizationType) -> float:
    atoms = list(mol.GetAtoms())
    if not atoms:
        return 0.0
    count = sum(atom.GetHybridization() == hybridization for atom in atoms)
    return count / len(atoms)


def compute_substructure_flags(mol: Chem.Mol) -> dict[str, bool]:
    """Boolean presence flag for each pattern in SUBSTRUCTURE_SMARTS."""
    return {
        name: mol.HasSubstructMatch(pattern)
        for name, pattern in _SUBSTRUCTURE_PATTERNS.items()
    }


# ── Morgan fingerprints (kept separate — packed BLOBs / ML vectors) ───────────

def morgan_fingerprint(smiles: str, radius: int = 2, nbits: int = 2048) -> list[int]:
    """Binary Morgan fingerprint as a list of 0/1 ints of length nbits."""
    mol = mol_from_smiles(smiles)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    arr = np.zeros((nbits,), dtype=int)
    DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), arr)
    return arr.tolist()


def morgan_count_fingerprint(smiles: str, radius: int = 2, nbits: int = 2048) -> list[int]:
    """Count Morgan fingerprint as a list of ints of length nbits."""
    mol = mol_from_smiles(smiles)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    sparse_fp = generator.GetCountFingerprint(mol)
    counts = [0] * nbits
    for bit_id, count in sparse_fp.GetNonzeroElements().items():
        counts[bit_id] = count
    return counts


# ── Tabular feature set ───────────────────────────────────────────────────────

def compute_features(smiles: str) -> dict[str, Any]:
    """Compute the tabular feature set for one molecule (no fingerprints).

    Returns canonical SMILES, element composition, scalar descriptors, and
    boolean substructure flags. Use morgan_fingerprint / morgan_count_fingerprint
    for the fingerprint vectors.
    """
    mol = mol_from_smiles(smiles)
    mol_h = with_explicit_hydrogens(mol)
    unique_elements, unique_elements_with_counts = unique_elements_and_counts(mol_h)

    return {
        "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
        "unique_elements": unique_elements,
        "unique_elements_with_counts": unique_elements_with_counts,
        "total_atom_count": mol_h.GetNumAtoms(),
        "heavy_atom_count": mol.GetNumHeavyAtoms(),
        "molecular_weight": Descriptors.MolWt(mol),
        "aromatic_ring_count": count_aromatic_rings(mol),
        "fused_ring_count": count_fused_rings(mol),
        "aromatic_atom_count": count_aromatic_atoms(mol),
        "fraction_sp2": fraction_hybridization(mol, HybridizationType.SP2),
        "fraction_sp": fraction_hybridization(mol, HybridizationType.SP),
        "h_bond_donor_count": rdMolDescriptors.CalcNumLipinskiHBD(mol),
        "h_bond_acceptor_count": rdMolDescriptors.CalcNumLipinskiHBA(mol),
        "rotatable_bond_count": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "substructure_flags": compute_substructure_flags(mol),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute molecular features from a SMILES string."
    )
    parser.add_argument("smiles", help="Input SMILES string")
    parser.add_argument("--radius", type=int, default=2, help="Morgan radius (default: 2)")
    parser.add_argument("--nbits", type=int, default=2048, help="Morgan length in bits (default: 2048)")
    parser.add_argument(
        "--fingerprints", action="store_true",
        help="Also include the Morgan bit and count fingerprints in the output.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    args = parser.parse_args()

    features = compute_features(args.smiles)
    if args.fingerprints:
        features["morgan_fingerprint"] = morgan_fingerprint(args.smiles, args.radius, args.nbits)
        features["morgan_count_fingerprint"] = morgan_count_fingerprint(
            args.smiles, args.radius, args.nbits
        )

    print(json.dumps(features, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()