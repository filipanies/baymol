"""Core data structures for BayMol."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Precursor:
    smiles: str
    name: str | None = None
    supplier: str | None = None


@dataclass(frozen=True)
class Product:
    smiles: str
    precursor_a: str
    precursor_b: str
    reaction_name: str
