# BayMol

BayMol is a Python/RDKit toolkit for building synthesis-constrained molecular libraries for AI-guided molecular and materials discovery.

The project grew out of a first use case: discovering candidate organic electron-transport materials for perovskite solar cells. Given a set of purchasable precursors, BayMol detects their reactive motifs, enumerates the products they can form under one-step coupling and condensation reactions, and deduplicates the resulting library — producing synthesis-aware candidate molecules for downstream screening.

The design is reaction- and objective-agnostic: different precursor sets and reaction SMARTS can target other one-step molecular discovery problems.

> **Status — early release.** This version ships **candidate generation** (reactive-site detection, reaction enumeration, deduplication) and **molecular featurisation** (RDKit descriptors, substructure flags, and Morgan fingerprints, stored in SQLite feature tables). Property-prediction and Bayesian-optimisation stages are planned (see [Roadmap](#roadmap)).

## What BayMol does (this release)

- Detects reactive precursor motifs — aryl/alkene halides, boronic acids/esters, stannanes, terminal alkynes, aryl aldehydes, and activated methylene motifs — using SMARTS patterns
- Flags self-polymerisable precursors (those carrying both a halide and a coupling nucleophile) so they can be excluded from the library
- Enumerates candidate products via one-step RDKit reaction SMARTS for the Suzuki, Stille, and Sonogashira couplings and the Knoevenagel condensation
- Deduplicates products by canonical SMILES, merging the precursors and reactions that lead to each product
- Computes molecular features for each product — RDKit scalar descriptors, element composition, boolean substructure flags, and Morgan fingerprints
- Stores features and fingerprints in SQLite tables alongside the products, computed in parallel and resumably

## Roadmap

- Surrogate property prediction (e.g. HOMO/LUMO frontier orbital energies) and active learning
- Property-based filtering to down-select a promising candidate subset
- Ask–tell Bayesian optimisation over the down-selected subset

## Installation

```bash
git clone https://github.com/filipanies/baymol.git
cd baymol
pip install -e .          # core
pip install -e ".[dev]"   # plus pytest + ruff
```

Requires Python ≥ 3.10. Core dependencies: RDKit and NumPy.

## Usage

### Detect reactive sites

```python
from baymol.reactive_sites import count_reactive_sites, is_self_polymerisable

count_reactive_sites("Brc1ccccc1")
# {'aryl_hal': 1, 'alkene_hal': 0, 'aryl_bo2': 0, ...}

# A molecule carrying both a halide and a coupling nucleophile can react
# with itself, so it is excluded from the precursor library:
sites = count_reactive_sites("OB(O)c1ccc(Br)cc1")
is_self_polymerisable(sites)   # True
```

### Enumerate a coupling product

```python
from baymol.reactions import chemical_reaction, SUZUKI

# Suzuki coupling: aryl halide + boronic acid -> biaryl
chemical_reaction(SUZUKI, "Brc1ccccc1", "OB(O)c1ccccc1")
# ['c1ccc(-c2ccccc2)cc1', ...]   biphenyl; symmetry-equivalent
#                                duplicates are collapsed by the dedup step
```

### Batch generation (CLI)

For large libraries, the `reactions` module generates and deduplicates products over a SQLite precursor database:

```bash
python -m baymol.reactions generate --precursors-db precursors.db --products-db products.db
python -m baymol.reactions dedup     --products-db products.db --output-db products_dedup.db
```

> The pipeline that builds the precursor database from raw sources (curation/scraping) is not part of this release.

### Compute molecular features

```python
from baymol.features import compute_features, morgan_fingerprint

compute_features("c1ccncc1C#N")
# {'canonical_smiles': 'N#Cc1cccnc1', 'heavy_atom_count': 8,
#  'aromatic_ring_count': 1, 'substructure_flags': {'nitrile': True, ...}, ...}

morgan_fingerprint("c1ccncc1C#N")   # 2048-bit Morgan fingerprint as a list of 0/1
```

A single molecule can also be featurised from the command line:

```bash
python -m baymol.features "c1ccncc1C#N" --pretty
```

### Featurise a product library (CLI)

Compute and store features for every product in a SQLite database. `molecular_features`
(scalar descriptors, element composition, substructure flags) is always populated; Morgan
fingerprints are opt-in:

```bash
python -m baymol.featurise products.db                 # descriptors, flags, elements
python -m baymol.featurise products.db --fingerprints  # also Morgan fingerprints (slower, larger)
```

Featurisation is resumable — re-running only processes products not yet featurised.

## Development

```bash
pip install -e ".[dev]"
pytest        # run the test suite
ruff check .  # lint
```

## License

Released under the MIT License — see [LICENSE](LICENSE).
