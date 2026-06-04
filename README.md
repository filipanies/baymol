# BayMol

[![CI](https://github.com/filipanies/baymol/actions/workflows/ci.yml/badge.svg)](https://github.com/filipanies/baymol/actions/workflows/ci.yml)

BayMol is a Python/RDKit toolkit for building synthesis-constrained molecular libraries for AI-guided molecular and materials discovery.

The project grew out of a first use case: discovering candidate organic electron-transport materials for perovskite solar cells. Given a set of purchasable precursors, BayMol detects their reactive motifs, enumerates the products they can form under one-step coupling and condensation reactions, and deduplicates the resulting library, producing synthesis-aware candidate molecules for downstream screening.

The design is reaction- and objective-agnostic: different precursor sets and reaction SMARTS can target other one-step molecular discovery problems.

> **Status — early release.** Ships **candidate generation** (reactive-site detection, reaction enumeration, deduplication), **molecular featurisation** (RDKit descriptors, substructure flags, and Morgan fingerprints, stored in SQLite feature tables), and **property prediction** (HOMO/LUMO/gap via an injectable predictor, with an optional Chemprop adapter behind the `[ml]` extra, as baymol consumes trained models rather than training them). Active-learning and Bayesian optimisation stages are planned (see [Roadmap](#roadmap)).

## What BayMol does (this release)

- Detects reactive precursor motifs using SMARTS patterns; aryl/alkene halides, boronic acids/esters, stannanes, terminal alkynes, aryl aldehydes, and activated methylene motifs 
- Flags self-polymerisable precursors (those carrying both a halide and a coupling nucleophile) so they can be excluded from the library
- Enumerates candidate products via one-step RDKit reaction SMARTS for the Suzuki, Stille, and Sonogashira couplings and the Knoevenagel condensation
- Deduplicates products by canonical SMILES, merging the precursors and reactions that lead to each product
- Computes molecular features for each product: RDKit scalar descriptors, element composition, boolean substructure flags, and Morgan fingerprints
- Stores features and fingerprints in SQLite tables alongside the products, computed in parallel and resumably
- Predicts frontier-orbital properties (HOMO/LUMO/gap) for each product by handing SMILES to an injectable predictor, storing results per model in a `predictions` table (torch-free by default, with an optional Chemprop adapter (`[ml]` extra) for trained models)

## Roadmap

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

### Quickstart

The quickest way to see the whole toolkit in action is the runnable demo. From
100 real precursors (frozen in `examples/example_precursors.csv`) it detects
reactive sites and drops self-polymerisable compounds, enumerates products
across all four reactions, deduplicates them (merging products reachable by more
than one route, e.g. the same molecule via Suzuki *and* Stille coupling),
computes molecular features (descriptors + Morgan fingerprints), and reports
HOMO/LUMO/gap from an OE62+CEPDB10k-trained Chemprop model (real values, frozen
in `examples/example_predictions.csv` so the demo needs no external data, no
PyTorch, core install only):

```bash
python examples/quickstart.py
```

### Library API

The building blocks the quickstart strings together, each usable on its own.

#### Detect reactive sites

```python
from baymol.reactive_sites import count_reactive_sites, is_self_polymerisable

count_reactive_sites("Brc1ccccc1")
# {'aryl_hal': 1, 'alkene_hal': 0, 'aryl_bo2': 0, ...}

# A molecule carrying both a halide and a coupling nucleophile can react
# with itself, so it is excluded from the precursor library:
sites = count_reactive_sites("OB(O)c1ccc(Br)cc1")
is_self_polymerisable(sites)   # True
```

#### Enumerate a coupling product

```python
from baymol.reactions import chemical_reaction, SUZUKI

# Suzuki coupling: aryl halide + boronic acid -> biaryl
chemical_reaction(SUZUKI, "Brc1ccccc1", "OB(O)c1ccccc1")
# ['c1ccc(-c2ccccc2)cc1', ...]
# biphenyl; symmetry-equivalent duplicates are collapsed by the dedup step
```

#### Generate and deduplicate a library (CLI)

For large libraries, the `reactions` module generates and deduplicates products over a SQLite precursor database:

```bash
python -m baymol.reactions generate --precursors-db precursors.db --products-db products.db
python -m baymol.reactions dedup     --products-db products.db --output-db products_dedup.db
```

> The pipeline that builds the precursor database from raw sources (curation/scraping) is not part of this release.

#### Compute features

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

Featurisation also works for a large product database, `molecular_features` (scalar descriptors,
element composition, substructure flags) is always populated, Morgan fingerprints
are opt-in, and the run is resumable:

```bash
python -m baymol.featurise products.db                 # descriptors, flags, elements
python -m baymol.featurise products.db --fingerprints  # also Morgan fingerprints (slower, larger)
```

#### Predict properties

`predict_properties` hands product SMILES to any predictor (`list[str] -> (homo,
lumo, gap)` triples) and stores the results per model, staying torch-free unless
you opt into the Chemprop adapter:

```python
from baymol.predict import predict_properties, load_chemprop_predictor

# Bring your own predictor (SMILES list -> (homo, lumo, gap) triples):
predict_properties("products.db", my_predictor, model_name="my_model")

# ...or load a trained Chemprop model (needs the [ml] extra):
predict_properties("products.db",
                   load_chemprop_predictor("path/to/chemprop_model"),
                   model_name="oe62_cepdb10k")
```

## Development

```bash
pip install -e ".[dev]"
pytest        # run the test suite
ruff check .  # lint
```

## License

Released under the MIT License; see [LICENSE](LICENSE).
