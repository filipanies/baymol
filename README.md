# BayMol

BayMol is a Python/RDKit toolkit for building synthesis-constrained molecular libraries for AI-guided molecular and materials discovery.

The project was developed around a first use case: discovering candidate organic electron-transport materials for perovskite solar cells. In that workflow, BayMol generates candidate molecules from purchasable precursors, applies reaction SMARTS, deduplicates products, and computes molecular descriptors suitable for downstream machine-learning workflows.

Although the current configuration targets small-molecule organic semiconductor candidates, the core design is reaction- and objective-agnostic: different precursor sets, SMARTS definitions, and optimisation targets can be used for other one-step molecular discovery problems.

BayMol currently focuses on the candidate-generation, featurisation, and property-prediction stages of a planned ask-tell Bayesian optimisation workflow.

## What BayMol does

- Detects reactive precursor motifs using SMARTS patterns
- Enumerates candidate products from configurable one-step RDKit reaction SMARTS
- Deduplicates products using canonical SMILES
- Computes RDKit molecular descriptors, substructure flags, and Morgan fingerprints
- Produces ML-ready molecular feature tables
- Supports downstream property-prediction and active-learning workflows
- Includes experimental Chemprop-based HOMO/LUMO prediction workflows

## Current use case

The initial BayMol workflow targets organic electron-transport materials for perovskite solar cells.

In this setting, generated molecules can be screened using predicted frontier molecular orbital energies, especially HOMO and LUMO levels, before further computational evaluation or experimental follow-up. The long-term goal is to combine synthesis-constrained library generation, surrogate property prediction, and Bayesian optimisation to propose high-value candidates for synthesis and testing.

## Current status

This repository is being cleaned and refactored from an internal research codebase.

The first public version will focus on a minimal, reproducible pipeline:

```text
example precursors
    ↓
reaction enumeration
    ↓
product deduplication
    ↓
molecular feature generation
    ↓
property prediction / ML-ready output table
