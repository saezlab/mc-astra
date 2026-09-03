# mc-ASTRA - multicellular Analysis of Sample Tissue Representations and Associations

<p align="center">
  <img src="docs/_static/img/mc-astra-logo.jpg" alt="mc-ASTRA logo" width="190">
</p>

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/saezlab/mc-astra/test.yaml?branch=main
[badge-docs]: https://img.shields.io/badge/docs-GitHub%20Pages-blue

mc-ASTRA (`mc_astra`, commonly imported as `mca`) is a Python package for building interpretable maps of tissue and sample variability from single-cell and spatial omics data.

It integrates molecular, compositional, and spatial tissue descriptors together with sample-level information, such as clinical variables or technical covariates, to identify the main sources of variation across a collection of tissues. Using flexible factor models and downstream biological interpretation, mc-ASTRA connects these differences to coordinated multicellular programs and changes in tissue organization.

The package provides modular workflows for preprocessing, constructing multi-view tissue representations, fitting and exploring tissue-state maps, and interpreting the multicellular processes underlying them. It integrates with the [`scverse`](https://scverse.org/) ecosystem, uses [`MOFA-FLEX`](https://github.com/bioFAM/mofaflex) for flexible factor modeling, and supports the incorporation of biological and technical prior knowledge.

## Installation

mc-ASTRA currently targets Python 3.12 and 3.13.
mc-ASTRA currently targets the dev branch of MOFA-FLEX: https://github.com/bioFAM/mofaflex.git@main.

Install mc-ASTRA latest development version:

```bash
pip install git+https://github.com/saezlab/mc-astra.git@main
```

PyPI installation coming soon!

Import the package as:

```python
import mc_astra as mca
```

## Citation

> Ricardo Omar Ramirez Flores, Jan David Lanzer, Daniel Dimitrov, Britta Velten, Julio Saez-Rodriguez (2023) Multicellular factor analysis of single-cell data for a tissue-centric understanding of disease eLife 12:e93161

[uv]: https://github.com/astral-sh/uv
[tests]: https://github.com/saezlab/mc-astra/actions/workflows/test.yaml
[documentation]: https://saezlab.github.io/mc-astra/
