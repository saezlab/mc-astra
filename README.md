# mc-ASTRA

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/saezlab/mc-astra/test.yaml?branch=main
[badge-docs]: https://img.shields.io/badge/docs-GitHub%20Pages-blue

## Multicellular factor analysis for tissue-state representations
mc-ASTRA (`mc_astra`) provides a bridge between single-cell data analysis workflows from `scverse`, factor-based models from `MOFA-FLEX`, and prior knowledge to generate tissue-centric descriptions from single-cell data.

This package facilitates the implementation of [Multicellular Factor Analysis](https://elifesciences.org/articles/93161) by providing functions to process and format single-cell data into a multi-view format, together with additional visualization and downstream tasks to analyse and interpret multicellular programs.

## Installation

You need to have Python 3.12 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv].

mc-ASTRA currently targets the development version of MOFA-FLEX.

## Version status

mc-ASTRA is currently pre-1.0. APIs may change between minor versions.

The `0.1.x` series introduces a revised API and is not fully backward-compatible with `0.0.x`.

Install mc-ASTRA from PyPI with:

```bash
pip install mc-astra
```

Or install the latest development version:


```bash
pip install git+https://github.com/saezlab/mc-astra.git@main
```

Import the package as:

```python
import mc_astra
```

## Citation

> Ricardo Omar Ramirez Flores, Jan David Lanzer, Daniel Dimitrov, Britta Velten, Julio Saez-Rodriguez (2023) Multicellular factor analysis of single-cell data for a tissue-centric understanding of disease eLife 12:e93161

[uv]: https://github.com/astral-sh/uv
[tests]: https://github.com/saezlab/mc-astra/actions/workflows/test.yaml
[documentation]: https://saezlab.github.io/mc-astra/
