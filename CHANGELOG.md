# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Fixed

- `mina.down.kendall_info()` now drops missing `(factor, ordinal)` pairs before running Kendall tau, preventing single missing factor values from turning `tau` and `pvalue` into `NaN` for the whole test.

### Added

- MuVIcell-derived downstream helpers for factor score tables, reconstruction summaries, variance-by-view summaries, feature-class variance summaries, variable loading tables, selected-feature extraction, Kruskal and Kendall tests, confidence ellipse coordinates, and top-feature ranking by view or feature class.
- MuVIcell-derived plotting helpers for reconstruction summaries, variance-by-view tiles, feature-class variance tiles, top-loading heatmaps, selected-feature heatmaps, factor violins, and confidence ellipses.
- Documentation examples: reproduction of the MuVIcell tutorial with MINA (shipping the exported synthetic MuData), a full MIBI colorectal-cancer spatial-proteomics case study (cell-type-stratified pseudobulk, spatial neighbourhood-interaction view, multicellular factors and coordination networks), and a best-practices guide for spatial proteomics.


## [0.1.0] - 2026-06-02

### Added

- Preprocessing utilities for filtering views, samples, genes, highly variable genes, and cell-type-specific AnnData objects.
- Multi-view construction helpers for raw count storage, view annotation, merging, functional view conversion, membership matrices, and spatial neighborhood enrichment features.
- Downstream analysis functions for factor associations, explained variance summaries, p-value matrices, gene set loading summaries, information networks, multicellular networks, and factor projection.
- Plotting utilities for view-level coverage, p-value tiles, multicellular functional communication summaries, multicellular networks, feature counts per view, and communication overview plots.
- Tutorial notebooks for MOFA, SOFA-guided factors, experimental groups, functional views, pathway-guided functional views, spatial data, and patpy patient-map evaluation.
- Patpy integration through a precomputed sample representation adapter.
