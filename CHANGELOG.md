# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## Unreleased

### Added

- MuVIcell-derived downstream and plotting helpers (factor scores, reconstruction, variance summaries, loadings, statistical tests, confidence ellipses, top-feature ranking).
- Documentation examples: reproducing the full MuVIcell MIBI colorectal-cancer spatial-proteomics case study, and a best-practices guide for spatial proteomics.

### Changed

- Documentation navigation reorganised into **Tutorials** (learning-oriented notebooks) and **Case studies** (real-data reproductions); `tutorials.md` now lists every tutorial and case study.
- `mina.pl.plot_variance_by_view()` and `mina.pl.plot_featureclass_variance()` order factor columns by marginal explained variance (descending); `plot_variance_by_view()` defaults its y-axis label to `"Cell type"` (configurable via `ylab`).
- `mina.down.kruskal_info()` and `mina.down.kendall_info()` report multiple-testing correction in a dedicated column (`pvalue_bonferroni` by default) and accept a `correction` parameter (`"bonferroni"`, `"fdr_bh"`, or `None`), keeping the raw `pvalue` column.

### Fixed

- `mina.down.kendall_info()` now drops missing `(factor, ordinal)` pairs before running Kendall tau.

## [0.1.0] - 2026-06-02

### Added

- Preprocessing utilities for filtering views, samples, genes, highly variable genes, and cell-type-specific AnnData objects.
- Multi-view construction helpers for raw count storage, view annotation, merging, functional view conversion, membership matrices, and spatial neighborhood enrichment features.
- Downstream analysis functions for factor associations, explained variance summaries, p-value matrices, gene set loading summaries, information networks, multicellular networks, and factor projection.
- Plotting utilities for view-level coverage, p-value tiles, multicellular functional communication summaries, multicellular networks, feature counts per view, and communication overview plots.
- Tutorial notebooks for MOFA, SOFA-guided factors, experimental groups, functional views, pathway-guided functional views, spatial data, and patpy patient-map evaluation.
- Patpy integration through a precomputed sample representation adapter.
