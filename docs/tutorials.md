# Tutorials & case studies

This page is the starting point for learning MINA. The **tutorials** teach the
workflow and individual capabilities step by step, mostly on small or bundled
datasets. The **case studies** apply MINA end to end to real published datasets
and reproduce their analyses.

New to MINA? Start with *Multicellular factor analysis with MOFA*, then pick the
tutorials that match your data and question.

## Getting started

- [Multicellular factor analysis with MOFA](notebooks/GetStarted_MOFA.ipynb) —
  the core workflow: build per-cell-type views, fit a factor model, and
  interpret multicellular programs.

## Guiding and structuring factors

- [Guided by patient covariates (SOFA)](notebooks/GetStarted_SOFA.ipynb) —
  steer factors using sample-level metadata.
- [Across experimental groups](notebooks/GetStarted_groups.ipynb) —
  compare multicellular programs between groups of samples.
- [Functional views](notebooks/GetStarted_FunctionalViews.ipynb) —
  turn expression into pathway/regulatory activity views.
- [Guided by pathway annotations](notebooks/GetStarted_FunctionalViews_guided.ipynb) —
  inform factors with prior knowledge of pathways.

## Spatial data

- [Spatial data](notebooks/GetStarted_spatial.ipynb) —
  add spatial descriptors (e.g. neighbourhood enrichment) as views.
- [Best practices for spatial proteomics](notebooks/BestPractices_SpatialProteomics.ipynb) —
  practical recommendations for QC, normalization, factor selection and
  interpretation on imaging-based proteomics.

## Evaluating and comparing models

- [Evaluating patient maps (patpy)](notebooks/EvaluationOfModels.ipynb) —
  integrate MINA representations with patpy and assess patient maps.

## Coming from MuVIcell

- [From MuVIcell to MINA](notebooks/Example_MuVIcell.ipynb) —
  reproduce the MuVIcell tutorial with MINA, mapping each step to its MINA
  equivalent.

## Case studies

End-to-end analyses on real datasets, reproducing published results with MINA:

- [Spatial proteomics: MIBI colorectal cancer](notebooks/Example_MIBI_CRC.ipynb) —
  build cell-type-stratified and spatial views from a single-cell table and
  learn stage-associated multicellular programs and coordination networks.
- [CRC multicellular factor analysis (MOFACell reproduction)](notebooks/Example_CRC_MOFACell.ipynb) —
  reproduce the colorectal-cancer MOFACell analysis from the precomputed feature
  object.
