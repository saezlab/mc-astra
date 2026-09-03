# Tutorials

The vignettes are organized around the main choices in a mc-ASTRA analysis: which model to fit, which tissue descriptors to represent, and how to evaluate or interpret the resulting patient map.

## Using Different Models

<div class="tutorial-grid" markdown>

<div class="tutorial-card" markdown>
### [Core MOFA workflow](notebooks/GetStarted_MOFA.ipynb)
Build an unsupervised multicellular factor model from single-cell data.
</div>

<div class="tutorial-card" markdown>
### [Guided sample-level factors with SOFA](notebooks/GetStarted_SOFA.ipynb)
Use patient covariates to guide factor discovery and interpretation.
</div>

<div class="tutorial-card" markdown>
### [Models across experimental groups](notebooks/GetStarted_groups.ipynb)
Align datasets and compare multicellular programs across groups.
</div>

<div class="tutorial-card" markdown>
### [Pathway-guided factors with MuVI](notebooks/GetStarted_FunctionalViews_guided.ipynb)
Use biological prior knowledge to guide feature-level factors.
</div>

<div class="tutorial-card" markdown>
### [Spatiotemporal models with MEFISTO](notebooks/Spatiotemporal_notebook.ipynb)
Model temporal structure together with spatial tissue descriptors.
</div>

</div>

## Using Different Tissue Descriptors

<div class="tutorial-grid" markdown>

<div class="tutorial-card" markdown>
### [Functional views](notebooks/GetStarted_FunctionalViews.ipynb)
Represent samples through pathway activities and other functional summaries.
</div>

<div class="tutorial-card" markdown>
### [Spatial descriptors](notebooks/GetStarted_spatial.ipynb)
Add neighborhood enrichment and cell-composition views to patient maps.
</div>

<div class="tutorial-card" markdown>
### [Multimodal spatial proteomics in CRC](notebooks/CRC_morpho.ipynb)
Combine marker intensity, morphology, and spatial feature types.
</div>

</div>

## Evaluation and Downstream Analysis

<div class="tutorial-grid" markdown>

<div class="tutorial-card" markdown>
### [Integration with patpy and model evaluation](notebooks/EvaluationOfModels.ipynb)
Evaluate patient maps and connect mc-ASTRA outputs to patpy workflows.
</div>

<div class="tutorial-card" markdown>
### [Patient archetypes](notebooks/PatientArchetypes.ipynb)
Identify and interpret extreme tissue states in multicellular factor space.
</div>

</div>
