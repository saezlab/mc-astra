# Dependencies
import re
import warnings

import numpy as np
import pandas as pd
import scanpy as sc

# Upstream processing functions


# Generate study metadata from pseudobulks
def extract_metadata_from_obs(obs: pd.DataFrame, groupby: str, sort: bool = False) -> pd.DataFrame:
    """
    Extract group-level metadata from an observation table.

    Only columns with a single unique value per group are retained.

    Parameters
    ----------
    obs : pandas.DataFrame
        Observation metadata (e.g., ``AnnData.obs``).
    groupby : str
        Column used to define groups.
    sort : bool
        Whether to apply natural sorting to group identifiers.

    Returns
    -------
    pandas.DataFrame
        Group-level metadata table.
    """
    stable_cols = []

    for col in obs.columns:
        if col == groupby:
            continue

        # Group values and check if each group has only one unique value
        is_stable = obs.groupby(groupby, observed=False)[col].apply(lambda x: x.dropna().nunique() <= 1).all()

        if is_stable:
            stable_cols.append(col)

    if not stable_cols:
        print("⚠️ No stable columns found other than the group ID.")

    # Now collect the first value from each group for these columns
    metadata = obs.groupby(groupby, observed=False)[stable_cols].first().reset_index()

    metadata = metadata.set_index(groupby, drop=False)

    metadata.index.name = None

    # Sort naturally by extracting numeric part after underscore (if format like 'patient_11')
    if sort:

        def extract_number(x):
            match = re.search(r"(\d+)", x)
            return int(match.group(1)) if match else float("inf")

        metadata = metadata.sort_values(by=groupby, key=lambda col: col.map(extract_number)).reset_index(drop=True)

    return metadata


# From pseudobulk to list of anndatas


def split_anndata_by_celltype(pdata, grouping="cell_type"):
    """
    Split an AnnData object into multiple AnnData objects by cell type.

    Parameters
    ----------
    pdata : anndata.AnnData
        Input AnnData object.
    grouping : str
        Column in ``pdata.obs`` defining cell types.

    Returns
    -------
    dict[str, anndata.AnnData]
        Dictionary mapping cell types to AnnData objects.
    """
    if grouping not in pdata.obs.columns:
        raise ValueError(f"'{grouping}' not found in `pdata.obs`.")

    celltype_adata_dict = {}

    for cell_type in pdata.obs[grouping].unique():
        celltype_adata_dict[cell_type] = pdata[pdata.obs[grouping] == cell_type].copy()

    return celltype_adata_dict


def norm_log(
    anndata_dict,
    target_sum=1e6,
    exclude_highly_expressed=False,
    max_value=None,
    center=True,
    method: str = "log1p_scale",
):
    """
    Normalize AnnData objects in place using one of MINA's supported modes.

    Parameters
    ----------
    anndata_dict : dict[str, anndata.AnnData]
        Dictionary of AnnData objects.
    target_sum : float
        Target total count per sample after normalization.
    exclude_highly_expressed : bool
        Whether to exclude highly expressed genes during normalization.
    max_value : float or None
        Maximum value after scaling to clip outliers.
    center : bool
        Whether to center features during scaling.
    method : {``"log1p_scale"``, ``"zscore"``}
        Normalization mode. ``"log1p_scale"`` reproduces MINA's existing
        total-count normalization followed by log1p and optional scaling.
        ``"zscore"`` applies per-feature z-score normalization directly to the
        current matrix, matching MuVIcell's view-wise centering and scaling.

    Returns
    -------
    None
        The input dictionary is modified in place.
    """
    if method not in {"log1p_scale", "zscore"}:
        raise ValueError("method must be one of ['log1p_scale', 'zscore']")

    for _key, adata in anndata_dict.items():
        if method == "zscore":
            X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X, dtype=float)
            means = np.nanmean(X, axis=0, keepdims=True)
            stds = np.nanstd(X, axis=0, keepdims=True)
            stds = np.where(stds == 0, 1.0, stds)
            adata.X = (X - means) / stds
            continue

        # Step 1: Perform total count normalization
        sc.pp.normalize_total(adata, target_sum=target_sum, exclude_highly_expressed=exclude_highly_expressed)

        # Step 2: Log-transform the normalized data
        sc.pp.log1p(adata)

        # Step 3: Center and scale the data
        if center:
            sc.pp.scale(adata, max_value=max_value)

    # Optionally: Print confirmation that normalization, log-transformation, and scaling are complete
    if method == "zscore":
        print("Per-view z-score normalization complete for all AnnData objects.")
    elif center:
        print(
            f"Normalization, log-transformation, and scaling complete for all AnnData objects with target_sum = {target_sum}."
        )

    else:
        print(f"Normalization and log-transformation complete for all AnnData objects with target_sum = {target_sum}.")


def get_view_info(anndata_dict) -> pd.DataFrame:
    """
    Summarize view dimensions and example feature names.

    Parameters
    ----------
    anndata_dict : dict[str, anndata.AnnData]
        Dictionary mapping view names to AnnData objects.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``view_name``, ``n_obs``, ``n_vars``, and
        ``var_names`` (first five feature names).
    """
    rows = []
    for view_name, view_data in anndata_dict.items():
        rows.append(
            {
                "view_name": view_name,
                "n_obs": view_data.n_obs,
                "n_vars": view_data.n_vars,
                "var_names": list(view_data.var_names[:5]),
            }
        )
    return pd.DataFrame(rows)


def validate_views(anndata_dict) -> dict[str, bool]:
    """
    Validate a MINA-style multiview dictionary before fitting.

    Parameters
    ----------
    anndata_dict : dict[str, anndata.AnnData]
        Dictionary mapping view names to AnnData objects.

    Returns
    -------
    dict[str, bool]
        Validation summary with keys ``has_multiple_views``,
        ``consistent_observations``, ``views_have_variables``, and
        ``consistent_obs_names``.
    """
    obs_counts = [adata.n_obs for adata in anndata_dict.values()]
    obs_name_sets = [tuple(adata.obs_names.astype(str).tolist()) for adata in anndata_dict.values()]
    return {
        "has_multiple_views": len(anndata_dict) >= 2,
        "consistent_observations": len(set(obs_counts)) <= 1,
        "views_have_variables": all(adata.n_vars > 0 for adata in anndata_dict.values()),
        "consistent_obs_names": len(set(obs_name_sets)) <= 1,
    }


def filter_views_qc(
    anndata_dict,
    min_cells_per_gene: int = 3,
    min_genes_per_cell: int = 200,
    max_genes_per_cell: int | None = None,
    view_specific_filters: dict[str, dict] | None = None,
):
    """
    Apply gene and cell QC filters to each view in place.

    Parameters
    ----------
    anndata_dict : dict[str, anndata.AnnData]
        Dictionary mapping view names to AnnData objects.
    min_cells_per_gene : int
        Minimum number of cells expressing a gene.
    min_genes_per_cell : int
        Minimum number of genes detected per cell.
    max_genes_per_cell : int or None
        Optional upper bound on genes detected per cell.
    view_specific_filters : dict[str, dict] or None
        Optional per-view overrides with the same parameter names.

    Returns
    -------
    None
        The dictionary is modified in place.
    """
    for view_name, view_data in anndata_dict.items():
        filters = view_specific_filters.get(view_name, {}) if view_specific_filters else {}
        min_cells = filters.get("min_cells_per_gene", min_cells_per_gene)
        min_genes = filters.get("min_genes_per_cell", min_genes_per_cell)
        max_genes = filters.get("max_genes_per_cell", max_genes_per_cell)

        sc.pp.filter_genes(view_data, min_cells=min_cells)
        sc.pp.filter_cells(view_data, min_genes=min_genes)
        if max_genes is not None:
            sc.pp.filter_cells(view_data, max_genes=max_genes)


def find_highly_variable_genes(
    anndata_dict,
    n_top_genes: int = 2000,
    view_specific_n_genes: dict[str, int] | None = None,
    flavor: str = "seurat",
    fallback_manual: bool = True,
):
    """
    Mark highly variable genes in each view without subsetting.

    Parameters
    ----------
    anndata_dict : dict[str, anndata.AnnData]
        Dictionary mapping view names to AnnData objects.
    n_top_genes : int
        Default number of highly variable genes to mark per view.
    view_specific_n_genes : dict[str, int] or None
        Optional per-view overrides for ``n_top_genes``.
    flavor : str
        Scanpy HVG flavor passed to ``sc.pp.highly_variable_genes``.
    fallback_manual : bool
        Whether to fall back to simple variance ranking when scanpy HVG
        selection fails.

    Returns
    -------
    None
        The dictionary is modified in place.
    """
    for view_name, view_data in anndata_dict.items():
        n_genes = view_specific_n_genes.get(view_name, n_top_genes) if view_specific_n_genes else n_top_genes
        n_genes = min(int(n_genes), int(view_data.n_vars)) if view_data.n_vars else 0
        if n_genes <= 0:
            view_data.var["highly_variable"] = False
            continue

        try:
            sc.pp.highly_variable_genes(
                view_data,
                flavor=flavor,
                n_top_genes=n_genes,
                subset=False,
                inplace=True,
            )
        except Exception as exc:
            if not fallback_manual:
                raise
            warnings.warn(
                f"Could not find highly variable genes for view '{view_name}': {exc}. Marking the top {n_genes} genes by variance.",
                stacklevel=2,
            )
            X = view_data.X.toarray() if hasattr(view_data.X, "toarray") else np.asarray(view_data.X, dtype=float)
            gene_vars = np.var(X, axis=0)
            top_indices = np.argsort(gene_vars)[::-1][:n_genes]
            view_data.var["highly_variable"] = False
            view_data.var.iloc[top_indices, view_data.var.columns.get_loc("highly_variable")] = True
            view_data.var["means"] = np.mean(X, axis=0)
            view_data.var["dispersions"] = gene_vars
            mean_var = float(np.mean(gene_vars)) if np.mean(gene_vars) > 0 else 1.0
            view_data.var["dispersions_norm"] = gene_vars / mean_var


def subset_to_hvg(anndata_dict):
    """
    Subset each view to previously marked highly variable genes.

    Parameters
    ----------
    anndata_dict : dict[str, anndata.AnnData]
        Dictionary mapping view names to AnnData objects.

    Returns
    -------
    None
        The dictionary is modified in place.
    """
    for view_name, view_data in list(anndata_dict.items()):
        if "highly_variable" not in view_data.var.columns:
            warnings.warn(
                f"No highly variable genes found in view '{view_name}'. Run find_highly_variable_genes() first.",
                stacklevel=2,
            )
            continue
        anndata_dict[view_name] = view_data[:, view_data.var["highly_variable"].to_numpy()].copy()


def preprocess_views(
    anndata_dict,
    *,
    filter_views: bool = True,
    normalize: bool = True,
    find_hvg: bool = True,
    subset_hvg: bool = True,
    norm_method: str = "zscore",
    **kwargs,
):
    """
    Run a MuVIcell-style preprocessing pipeline on MINA view dictionaries.

    Parameters
    ----------
    anndata_dict : dict[str, anndata.AnnData]
        Dictionary mapping view names to AnnData objects.
    filter_views : bool
        Whether to run ``filter_views_qc``.
    normalize : bool
        Whether to normalize using ``norm_log``.
    find_hvg : bool
        Whether to mark highly variable genes.
    subset_hvg : bool
        Whether to subset to previously marked highly variable genes.
    norm_method : {``"zscore"``, ``"log1p_scale"``}
        Method passed through to ``norm_log``.
    **kwargs : dict
        Additional keyword arguments forwarded to the selected preprocessing
        helpers.

    Returns
    -------
    None
        The dictionary is modified in place.
    """
    if filter_views:
        filter_views_qc(
            anndata_dict,
            **{
                k: v
                for k, v in kwargs.items()
                if k in {"min_cells_per_gene", "min_genes_per_cell", "max_genes_per_cell", "view_specific_filters"}
            },
        )

    if normalize:
        norm_log(
            anndata_dict,
            method=norm_method,
            **{
                k: v
                for k, v in kwargs.items()
                if k in {"target_sum", "exclude_highly_expressed", "max_value", "center"}
            },
        )

    if find_hvg:
        find_highly_variable_genes(
            anndata_dict,
            **{
                k: v
                for k, v in kwargs.items()
                if k in {"n_top_genes", "view_specific_n_genes", "flavor", "fallback_manual"}
            },
        )

    if subset_hvg:
        subset_to_hvg(anndata_dict)
