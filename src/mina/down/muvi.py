from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.stats import kendalltau, kruskal
from statsmodels.stats.multitest import multipletests

from .utils import split_by_view


def _nan_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    return np.corrcoef(x[mask], y[mask])[0, 1]


# Supported multiple-testing corrections across factors and the column name
# used for the adjusted values, so the output never mislabels a corrected
# value as a raw p-value.
_CORRECTION_COLUMNS = {"bonferroni": "pvalue_bonferroni", "fdr_bh": "FDR"}


def _adjust_pvalues(pvalues: pd.Series | np.ndarray, method: str) -> np.ndarray:
    """Adjust p-values across factors with ``method``, ignoring NaN entries."""
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.full(p.shape, np.nan)
    mask = np.isfinite(p)
    if mask.any():
        adjusted[mask] = multipletests(p[mask], method=method)[1]
    return adjusted


def _resolve_correction(correction: str | None) -> str | None:
    """Validate and normalise the ``correction`` argument."""
    if correction is None:
        return None
    key = str(correction).lower()
    aliases = {"fdr": "fdr_bh", "bh": "fdr_bh", "none": None}
    key = aliases.get(key, key)
    if key is not None and key not in _CORRECTION_COLUMNS:
        raise ValueError(
            f"correction must be one of {sorted(_CORRECTION_COLUMNS)}, 'fdr', or None; got {correction!r}"
        )
    return key


def _loadings_by_view(model_adata: AnnData) -> dict[str, pd.DataFrame]:
    factor_order = model_adata.var_names.astype(str).tolist()
    if "gene_loadings" not in model_adata.varm:
        raise KeyError("model_adata.varm['gene_loadings'] is missing.")
    if "gene_loadings_columns" not in model_adata.uns:
        raise KeyError("model_adata.uns['gene_loadings_columns'] is missing.")

    loadings = pd.DataFrame(
        model_adata.varm["gene_loadings"],
        index=factor_order,
        columns=list(model_adata.uns["gene_loadings_columns"]),
    )
    if not all(":" in str(col) for col in loadings.columns):
        raise ValueError("gene_loadings_columns must be encoded as 'view:feature'.")
    return split_by_view(loadings)


def factor_scores_info(
    model_adata: AnnData,
    obs_keys: Sequence[str] | None = None,
    factor_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Return factor scores from a MINA model AnnData as a tidy dataframe.

    Parameters
    ----------
    model_adata
        AnnData returned by ``mina.down.model_to_anndata``.
    obs_keys
        Optional observation columns from ``model_adata.obs`` to append.
    factor_names
        Optional subset of factor names to include. Defaults to all factors.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by sample with factor score columns and, when
        requested, selected observation metadata columns.
    """
    factor_index = list(factor_names) if factor_names is not None else model_adata.var_names.astype(str).tolist()
    missing_factors = [factor for factor in factor_index if factor not in model_adata.var_names]
    if missing_factors:
        raise KeyError(f"Unknown factor names: {missing_factors}")

    scores = pd.DataFrame(
        model_adata.X,
        index=model_adata.obs_names.astype(str),
        columns=model_adata.var_names.astype(str),
    ).loc[:, factor_index]

    if obs_keys is None:
        return scores

    missing_obs = [key for key in obs_keys if key not in model_adata.obs.columns]
    if missing_obs:
        raise KeyError(f"Unknown obs columns: {missing_obs}")

    return scores.join(model_adata.obs.loc[:, list(obs_keys)])


def reconstruction_info(
    model_adata: AnnData,
    views: Sequence[str] | None = None,
) -> dict[str, object]:
    """
    Compute per-view reconstruction correlation and $R^2$ from MINA outputs.

    Parameters
    ----------
    model_adata
        AnnData returned by ``mina.down.model_to_anndata``.
    views
        Optional subset of views to analyze. Defaults to all views with both
        stored matrices and loadings.

    Returns
    -------
    dict[str, object]
        Dictionary with ``by_view`` and ``macro`` summaries.
    """
    scores = np.asarray(model_adata.X, dtype=float)
    loadings_by_view = _loadings_by_view(model_adata)
    selected_views = list(views) if views is not None else [view for view in loadings_by_view if view in model_adata.obsm]

    rows = []
    for view in selected_views:
        if view not in loadings_by_view:
            raise KeyError(f"Unknown loadings view: {view}")
        if view not in model_adata.obsm:
            raise KeyError(f"Unknown reconstructed view in model_adata.obsm: {view}")
        observed = np.asarray(model_adata.obsm[view], dtype=float)
        reconstructed = scores @ loadings_by_view[view].to_numpy(dtype=float)
        corr = _nan_pearsonr(observed.ravel(), reconstructed.ravel())
        rows.append({"view": view, "R": corr, "R2": np.nan if pd.isna(corr) else corr * corr})

    by_view = pd.DataFrame(rows)
    macro_r = by_view["R"].mean(skipna=True) if not by_view.empty else np.nan
    macro_r2 = by_view["R2"].mean(skipna=True) if not by_view.empty else np.nan
    return {"by_view": by_view, "macro": {"R": macro_r, "R2": macro_r2}}


def variance_by_view_info(
    model_adata: AnnData,
    *,
    aggregate_groups: bool = True,
    agg: str = "sum",
) -> pd.DataFrame:
    """
    Return explained variance per factor in a long format dataframe.

    Parameters
    ----------
    model_adata
        AnnData returned by ``mina.down.model_to_anndata``.
    aggregate_groups
        Whether to aggregate columns encoded as ``"view:group"`` into a
        single value per ``Factor`` and ``View``.
    agg
        Aggregation applied when ``aggregate_groups`` is True. Supported values
        are ``"sum"``, ``"mean"``, ``"median"``, and ``"max"``.

    Returns
    -------
    pandas.DataFrame
        Long dataframe containing explained variance summaries.
    """
    allowed_aggs = {"sum", "mean", "median", "max"}
    if agg not in allowed_aggs:
        raise ValueError(f"agg must be one of {sorted(allowed_aggs)}")

    factor_order = model_adata.var_names.astype(str).tolist()
    variance = model_adata.var.copy()
    variance.index = factor_order
    variance.index.name = "Factor"

    variance_long = variance.reset_index().melt(
        id_vars="Factor",
        var_name="View_group",
        value_name="Variance",
    )
    view_group = variance_long["View_group"].astype(str).str.split(":", n=1, expand=True)
    variance_long["View"] = view_group[0]
    variance_long["Group"] = view_group[1].fillna("")
    variance_long["Factor"] = pd.Categorical(
        variance_long["Factor"],
        categories=factor_order,
        ordered=True,
    )

    if not aggregate_groups:
        return variance_long.loc[:, ["Factor", "View", "Group", "View_group", "Variance"]]

    grouped = (
        variance_long.groupby(["Factor", "View"], observed=False, as_index=False)["Variance"]
        .agg(agg)
    )
    grouped["Factor"] = pd.Categorical(grouped["Factor"], categories=factor_order, ordered=True)
    return grouped.loc[:, ["Factor", "View", "Variance"]]


def featureclass_variance_info(
    model_adata: AnnData,
    feature_type_map: dict[str, list[str]],
    *,
    aggregator: str = "median",
) -> pd.DataFrame:
    """
    Estimate per-factor reconstruction $R^2$ across feature classes.

    Parameters
    ----------
    model_adata
        AnnData returned by ``mina.down.model_to_anndata``.
    feature_type_map
        Mapping from feature-class names to feature lists.
    aggregator
        Aggregation across views. Supported values are ``"median"`` and
        ``"mean"``.

    Returns
    -------
    pandas.DataFrame
        Long dataframe with columns ``Factor``, ``Feature_type``, and
        ``Variance``.
    """
    if aggregator not in {"median", "mean"}:
        raise ValueError("aggregator must be one of ['mean', 'median']")

    factor_order = model_adata.var_names.astype(str).tolist()
    scores = np.asarray(model_adata.X, dtype=float)
    loadings_by_view = _loadings_by_view(model_adata)
    rows = []

    for feature_type, features in feature_type_map.items():
        per_view = []
        for view, view_loadings in loadings_by_view.items():
            if view not in model_adata.obsm:
                continue
            view_features = [feature for feature in features if feature in view_loadings.columns]
            if not view_features:
                continue
            columns_key = f"{view}_columns"
            if columns_key not in model_adata.uns:
                continue
            observed = pd.DataFrame(
                np.asarray(model_adata.obsm[view], dtype=float),
                index=model_adata.obs_names.astype(str),
                columns=[str(col) for col in model_adata.uns[columns_key]],
            ).loc[:, view_features].to_numpy()
            weights = view_loadings.loc[:, view_features].to_numpy(dtype=float)
            factor_r2 = []
            for idx in range(len(factor_order)):
                reconstructed = scores[:, [idx]] @ weights[[idx], :]
                corr = _nan_pearsonr(observed.ravel(), reconstructed.ravel())
                factor_r2.append(np.nan if pd.isna(corr) else corr * corr)
            per_view.append(np.asarray(factor_r2, dtype=float))

        if not per_view:
            continue

        stacked = np.vstack(per_view)
        aggregated = []
        for idx in range(stacked.shape[1]):
            column = stacked[:, idx]
            finite = column[np.isfinite(column)]
            if finite.size == 0:
                aggregated.append(np.nan)
            elif aggregator == "median":
                aggregated.append(float(np.median(finite)))
            else:
                aggregated.append(float(np.mean(finite)))
        rows.extend(
            {
                "Factor": factor,
                "Feature_type": feature_type,
                "Variance": value,
            }
            for factor, value in zip(factor_order, aggregated, strict=False)
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["Factor", "Feature_type", "Variance"])
    out["Factor"] = pd.Categorical(out["Factor"], categories=factor_order, ordered=True)
    return out


def variable_loadings_info(model_adata: AnnData) -> pd.DataFrame:
    """
    Return model loadings in a wide dataframe annotated by view and variable.

    Parameters
    ----------
    model_adata
        AnnData returned by ``mina.down.model_to_anndata``.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per variable and factor columns containing the
        corresponding loadings.
    """
    if "gene_loadings" not in model_adata.varm:
        raise KeyError("model_adata.varm['gene_loadings'] is missing.")
    if "gene_loadings_columns" not in model_adata.uns:
        raise KeyError("model_adata.uns['gene_loadings_columns'] is missing.")

    factor_order = model_adata.var_names.astype(str).tolist()
    loading_columns = list(model_adata.uns["gene_loadings_columns"])
    if not loading_columns:
        return pd.DataFrame(columns=["view", "variable", *factor_order])
    if not all(":" in str(col) for col in loading_columns):
        raise ValueError("gene_loadings_columns must be encoded as 'view:feature'.")

    loadings = pd.DataFrame(
        model_adata.varm["gene_loadings"],
        index=factor_order,
        columns=loading_columns,
    ).T
    split_cols = loadings.index.to_series().astype(str).str.split(":", n=1, expand=True)
    loadings.insert(0, "variable", split_cols[1].to_numpy())
    loadings.insert(0, "view", split_cols[0].to_numpy())
    return loadings.reset_index(drop=True)


def selected_features_info(
    variable_loadings: pd.DataFrame,
    selections: Sequence[tuple[str, str]],
) -> pd.DataFrame:
    """
    Return loadings for selected features in a long format dataframe.

    Parameters
    ----------
    variable_loadings
        Output of ``variable_loadings_info``.
    selections
        Sequence of ``(variable, view)`` pairs to extract.

    Returns
    -------
    pandas.DataFrame
        Long dataframe with columns ``variable``, ``view``, ``variable_view``,
        ``factor``, and ``loading``.
    """
    factor_cols = [col for col in variable_loadings.columns if str(col).startswith("Factor")]
    if not factor_cols:
        return pd.DataFrame(columns=["variable", "view", "variable_view", "factor", "loading"])

    base = variable_loadings.set_index(["variable", "view"])
    selected_rows = []
    for variable, view in selections:
        if (variable, view) not in base.index:
            continue
        row = base.loc[(variable, view), factor_cols]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        selected_rows.append(
            pd.DataFrame(
                {
                    "variable": [variable] * len(factor_cols),
                    "view": [view] * len(factor_cols),
                    "variable_view": [f"{variable}/{view}"] * len(factor_cols),
                    "factor": factor_cols,
                    "loading": row.to_numpy(),
                }
            )
        )

    if not selected_rows:
        return pd.DataFrame(columns=["variable", "view", "variable_view", "factor", "loading"])

    return pd.concat(selected_rows, ignore_index=True)


def kruskal_info(
    scores_df: pd.DataFrame,
    group_col: str,
    factors: Sequence[str] | None = None,
    *,
    correction: str | None = "bonferroni",
) -> pd.DataFrame:
    """
    Run Kruskal-Wallis tests for factor scores across groups.

    Parameters
    ----------
    scores_df
        Dataframe containing factor scores and grouping metadata.
    group_col
        Column in ``scores_df`` defining the comparison groups.
    factors
        Optional subset of factor columns to test. Defaults to all columns that
        start with ``"Factor"``.
    correction
        Multiple-testing correction applied across the tested factors. One of
        ``"bonferroni"`` (default), ``"fdr_bh"`` (Benjamini-Hochberg, also
        accepted as ``"fdr"``), or ``None`` for uncorrected p-values only.

    Returns
    -------
    pandas.DataFrame
        Dataframe with a ``factor`` column and the raw ``pvalue`` column. When
        a correction is requested, the adjusted values are added in a
        dedicated, clearly named column (``pvalue_bonferroni`` or ``FDR``) so
        corrected values are never mistaken for raw p-values. Sorted by the
        adjusted column when present, otherwise by ``pvalue``.
    """
    if group_col not in scores_df.columns:
        raise KeyError(f"Unknown group column: {group_col}")
    method = _resolve_correction(correction)

    factor_cols = list(factors) if factors is not None else [col for col in scores_df.columns if str(col).startswith("Factor")]
    groups = scores_df[group_col].dropna().unique().tolist()
    rows = []

    for factor in factor_cols:
        if factor not in scores_df.columns:
            raise KeyError(f"Unknown factor column: {factor}")
        samples = [scores_df.loc[scores_df[group_col] == group, factor].dropna().to_numpy() for group in groups]
        samples = [sample for sample in samples if len(sample) > 0]
        if len(samples) < 2:
            pvalue = np.nan
        else:
            _, pvalue = kruskal(*samples)
        rows.append({"factor": factor, "pvalue": pvalue})

    result = pd.DataFrame(rows, columns=["factor", "pvalue"])
    if method is not None:
        adj_col = _CORRECTION_COLUMNS[method]
        result[adj_col] = _adjust_pvalues(result["pvalue"], method)
        return result.sort_values(adj_col, na_position="last").reset_index(drop=True)
    return result.sort_values("pvalue", na_position="last").reset_index(drop=True)


def kendall_info(
    scores_df: pd.DataFrame,
    ordinal_col: str,
    factors: Sequence[str] | None = None,
    *,
    correction: str | None = "bonferroni",
) -> pd.DataFrame:
    """
    Run Kendall tau tests between factor scores and an ordinal variable.

    Parameters
    ----------
    scores_df
        Dataframe containing factor scores and metadata.
    ordinal_col
        Column in ``scores_df`` defining the ordinal variable.
    factors
        Optional subset of factor columns to test. Defaults to all columns that
        start with ``"Factor"``.
    correction
        Multiple-testing correction applied across the tested factors. One of
        ``"bonferroni"`` (default), ``"fdr_bh"`` (Benjamini-Hochberg, also
        accepted as ``"fdr"``), or ``None`` for uncorrected p-values only.

    Returns
    -------
    pandas.DataFrame
        Dataframe with ``factor``, ``tau``, and the raw ``pvalue`` columns. When
        a correction is requested, the adjusted values are added in a
        dedicated, clearly named column (``pvalue_bonferroni`` or ``FDR``) so
        corrected values are never mistaken for raw p-values. Sorted by the
        adjusted column when present, otherwise by ``pvalue``.
    """
    if ordinal_col not in scores_df.columns:
        raise KeyError(f"Unknown ordinal column: {ordinal_col}")
    method = _resolve_correction(correction)

    factor_cols = list(factors) if factors is not None else [col for col in scores_df.columns if str(col).startswith("Factor")]
    rows = []

    for factor in factor_cols:
        if factor not in scores_df.columns:
            raise KeyError(f"Unknown factor column: {factor}")
        paired = scores_df.loc[:, [factor, ordinal_col]].dropna()
        if paired.empty:
            tau, pvalue = np.nan, np.nan
        else:
            ordinal_codes = pd.Categorical(paired[ordinal_col]).codes
            tau, pvalue = kendalltau(paired[factor], ordinal_codes)
        rows.append({"factor": factor, "tau": tau, "pvalue": pvalue})

    result = pd.DataFrame(rows, columns=["factor", "tau", "pvalue"])
    if method is not None:
        adj_col = _CORRECTION_COLUMNS[method]
        result[adj_col] = _adjust_pvalues(result["pvalue"], method)
        return result.sort_values(adj_col, na_position="last").reset_index(drop=True)
    return result.sort_values("pvalue", na_position="last").reset_index(drop=True)


def confidence_ellipses_info(
    scores_df: pd.DataFrame,
    x_factor: str,
    y_factor: str,
    group_col: str,
    *,
    nstd: float = 2.0,
    num_points: int = 100,
) -> pd.DataFrame:
    """
    Compute confidence-ellipse coordinates for factor pairs by group.

    Parameters
    ----------
    scores_df
        Dataframe containing factor scores and grouping metadata.
    x_factor
        Factor column plotted on the x-axis.
    y_factor
        Factor column plotted on the y-axis.
    group_col
        Column in ``scores_df`` defining the groups.
    nstd
        Number of standard deviations used for the ellipse radius.
    num_points
        Number of points sampled along each ellipse.

    Returns
    -------
    pandas.DataFrame
        Dataframe with columns ``x``, ``y``, and ``group``.
    """
    required = [x_factor, y_factor, group_col]
    missing = [column for column in required if column not in scores_df.columns]
    if missing:
        raise KeyError(f"Unknown columns: {missing}")

    rows = []
    for group, sub_df in scores_df.groupby(group_col, observed=False):
        coords = sub_df.loc[:, [x_factor, y_factor]].dropna().to_numpy()
        if coords.shape[0] < 2:
            continue
        center = coords.mean(axis=0)
        cov = np.cov(coords.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
        transform = eigvecs @ np.diag(nstd * np.sqrt(eigvals))
        angles = np.linspace(0.0, 2.0 * np.pi, num_points)
        circle = np.column_stack([np.cos(angles), np.sin(angles)])
        ellipse = circle @ transform.T + center
        rows.append(pd.DataFrame({"x": ellipse[:, 0], "y": ellipse[:, 1], "group": group}))

    if not rows:
        return pd.DataFrame(columns=["x", "y", "group"])

    return pd.concat(rows, ignore_index=True)


def top_features_by_view_info(
    variable_loadings: pd.DataFrame,
    factors: Sequence[str],
    *,
    top_per_view: int = 5,
    by_abs: bool = True,
) -> pd.DataFrame:
    """
    Rank top features per view across selected factors.

    Parameters
    ----------
    variable_loadings
        Output of ``variable_loadings_info``.
    factors
        Factor columns to analyze.
    top_per_view
        Maximum number of features retained per view.
    by_abs
        Whether to rank by absolute loading magnitude.

    Returns
    -------
    pandas.DataFrame
        Dataframe with columns ``variable``, ``view``, ``weight``, and
        ``factor``.
    """
    rows = []
    for factor in factors:
        if factor not in variable_loadings.columns:
            raise KeyError(f"Unknown factor column: {factor}")
        factor_df = variable_loadings.loc[:, ["variable", "view", factor]].copy()
        factor_df["score"] = factor_df[factor].abs() if by_abs else factor_df[factor]
        top = factor_df.sort_values("score", ascending=False).groupby("view", group_keys=False).head(top_per_view)
        top = top.drop(columns="score").rename(columns={factor: "weight"})
        top["factor"] = factor
        rows.append(top)

    if not rows:
        return pd.DataFrame(columns=["variable", "view", "weight", "factor"])

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(by="weight", key=lambda col: col.abs(), ascending=False)
    out = out.drop_duplicates(subset=["variable", "view"])
    return out.groupby("view", group_keys=False).head(top_per_view).reset_index(drop=True)


def top_features_by_class_info(
    variable_loadings: pd.DataFrame,
    feature_types: dict[str, str],
    factors: Sequence[str],
    *,
    top_per_class: int = 5,
    by_abs: bool = True,
) -> pd.DataFrame:
    """
    Rank top features per feature class across selected factors.

    Parameters
    ----------
    variable_loadings
        Output of ``variable_loadings_info``.
    feature_types
        Mapping from variable names to feature-class labels.
    factors
        Factor columns to analyze.
    top_per_class
        Maximum number of features retained per feature class.
    by_abs
        Whether to rank by absolute loading magnitude.

    Returns
    -------
    pandas.DataFrame
        Dataframe with columns ``variable``, ``view``, ``feature_type``,
        ``weight``, and ``factor``.
    """
    rows = []
    for factor in factors:
        if factor not in variable_loadings.columns:
            raise KeyError(f"Unknown factor column: {factor}")
        factor_df = variable_loadings.loc[:, ["variable", "view", factor]].copy()
        factor_df["feature_type"] = factor_df["variable"].map(feature_types).fillna("NA")
        factor_df["score"] = factor_df[factor].abs() if by_abs else factor_df[factor]
        top = factor_df.sort_values("score", ascending=False).groupby("feature_type", group_keys=False).head(top_per_class)
        top = top.drop(columns="score").rename(columns={factor: "weight"})
        top["factor"] = factor
        rows.append(top)

    if not rows:
        return pd.DataFrame(columns=["variable", "view", "feature_type", "weight", "factor"])

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(by="weight", key=lambda col: col.abs(), ascending=False)
    out = out.drop_duplicates(subset=["variable", "view"])
    out = out.groupby("feature_type", group_keys=False).head(top_per_class).reset_index(drop=True)
    return out.loc[:, ["variable", "view", "feature_type", "weight", "factor"]]


def build_selected_anndata(model_adata: AnnData, selection_df: pd.DataFrame) -> AnnData:
    """
    Build an AnnData from selected features stored in ``model_adata.obsm``.

    Parameters
    ----------
    model_adata
        AnnData returned by ``mina.down.model_to_anndata``.
    selection_df
        Dataframe containing either ``variable`` and ``view`` columns or the
        title-case aliases ``Variable`` and ``View``.

    Returns
    -------
    anndata.AnnData
        AnnData with selected features as variables and the original sample
        annotations copied from ``model_adata.obs``.
    """
    rename_map = {}
    if "variable" not in selection_df.columns and "Variable" in selection_df.columns:
        rename_map["Variable"] = "variable"
    if "view" not in selection_df.columns and "View" in selection_df.columns:
        rename_map["View"] = "view"
    selection_df = selection_df.rename(columns=rename_map)

    required = {"variable", "view"}
    missing = required - set(selection_df.columns)
    if missing:
        raise ValueError(f"selection_df is missing required columns: {sorted(missing)}")

    matrices = []
    var_rows = []
    obs_index = model_adata.obs_names.astype(str)
    for row in selection_df.loc[:, ["variable", "view"]].drop_duplicates().itertuples(index=False):
        view = row.view
        variable = row.variable
        if view not in model_adata.obsm:
            raise KeyError(f"Unknown view in model_adata.obsm: {view}")
        columns_key = f"{view}_columns"
        if columns_key not in model_adata.uns:
            raise KeyError(f"Missing feature names in model_adata.uns['{columns_key}']")
        view_columns = [str(col) for col in model_adata.uns[columns_key]]
        if variable not in view_columns:
            raise KeyError(f"Variable '{variable}' not found in view '{view}'")
        column_idx = view_columns.index(variable)
        matrix = np.asarray(model_adata.obsm[view])[:, [column_idx]]
        matrices.append(matrix)
        var_rows.append({"variable": variable, "view": view, "variable_view": f"{variable}/{view}"})

    if not matrices:
        raise ValueError("selection_df does not contain any selectable rows.")

    X = np.hstack(matrices)
    var = pd.DataFrame(var_rows, index=[row["variable_view"] for row in var_rows])
    obs = model_adata.obs.copy()
    obs.index = obs_index
    return AnnData(X=X, obs=obs, var=var)