from __future__ import annotations

import numpy as np
import pandas as pd


def _require_plotnine():
    try:
        import plotnine as p9
    except ImportError as e:
        raise ImportError(
            "MuVIcell-derived plotting functions require plotnine. "
            "Install the optional plotting dependencies with:\n\n"
            "    pip install 'mina[spatial]'\n\n"
            "or, in a uv-managed development environment:\n\n"
            "    uv sync --extra spatial"
        ) from e
    return p9


def _ggsave_if(plot, save_path: str | None, width: float, height: float, dpi: int, verbose: bool = False) -> None:
    if save_path:
        p9 = _require_plotnine()
        p9.ggsave(save_path, plot=plot, width=width, height=height, dpi=dpi, verbose=verbose)


def plot_reconstruction(
    stats_df: pd.DataFrame,
    title: str = "Reconstruction R2 by view",
    save_path: str | None = None,
    width: float = 6,
    height: float = 4,
    dpi: int = 300,
):
    p9 = _require_plotnine()
    plot = (
        p9.ggplot(stats_df, p9.aes(x="view", y="R2"))
        + p9.geom_col()
        + p9.theme_classic()
        + p9.theme(axis_text_x=p9.element_text(angle=45, hjust=1))
        + p9.labs(title=title, x="View", y="R2")
    )
    _ggsave_if(plot, save_path, width, height, dpi)
    return plot


def plot_variance_by_view(
    variance_df: pd.DataFrame,
    subtitle: str | None = None,
    save_path: str | None = None,
    width: float = 6,
    height: float = 5,
    dpi: int = 300,
):
    p9 = _require_plotnine()
    row_sums = variance_df.groupby("View", as_index=False)["Variance"].sum()
    row_sums["Factor"] = "Sum"
    col_sums = variance_df.groupby("Factor", as_index=False)["Variance"].sum()
    col_sums["View"] = "Sum"
    sorted_views = row_sums.sort_values("Variance", ascending=False)["View"].tolist()
    factor_levels = list(variance_df["Factor"].cat.categories if isinstance(variance_df["Factor"].dtype, pd.CategoricalDtype) else variance_df["Factor"].unique()) + ["Sum"]
    view_levels = sorted_views + ["Sum"]
    dfm = pd.concat([variance_df, row_sums, col_sums], ignore_index=True)
    dfm["Factor"] = pd.Categorical(dfm["Factor"], categories=factor_levels, ordered=True)
    dfm["View"] = pd.Categorical(dfm["View"], categories=view_levels, ordered=True)
    main = dfm[(dfm["Factor"] != "Sum") & (dfm["View"] != "Sum")].copy()
    rowb = dfm[(dfm["Factor"] == "Sum") & (dfm["View"] != "Sum")].copy()
    colb = dfm[(dfm["View"] == "Sum") & (dfm["Factor"] != "Sum")].copy()
    rowb["bar_length"] = rowb["Variance"] / rowb["Variance"].max()
    colb["bar_length"] = colb["Variance"] / colb["Variance"].max()
    rowb["Variance_label"] = rowb["Variance"].round(2).astype(str)
    colb["Variance_label"] = colb["Variance"].round(2).astype(str)
    fac_no_sum = [value for value in factor_levels if value != "Sum"]
    view_no_sum = [value for value in view_levels if value != "Sum"]
    fpos = {value: idx for idx, value in enumerate(fac_no_sum)}
    vpos = {value: idx for idx, value in enumerate(view_no_sum)}
    main["x"] = main["Factor"].map(fpos)
    main["y"] = main["View"].map(vpos)
    colb["x"] = colb["Factor"].map(fpos)
    rowb["y"] = rowb["View"].map(vpos)
    main["xmin"] = main["x"] - 0.5
    main["xmax"] = main["x"] + 0.5
    main["ymin"] = main["y"] - 0.5
    main["ymax"] = main["y"] + 0.5
    colb["xmin"] = colb["x"] - 0.5
    colb["xmax"] = colb["x"] + 0.5
    colb["ymin"] = len(view_no_sum) - 0.5
    colb["ymax"] = colb["ymin"] + colb["bar_length"]
    rowb["ymin"] = rowb["y"] - 0.5
    rowb["ymax"] = rowb["y"] + 0.5
    rowb["xmin"] = len(fac_no_sum) - 0.5
    rowb["xmax"] = rowb["xmin"] + rowb["bar_length"]
    plot = (
        p9.ggplot()
        + p9.geom_rect(main, p9.aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax", fill="Variance"))
        + p9.geom_rect(colb, p9.aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"), fill="#acabab")
        + p9.geom_text(colb, p9.aes(x="x", y=main["ymax"].max() + 0.2, label="Variance_label"), va="bottom", size=8)
        + p9.geom_rect(rowb, p9.aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"), fill="#acabab")
        + p9.geom_text(rowb, p9.aes(x=main["xmax"].max() + 0.2, y="y", label="Variance_label"), ha="left", size=8)
        + p9.scale_fill_gradientn(colors=["#EFF822", "#CC4977", "#0F0782"])
        + p9.scale_x_continuous(breaks=list(range(len(fac_no_sum))), labels=fac_no_sum)
        + p9.scale_y_continuous(breaks=list(range(len(view_no_sum))), labels=view_no_sum)
        + p9.theme_classic()
        + p9.theme(axis_text_x=p9.element_text(angle=45, hjust=1))
        + p9.labs(title="Variance explained by MINA factors", subtitle=subtitle, x="Factor", y="View")
    )
    _ggsave_if(plot, save_path, width, height, dpi)
    return plot


def plot_featureclass_variance(
    featureclass_df: pd.DataFrame,
    save_path: str | None = None,
    width: float = 5,
    height: float = 5,
    dpi: int = 300,
):
    p9 = _require_plotnine()
    row_sums = featureclass_df.groupby("Feature_type", as_index=False)["Variance"].sum()
    row_sums["Factor"] = "Sum"
    col_sums = featureclass_df.groupby("Factor", as_index=False)["Variance"].sum()
    col_sums["Feature_type"] = "Sum"
    sorted_types = row_sums.sort_values("Variance", ascending=False)["Feature_type"].tolist()
    factor_levels = list(pd.unique(featureclass_df["Factor"])) + ["Sum"]
    type_levels = sorted_types + ["Sum"]
    dfm = pd.concat([featureclass_df, row_sums, col_sums], ignore_index=True)
    dfm["Factor"] = pd.Categorical(dfm["Factor"], categories=factor_levels, ordered=True)
    dfm["Feature_type"] = pd.Categorical(dfm["Feature_type"], categories=type_levels, ordered=True)
    main = dfm[(dfm["Factor"] != "Sum") & (dfm["Feature_type"] != "Sum")].copy()
    rowb = dfm[(dfm["Factor"] == "Sum") & (dfm["Feature_type"] != "Sum")].copy()
    colb = dfm[(dfm["Feature_type"] == "Sum") & (dfm["Factor"] != "Sum")].copy()
    rowb["bar_length"] = rowb["Variance"] / rowb["Variance"].max()
    colb["bar_length"] = colb["Variance"] / colb["Variance"].max()
    rowb["Variance_label"] = rowb["Variance"].round(2).astype(str)
    colb["Variance_label"] = colb["Variance"].round(2).astype(str)
    fac_no_sum = [value for value in factor_levels if value != "Sum"]
    type_no_sum = [value for value in type_levels if value != "Sum"]
    fpos = {value: idx for idx, value in enumerate(fac_no_sum)}
    tpos = {value: idx for idx, value in enumerate(type_no_sum)}
    main["x"] = main["Factor"].map(fpos)
    main["y"] = main["Feature_type"].map(tpos)
    colb["x"] = colb["Factor"].map(fpos)
    rowb["y"] = rowb["Feature_type"].map(tpos)
    main["xmin"] = main["x"] - 0.5
    main["xmax"] = main["x"] + 0.5
    main["ymin"] = main["y"] - 0.5
    main["ymax"] = main["y"] + 0.5
    colb["xmin"] = colb["x"] - 0.5
    colb["xmax"] = colb["x"] + 0.5
    colb["ymin"] = len(type_no_sum) - 0.5
    colb["ymax"] = colb["ymin"] + colb["bar_length"]
    rowb["ymin"] = rowb["y"] - 0.5
    rowb["ymax"] = rowb["y"] + 0.5
    rowb["xmin"] = len(fac_no_sum) - 0.5
    rowb["xmax"] = rowb["xmin"] + rowb["bar_length"]
    plot = (
        p9.ggplot()
        + p9.geom_rect(main, p9.aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax", fill="Variance"))
        + p9.geom_rect(colb, p9.aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"), fill="#acabab")
        + p9.geom_text(colb, p9.aes(x="x", y=main["ymax"].max() + 0.2, label="Variance_label"), va="bottom", size=8)
        + p9.geom_rect(rowb, p9.aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"), fill="#acabab")
        + p9.geom_text(rowb, p9.aes(x=main["xmax"].max() + 0.2, y="y", label="Variance_label"), ha="left", size=8)
        + p9.scale_fill_gradientn(colors=["#EFF822", "#CC4977", "#0F0782"])
        + p9.scale_x_continuous(breaks=list(range(len(fac_no_sum))), labels=fac_no_sum)
        + p9.scale_y_continuous(breaks=list(range(len(type_no_sum))), labels=type_no_sum)
        + p9.theme_classic()
        + p9.theme(axis_text_x=p9.element_text(angle=45, hjust=1))
        + p9.labs(title="Variance explained by MINA factors", x="Factor", y="Feature type")
    )
    _ggsave_if(plot, save_path, width, height, dpi)
    return plot


def plot_top_loadings_heatmap(
    variable_loadings: pd.DataFrame,
    factor: str = "Factor1",
    top_n: int = 30,
    by_abs: bool = True,
    save_path: str | None = None,
    width: float = 5,
    height: float = 5,
    dpi: int = 300,
):
    p9 = _require_plotnine()
    if factor not in variable_loadings.columns:
        raise ValueError(f"Factor column not found: {factor}")
    ranking = variable_loadings[factor].abs() if by_abs else variable_loadings[factor]
    top_vars = variable_loadings.assign(score=ranking).sort_values("score", ascending=False).head(top_n)["variable"].tolist()
    plot_df = variable_loadings[variable_loadings["variable"].isin(top_vars)].loc[:, ["variable", "view", factor]].copy()
    plot = (
        p9.ggplot(plot_df)
        + p9.aes(x="view", y="variable", fill=factor)
        + p9.geom_tile()
        + p9.scale_fill_gradient2(low="#1f77b4", mid="lightgray", high="#c20019", limits=[-1.1, 1.1])
        + p9.theme_classic()
        + p9.theme(axis_text_x=p9.element_text(angle=45, hjust=1))
        + p9.labs(title=factor, x="View", y="Feature", fill="Loading")
        + p9.coord_fixed()
    )
    _ggsave_if(plot, save_path, width, height, dpi)
    return plot


def plot_selected_features(
    selected_df: pd.DataFrame,
    save_path: str | None = None,
    width: float = 6,
    height: float = 5,
    dpi: int = 300,
):
    p9 = _require_plotnine()
    plot_df = selected_df.rename(columns={"variable_view": "Variable_view", "factor": "Factor", "loading": "loading"})
    plot = (
        p9.ggplot(plot_df, p9.aes(x="Factor", y="Variable_view", fill="loading"))
        + p9.geom_tile()
        + p9.scale_fill_gradient2(low="#1f77b4", mid="lightgray", high="#c20019", limits=[-1.1, 1.1])
        + p9.theme_classic()
        + p9.theme(axis_text_x=p9.element_text(angle=45, hjust=1), legend_position="bottom")
        + p9.labs(title="Selected features loadings", x="Factor", y="Feature/View", fill="Loading")
        + p9.coord_fixed()
    )
    _ggsave_if(plot, save_path, width, height, dpi)
    return plot


def plot_factor_violin(
    scores_df: pd.DataFrame,
    factor: str,
    group_col: str,
    palette: list[str] | None = None,
    pvalue: float | None = None,
    save_path: str | None = None,
    width: float = 4.5,
    height: float = 4.5,
    dpi: int = 300,
):
    p9 = _require_plotnine()
    plot = (
        p9.ggplot(scores_df, p9.aes(y=factor, x=group_col, fill=group_col))
        + p9.geom_violin(style="right", scale="width", width=1.25)
        + p9.theme_classic()
        + p9.coord_flip()
        + p9.guides(fill=False)
        + p9.labs(
            title=f"{factor}" + (f" adjusted p = {np.round(pvalue, 5)}" if pvalue is not None else ""),
            x=group_col,
            y=factor,
        )
    )
    if palette is not None:
        plot = plot + p9.scale_fill_manual(values=palette)
    _ggsave_if(plot, save_path, width, height, dpi)
    return plot


def plot_confidence_ellipses(
    scores_df: pd.DataFrame,
    ellipses_df: pd.DataFrame,
    x_factor: str,
    y_factor: str,
    group_col: str,
    palette: list[str] | None = None,
    save_path: str | None = None,
    width: float = 4.5,
    height: float = 4.5,
    dpi: int = 300,
):
    p9 = _require_plotnine()
    ellipse_group_col = group_col if group_col in ellipses_df.columns else "group"
    plot = (
        p9.ggplot(scores_df, p9.aes(x=x_factor, y=y_factor, color=group_col))
        + p9.geom_path(ellipses_df, p9.aes(x="x", y="y", group=ellipse_group_col, color=ellipse_group_col), size=3)
        + p9.theme_classic()
        + p9.ggtitle("Confidence ellipses by group")
        + p9.coord_equal()
    )
    if palette is not None:
        plot = plot + p9.scale_color_manual(values=palette)
    _ggsave_if(plot, save_path, width, height, dpi)
    return plot