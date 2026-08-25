"""Plotting functions for mc-ASTRA upstream and downstream summaries."""

from collections.abc import Mapping, Sequence

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm, to_hex
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from pycirclize import Circos
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

# Plotting


def plot_view_samples(
    anndata_dict, min_samples, table=False, figsize=(5, 5), dpi=100, ax=None, return_fig=False, **kwargs
):
    """
    Quality control plot to assess the quality of the obtained pseudobulk samples.

    Parameters
    ----------
    anndata_dict : dict[str, anndata.AnnData]
        Dictionary mapping view names to AnnData objects.
    min_samples : int
        Minimum number of samples required for a view to be included.
    table : bool
        Whether to return the underlying summary table instead of plotting.
        Default is False.
    figsize : tuple[int, int]
        Size of the figure in inches. Default is (5, 5).
    dpi : int
        Resolution of the figure in dots per inch. Default is 100.
    ax : matplotlib.axes.Axes or None
        Matplotlib Axes object to plot on. If None, a new figure and axes
        are created.
    return_fig : bool
        Whether to return the Figure object. Default is False.
    **kwargs : dict
        Additional keyword arguments passed to ``seaborn.scatterplot``.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The created Figure object if ``return_fig`` is True, otherwise None.
    """
    log_view_counts = []
    view_counts = []
    view_samples = []
    view_names = []
    for x, y in anndata_dict.items():
        df = y.var.copy()

        # Transform to log10
        log_view_counts.append(np.log10(sum(df["total_counts"])))
        view_counts.append(sum(df["total_counts"]))
        view_samples.append(y.shape[0])
        view_names.append(x)

    data = pd.DataFrame(
        list(zip(view_samples, view_counts, log_view_counts, strict=False)),
        columns=["Samples", "Total Counts", "Log Total Counts"],
        index=view_names,
    )

    # Plot
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    ax.grid(zorder=0)
    ax.set_axisbelow(True)
    sns.scatterplot(x="Samples", y="Log Total Counts", hue=data.index, ax=ax, data=data, zorder=1, **kwargs)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, title="View")
    ax.axvline(x=min_samples, c="gray", ls="--")
    ax.set_xlabel("Total samples per view")
    ax.set_ylabel("Log10 total sum of counts")

    if return_fig:
        return fig

    if table:
        return data


def plot_view_genes(anndata_dict, min_genes, table=False, figsize=(5, 5), dpi=100, ax=None, return_fig=False, **kwargs):
    """
    Quality control plot to assess the quality of the obtained pseudobulk samples.

    Parameters
    ----------
    anndata_dict : dict[str, anndata.AnnData]
        Dictionary mapping view names to AnnData objects.
    min_genes : int
        Minimum number of genes required for a view to be included.
    table : bool
        Whether to return the underlying summary table instead of plotting.
        Default is False.
    figsize : tuple[int, int]
        Size of the figure in inches. Default is (5, 5).
    dpi : int
        Resolution of the figure in dots per inch. Default is 100.
    ax : matplotlib.axes.Axes or None
        Matplotlib Axes object to plot on. If None, a new figure and axes
        are created.
    return_fig : bool
        Whether to return the Figure object. Default is False.
    **kwargs : dict
        Additional keyword arguments passed to ``seaborn.scatterplot``.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The created Figure object if ``return_fig`` is True, otherwise None.
    """
    # Extract obs
    log_view_counts = []
    view_counts = []
    view_genes = []
    view_names = []
    for x, y in anndata_dict.items():
        df = y.var.copy()

        # Transform to log10
        log_view_counts.append(np.log10(sum(df["total_counts"])))
        view_counts.append(sum(df["total_counts"]))
        view_genes.append(y.shape[1])
        view_names.append(x)

    data = pd.DataFrame(
        list(zip(view_genes, view_counts, log_view_counts, strict=False)),
        columns=["Genes", "Total Counts", "Log Total Counts"],
        index=view_names,
    )

    # Plot
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    ax.grid(zorder=0)
    ax.set_axisbelow(True)
    sns.scatterplot(x="Genes", y="Log Total Counts", hue=data.index, ax=ax, data=data, zorder=1, **kwargs)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, title="View")
    ax.axvline(x=min_genes, c="gray", ls="--")
    ax.set_xlabel("Total genes per view")
    ax.set_ylabel("Log10 total sum of counts")

    if return_fig:
        return fig

    if table:
        return data


def plot_sample_coverage(
    anndata_dict, threshold, proportion, table=False, figsize=(5, 5), dpi=100, return_fig=False, **kwargs
):
    """
    Visualize sample coverage for each AnnData view.

    Samples below the requested proportion threshold are highlighted. One
    figure is produced per dictionary key.

    Parameters
    ----------
    anndata_dict : dict[str, anndata.AnnData]
        Dictionary mapping view names to AnnData objects.
    threshold : float or dict[str, float]
        Gene expression threshold. If a dict, must contain all keys of
        ``anndata_dict``.
    proportion : float or dict[str, float]
        Minimum proportion of genes above ``threshold``. If a dict, must
        contain all keys.
    table : bool
        If True, return summary tables instead of plotting.
        Default is False.
    figsize : tuple[int, int]
        Figure size per subplot. Default is (5, 5).
    dpi : int
        Figure resolution in dots per inch. Default is 100.
    return_fig : bool
        If True, return the generated Figure objects. Default is False.
    **kwargs : dict
        Additional keyword arguments passed to ``matplotlib.axes.Axes.scatter``.

    Returns
    -------
    dict[str, pandas.DataFrame] or dict[str, matplotlib.figure.Figure] or None
        Summary tables if ``table`` is True, figures if ``return_fig`` is True,
        otherwise None.
    """
    # Validate dict-style thresholds if provided
    if isinstance(threshold, dict):
        missing = set(anndata_dict.keys()) - set(threshold.keys())
        if missing:
            raise KeyError(f"'threshold' missing keys: {sorted(missing)}")
    if isinstance(proportion, dict):
        missing = set(anndata_dict.keys()) - set(proportion.keys())
        if missing:
            raise KeyError(f"'proportion' missing keys: {sorted(missing)}")

    tables = {}
    figs = {}

    for key, adata in anndata_dict.items():
        th = threshold[key] if isinstance(threshold, dict) else threshold
        prop = proportion[key] if isinstance(proportion, dict) else proportion

        counts = adata.X

        # Count genes > threshold per sample, robust to sparse/dense
        if hasattr(counts, "toarray") or str(type(counts)).endswith("spmatrix'>"):
            # Sparse path
            num_genes_above = np.asarray((counts > th).sum(axis=1)).ravel()
        else:
            # Dense path
            num_genes_above = np.sum(counts > th, axis=1)
            num_genes_above = np.asarray(num_genes_above).ravel()

        total_genes = counts.shape[1]
        prop_above = num_genes_above / float(total_genes)

        data = pd.DataFrame(
            {
                "Genes Above Threshold": num_genes_above,
                "Proportion Above Threshold": prop_above,
            },
            index=adata.obs.index,
        )
        tables[key] = data

        if table:
            continue

        # Create per-key figure/axis (like your old usage)
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        ax.grid(zorder=0)
        ax.set_axisbelow(True)

        below = data[data["Proportion Above Threshold"] < prop]
        above = data[data["Proportion Above Threshold"] >= prop]

        # Plot above-threshold samples in neutral gray (no legend)
        ax.scatter(
            above["Genes Above Threshold"],
            above["Proportion Above Threshold"],
            color="gray",
            s=40,
            zorder=1,
            label=None,
            **kwargs,
        )

        # Plot below-threshold samples individually (unique colors + legend)
        cmap = plt.get_cmap("tab10")
        for i, (sample, row) in enumerate(below.iterrows()):
            c = cmap(i % 10)
            ax.scatter(
                row["Genes Above Threshold"],
                row["Proportion Above Threshold"],
                color=c,
                s=70,
                zorder=2,
                label=sample,
                **kwargs,
            )

        # Threshold line
        ax.axhline(y=prop, c="gray", ls="--")

        # Labels & legend (only for below-threshold points)
        ax.set_xlabel("Genes Above Threshold")
        ax.set_ylabel("Proportion Above Threshold")
        if not below.empty:
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                frameon=False,
                title="Below threshold",
            )

        # Title like your old suptitle usage
        fig.suptitle(f"{key}", fontsize=14, fontweight="bold")
        figs[key] = fig

    if table:
        return tables
    if return_fig:
        return figs


# Downstream plotting functions

# Associations


def plot_pval_tiles(p_df: pd.DataFrame, star_threshold: float = 0.05, ax=None, title: str | None = None):
    """
    Create a tile plot of ``-log10(p)`` values.

    Parameters
    ----------
    p_df : pandas.DataFrame
        DataFrame of p-values with rows and columns defining the tile grid.
    star_threshold : float
        P-value threshold for star annotation. Default is 0.05.
    ax : matplotlib.axes.Axes or None
        Axes to draw on. If None, a new figure and axes are created.
    title : str or None
        Optional title for the plot.

    Returns
    -------
    axes : matplotlib.axes.Axes or tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Existing axes when ``ax`` is provided, otherwise the created figure and axes.
    """
    # Copy to avoid modifying the input
    p = p_df.copy()

    # Handle zeros or non-positive values to avoid -log10 issues
    # Replace any p <= 0 with the smallest positive float
    min_positive = np.nextafter(0, 1)
    p = p.mask(p <= 0, min_positive)

    # Compute -log10(p)
    neglog10 = -np.log10(p.astype(float))

    # Prepare axes
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, 0.6 * neglog10.shape[1]), max(3.5, 0.6 * neglog10.shape[0])))
        created_fig = True

    # Plot tiles using imshow (matplotlib default colormap)
    im = ax.imshow(neglog10.values, aspect="auto")

    # Ticks & tick labels
    ax.set_xticks(range(neglog10.shape[1]))
    ax.set_yticks(range(neglog10.shape[0]))
    ax.set_xticklabels(neglog10.columns, rotation=45, ha="right")
    ax.set_yticklabels(neglog10.index)

    # Grid lines (optional, light)
    ax.set_xticks(np.arange(-0.5, neglog10.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, neglog10.shape[0], 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("-log10(p-value)")

    # Annotate stars for significant p-values
    star_mask = p.values <= star_threshold
    for i in range(star_mask.shape[0]):
        for j in range(star_mask.shape[1]):
            if star_mask[i, j]:
                ax.text(j, i, "★", ha="center", va="center")

    if title:
        ax.set_title(title)

    plt.tight_layout()
    if created_fig:
        return fig, ax
    return ax


# Functional enrichment


def plot_mcell_funcomics(
    result_dict: dict[str, dict[str, pd.DataFrame]],
    result_key: str = "pw_acts",
    pval_key: str = "pw_padj",
    p_threshold: float = 0.05,
    top_n: int = 10,
    cmap: str = "coolwarm",
    figsize: tuple = (14, 5),
    ytick_rotation: int = 0,
    use_var: bool = False,
    share_color_scale: bool = True,
    center: float | None = 0.0,
):
    """
    Plot grouped heatmaps per view using a selected result matrix.

    Features are filtered by adjusted p-value and ranked either by
    mean absolute value or variance.

    Parameters
    ----------
    result_dict : dict[str, dict[str, pandas.DataFrame]]
        Output of ``run_ulm_per_view`` with one entry per view.
    result_key : str
        Key within each view result containing values to plot.
    pval_key : str
        Key within each view result containing adjusted p-values.
    p_threshold : float
        Adjusted p-value significance threshold.
    top_n : int
        Number of top significant features per view to display.
    cmap : str
        Colormap for the heatmaps.
    figsize : tuple[int, int]
        Overall figure size.
    ytick_rotation : int
        Rotation angle for y-axis tick labels.
    use_var : bool
        If True, rank features by variance instead of mean absolute value.
    share_color_scale : bool
        Whether all heatmaps use a shared color scale.
    center : float or None
        Center value for diverging color scaling. If None, no center is used.

    Returns
    -------
    None
        The function displays the plot and does not return an object.
    """
    views = []
    filtered_data = {}
    significance_masks = {}

    # Step 1: collect views that pass p-value filtering
    for view, result in result_dict.items():
        data = result[result_key]  # factors × features
        pvals = result[pval_key]  # factors × features

        sig_mask = (pvals < p_threshold).any(axis=0)
        sig_features = sig_mask[sig_mask].index

        if len(sig_features) == 0:
            continue

        # Rank by mean(abs) or variance
        if use_var:
            feature_score = data[sig_features].var(axis=0)
        else:
            feature_score = data[sig_features].abs().mean(axis=0)

        top_features = feature_score.sort_values(ascending=False).head(top_n).index
        filtered_data[view] = data[top_features]
        significance_masks[view] = pvals[top_features] < p_threshold
        views.append(view)

    n_views = len(views)
    if n_views == 0:
        print("No views with significant features found.")
        return

    # Global color scale if requested
    if share_color_scale:
        all_vals = pd.concat(filtered_data.values(), axis=1)
        values = all_vals.to_numpy().ravel()
        values = values[np.isfinite(values)]

        if center is None:
            vmin, vmax = np.nanmin(values), np.nanmax(values)
        else:
            max_abs = np.nanmax(np.abs(values - center))
            vmin, vmax = center - max_abs, center + max_abs
    else:
        vmin, vmax = None, None

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, n_views, wspace=0.4)

    for i, view in enumerate(views):
        plot_data = filtered_data[view]

        # Per-view centered color scale if not sharing scale
        if not share_color_scale:
            values = plot_data.to_numpy().ravel()
            values = values[np.isfinite(values)]

            if center is None:
                view_vmin, view_vmax = np.nanmin(values), np.nanmax(values)
            else:
                max_abs = np.nanmax(np.abs(values - center))
                view_vmin, view_vmax = center - max_abs, center + max_abs
        else:
            view_vmin, view_vmax = vmin, vmax

        ax = fig.add_subplot(gs[i])
        sns.heatmap(
            plot_data,
            cmap=cmap,
            vmin=view_vmin,
            vmax=view_vmax,
            center=center,
            cbar=False,  # suppress individual colorbars
            ax=ax,
            xticklabels=True,
            yticklabels=(i == 0),
        )

        sig_mask = significance_masks[view].to_numpy()
        for row_idx, col_idx in np.argwhere(sig_mask):
            ax.text(col_idx + 0.5, row_idx + 0.5, "★", ha="center", va="center")

        ax.set_title(view, fontsize=10)
        ax.tick_params(axis="x", labelsize=7, rotation=90)
        ax.tick_params(axis="y", labelsize=7, rotation=ytick_rotation)

        if i > 0:
            ax.set_ylabel("")

    # Shared colorbar
    if share_color_scale:
        plt.tight_layout(rect=[0, 0, 0.9, 1])

        cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.5])
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, label=result_key)
    else:
        plt.tight_layout()

    plt.show()


# Multicellular information networks


def plot_mcell_network(
    df: pd.DataFrame,
    weight_col: str = "coef",
    abs_cutoff: float = 0.0,
    keep_negative: bool = True,
    edge_width_range: tuple = (0.8, 6),
    node_size: int = 1100,
    arrowsize: int = 18,
    reciprocal_curvature: float = 0.25,
    default_curvature: float = 0.04,
    positive_color: str = "tab:purple",
    negative_color: str = "tab:red",
    show_edge_labels: bool = False,
    label_fmt: str = "{:.2f}",
    title: str | None = None,
    save_path: str | None = None,
    edge_margin_factor: float = 0.55,
    arrows_on_top: bool = True,
):
    """
    Plot an inferred multicellular information network.

    The results are shown solely from one subset (positive or negative loadings).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame defining directed edges. Must contain at least source,
        target, and edge weight columns.
    weight_col : str
        Column name containing edge weights. Default is "coef".
    abs_cutoff : float
        Minimum absolute weight required to keep an edge.
    keep_negative : bool
        Whether to retain negatively weighted edges.
    edge_width_range : tuple[float, float]
        Minimum and maximum edge widths used for scaling.
    node_size : int
        Size of network nodes.
    arrowsize : int
        Size of arrow heads.
    reciprocal_curvature : float
        Curvature used for reciprocal edges.
    default_curvature : float
        Curvature used for non-reciprocal edges.
    positive_color : str
        Color for positively weighted edges.
    negative_color : str
        Color for negatively weighted edges.
    show_edge_labels : bool
        Whether to display edge weight labels.
    label_fmt : str
        Format string used for edge labels.
    title : str or None
        Optional plot title.
    save_path : str or None
        If provided, save the figure to this path.
    edge_margin_factor : float
        Factor controlling spacing between nodes and edges.
    arrows_on_top : bool
        Whether arrows are drawn above nodes.

    Returns
    -------
    graph : networkx.DiGraph
        Directed graph built from the filtered network table.
    """
    required_cols = {"target", "predictor", weight_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    d = df[["target", "predictor", weight_col]].copy()
    d = d.dropna(subset=["target", "predictor", weight_col])
    d["weight"] = d[weight_col].astype(float)
    d = d[np.abs(d["weight"]) >= float(abs_cutoff)]
    if not keep_negative:
        d = d[d["weight"] >= 0]

    G = nx.DiGraph()
    for _, r in d.iterrows():
        G.add_edge(r["predictor"], r["target"], weight=float(r["weight"]))

    if G.number_of_edges() == 0:
        plt.figure(figsize=(6, 4))
        plt.axis("off")
        plt.text(0.5, 0.5, "No edges after filtering.", ha="center", va="center")
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=200)
        plt.show()
        return G

    pos = nx.circular_layout(G)

    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.set_axis_off()
    node_coll = nx.draw_networkx_nodes(
        G, pos, node_color="#E9ECF6", node_size=node_size, edgecolors="#D2D6EA", linewidths=1.2, ax=ax
    )
    node_coll.set_zorder(2)
    label_dict = nx.draw_networkx_labels(G, pos, font_size=11, font_weight="bold", ax=ax)
    for t in label_dict.values():
        t.set_zorder(3)

    edges = list(G.edges(data=True))
    weights = np.array([abs(a["weight"]) for _, _, a in edges], dtype=float)
    if weights.max() == weights.min():
        widths = np.full_like(weights, np.mean(edge_width_range), dtype=float)
    else:
        wmin, wmax = map(float, edge_width_range)
        widths = wmin + (weights - weights.min()) * (wmax - wmin) / (weights.max() - weights.min())

    colors = [positive_color if a["weight"] >= 0 else negative_color for _, _, a in edges]

    reciprocals = {tuple(sorted((u, v))) for u, v in G.edges() if G.has_edge(v, u)}
    curvatures = []
    for u, v, _ in edges:
        if tuple(sorted((u, v))) in reciprocals:
            curv = reciprocal_curvature if (u < v) else -reciprocal_curvature
        else:
            curv = default_curvature
        curvatures.append(curv)

    base = np.sqrt(node_size)
    margin = edge_margin_factor * base

    data = list(zip(edges, widths, colors, curvatures, strict=False))
    edge_z = 4 if arrows_on_top else 1

    def draw_batch(batch, rad):
        """
        Draw a batch of network edges with a shared curvature.

        Parameters
        ----------
        batch : list[tuple]
            Edge tuples with width, color, and curvature metadata.
        rad : float
            Curvature radius passed to NetworkX edge drawing.

        Returns
        -------
        None
            Edges are drawn directly on the active axes.
        """
        if not batch:
            return
        edgelist = [(u, v) for (u, v, _), _, _, _ in batch]
        widthlist = [w for _, w, _, _ in batch]
        colorlist = [c for _, _, c, _ in batch]
        arts = nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edgelist,
            width=widthlist,
            edge_color=colorlist,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=arrowsize,
            connectionstyle=f"arc3,rad={rad}",
            min_source_margin=margin,
            min_target_margin=margin,
            ax=ax,
            alpha=0.95,
        )
        if arts is not None and arrows_on_top:
            try:
                for art in arts:
                    art.set_zorder(edge_z)
            except TypeError:
                arts.set_zorder(edge_z)

    neg = [(e, w, c, cv) for (e, w, c, cv) in data if cv < 0]
    posb = [(e, w, c, cv) for (e, w, c, cv) in data if cv > 0]
    flat = [(e, w, c, cv) for (e, w, c, cv) in data if abs(cv) <= 1e-9]
    draw_batch(neg, -abs(reciprocal_curvature))
    draw_batch(posb, abs(reciprocal_curvature))
    draw_batch(flat, default_curvature)

    if show_edge_labels:
        lbls = {(u, v): label_fmt.format(a["weight"]) for u, v, a in edges}
        edlbls = nx.draw_networkx_edge_labels(G, pos, edge_labels=lbls, font_size=8)
        for t in edlbls.values():
            t.set_zorder(5)

    if title:
        ax.set_title(title, pad=10)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=220)
    plt.show()
    return G

# Simpler network representations

def plot_interaction_tileplot(
    df: pd.DataFrame,
    value_col: str = "cor_estimate",
    row_col: str = "target",
    col_col: str = "predictor",
    cluster_by: str | None = "rows",  # "rows", "columns", or None
    same_order_for_rows_cols: bool = True,
    fill_missing_for_clustering: float = 0.0,
    cmap: str = "RdBu_r",
    center: float = 0.0,
    vlim: float | None = None,
    show_values: bool = True,
    value_decimals: int = 2,
    figsize: tuple[float, float] | None = None,
    cbar_label: str | None = None,
    linewidth: float = 0.5,
    linecolor: str = "lightgrey",
    text_kwargs: dict | None = None,
):
    """
    Plot a target × predictor interaction matrix as a diverging clustered tile plot.

    By default:
    - tile fill uses `cor_estimate`
    - colors are centered at 0
    - 0 is white
    - rows are clustered
    - the inferred row order is also applied to columns

    Parameters
    ----------
    df
        Long-format dataframe with one row per target-predictor interaction.
    value_col
        Column used for tile color.
    row_col
        Column defining rows.
    col_col
        Column defining columns.
    cluster_by
        One of {"rows", "columns", None}.
        If "rows", cluster row profiles and apply that order.
        If "columns", cluster column profiles and apply that order.
        If None, keep original order.
    same_order_for_rows_cols
        If True, apply the chosen clustering order to both rows and columns.
        This is appropriate when rows and columns represent the same entities,
        e.g. cell types.
    fill_missing_for_clustering
        Value used only for clustering missing interactions.
        Missing plotted tiles remain NaN and are shown as white.
    cmap
        Diverging colormap.
    center
        Center of the color scale.
    vlim
        Symmetric color limit. If None, inferred from max absolute value.
    show_values
        Whether to write values in tiles.
    value_decimals
        Number of decimals shown inside tiles.
    """
    required = {row_col, col_col, value_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if cluster_by not in {"rows", "columns", None}:
        raise ValueError("cluster_by must be one of {'rows', 'columns', None}")

    # No aggregation: repeated target-predictor pairs are an error.
    duplicated = df.duplicated(subset=[row_col, col_col], keep=False)
    if duplicated.any():
        dup_pairs = (
            df.loc[duplicated, [row_col, col_col]]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            "Repeated target-predictor pairs found. "
            f"Examples: {dup_pairs}"
        )

    # Preserve first-seen order before any clustering
    row_order_original = pd.Index(df[row_col].drop_duplicates())
    col_order_original = pd.Index(df[col_col].drop_duplicates())

    mat = (
        df.set_index([row_col, col_col])[value_col]
        .unstack(col_col)
        .reindex(index=row_order_original, columns=col_order_original)
    )

    # Make missing values visually white
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("white")

    # Symmetric color scale around center
    if vlim is None:
        vlim = np.nanmax(np.abs(mat.to_numpy()))
        if not np.isfinite(vlim) or vlim == 0:
            vlim = 1.0

    norm = TwoSlopeNorm(vmin=-vlim, vcenter=center, vmax=vlim)

    # Clustering
    if cluster_by is not None:
        cluster_mat = mat.fillna(fill_missing_for_clustering)

        if cluster_by == "rows":
            if cluster_mat.shape[0] > 1:
                distances = pdist(cluster_mat.to_numpy(), metric="euclidean")
                linkage_matrix = linkage(distances, method="average")
                ordered_labels = mat.index[leaves_list(linkage_matrix)]
            else:
                ordered_labels = mat.index

        elif cluster_by == "columns":
            if cluster_mat.shape[1] > 1:
                distances = pdist(cluster_mat.T.to_numpy(), metric="euclidean")
                linkage_matrix = linkage(distances, method="average")
                ordered_labels = mat.columns[leaves_list(linkage_matrix)]
            else:
                ordered_labels = mat.columns

        if same_order_for_rows_cols:
            # Apply the selected order to both axes.
            # Labels absent from the clustered axis are appended in original order.
            all_labels = pd.Index(
                list(row_order_original) +
                [x for x in col_order_original if x not in row_order_original]
            )

            ordered_labels = pd.Index(
                list(ordered_labels) +
                [x for x in all_labels if x not in ordered_labels]
            )

            mat = mat.reindex(index=ordered_labels, columns=ordered_labels)
        else:
            if cluster_by == "rows":
                mat = mat.reindex(index=ordered_labels)
            elif cluster_by == "columns":
                mat = mat.reindex(columns=ordered_labels)

    if figsize is None:
        figsize = (
            max(5, 0.55 * mat.shape[1]),
            max(4, 0.45 * mat.shape[0]),
        )

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        mat.to_numpy(),
        cmap=cmap_obj,
        norm=norm,
        aspect="auto",
    )

    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax.set_yticklabels(mat.index)

    ax.set_xlabel(col_col)
    ax.set_ylabel(row_col)

    # Tile grid
    ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
    ax.grid(which="minor", color=linecolor, linewidth=linewidth)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Adaptive text color based on tile luminance
    if show_values:
        default_text_kwargs = {
            "ha": "center",
            "va": "center",
            "fontsize": 9,
        }
        if text_kwargs is not None:
            default_text_kwargs.update(text_kwargs)

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                value = mat.iat[i, j]

                if pd.isna(value):
                    continue

                rgba = cmap_obj(norm(value))
                r, g, b = rgba[:3]

                # Perceived luminance
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                text_color = "black" if luminance > 0.5 else "white"

                ax.text(
                    j,
                    i,
                    f"{value:.{value_decimals}f}",
                    color=text_color,
                    **default_text_kwargs,
                )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label or value_col)

    fig.tight_layout()

    return None


def plot_features_per_view(
    df_dict: dict[str, pd.DataFrame],
    features: list[str],
    cmap: str = "coolwarm",
    figsize: tuple[int, int] = (14, 5),
    ytick_rotation: int = 0,
    xtick_rotation: int = 90,
    share_color_scale: bool = True,
    center: float | None = 0.0,
):
    """
    Plot grouped heatmaps for selected features across multiple views.

    Each entry in ``df_dict`` is a dataframe for one view, with rows as samples
    and columns as features. For each view, only the requested features present
    in the dataframe are plotted.

    Parameters
    ----------
    df_dict : dict[str, pandas.DataFrame]
        Dictionary mapping view names to sample-by-feature matrices.
    features : list[str]
        Feature names to plot in each view when present.
    cmap : str
        Matplotlib colormap name.
    figsize : tuple[int, int]
        Overall figure size.
    ytick_rotation : int
        Rotation angle for y-axis tick labels.
    xtick_rotation : int
        Rotation angle for x-axis tick labels.
    share_color_scale : bool
        Whether all heatmaps use a shared color scale.
    center : float or None
        Center value for diverging color scaling. If None, no center is used.

    Returns
    -------
    None
        The function displays the plot and does not return an object.
    """
    filtered_data = {}
    views = []

    # Keep only views with at least one requested feature
    for view, df in df_dict.items():
        view_features = [f for f in features if f in df.columns]
        if len(view_features) == 0:
            continue

        filtered_data[view] = df[view_features]
        views.append(view)

    if len(views) == 0:
        print("No requested features were found in any view.")
        return

    # Global color scale if requested
    if share_color_scale:
        all_vals = pd.concat(filtered_data.values(), axis=1)
        values = all_vals.to_numpy().ravel()
        values = values[np.isfinite(values)]

        if center is None:
            vmin, vmax = np.nanmin(values), np.nanmax(values)
        else:
            max_abs = np.nanmax(np.abs(values - center))
            vmin, vmax = center - max_abs, center + max_abs
    else:
        vmin, vmax = None, None

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, len(views), wspace=0.4)

    for i, view in enumerate(views):
        plot_data = filtered_data[view]

        # Per-view centered color scale if not sharing scale
        if not share_color_scale:
            values = plot_data.to_numpy().ravel()
            values = values[np.isfinite(values)]

            if center is None:
                view_vmin, view_vmax = np.nanmin(values), np.nanmax(values)
            else:
                max_abs = np.nanmax(np.abs(values - center))
                view_vmin, view_vmax = center - max_abs, center + max_abs
        else:
            view_vmin, view_vmax = vmin, vmax

        ax = fig.add_subplot(gs[i])
        sns.heatmap(
            plot_data,
            cmap=cmap,
            vmin=view_vmin,
            vmax=view_vmax,
            center=center,
            cbar=False,
            ax=ax,
            xticklabels=True,
            yticklabels=(i == 0),
        )

        ax.set_title(view, fontsize=10)
        ax.tick_params(axis="x", labelsize=7, rotation=xtick_rotation)
        ax.tick_params(axis="y", labelsize=7, rotation=ytick_rotation)

        if i > 0:
            ax.set_ylabel("")

    # Shared colorbar
    if share_color_scale:
        cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.5])
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax)

        plt.tight_layout(rect=[0, 0, 0.9, 1])
    else:
        plt.tight_layout()

    plt.show()


def plot_comm_overview(
    plot_df,
    tile_width=0.6,
    tile_height=0.6,
    text_size=5,
    figsize=None,
    source_label="source",
    target_label="target",
    ax=None,
):
    """
    Plot ligand-receptor coherent interactions as source/target tiles.

    Parameters
    ----------
    plot_df : pandas.DataFrame
        Output from `generate_lr_plot_df`.
    tile_width : float
        Width of each rectangular tile.
    tile_height : float
        Height of each rectangular tile.
    text_size : float
        Font size of + / - labels inside tiles.
    figsize : tuple or None
        Matplotlib figure size. If None, size is inferred from data dimensions.
    source_label : str
        Label for source side of x-axis.
    target_label : str
        Label for target side of x-axis.
    ax : matplotlib.axes.Axes or None
        Existing axis to plot into.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the communication overview.
    ax : matplotlib.axes.Axes
        Axes containing the communication overview.
    """
    required_cols = {
        "source",
        "target",
        "interaction",
        "coherent_sign",
    }

    missing_cols = required_cols - set(plot_df.columns)

    if missing_cols:
        raise ValueError(f"`plot_df` is missing required columns: {missing_cols}")

    if plot_df.empty:
        raise ValueError("`plot_df` is empty. Nothing to plot.")

    interaction_order = plot_df["interaction"].drop_duplicates().tolist()

    source_order = plot_df["source"].drop_duplicates().tolist()
    target_order = plot_df["target"].drop_duplicates().tolist()

    source_x = {source: i for i, source in enumerate(source_order)}

    target_x = {target: i + len(source_order) + 1 for i, target in enumerate(target_order)}

    y_pos = {interaction: i for i, interaction in enumerate(interaction_order[::-1])}

    if figsize is None:
        width = max(6, 0.35 * (len(source_order) + len(target_order) + 1))
        height = max(4, 0.35 * len(interaction_order))
        figsize = (width, height)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sign_colors = {
        1: "#3b82f6",  # positive
        -1: "#7e22ce",  # negative
    }

    for _, row in plot_df.iterrows():
        y = y_pos[row["interaction"]]
        sign = int(row["coherent_sign"])

        if sign not in sign_colors:
            continue

        color = sign_colors[sign]
        label = "+" if sign > 0 else "-"

        x_values = [
            source_x[row["source"]],
            target_x[row["target"]],
        ]

        for x in x_values:
            ax.add_patch(
                Rectangle(
                    (x - tile_width / 2, y - tile_height / 2),
                    tile_width,
                    tile_height,
                    facecolor=color,
                    edgecolor="none",
                )
            )

            ax.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                color="white",
                fontsize=text_size,
                fontweight="bold",
            )

    # x-axis labels
    x_labels = source_order + [""] + target_order

    x_positions = list(range(len(source_order))) + [len(source_order)] + list(target_x.values())

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")

    # y-axis labels
    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels(list(y_pos.keys()))

    # panel separators
    ax.axvline(len(source_order) - 0.5, color="black", linewidth=1)
    ax.axvline(len(source_order) + 0.5, color="black", linewidth=1)

    # limits
    ax.set_xlim(
        -0.5,
        len(source_order) + len(target_order) + 0.5,
    )

    ax.set_ylim(
        -0.5,
        len(interaction_order) - 0.5,
    )

    ax.set_xlabel(f"{source_label} / {target_label}")
    ax.set_facecolor("#d1d5db")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    return fig, ax


def plot_lr_circos(
    plot_df: pd.DataFrame,
    sector_order: Sequence[str] | None = None,
    sector_colors: Mapping[str, str] | None = None,
    cmap: str = "tab10",
    sector_gap: float = 4,
    sector_r_lim: tuple[float, float] = (96, 100),
    gene_label_size: float = 7,
    link_color: str = "black",
    link_width: float = 0.8,
    link_alpha: float = 0.7,
    arrow_height: float = 3,
    arrow_width: float = 2,
    figsize: tuple[float, float] = (10, 10),
    start: float = 0,
    end: float = 360,
    dpi: int = 100,
    ax: Axes | None = None,
    show_sector_legend: bool = True,
    legend_kwargs: dict | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot ligand-receptor interactions as a directed Circos plot.

    Each source or target cell type is represented by one sector. Ligands and
    receptors are placed as labels within their corresponding cell-type sector.
    Each row creates a directed connection:

        source ligand -> target receptor
    """
    required_cols = {
        "source",
        "target",
        "ligand",
        "receptor",
    }

    missing_cols = required_cols - set(plot_df.columns)

    if missing_cols:
        raise ValueError(
            f"`plot_df` is missing required columns: {sorted(missing_cols)}"
        )

    if plot_df.empty:
        raise ValueError("`plot_df` is empty. Nothing to plot.")

    if ax is not None and getattr(ax, "name", None) != "polar":
        raise ValueError(
            "`ax` must be a polar Matplotlib axis. Create it with "
            "`subplot_kw={'projection': 'polar'}`."
        )

    df = plot_df[
        ["source", "target", "ligand", "receptor"]
    ].copy()

    df = df.dropna(
        subset=["source", "target", "ligand", "receptor"]
    )

    if df.empty:
        raise ValueError(
            "No complete ligand-receptor interactions remain after "
            "removing missing values."
        )

    for column in ["source", "target", "ligand", "receptor"]:
        df[column] = df[column].astype(str)

    df = df.drop_duplicates().reset_index(drop=True)

    observed_sectors = pd.unique(
        pd.concat(
            [df["source"], df["target"]],
            ignore_index=True,
        )
    ).tolist()

    if sector_order is None:
        resolved_sector_order = observed_sectors
    else:
        if isinstance(sector_order, str):
            raise TypeError(
                "`sector_order` must be a sequence of cell-type names, "
                "not a single string."
            )

        requested_order = [str(value) for value in sector_order]

        unknown_sectors = sorted(
            set(requested_order) - set(observed_sectors)
        )

        if unknown_sectors:
            raise ValueError(
                "`sector_order` contains cell types absent from "
                f"`plot_df`: {unknown_sectors}"
            )

        resolved_sector_order = requested_order + [
            sector
            for sector in observed_sectors
            if sector not in requested_order
        ]

    sector_genes = {
        sector: []
        for sector in resolved_sector_order
    }

    for row in df.itertuples(index=False):
        if row.ligand not in sector_genes[row.source]:
            sector_genes[row.source].append(row.ligand)

        if row.receptor not in sector_genes[row.target]:
            sector_genes[row.target].append(row.receptor)

    sectors = {
        sector: len(sector_genes[sector])
        for sector in resolved_sector_order
    }

    if sector_gap * len(sectors) >= end - start:
        raise ValueError(
            "`sector_gap` is too large for the number of sectors."
        )

    cmap_obj = plt.get_cmap(cmap)

    generated_colors = {
        sector: to_hex(
            cmap_obj(index / max(len(resolved_sector_order) - 1, 1))
        )
        for index, sector in enumerate(resolved_sector_order)
    }

    if sector_colors is not None:
        generated_colors.update(
            {
                str(sector): color
                for sector, color in sector_colors.items()
            }
        )

    circos = Circos(
        sectors=sectors,
        start=start,
        end=end,
        space=sector_gap,
    )

    gene_positions: dict[tuple[str, str], float] = {}

    for sector in circos.sectors:
        genes = sector_genes[sector.name]
        color = generated_colors[sector.name]

        track = sector.add_track(
            sector_r_lim,
            r_pad_ratio=0,
        )

        track.axis(
            fc=color,
            ec="none",
        )

        positions = np.arange(len(genes), dtype=float) + 0.5

        track.xticks(
            positions,
            labels=genes,
            outer=True,
            tick_length=2.5,
            label_margin=1.5,
            label_size=gene_label_size,
            label_orientation="vertical",
            line_kws={
                "ec": color,
                "lw": 0.8,
            },
            text_kws={
                "color": "black",
            },
        )

        for gene, position in zip(genes, positions, strict=True):
            gene_positions[(sector.name, gene)] = float(position)

    for row in df.itertuples(index=False):
        ligand_position = gene_positions[
            (row.source, row.ligand)
        ]

        receptor_position = gene_positions[
            (row.target, row.receptor)
        ]

        circos.link_line(
            (row.source, ligand_position),
            (row.target, receptor_position),
            r1=sector_r_lim[0] - 1,
            r2=sector_r_lim[0] - 1,
            direction=1,
            color=link_color,
            lw=link_width,
            alpha=link_alpha,
            arrow_height=arrow_height,
            arrow_width=arrow_width,
        )

    fig = circos.plotfig(
        figsize=figsize,
        dpi=dpi,
        ax=ax,
    )

    if show_sector_legend:
        legend_handles = [
            Patch(
                facecolor=generated_colors[sector],
                edgecolor="none",
                label=sector,
            )
            for sector in resolved_sector_order
        ]

        if legend_kwargs is None:
            legend_kwargs = {
                "loc": "center left",
                "bbox_to_anchor": (1.02, 0.5),
                "frameon": False,
                "title": "Cell type",
            }

        circos.ax.legend(
            handles=legend_handles,
            **legend_kwargs,
        )

    return fig, circos.ax

def plot_feats_through_cov(
    adata,
    features,
    view,
    covariate="time",
    group_by=None,
    ci_opacity=0.3,
    line_size=0.8,
    dot_size=20,
    figsize=None,
    line_color="#08306b",
    point_color=None,
    ribbon_color="#d3d3d3",
    group_colors=None,
    covariate_order=None,
    sharey=True,
):
    """
    Plot selected features against a one-dimensional covariate.

    For each feature, the function plots the mean values connected by a line,
    together with mean points and an approximate 95% confidence interval:

        mean ± 1.96 * std / sqrt(n)

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing:

        - ``adata.obsm[view]``: sample-by-feature matrix
        - ``adata.uns[f"{view}_columns"]``: feature names
        - ``adata.obs[covariate]``: covariate values

    features : sequence of str
        Features to plot. Panel order follows the order provided here.

    view : str
        Key in ``adata.obsm`` containing the sample-by-feature matrix.

    covariate : str, default="time"
        Column in ``adata.obs`` used for the x-axis.

    group_by : str or None, default=None
        Optional column in ``adata.obs`` defining separate trajectories.

    ci_opacity : float, default=0.3
        Confidence interval opacity.

    line_size : float, default=0.8
        Width of the connecting lines.

    dot_size : float, default=20
        Scatter-point area in points squared.

    figsize : tuple or None
        Figure size. Defaults to approximately four inches per feature.

    line_color : str, default="#08306b"
        Fixed line colour when ``group_by=None``.

    point_color : str or None, default=None
        Fixed point colour when ``group_by=None``. When None, uses
        ``line_color``.

    ribbon_color : str, default="#d3d3d3"
        Confidence interval colour when ``group_by=None``.

    group_colors : sequence, mapping, or None
        Colours used when ``group_by`` is provided.

        - If a mapping, keys should be group labels.
        - If a sequence, colours are assigned in group order.
        - If None, colours are taken from Matplotlib's ``tab10`` colormap.

    covariate_order : sequence or None
        Explicit ordering for a categorical covariate.

    sharey : bool, default=True
        Whether feature panels share the same y-axis scale.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure.

    axes : numpy.ndarray
        One-dimensional array of Matplotlib axes.

    summary : pandas.DataFrame
        Summary statistics used for plotting.
    """
    features = list(features)

    if not features:
        raise ValueError("features must contain at least one feature.")

    if view not in adata.obsm:
        raise KeyError(f"{view!r} was not found in adata.obsm.")

    if covariate not in adata.obs.columns:
        raise KeyError(f"{covariate!r} was not found in adata.obs.")

    if group_by is not None and group_by not in adata.obs.columns:
        raise KeyError(f"{group_by!r} was not found in adata.obs.")

    columns_key = f"{view}_columns"

    if columns_key not in adata.uns:
        raise KeyError(f"{columns_key!r} was not found in adata.uns.")

    if point_color is None:
        point_color = line_color

    # ------------------------------------------------------------------
    # Build the sample-by-feature dataframe
    # ------------------------------------------------------------------
    view_data = adata.obsm[view]
    view_features = list(adata.uns[columns_key])

    if isinstance(view_data, pd.DataFrame):
        view_df = view_data.copy()
        view_df.index = adata.obs_names

        if view_df.shape[1] != len(view_features):
            raise ValueError(
                f"adata.obsm[{view!r}] contains {view_df.shape[1]} columns, "
                f"but adata.uns[{columns_key!r}] contains "
                f"{len(view_features)} feature names."
            )

        view_df.columns = view_features

    else:
        if hasattr(view_data, "toarray"):
            view_data = view_data.toarray()

        view_array = np.asarray(view_data)

        if view_array.ndim != 2:
            raise ValueError(
                f"adata.obsm[{view!r}] must be a two-dimensional matrix."
            )

        if view_array.shape[0] != adata.n_obs:
            raise ValueError(
                f"adata.obsm[{view!r}] contains {view_array.shape[0]} rows, "
                f"but adata contains {adata.n_obs} observations."
            )

        if view_array.shape[1] != len(view_features):
            raise ValueError(
                f"adata.obsm[{view!r}] contains {view_array.shape[1]} columns, "
                f"but adata.uns[{columns_key!r}] contains "
                f"{len(view_features)} feature names."
            )

        view_df = pd.DataFrame(
            view_array,
            columns=view_features,
            index=adata.obs_names,
        )

    missing_features = [
        feature for feature in features
        if feature not in view_df.columns
    ]

    if missing_features:
        raise KeyError(
            f"The following features were not found in view {view!r}: "
            f"{missing_features}"
        )

    # ------------------------------------------------------------------
    # Convert to long format and add observation metadata
    # ------------------------------------------------------------------
    plot_df = (
        view_df.loc[:, features]
        .rename_axis("sample")
        .reset_index()
        .melt(
            id_vars="sample",
            var_name="feature",
            value_name="feature_value",
        )
    )

    metadata_columns = [covariate]

    if group_by is not None:
        metadata_columns.append(group_by)

    metadata_df = adata.obs.loc[:, metadata_columns].copy()
    metadata_df.index = metadata_df.index.astype(str)

    plot_df["sample"] = plot_df["sample"].astype(str)

    plot_df = plot_df.merge(
        metadata_df,
        left_on="sample",
        right_index=True,
        how="left",
        validate="many_to_one",
    )

    plot_df["feature"] = pd.Categorical(
        plot_df["feature"],
        categories=features,
        ordered=True,
    )

    # ------------------------------------------------------------------
    # Calculate mean and approximate 95% confidence interval
    # ------------------------------------------------------------------
    grouping_columns = [covariate, "feature"]

    if group_by is not None:
        grouping_columns.append(group_by)

    summary = (
        plot_df
        .groupby(
            grouping_columns,
            observed=True,
            sort=False,
            dropna=False,
        )
        .agg(
            mean=("feature_value", "mean"),
            std=("feature_value", "std"),
            count=("feature_value", "count"),
        )
        .reset_index()
    )

    # Standard deviation is undefined for groups containing one observation.
    summary["std"] = summary["std"].fillna(0.0)

    summary["ci95"] = (
        1.96
        * summary["std"]
        / np.sqrt(summary["count"])
    )

    required_columns = [covariate, "mean"]

    if group_by is not None:
        required_columns.append(group_by)

    summary = summary.dropna(subset=required_columns).copy()

    if summary.empty:
        raise ValueError(
            "No valid observations remained after grouping and removing "
            "missing values."
        )

    # ------------------------------------------------------------------
    # Convert the covariate into numeric plotting positions
    # ------------------------------------------------------------------
    covariate_values = plot_df[covariate]

    if pd.api.types.is_numeric_dtype(covariate_values):
        summary["_x"] = pd.to_numeric(
            summary[covariate],
            errors="coerce",
        )

        summary = summary.dropna(subset=["_x"]).copy()

        categorical_x = False
        category_labels = None

    else:
        categorical_x = True

        if covariate_order is not None:
            category_labels = list(covariate_order)

        elif isinstance(covariate_values.dtype, pd.CategoricalDtype):
            category_labels = list(
                covariate_values.cat.categories
            )

        else:
            category_labels = list(
                pd.unique(covariate_values.dropna())
            )

        category_positions = {
            category: position
            for position, category in enumerate(category_labels)
        }

        summary["_x"] = summary[covariate].map(category_positions)

        unknown_categories = summary.loc[
            summary["_x"].isna(),
            covariate,
        ].unique()

        if len(unknown_categories) > 0:
            raise ValueError(
                "The following covariate values were not included in "
                f"covariate_order: {list(unknown_categories)}"
            )

        summary["_x"] = summary["_x"].astype(float)

    # ------------------------------------------------------------------
    # Create figure
    # ------------------------------------------------------------------
    n_features = len(features)

    if figsize is None:
        figsize = (4 * n_features, 4)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_features,
        figsize=figsize,
        sharex=True,
        sharey=sharey,
        squeeze=False,
    )

    axes = axes.ravel()

    # ------------------------------------------------------------------
    # Plot a single trajectory per feature
    # ------------------------------------------------------------------
    if group_by is None:
        for ax, feature in zip(axes, features, strict=True):
            feature_summary = (
                summary.loc[summary["feature"] == feature]
                .sort_values("_x")
            )

            x = feature_summary["_x"].to_numpy(dtype=float)
            mean = feature_summary["mean"].to_numpy(dtype=float)
            ci = feature_summary["ci95"].to_numpy(dtype=float)

            ax.fill_between(
                x,
                mean - ci,
                mean + ci,
                color=ribbon_color,
                alpha=ci_opacity,
                linewidth=0,
                zorder=1,
            )

            ax.plot(
                x,
                mean,
                color=line_color,
                linewidth=line_size,
                zorder=2,
            )

            ax.scatter(
                x,
                mean,
                color=point_color,
                s=dot_size,
                zorder=3,
            )

            ax.set_title(
                str(feature),
                fontweight="bold",
                fontsize=10,
            )

    # ------------------------------------------------------------------
    # Plot separate trajectories for each group
    # ------------------------------------------------------------------
    else:
        groups = list(pd.unique(summary[group_by]))

        if isinstance(group_colors, dict):
            missing_group_colors = [
                group for group in groups
                if group not in group_colors
            ]

            if missing_group_colors:
                raise KeyError(
                    "No colour was provided for the following groups: "
                    f"{missing_group_colors}"
                )

            group_color_map = {
                group: group_colors[group]
                for group in groups
            }

        else:
            if group_colors is None:
                cmap = plt.get_cmap("tab10")
                colors = [
                    cmap(index % cmap.N)
                    for index in range(len(groups))
                ]
            else:
                colors = list(group_colors)

                if len(colors) < len(groups):
                    raise ValueError(
                        f"group_colors contains {len(colors)} colours, "
                        f"but {len(groups)} groups are present."
                    )

            group_color_map = dict(zip(groups, colors, strict=False))

        for ax, feature in zip(axes, features, strict=True):
            feature_summary = summary.loc[
                summary["feature"] == feature
            ]

            for group in groups:
                group_summary = (
                    feature_summary.loc[
                        feature_summary[group_by] == group
                    ]
                    .sort_values("_x")
                )

                if group_summary.empty:
                    continue

                x = group_summary["_x"].to_numpy(dtype=float)
                mean = group_summary["mean"].to_numpy(dtype=float)
                ci = group_summary["ci95"].to_numpy(dtype=float)
                color = group_color_map[group]

                ax.fill_between(
                    x,
                    mean - ci,
                    mean + ci,
                    color=color,
                    alpha=ci_opacity,
                    linewidth=0,
                    zorder=1,
                )

                ax.plot(
                    x,
                    mean,
                    color=color,
                    linewidth=line_size,
                    zorder=2,
                )

                ax.scatter(
                    x,
                    mean,
                    color=color,
                    s=dot_size,
                    zorder=3,
                )

            ax.set_title(
                str(feature),
                fontweight="bold",
                fontsize=10,
            )

        legend_handles = [
            Line2D(
                [0],
                [0],
                color=group_color_map[group],
                marker="o",
                linewidth=line_size,
                markersize=np.sqrt(dot_size),
                label=str(group),
            )
            for group in groups
        ]

        fig.legend(
            handles=legend_handles,
            title=group_by,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            frameon=False,
        )

    # ------------------------------------------------------------------
    # Style axes
    # ------------------------------------------------------------------
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

        if categorical_x:
            positions = np.arange(len(category_labels))
            ax.set_xticks(positions)
            ax.set_xticklabels(category_labels)

    fig.supxlabel(covariate)
    fig.supylabel("Feature value (original scale)")

    if group_by is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=(0, 0, 0.9, 1))

    return fig, axes
