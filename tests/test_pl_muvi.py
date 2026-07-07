import pandas as pd
from tests.test_down_muvi import _make_model_adata

from mina.down import (
    confidence_ellipses_info,
    factor_scores_info,
    featureclass_variance_info,
    reconstruction_info,
    selected_features_info,
    variable_loadings_info,
    variance_by_view_info,
)
from mina.pl import (
    plot_confidence_ellipses,
    plot_factor_violin,
    plot_featureclass_variance,
    plot_reconstruction,
    plot_selected_features,
    plot_top_loadings_heatmap,
    plot_variance_by_view,
)


def test_plotnine_smoke_for_muvi_ported_plots():
    model_adata = _make_model_adata()
    scores = factor_scores_info(model_adata).join(pd.DataFrame({"group": ["A", "B"]}, index=model_adata.obs_names))
    variance_df = variance_by_view_info(model_adata)
    loadings = variable_loadings_info(model_adata)
    selected = selected_features_info(loadings, [("gene1", "A"), ("gene3", "B")])
    ellipses = confidence_ellipses_info(scores, "Factor1", "Factor2", "group", num_points=12)
    reconstruction = reconstruction_info(model_adata)["by_view"]
    featureclass_df = featureclass_variance_info(
        model_adata,
        feature_type_map={"alpha": ["gene1", "gene3"], "beta": ["gene2"]},
    )

    plots = [
        plot_reconstruction(reconstruction),
        plot_variance_by_view(variance_df),
        plot_featureclass_variance(featureclass_df),
        plot_top_loadings_heatmap(loadings, factor="Factor1", top_n=2),
        plot_selected_features(selected),
        plot_factor_violin(scores, factor="Factor1", group_col="group"),
        plot_confidence_ellipses(scores, ellipses, x_factor="Factor1", y_factor="Factor2", group_col="group"),
    ]

    for plot in plots:
        assert plot.__class__.__name__ == "ggplot"
        assert hasattr(plot, "draw")
