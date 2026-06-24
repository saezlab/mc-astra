import anndata as ad
import numpy as np
import pandas as pd
import pytest

from mina.down import (
    build_selected_anndata,
    confidence_ellipses_info,
    factor_scores_info,
    featureclass_variance_info,
    kendall_info,
    kruskal_info,
    model_to_anndata,
    reconstruction_info,
    selected_features_info,
    top_features_by_class_info,
    top_features_by_view_info,
    variable_loadings_info,
    variance_by_view_info,
)
from mina.down.utils import split_by_view


class _FakeModel:
    def get_factors(self):
        return {
            "all": pd.DataFrame(
                [[1.0, 2.0], [3.0, 4.0]],
                index=["s1", "s2"],
                columns=["1", "2"],
            )
        }

    def get_r2(self):
        return pd.DataFrame(
            [
                {"group": "g1", "view": "A", "component": "1", "R2": 0.1},
                {"group": "g2", "view": "A", "component": "1", "R2": 0.2},
                {"group": "g1", "view": "B", "component": "1", "R2": 0.3},
                {"group": "g1", "view": "A", "component": "2", "R2": 0.4},
                {"group": "g2", "view": "A", "component": "2", "R2": 0.5},
                {"group": "g1", "view": "B", "component": "2", "R2": 0.6},
            ]
        )

    def get_weights(self):
        return {
            "A": pd.DataFrame(
                [[0.1, 0.2], [0.3, 0.4]],
                index=["gene1", "gene2"],
                columns=["1", "2"],
            ),
            "B": pd.DataFrame(
                [[0.5, 0.6]],
                index=["gene3"],
                columns=["1", "2"],
            ),
        }


class _PrefixedFeatureModel:
    def get_factors(self):
        return {
            "all": pd.DataFrame(
                [[1.0, 2.0], [3.0, 4.0]],
                index=["s1", "s2"],
                columns=["1", "2"],
            )
        }

    def get_r2(self):
        return _FakeModel().get_r2()

    def get_weights(self):
        return {
            "A": pd.DataFrame(
                [[0.1, 0.2], [0.3, 0.4]],
                index=["A:gene1", "A:gene2"],
                columns=["1", "2"],
            ),
            "B": pd.DataFrame(
                [[0.5, 0.6]],
                index=["B:gene3"],
                columns=["1", "2"],
            ),
        }


def _make_model_adata():
    adata_a = ad.AnnData(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        obs=pd.DataFrame(index=["s1", "s2"]),
        var=pd.DataFrame(index=["gene1", "gene2"]),
    )
    adata_b = ad.AnnData(
        np.array([[5.0], [6.0]]),
        obs=pd.DataFrame(index=["s1", "s2"]),
        var=pd.DataFrame(index=["gene3"]),
    )
    metadata = pd.DataFrame({"condition": ["x", "y"]}, index=["s1", "s2"])
    return model_to_anndata({"A": adata_a, "B": adata_b}, metadata, _FakeModel())


def _make_prefixed_model_adata():
    adata_a = ad.AnnData(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        obs=pd.DataFrame(index=["s1", "s2"]),
        var=pd.DataFrame(index=["A:gene1", "A:gene2"]),
    )
    adata_b = ad.AnnData(
        np.array([[5.0], [6.0]]),
        obs=pd.DataFrame(index=["s1", "s2"]),
        var=pd.DataFrame(index=["B:gene3"]),
    )
    metadata = pd.DataFrame({"condition": ["x", "y"]}, index=["s1", "s2"])
    return model_to_anndata({"A": adata_a, "B": adata_b}, metadata, _PrefixedFeatureModel())


def test_model_to_anndata_preserves_view_encoded_loading_columns():
    model_adata = _make_model_adata()

    assert model_adata.uns["gene_loadings_columns"] == ["A:gene1", "A:gene2", "B:gene3"]

    wide = pd.DataFrame(model_adata.varm["gene_loadings"], columns=model_adata.uns["gene_loadings_columns"])
    split = split_by_view(wide)

    assert list(split) == ["A", "B"]
    assert list(split["A"].columns) == ["gene1", "gene2"]
    assert list(split["B"].columns) == ["gene3"]


def test_model_to_anndata_avoids_double_prefixing_already_prefixed_features():
    model_adata = _make_prefixed_model_adata()

    assert model_adata.uns["gene_loadings_columns"] == ["A:gene1", "A:gene2", "B:gene3"]
    assert model_adata.uns["A_columns"] == ["gene1", "gene2"]
    assert model_adata.uns["B_columns"] == ["gene3"]

    wide = pd.DataFrame(model_adata.varm["gene_loadings"], columns=model_adata.uns["gene_loadings_columns"])
    split = split_by_view(wide)

    assert list(split["A"].columns) == ["gene1", "gene2"]
    assert list(split["B"].columns) == ["gene3"]


def test_factor_scores_info_returns_scores_and_selected_obs_columns():
    model_adata = _make_model_adata()

    scores = factor_scores_info(model_adata, obs_keys=["condition"], factor_names=["Factor2"])

    assert list(scores.columns) == ["Factor2", "condition"]
    assert list(scores.index) == ["s1", "s2"]
    assert scores.loc["s1", "Factor2"] == 2.0
    assert scores.loc["s2", "condition"] == "y"


def test_factor_scores_info_raises_for_missing_factor_or_obs_key():
    model_adata = _make_model_adata()

    with pytest.raises(KeyError, match="Unknown factor names"):
        factor_scores_info(model_adata, factor_names=["Factor9"])

    with pytest.raises(KeyError, match="Unknown obs columns"):
        factor_scores_info(model_adata, obs_keys=["missing_col"])


def test_variance_by_view_info_aggregates_and_preserves_group_detail():
    model_adata = _make_model_adata()

    aggregated = variance_by_view_info(model_adata)
    detailed = variance_by_view_info(model_adata, aggregate_groups=False)

    expected = {
        ("Factor1", "A"): 0.3,
        ("Factor1", "B"): 0.3,
        ("Factor2", "A"): 0.9,
        ("Factor2", "B"): 0.6,
    }
    observed = {
        (row.Factor, row.View): row.Variance
        for row in aggregated.itertuples(index=False)
    }

    assert observed.keys() == expected.keys()
    for key, value in expected.items():
        assert observed[key] == pytest.approx(value)
    assert set(detailed.columns) == {"Factor", "View", "Group", "View_group", "Variance"}
    assert set(detailed["Group"]) == {"g1", "g2"}


def test_variance_by_view_info_validates_aggregation_name():
    model_adata = _make_model_adata()

    with pytest.raises(ValueError, match="agg must be one of"):
        variance_by_view_info(model_adata, agg="invalid")


def test_variable_loadings_info_returns_view_and_variable_columns():
    model_adata = _make_model_adata()

    loadings = variable_loadings_info(model_adata)

    assert list(loadings.columns) == ["view", "variable", "Factor1", "Factor2"]
    assert list(loadings[["view", "variable"]].itertuples(index=False, name=None)) == [
        ("A", "gene1"),
        ("A", "gene2"),
        ("B", "gene3"),
    ]
    assert loadings.loc[0, "Factor1"] == 0.1
    assert loadings.loc[2, "Factor2"] == 0.6


def test_variable_loadings_info_rejects_unqualified_columns():
    model_adata = _make_model_adata()
    model_adata.uns["gene_loadings_columns"] = ["gene1", "gene2", "gene3"]

    with pytest.raises(ValueError, match="view:feature"):
        variable_loadings_info(model_adata)


def test_selected_features_info_extracts_requested_pairs_only():
    model_adata = _make_model_adata()
    loadings = variable_loadings_info(model_adata)

    selected = selected_features_info(loadings, [("gene1", "A"), ("gene3", "B"), ("missing", "A")])

    assert list(selected.columns) == ["variable", "view", "variable_view", "factor", "loading"]
    assert set(selected["variable_view"]) == {"gene1/A", "gene3/B"}
    assert set(selected["factor"]) == {"Factor1", "Factor2"}


def test_kruskal_and_kendall_info_return_expected_columns_and_validate_inputs():
    scores_df = pd.DataFrame(
        {
            "Factor1": [1.0, 1.1, 5.0, 5.2],
            "Factor2": [2.0, 2.1, 2.2, 2.3],
            "group": ["A", "A", "B", "B"],
            "stage": ["low", "low", "high", "high"],
        },
        index=["s1", "s2", "s3", "s4"],
    )

    kruskal_df = kruskal_info(scores_df, "group")
    kendall_df = kendall_info(scores_df, "stage")

    assert list(kruskal_df.columns) == ["factor", "pvalue"]
    assert list(kendall_df.columns) == ["factor", "tau", "pvalue"]
    assert set(kruskal_df["factor"]) == {"Factor1", "Factor2"}
    assert set(kendall_df["factor"]) == {"Factor1", "Factor2"}

    with pytest.raises(KeyError, match="Unknown group column"):
        kruskal_info(scores_df, "missing")

    with pytest.raises(KeyError, match="Unknown ordinal column"):
        kendall_info(scores_df, "missing")


def test_confidence_ellipses_info_builds_grouped_coordinates():
    scores_df = pd.DataFrame(
        {
            "Factor1": [0.0, 1.0, 2.0, 3.0],
            "Factor2": [0.0, 1.5, 2.0, 3.5],
            "group": ["A", "A", "B", "B"],
        }
    )

    ellipses = confidence_ellipses_info(scores_df, "Factor1", "Factor2", "group", num_points=12)

    assert list(ellipses.columns) == ["x", "y", "group"]
    assert set(ellipses["group"]) == {"A", "B"}
    assert len(ellipses) == 24


def test_top_feature_helpers_rank_features_from_variable_loadings():
    model_adata = _make_model_adata()
    loadings = variable_loadings_info(model_adata)

    top_by_view = top_features_by_view_info(loadings, ["Factor1"], top_per_view=1)
    top_by_class = top_features_by_class_info(
        loadings,
        feature_types={"gene1": "alpha", "gene2": "beta", "gene3": "alpha"},
        factors=["Factor2"],
        top_per_class=1,
    )

    assert list(top_by_view.columns) == ["variable", "view", "weight", "factor"]
    assert set(top_by_view["view"]) == {"A", "B"}
    assert list(top_by_class.columns) == ["variable", "view", "feature_type", "weight", "factor"]
    assert set(top_by_class["feature_type"]) == {"alpha", "beta"}


def test_build_selected_anndata_reconstructs_matrix_from_obsm():
    model_adata = _make_model_adata()
    selected = pd.DataFrame(
        {
            "variable": ["gene2", "gene3"],
            "view": ["A", "B"],
        }
    )

    selected_adata = build_selected_anndata(model_adata, selected)

    assert selected_adata.shape == (2, 2)
    assert list(selected_adata.var["variable_view"]) == ["gene2/A", "gene3/B"]
    np.testing.assert_array_equal(selected_adata.X, np.array([[2.0, 5.0], [4.0, 6.0]]))


def test_build_selected_anndata_accepts_title_case_columns_and_validates_views():
    model_adata = _make_model_adata()
    selected = pd.DataFrame({"Variable": ["gene1"], "View": ["A"]})

    selected_adata = build_selected_anndata(model_adata, selected)
    assert selected_adata.shape == (2, 1)

    with pytest.raises(KeyError, match="Unknown view"):
        build_selected_anndata(model_adata, pd.DataFrame({"variable": ["gene1"], "view": ["missing"]}))


def test_reconstruction_info_returns_per_view_and_macro_summaries():
    model_adata = _make_model_adata()

    reconstruction = reconstruction_info(model_adata)

    assert set(reconstruction) == {"by_view", "macro"}
    assert list(reconstruction["by_view"].columns) == ["view", "R", "R2"]
    assert set(reconstruction["by_view"]["view"]) == {"A", "B"}
    assert np.isfinite(reconstruction["macro"]["R"])
    assert np.isfinite(reconstruction["macro"]["R2"])


def test_featureclass_variance_info_estimates_per_factor_r2_by_feature_class():
    model_adata = _make_model_adata()

    featureclass_df = featureclass_variance_info(
        model_adata,
        feature_type_map={"alpha": ["gene1", "gene3"], "beta": ["gene2"]},
        aggregator="mean",
    )

    assert list(featureclass_df.columns) == ["Factor", "Feature_type", "Variance"]
    assert set(featureclass_df["Feature_type"]) == {"alpha", "beta"}
    assert set(featureclass_df["Factor"]) == {"Factor1", "Factor2"}

    with pytest.raises(ValueError, match="aggregator"):
        featureclass_variance_info(model_adata, {"alpha": ["gene1"]}, aggregator="sum")