import anndata as ad
import numpy as np
import pandas as pd
import pytest

from mina.up import pp, utils


def _make_adata(x, obs_names, var_names, obs_extra=None):
    obs = pd.DataFrame(index=pd.Index(obs_names, dtype=str))
    if obs_extra:
        for k, v in obs_extra.items():
            obs[k] = v
    var = pd.DataFrame(index=pd.Index(var_names))
    return ad.AnnData(X=np.array(x, dtype=float), obs=obs, var=var)


def test_extract_metadata_from_obs_keeps_only_stable_columns():
    obs = pd.DataFrame(
        {
            "donor_id": ["patient_2", "patient_10", "patient_2", "patient_10"],
            "age": [30, 40, 30, 40],
            "sex": [np.nan, "F", np.nan, "F"],
            "unstable": [1, 1, 2, 1],
        }
    )

    metadata = pp.extract_metadata_from_obs(obs=obs, groupby="donor_id", sort=False)

    assert set(metadata.columns) == {"donor_id", "age", "sex"}
    assert "unstable" not in metadata.columns
    assert metadata.index.name is None
    assert list(metadata["donor_id"]) == ["patient_10", "patient_2"]


def test_extract_metadata_from_obs_sort_natural_order():
    obs = pd.DataFrame(
        {
            "donor_id": ["patient_10", "patient_2", "patient_10", "patient_2"],
            "site": ["A", "B", "A", "B"],
        }
    )

    metadata = pp.extract_metadata_from_obs(obs=obs, groupby="donor_id", sort=True)

    assert list(metadata["donor_id"]) == ["patient_2", "patient_10"]


def test_extract_metadata_from_obs_warns_if_no_stable_columns(capsys):
    obs = pd.DataFrame(
        {
            "donor_id": ["d1", "d1", "d2", "d2"],
            "value": [1, 2, 3, 4],
        }
    )

    metadata = pp.extract_metadata_from_obs(obs=obs, groupby="donor_id")

    captured = capsys.readouterr()
    assert "No stable columns found" in captured.out
    assert list(metadata.columns) == ["donor_id"]


def test_split_anndata_by_celltype_splits_and_returns_copies():
    pdata = _make_adata(
        x=[[1, 2], [3, 4], [5, 6]],
        obs_names=["c1", "c2", "c3"],
        var_names=["g1", "g2"],
        obs_extra={"cell_type": ["T", "B", "T"]},
    )

    split = pp.split_anndata_by_celltype(pdata, grouping="cell_type")

    assert set(split.keys()) == {"T", "B"}
    assert list(split["T"].obs_names) == ["c1", "c3"]
    assert list(split["B"].obs_names) == ["c2"]
    split["T"].X[0, 0] = 999
    assert pdata.X[0, 0] == 1


def test_split_anndata_by_celltype_missing_grouping_raises():
    pdata = _make_adata(x=[[1]], obs_names=["c1"], var_names=["g1"])

    with pytest.raises(ValueError, match="not found"):
        pp.split_anndata_by_celltype(pdata, grouping="cell_type")


def test_norm_log_calls_scanpy_pipeline_center_true(monkeypatch, capsys):
    views = {
        "A": _make_adata([[1, 2]], ["c1"], ["g1", "g2"]),
        "B": _make_adata([[3, 4]], ["c2"], ["g1", "g2"]),
    }
    calls = []

    def fake_normalize_total(adata, target_sum, exclude_highly_expressed):
        calls.append(("normalize_total", target_sum, exclude_highly_expressed, adata.n_obs))

    def fake_log1p(adata):
        calls.append(("log1p", adata.n_obs))

    def fake_scale(adata, max_value):
        calls.append(("scale", max_value, adata.n_obs))

    monkeypatch.setattr(pp.sc.pp, "normalize_total", fake_normalize_total)
    monkeypatch.setattr(pp.sc.pp, "log1p", fake_log1p)
    monkeypatch.setattr(pp.sc.pp, "scale", fake_scale)

    pp.norm_log(views, target_sum=1234, exclude_highly_expressed=True, max_value=5.0, center=True)

    assert [c[0] for c in calls].count("normalize_total") == 2
    assert [c[0] for c in calls].count("log1p") == 2
    assert [c[0] for c in calls].count("scale") == 2
    assert ("normalize_total", 1234, True, 1) in calls
    assert ("scale", 5.0, 1) in calls
    assert "scaling complete" in capsys.readouterr().out


def test_norm_log_skips_scale_when_center_false(monkeypatch, capsys):
    views = {"A": _make_adata([[1, 2]], ["c1"], ["g1", "g2"])}
    calls = []

    monkeypatch.setattr(pp.sc.pp, "normalize_total", lambda *args, **kwargs: calls.append("normalize_total"))
    monkeypatch.setattr(pp.sc.pp, "log1p", lambda *args, **kwargs: calls.append("log1p"))
    monkeypatch.setattr(pp.sc.pp, "scale", lambda *args, **kwargs: calls.append("scale"))

    pp.norm_log(views, center=False)

    assert calls == ["normalize_total", "log1p"]
    assert "Normalization and log-transformation complete" in capsys.readouterr().out


def test_norm_log_supports_zscore_mode(capsys):
    views = {"A": _make_adata([[1, 2], [3, 4], [5, 6]], ["c1", "c2", "c3"], ["g1", "g2"])}

    pp.norm_log(views, method="zscore")

    np.testing.assert_allclose(views["A"].X.mean(axis=0), np.array([0.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(views["A"].X.std(axis=0), np.array([1.0, 1.0]), atol=1e-12)
    assert "z-score normalization complete" in capsys.readouterr().out


def test_get_view_info_and_validate_views_report_structure():
    views = {
        "A": _make_adata([[1, 2], [3, 4]], ["s1", "s2"], ["g1", "g2"]),
        "B": _make_adata([[5], [6]], ["s1", "s2"], ["g3"]),
    }

    info = pp.get_view_info(views)
    validation = pp.validate_views(views)

    assert list(info.columns) == ["view_name", "n_obs", "n_vars", "var_names"]
    assert set(info["view_name"]) == {"A", "B"}
    assert validation == {
        "has_multiple_views": True,
        "consistent_observations": True,
        "views_have_variables": True,
        "consistent_obs_names": True,
    }


def test_validate_views_detects_mismatched_obs_names_and_zero_var_view():
    views = {
        "A": _make_adata([[1.0], [2.0]], ["s1", "s2"], ["g1"]),
        "B": _make_adata(np.empty((1, 0)), ["other"], []),
    }

    validation = pp.validate_views(views)

    assert validation["consistent_observations"] is False
    assert validation["views_have_variables"] is False
    assert validation["consistent_obs_names"] is False


def test_filter_views_qc_applies_defaults_and_view_specific_overrides(monkeypatch):
    views = {
        "A": _make_adata([[1, 0], [0, 1]], ["c1", "c2"], ["g1", "g2"]),
        "B": _make_adata([[1, 1], [1, 0]], ["d1", "d2"], ["g1", "g2"]),
    }
    calls = []

    def fake_filter_genes(adata, min_cells):
        calls.append(("genes", min_cells, adata.n_obs))

    def fake_filter_cells(adata, min_genes=None, max_genes=None):
        calls.append(("cells", min_genes, max_genes, adata.n_obs))

    monkeypatch.setattr(pp.sc.pp, "filter_genes", fake_filter_genes)
    monkeypatch.setattr(pp.sc.pp, "filter_cells", fake_filter_cells)

    pp.filter_views_qc(
        views,
        min_cells_per_gene=3,
        min_genes_per_cell=200,
        max_genes_per_cell=500,
        view_specific_filters={"B": {"min_cells_per_gene": 5, "max_genes_per_cell": 300}},
    )

    assert ("genes", 3, 2) in calls
    assert ("genes", 5, 2) in calls
    assert ("cells", 200, None, 2) in calls
    assert ("cells", None, 500, 2) in calls
    assert ("cells", None, 300, 2) in calls


def test_find_highly_variable_genes_marks_features_and_supports_manual_fallback(monkeypatch):
    views = {
        "A": _make_adata([[1, 2, 5], [1, 2, 0], [1, 2, 10]], ["c1", "c2", "c3"], ["g1", "g2", "g3"]),
        "B": _make_adata([[0, 1], [0, 3]], ["d1", "d2"], ["h1", "h2"]),
    }

    def fake_hvg(adata, **kwargs):
        if adata.n_vars == 2:
            raise ValueError("too small")
        adata.var["highly_variable"] = [False, True, True]
        adata.var["means"] = [1.0, 2.0, 3.0]
        adata.var["dispersions"] = [0.1, 0.2, 0.3]
        adata.var["dispersions_norm"] = [0.1, 0.2, 0.3]

    monkeypatch.setattr(pp.sc.pp, "highly_variable_genes", fake_hvg)

    with pytest.warns(UserWarning, match="Could not find highly variable genes"):
        pp.find_highly_variable_genes(views, n_top_genes=1, view_specific_n_genes={"A": 2})

    assert views["A"].var["highly_variable"].tolist() == [False, True, True]
    assert views["B"].var["highly_variable"].sum() == 1


def test_subset_to_hvg_subsets_and_warns_when_missing(capsys):
    views = {
        "A": _make_adata([[1, 2, 3]], ["c1"], ["g1", "g2", "g3"]),
        "B": _make_adata([[1, 2]], ["d1"], ["h1", "h2"]),
    }
    views["A"].var["highly_variable"] = [True, False, True]

    with pytest.warns(UserWarning, match="Run find_highly_variable_genes"):
        pp.subset_to_hvg(views)

    assert list(views["A"].var_names) == ["g1", "g3"]
    assert list(views["B"].var_names) == ["h1", "h2"]


def test_preprocess_views_runs_selected_pipeline_steps(monkeypatch):
    views = {"A": _make_adata([[1, 2], [3, 4]], ["c1", "c2"], ["g1", "g2"])}
    calls = []

    monkeypatch.setattr(pp, "filter_views_qc", lambda *args, **kwargs: calls.append(("filter", kwargs)))
    monkeypatch.setattr(pp, "norm_log", lambda *args, **kwargs: calls.append(("norm", kwargs)))
    monkeypatch.setattr(pp, "find_highly_variable_genes", lambda *args, **kwargs: calls.append(("hvg", kwargs)))
    monkeypatch.setattr(pp, "subset_to_hvg", lambda *args, **kwargs: calls.append(("subset", {})))

    pp.preprocess_views(
        views,
        filter_views=True,
        normalize=True,
        find_hvg=True,
        subset_hvg=True,
        norm_method="zscore",
        min_cells_per_gene=7,
        n_top_genes=12,
    )

    assert calls == [
        ("filter", {"min_cells_per_gene": 7}),
        ("norm", {"method": "zscore"}),
        ("hvg", {"n_top_genes": 12}),
        ("subset", {}),
    ]


def test_save_raw_counts_creates_independent_layer(capsys):
    adata = _make_adata([[1, 2], [3, 4]], ["c1", "c2"], ["g1", "g2"])
    views = {"A": adata}

    utils.save_raw_counts(views)

    np.testing.assert_array_equal(views["A"].layers["raw_counts"], np.array([[1.0, 2.0], [3.0, 4.0]]))
    views["A"].X[0, 0] = 99
    assert views["A"].layers["raw_counts"][0, 0] == 1.0
    assert "Raw counts saved" in capsys.readouterr().out


def test_save_raw_counts_custom_layer_name():
    adata = _make_adata([[1]], ["c1"], ["g1"])
    views = {"A": adata}

    utils.save_raw_counts(views, layer_name="counts_backup")

    assert "counts_backup" in views["A"].layers


def test_append_view_to_var_prefixes_names_and_supports_custom_separator():
    adata = _make_adata([[1, 2]], ["c1"], [1, 2])
    views = {"Tcell": adata}

    utils.append_view_to_var(views, join="__")

    assert list(views["Tcell"].var_names) == ["Tcell__1", "Tcell__2"]


def _studies_for_merge():
    study1 = {
        "A": _make_adata(
            [[1, 2], [3, 4]],
            ["s1_a1", "s1_a2"],
            ["g1", "g2"],
            obs_extra={"donor": ["d1", "d2"], "age": [50, 60]},
        ),
        "B": _make_adata(
            [[9, 8]],
            ["s1_b1"],
            ["g2", "g3"],
            obs_extra={"donor": ["d1"], "age": [50]},
        ),
    }
    study2 = {
        "A": _make_adata(
            [[5, 6], [7, 8]],
            ["s2_a1", "s2_a2"],
            ["g2", "g3"],
            obs_extra={"donor": ["d3", "d4"], "age": [40, 41], "extra": ["x", "y"]},
        ),
        "C": _make_adata(
            [[10]],
            ["s2_c1"],
            ["g5"],
            obs_extra={"donor": ["d3"], "age": [40]},
        ),
    }
    return study1, study2


def test_merge_adata_views_validates_arguments():
    s1, s2 = _studies_for_merge()

    with pytest.raises(ValueError, match="same length"):
        utils.merge_adata_views([s1, s2], ["one"])

    with pytest.raises(ValueError, match="Unknown view_mode"):
        utils.merge_adata_views([s1, s2], ["one", "two"], view_mode="bad_mode")

    with pytest.raises(ValueError, match="min_view_studies must be >= 2"):
        utils.merge_adata_views([s1, s2], ["one", "two"], view_mode="min_n", min_view_studies=1)

    with pytest.raises(ValueError, match="min_var_studies must be >= 2"):
        utils.merge_adata_views([s1, s2], ["one", "two"], var_mode="min_n", min_var_studies=1)


def test_merge_adata_views_union_outer_and_obs_column_intersection():
    s1, s2 = _studies_for_merge()

    merged = utils.merge_adata_views([s1, s2], ["study1", "study2"], view_mode="union", var_mode="outer")

    assert set(merged.keys()) == {"A", "B", "C"}
    assert list(merged["A"].var_names) == ["g1", "g2", "g3"]
    assert list(merged["A"].obs.columns) == ["age", "donor", "study"]
    assert set(merged["A"].obs["study"]) == {"study1", "study2"}


def test_merge_adata_views_intersection_and_min_n_view_modes():
    s1, s2 = _studies_for_merge()

    merged_intersection = utils.merge_adata_views([s1, s2], ["study1", "study2"], view_mode="intersection")
    merged_min_n = utils.merge_adata_views([s1, s2], ["study1", "study2"], view_mode="min_n", min_view_studies=2)

    assert set(merged_intersection.keys()) == {"A"}
    assert set(merged_min_n.keys()) == {"A"}


def test_merge_adata_views_inner_and_min_n_var_modes():
    s1, s2 = _studies_for_merge()

    merged_inner = utils.merge_adata_views([s1, s2], ["study1", "study2"], view_mode="intersection", var_mode="inner")
    merged_min_n = utils.merge_adata_views(
        [s1, s2], ["study1", "study2"], view_mode="intersection", var_mode="min_n", min_var_studies=2
    )

    assert list(merged_inner["A"].var_names) == ["g2"]
    assert list(merged_min_n["A"].var_names) == ["g2"]


def test_merge_adata_views_returns_copies():
    s1, s2 = _studies_for_merge()

    merged = utils.merge_adata_views([s1, s2], ["study1", "study2"], view_mode="union")

    merged["B"].X[0, 0] = 999
    assert s1["B"].X[0, 0] == 9
