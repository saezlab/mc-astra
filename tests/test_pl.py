import matplotlib.pyplot as plt
import pandas as pd
import pytest

from mc_astra.pl import plot_mcell_funcomics


@pytest.fixture(autouse=True)
def _close_figures(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    yield
    plt.close("all")


def _view_result(values, pvals):
    index = ["Factor1", "Factor2"]
    columns = ["PathwayA", "PathwayB"]
    return {
        "pw_acts": pd.DataFrame(values, index=index, columns=columns),
        "pw_padj": pd.DataFrame(pvals, index=index, columns=columns),
    }


def _heatmap_axis(title):
    for ax in plt.gcf().axes:
        if ax.get_title() == title:
            return ax
    raise AssertionError(f"Could not find heatmap axis for {title!r}")


def _heatmap_clim(title):
    ax = _heatmap_axis(title)
    return ax.collections[0].get_clim()


def _starred_cells(title):
    ax = _heatmap_axis(title)
    features = [label.get_text() for label in ax.get_xticklabels()]
    factors = [label.get_text() for label in ax.get_yticklabels()]

    cells = set()
    for text in ax.texts:
        if text.get_text() != "★":
            continue
        col_idx = int(round(text.get_position()[0] - 0.5))
        row_idx = int(round(text.get_position()[1] - 0.5))
        cells.add((factors[row_idx], features[col_idx]))
    return cells


def test_plot_mcell_funcomics_uses_centered_shared_color_scale():
    result_dict = {
        "view_a": _view_result([[2.0, 4.0], [5.0, 6.0]], [[0.01, 0.01], [0.01, 0.01]]),
        "view_b": _view_result([[8.0, 7.0], [6.0, 5.0]], [[0.01, 0.01], [0.01, 0.01]]),
    }

    plot_mcell_funcomics(result_dict, top_n=2, center=5.0, share_color_scale=True)

    assert _heatmap_clim("view_a") == (2.0, 8.0)
    assert _heatmap_clim("view_b") == (2.0, 8.0)


def test_plot_mcell_funcomics_center_none_uses_observed_min_max():
    result_dict = {
        "view_a": _view_result([[1.0, 2.0], [4.0, 10.0]], [[0.01, 0.01], [0.01, 0.01]]),
    }

    plot_mcell_funcomics(result_dict, top_n=2, center=None, share_color_scale=True)

    assert _heatmap_clim("view_a") == (1.0, 10.0)


def test_plot_mcell_funcomics_marks_significant_factor_feature_cells():
    result_dict = {
        "view_a": _view_result([[10.0, 1.0], [2.0, 20.0]], [[0.01, 0.20], [0.20, 0.03]]),
    }

    plot_mcell_funcomics(result_dict, top_n=2, p_threshold=0.05)

    assert _starred_cells("view_a") == {
        ("Factor1", "PathwayA"),
        ("Factor2", "PathwayB"),
    }
