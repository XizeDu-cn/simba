"""Core plotting functions for the minimal scATAC workflow."""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from pandas.api.types import is_categorical_dtype, is_numeric_dtype, is_string_dtype

from .._settings import settings
from ._utils import generate_palette


def _resolve_plot_config(fig_size, save_fig, fig_path):
    if fig_size is None:
        fig_size = mpl.rcParams["figure.figsize"]
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, "figures")
    return fig_size, save_fig, fig_path


def _save_or_show(fig, save_fig, fig_path, fig_name):
    if save_fig:
        os.makedirs(fig_path, exist_ok=True)
        fig.savefig(os.path.join(fig_path, fig_name), pad_inches=1, bbox_inches="tight")
        plt.close(fig)


def violin(
    adata,
    list_obs=None,
    list_var=None,
    jitter=0.4,
    size=1,
    log=False,
    pad=1.08,
    w_pad=None,
    h_pad=3,
    fig_size=(3, 3),
    fig_ncol=3,
    save_fig=False,
    fig_path=None,
    fig_name="plot_violin.pdf",
    **kwargs,
):
    """Violin plots for selected obs/var columns."""
    fig_size, save_fig, fig_path = _resolve_plot_config(fig_size, save_fig, fig_path)
    list_obs = list_obs or []
    list_var = list_var or []

    for key in list_obs:
        if key not in adata.obs:
            raise ValueError(f"could not find {key} in `adata.obs_keys()`")
    for key in list_var:
        if key not in adata.var:
            raise ValueError(f"could not find {key} in `adata.var_keys()`")

    def _plot(df, keys, suffix):
        if not keys:
            return
        if log:
            df = pd.DataFrame(np.log1p(df.values), index=df.index, columns=df.columns)
        nrow = int(np.ceil(len(keys) / fig_ncol))
        fig = plt.figure(figsize=(fig_size[0] * fig_ncol * 1.05, fig_size[1] * nrow))
        for i, key in enumerate(keys):
            ax = fig.add_subplot(nrow, fig_ncol, i + 1)
            sns.violinplot(ax=ax, y=key, data=df, inner=None, **kwargs)
            sns.stripplot(ax=ax, y=key, data=df, color="black", jitter=jitter, s=size)
            ax.set_title(key)
            ax.set_ylabel("")
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
        plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
        _save_or_show(fig, save_fig, fig_path, f"{suffix}_{fig_name}")

    _plot(adata.obs[list_obs].copy(), list_obs, "obs")
    _plot(adata.var[list_var].copy(), list_var, "var")


def hist(
    adata,
    list_obs=None,
    list_var=None,
    kde=True,
    log=False,
    pad=1.08,
    w_pad=None,
    h_pad=3,
    fig_size=(3, 3),
    fig_ncol=3,
    save_fig=False,
    fig_path=None,
    fig_name="plot_histogram.pdf",
    **kwargs,
):
    """Histogram plots for selected obs/var columns."""
    fig_size, save_fig, fig_path = _resolve_plot_config(fig_size, save_fig, fig_path)
    list_obs = list_obs or []
    list_var = list_var or []

    for key in list_obs:
        if key not in adata.obs:
            raise ValueError(f"could not find {key} in `adata.obs_keys()`")
    for key in list_var:
        if key not in adata.var:
            raise ValueError(f"could not find {key} in `adata.var_keys()`")

    def _plot(df, keys, suffix):
        if not keys:
            return
        if log:
            df = pd.DataFrame(np.log1p(df.values), index=df.index, columns=df.columns)
        nrow = int(np.ceil(len(keys) / fig_ncol))
        fig = plt.figure(figsize=(fig_size[0] * fig_ncol * 1.05, fig_size[1] * nrow))
        for i, key in enumerate(keys):
            ax = fig.add_subplot(nrow, fig_ncol, i + 1)
            sns.histplot(ax=ax, x=key, data=df, kde=kde, **kwargs)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
        plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
        _save_or_show(fig, save_fig, fig_path, f"{suffix}_{fig_name}")

    _plot(adata.obs[list_obs].copy(), list_obs, "obs")
    _plot(adata.var[list_var].copy(), list_var, "var")


def pca_variance_ratio(
    adata,
    log=True,
    show_cutoff=True,
    fig_size=(4, 4),
    save_fig=None,
    fig_path=None,
    fig_name="plot_variance_ratio.pdf",
    pad=1.08,
    w_pad=None,
    h_pad=None,
    **kwargs,
):
    """Plot explained variance ratio of PCA components."""
    fig_size, save_fig, fig_path = _resolve_plot_config(fig_size, save_fig, fig_path)
    values = adata.uns["pca"]["variance_ratio"]
    n_components = len(values)

    fig = plt.figure(figsize=fig_size)
    y = np.log(values) if log else values
    plt.plot(range(n_components), y, **kwargs)
    if show_cutoff and "n_pcs" in adata.uns.get("pca", {}):
        plt.axvline(x=adata.uns["pca"]["n_pcs"] - 1, linestyle="--", color="#CE3746")
        print(f"the number of selected PC is: {adata.uns['pca']['n_pcs']}")
    plt.xlabel("Principal Component")
    plt.ylabel("Variance Ratio" if not log else "log(Variance Ratio)")
    plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
    _save_or_show(fig, save_fig, fig_path, fig_name)


def pcs_features(
    adata,
    fig_size=(2.2, 2.2),
    fig_ncol=6,
    save_fig=None,
    fig_path=None,
    fig_name="plot_pcs_features.pdf",
    pad=1.08,
    w_pad=None,
    h_pad=None,
):
    """Plot feature loading curves for selected PCs."""
    fig_size, save_fig, fig_path = _resolve_plot_config(fig_size, save_fig, fig_path)

    n_pcs = adata.uns["pca"]["n_pcs"]
    pcs = adata.uns["pca"]["PCs"]
    features = adata.uns["pca"].get("features", {})

    nrow = int(np.ceil(n_pcs / fig_ncol))
    fig = plt.figure(figsize=(fig_size[0] * fig_ncol * 1.05, fig_size[1] * nrow))

    for i in range(n_pcs):
        ax = fig.add_subplot(nrow, fig_ncol, i + 1)
        loadings = np.sort(np.abs(pcs[:, i]))[::-1]
        ax.plot(np.arange(len(loadings)), loadings, color="#4C72B0", linewidth=1.2)
        if f"pc_{i}" in features:
            cutoff = len(features[f"pc_{i}"])
            ax.axvline(cutoff, linestyle="--", linewidth=1, color="#CE3746")
        ax.set_title(f"PC.{i}", fontsize=8)
        ax.set_xlabel("Features", fontsize=7)
        ax.set_ylabel("Loading", fontsize=7)
        ax.tick_params(labelsize=7)

    plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
    _save_or_show(fig, save_fig, fig_path, fig_name)


def _scatterplot2d(
    df,
    x,
    y,
    list_hue=None,
    hue_palette=None,
    drawing_order="sorted",
    dict_drawing_order=None,
    size=8,
    fig_size=None,
    fig_ncol=3,
    fig_legend_ncol=1,
    fig_legend_order=None,
    vmin=None,
    vmax=None,
    alpha=0.8,
    pad=1.08,
    w_pad=None,
    h_pad=None,
    save_fig=None,
    fig_path=None,
    fig_name="scatterplot2d.pdf",
    copy=False,
    show_texts=False,
    texts=None,
    text_size=10,
    text_expand=(1.05, 1.2),
    **kwargs,
):
    """Internal 2D scatter helper shared by UMAP and query plots."""
    fig_size, save_fig, fig_path = _resolve_plot_config(fig_size, save_fig, fig_path)
    list_hue = list_hue or []
    hue_palette = hue_palette or {}
    dict_drawing_order = dict_drawing_order or {}

    if drawing_order not in ["original", "sorted", "random"]:
        raise ValueError("`drawing_order` must be one of ['original', 'sorted', 'random']")

    if not list_hue:
        list_hue = [None]

    nrow = int(np.ceil(len(list_hue) / fig_ncol))
    fig = plt.figure(figsize=(fig_size[0] * fig_ncol * 1.05, fig_size[1] * nrow))
    axes = []

    for i, hue in enumerate(list_hue):
        ax = fig.add_subplot(nrow, fig_ncol, i + 1)
        axes.append(ax)

        if hue is None:
            data = df
            ax.scatter(data[x], data[y], s=size, alpha=alpha, **kwargs)
            ax.set_title("embedding")
        else:
            if hue not in df.columns:
                raise ValueError(f"could not find {hue} in `df.columns`")

            order = dict_drawing_order.get(hue, drawing_order)
            if order == "sorted":
                data = df.sort_values(by=hue)
            elif order == "random":
                data = df.sample(frac=1, random_state=100)
            else:
                data = df

            if is_numeric_dtype(df[hue]):
                local_vmin = np.min(df[hue]) if vmin is None else vmin
                local_vmax = np.max(df[hue]) if vmax is None else vmax
                sc = ax.scatter(
                    data[x],
                    data[y],
                    c=data[hue],
                    s=size,
                    alpha=alpha,
                    vmin=local_vmin,
                    vmax=local_vmax,
                    **kwargs,
                )
                cbar = plt.colorbar(sc, ax=ax, pad=0.01, fraction=0.05, aspect=40)
                cbar.solids.set_edgecolor("face")
            else:
                palette = hue_palette.get(hue, generate_palette(df[hue]))
                if fig_legend_order and hue in fig_legend_order:
                    categories = fig_legend_order[hue]
                else:
                    categories = list(pd.unique(df[hue]))
                for cat in categories:
                    mask = data[hue] == cat
                    if np.any(mask):
                        ax.scatter(
                            data.loc[mask, x],
                            data.loc[mask, y],
                            s=size,
                            alpha=alpha,
                            c=palette.get(cat, "#333333"),
                            label=cat,
                            **kwargs,
                        )
                ax.legend(frameon=False, ncol=fig_legend_ncol, markerscale=1.2)

            ax.set_title(hue)

        if show_texts and texts is not None:
            txt = [
                ax.text(df.loc[t, x], df.loc[t, y], t, fontsize=text_size)
                for t in texts
                if t in df.index
            ]
            if txt:
                adjust_text(
                    txt,
                    ax=ax,
                    expand_text=text_expand,
                    expand_points=text_expand,
                    expand_objects=text_expand,
                    arrowprops={"arrowstyle": "-", "color": "black"},
                )

        ax.set_xlabel(x)
        ax.set_ylabel(y)

    plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
    _save_or_show(fig, save_fig, fig_path, fig_name)
    if copy:
        return axes
    return None


def umap(
    adata,
    color=None,
    dict_palette=None,
    n_components=None,
    size=8,
    drawing_order="sorted",
    dict_drawing_order=None,
    show_texts=False,
    texts=None,
    text_size=10,
    text_expand=(1.05, 1.2),
    fig_size=None,
    fig_ncol=3,
    fig_legend_ncol=1,
    fig_legend_order=None,
    vmin=None,
    vmax=None,
    alpha=1,
    pad=1.08,
    w_pad=None,
    h_pad=None,
    save_fig=None,
    fig_path=None,
    fig_name="plot_umap.pdf",
    plolty=False,
    **kwargs,
):
    """Plot UMAP coordinates with optional coloring."""
    if plolty:
        raise NotImplementedError("Plotly rendering is not supported in the minimal build.")

    if "X_umap" not in adata.obsm:
        raise ValueError("could not find `X_umap` in `adata.obsm`")

    fig_size, save_fig, fig_path = _resolve_plot_config(fig_size, save_fig, fig_path)
    color = list(dict.fromkeys(color or []))
    dict_palette = dict_palette or {}

    X = adata.obsm["X_umap"]
    if n_components is None:
        n_components = X.shape[1]
    if n_components < 2:
        raise ValueError("`n_components` must be >= 2")

    df_plot = pd.DataFrame(X[:, :n_components], index=adata.obs_names)
    df_plot.columns = [f"UMAP{i+1}" for i in range(n_components)]

    for key in color:
        if key in adata.obs:
            df_plot[key] = adata.obs[key]
            if not is_numeric_dtype(df_plot[key]):
                if "color" not in adata.uns:
                    adata.uns["color"] = {}
                if key in dict_palette:
                    adata.uns["color"][f"{key}_color"] = dict_palette[key]
                elif f"{key}_color" in adata.uns["color"]:
                    dict_palette[key] = adata.uns["color"][f"{key}_color"]
                else:
                    dict_palette[key] = generate_palette(df_plot[key])
                    adata.uns["color"][f"{key}_color"] = dict_palette[key]
        elif key in adata.var_names:
            df_plot[key] = adata.obs_vector(key)
        else:
            raise ValueError(f"could not find {key} in `adata.obs.columns` and `adata.var_names`")

    if not color:
        _scatterplot2d(
            df_plot,
            x="UMAP1",
            y="UMAP2",
            size=size,
            drawing_order=drawing_order,
            fig_size=fig_size,
            alpha=alpha,
            pad=pad,
            w_pad=w_pad,
            h_pad=h_pad,
            save_fig=save_fig,
            fig_path=fig_path,
            fig_name=fig_name,
            show_texts=show_texts,
            texts=texts,
            text_size=text_size,
            text_expand=text_expand,
            **kwargs,
        )
        return None

    _scatterplot2d(
        df_plot,
        x="UMAP1",
        y="UMAP2",
        list_hue=color,
        hue_palette=dict_palette,
        drawing_order=drawing_order,
        dict_drawing_order=dict_drawing_order,
        size=size,
        fig_size=fig_size,
        fig_ncol=fig_ncol,
        fig_legend_ncol=fig_legend_ncol,
        fig_legend_order=fig_legend_order,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        pad=pad,
        w_pad=w_pad,
        h_pad=h_pad,
        save_fig=save_fig,
        fig_path=fig_path,
        fig_name=fig_name,
        show_texts=show_texts,
        texts=texts,
        text_size=text_size,
        text_expand=text_expand,
        **kwargs,
    )
    return None
