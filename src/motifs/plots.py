import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from motifs.config import LOGGER

TF_IDF_PLOT_TYPES = ["group", "sep"]


def plot_tf_idf(
    tf_idf: pd.DataFrame,
    n_motifs: int = 20,
    plot_type: str = "sep",
    col_wrap: int = 3,
):
    if plot_type not in TF_IDF_PLOT_TYPES:
        LOGGER.error(
            f"{plot_type} is not implemented. For now you can use "
            f"only {TF_IDF_PLOT_TYPES}"
        )
        raise NotImplementedError

    temp = tf_idf.copy()
    temp = temp.groupby("piece").head(n_motifs)
    temp.sort_values(by="tfidf", ascending=False, inplace=True)

    if plot_type == "sep":
        g = sns.FacetGrid(
            temp,
            col="piece",
            sharey=False,
            col_wrap=col_wrap,
        )
        g.map(sns.barplot, "tfidf", "token")
        g.set_titles("{col_name}")
    elif plot_type == "group":
        sns.barplot(temp, y="token", x="tfidf", hue="piece")
    g.set(xlabel=None, ylabel=None)
    plt.show()
    # plt.close()


def plot_explained_variance_ratio(pca: PCA):
    plots = sns.barplot(pca.explained_variance_ratio_ * 100)
    plots.set(ylabel="Explained variance ratio", ylim=[0, 100])
    for bar in plots.patches:
        plots.annotate(
            format(bar.get_height(), ".0f") + "%",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="center",
            xytext=(0, 8),
            textcoords="offset points",
        )
    plt.show()
    # plt.close()


def plot_pca_projection(pca, var_names):
    loadings = pd.DataFrame(pca.components_.T, index=var_names)
    sns.heatmap(
        loadings, cmap="bwr", square=pca.components_.shape[0] == len(var_names)
    )
    plt.show()


def pca_variable_plot(data, pca, colwrap=3, max_plots=50):
    factors = pca.transform(data)
    pairs = list(itertools.combinations(list(range(factors.shape[-1])), 2))
    if len(pairs) > max_plots:
        LOGGER.error(
            f"Number of plots larger than max_plots={max_plots}. If you "
            f"really want to produce {len(pairs)} plots, then increase "
            f"max_plots. Aborting."
        )
        return
    if len(pairs) >= colwrap:
        ncol = colwrap
    else:
        ncol = len(pairs)
    nrow = math.ceil(len(pairs) / colwrap)

    row = 0
    fig, axs = plt.subplots(nrow, ncol, figsize=(5 * ncol, 5 * nrow))

    for i, pair in enumerate(pairs):
        if i % colwrap == 0 and i >= 3:
            row += 1
        col = i % colwrap
        pca_variable_2dplot(
            data, factors[:, [pair[0], pair[1]]], ax=axs[row, col]
        )

    # Remove extra empty axes
    if (nrow * ncol - len(pairs)) > 0:
        for i in range(colwrap - (nrow * ncol - len(pairs)), colwrap):
            axs[row, i].set_axis_off()

    plt.show()


def pca_variable_2dplot(data: pd.DataFrame, factors, ax):
    assert factors.shape[-1] == 2
    t = np.linspace(0, np.pi * 2, 100)
    corr_ = pd.DataFrame(
        [
            [
                np.corrcoef(data.values[:, i], factors[:, 0])[0, 1]
                for i in range(data.shape[-1])
            ],
            [
                np.corrcoef(data.values[:, i], factors[:, 1])[0, 1]
                for i in range(data.shape[-1])
            ],
        ],
        columns=data.columns,
        index=["comp_0", "comp_1"],
    ).T

    for i in range(data.shape[-1]):
        ax.annotate(
            corr_.index[i],
            xy=(corr_["comp_0"].values[i], corr_["comp_1"].values[i]),
            xytext=(corr_["comp_0"].values[i], corr_["comp_1"].values[i]),
        )

    for i in range(data.shape[-1]):
        ax.annotate(
            "",
            xy=(corr_["comp_0"].values[i], corr_["comp_1"].values[i]),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->"),
        )

    ax.plot(np.cos(t), np.sin(t), linewidth=1, c="black")
    # ax.set_box_aspect(1)
    ax.set_aspect("equal")
    ax.grid(True, which="both")
    ax.axhline(y=0, color="k")
    ax.axvline(x=0, color="k")


def pca_variable_plot_old(pca: PCA):
    t = np.linspace(0, np.pi * 2, 100)
    fig, ax = plt.subplots(1, 1)

    corr_ = pd.DataFrame(
        [
            [
                np.corrcoef(pca.transformed_data[:, i], pca.factors["comp_0"])[
                    0, 1
                ]
                for i in range(3)
            ],
            [
                np.corrcoef(pca.transformed_data[:, i], pca.factors["comp_1"])[
                    0, 1
                ]
                for i in range(3)
            ],
        ],
        columns=pca.loadings.index,
        index=["comp_0", "comp_1"],
    ).T

    for i in range(3):
        ax.annotate(
            corr_.index[i],
            xy=(corr_["comp_0"].values[i], corr_["comp_1"].values[i]),
            xytext=(corr_["comp_0"].values[i], corr_["comp_1"].values[i]),
        )

    for i in range(3):
        ax.annotate(
            "",
            xy=(corr_["comp_0"].values[i], corr_["comp_1"].values[i]),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->"),
        )

    ax.plot(np.cos(t), np.sin(t), linewidth=1, c="black")
    # ax.set_box_aspect(1)
    ax.set_aspect("equal")
    ax.grid(True, which="both")
    ax.axhline(y=0, color="k")
    ax.axvline(x=0, color="k")
    plt.show()
    # plt.close()
