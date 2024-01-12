import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from motifs.config import LOGGER

PLOT_TYPES = ["group", "sep"]


def plot_tf_idf(
    tf_idf: pd.DataFrame,
    n_tokens: int = 20,
    plot_type: str = "sep",
    col_wrap: int = 3,
):
    if plot_type not in PLOT_TYPES:
        LOGGER.error(
            f"{plot_type} is not implemented. For now you can use "
            f"only {PLOT_TYPES}"
        )
        raise NotImplementedError

    temp = tf_idf.copy()
    temp = temp.groupby("doc").head(n_tokens)
    temp.sort_values(by="tfidf", ascending=False, inplace=True)

    if plot_type == "sep":
        g = sns.FacetGrid(
            temp,
            col="doc",
            sharey=False,
            col_wrap=col_wrap,
        )
        g.map(sns.barplot, "tfidf", "token")
        g.set_titles("{col_name}")
    elif plot_type == "group":
        g = sns.barplot(temp, y="token", x="tfidf", hue="doc")
    g.set(xlabel=None, ylabel=None)
    plt.show()


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


def facet_bar_plot(
    data: pd.DataFrame,
    plot_type: str = "sep",
    col_wrap: int = 2,
    heigth: int = 4,
    y="token",
    x="count",
    ylabel="token",
    xlabel="count",
):
    """

    :param data: DataFrame of ngrams tokens with columns: [x, y, "doc"]
    where x has type supported by sns.barplot such as float.
    :param plot_type: "group" or "sep"
    :param col_wrap:
    :param heigth:
    :param y: Variable on the y axis
    :param x: Variables on the x axis
    :param ylabel:
    :param xlabel:
    :return:
    """

    if plot_type not in PLOT_TYPES:
        LOGGER.error(
            f"{plot_type} is not implemented. For now you can use "
            f"only {PLOT_TYPES}"
        )
        raise NotImplementedError

    if plot_type == "sep":
        g = sns.FacetGrid(
            data,
            col="doc",
            sharey=False,
            col_wrap=col_wrap,
            height=heigth,
            aspect=5 / heigth,
        )
        g.map(sns.barplot, x, y)
        g.set_titles("{col_name}")
    elif plot_type == "group":
        plt.figure(figsize=(5, len(data) / 5))
        g = sns.barplot(data, x="count", y="token", hue="doc")
    g.set(xlabel=xlabel, ylabel=ylabel)
    plt.show()


def plot_motif_histogram(
    ngrams: pd.DataFrame,
    stat: str = "count",
    n_tokens: int = 15,
    plot_type: str = "group",
):
    """

    :param ngrams: DataFrame of ngrams tokens with columns: ["token", "doc"]
    :param stat: One of ["count", "proportion", "percent"]
    :param n_tokens: Max number of tokens to plot
    :param plot_type:
    :return:
    """
    freq = ngrams.groupby("doc").token.value_counts()
    if stat == "proportion" or stat == "percent":
        freq = freq / freq.groupby("doc").sum()
        if stat == "percent":
            freq *= 100
    freq = (
        freq.sort_values(ascending=False)
        .groupby("doc")
        .head(n_tokens)
        .reset_index()
    )

    facet_bar_plot(freq, plot_type=plot_type, xlabel=stat)
