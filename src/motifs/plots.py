import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.multivariate.pca import PCA

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
    ratio = pca.eigenvals / np.sum(pca.eigenvals)
    plots = sns.barplot(ratio * 100)

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


def pca_variable_plot(pca: PCA):
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
