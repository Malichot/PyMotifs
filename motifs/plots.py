import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from motifs.config import LOGGER

TF_IDF_PLOT_TYPES = ["sep"]


def plot_tf_idf(
    tf_idf: pd.DataFrame,
    n_motifs: int = 20,
    plot_type: str = "sep",
    col_wrap: int = 3,
    show: bool = True,
):
    if plot_type not in TF_IDF_PLOT_TYPES:
        LOGGER.error(
            f"{plot_type} is not implemented. For now you can use "
            f"only {TF_IDF_PLOT_TYPES}"
        )
        raise NotImplementedError

    if plot_type == "sep":
        g = sns.FacetGrid(
            tf_idf.groupby("filename")
            .head(n_motifs)
            .sort_values(by="filename"),
            col="filename",
            sharey=False,
            col_wrap=col_wrap,
        )
        g.map(sns.barplot, "tfidf", "motif")
        g.set(xlabel=None, ylabel=None)
        g.set_titles("{col_name}")

    if show:
        plt.show()
