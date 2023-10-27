import numpy as np
import pandas as pd

from motifs.config import LOGGER


def ngrams(tokens: list, n: int) -> list[tuple]:
    return list(zip(*[tokens[i:] for i in range(n)]))


def ngrams_to_text(tokens: list, n: int):
    tokens = ngrams(tokens, n)
    return [" ".join(t) for t in tokens]


def transform_corpus_to_ngrams(data: pd.DataFrame, n: int) -> pd.DataFrame:
    """

    :param data:
    :param n: n-gram length
    :return:
    """
    data = pd.concat(
        [
            transform_token_to_ngrams(data[data["piece"] == f], n)
            for f in data.piece.unique()
        ],
        ignore_index=True,
    )
    data = data.drop("text", axis=1).rename(
        {"ngram_text": "text", "ngram_token": "token"}, axis=1
    )
    return data


def transform_token_to_ngrams(data: pd.DataFrame, n: int) -> pd.DataFrame:
    if "token" not in data.columns:
        LOGGER.error("The target column is not in the input data!")
        raise AssertionError

    df_ngrams = pd.DataFrame(
        [
            ngrams_to_text(data["text"].values.tolist(), n),
            ngrams_to_text(data["token"].values.tolist(), n),
        ],
        index=["ngram_text", "ngram_token"],
    ).T
    assert len(data) - len(df_ngrams) == n - 1

    df_ngrams["text"] = data["text"].values[: -(n - 1)]
    df_ngrams["piece"] = data["piece"].values[: -(n - 1)]

    return df_ngrams


def build_tfidf(data: pd.DataFrame, log: bool = True) -> pd.DataFrame:
    if not set(["token", "piece"]).issubset(set(data.columns)):
        LOGGER.error(
            "Wrong columns names: ['token', 'piece'] are not in the data"
        )
        raise AssertionError

    doc_names = list(set(data["piece"]))
    # frequency of each motif per document
    counts = (
        pd.DataFrame(data.groupby("piece")["token"].value_counts())
        .pivot_table(columns="piece", index="token", values="count")
        .fillna(0)
        .astype(int)
    )

    # Count total number of motif per document
    n_tokens_per_doc = np.sum(counts, axis=0)
    # Term freq: motifs freq per doc / total number of motifs per doc
    tf = counts / n_tokens_per_doc
    # Document freq: Count appearance of each motif over the document
    df = np.sum(counts != 0, axis=1)
    idf = len(doc_names) / df
    if log:
        idf = np.log(idf)

    tfidf = tf * idf.loc[tf.index].values.reshape(-1, 1)
    # Reorganise
    tfidf = pd.DataFrame(
        tfidf.stack().sort_values(ascending=False)
    ).reset_index()
    tfidf.columns = ["token", "piece", "tfidf"]

    return tfidf


def build_token_freq(data, freq_filter=2, n_motifs=None) -> pd.DataFrame():
    if not set(["token", "piece"]).issubset(set(data.columns)):
        LOGGER.error(
            "Wrong columns names: ['token', 'piece'] are not in the data"
        )
        raise AssertionError

    # Denumbering
    data = (
        data.groupby("piece")["token"]
        .value_counts(sort=True, ascending=False)
        .reset_index()
    )
    # Filtering low freq motifs
    if n_motifs is not None:
        # Select n first motifs for each text
        data = data.groupby("piece").head(10)
    else:
        assert freq_filter is not None, "You must give a freq_filter!"
        # Select motifs that appear at least freq_filter times
        data = data[data["count"] >= freq_filter]
    data.rename({"count": "freq"}, axis=1, inplace=True)

    return data
