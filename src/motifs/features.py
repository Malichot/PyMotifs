import numpy as np
import pandas as pd
from scipy.stats import hypergeom

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
            transform_token_to_ngrams(data[data["doc"] == f], n)
            for f in data.doc.unique()
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
    df_ngrams["doc"] = data["doc"].values[: -(n - 1)]

    return df_ngrams


def build_tfidf(data: pd.DataFrame, log: bool = True) -> pd.DataFrame:
    if not set(["token", "doc"]).issubset(set(data.columns)):
        LOGGER.error(
            "Wrong columns names: ['token', 'doc'] are not in the data"
        )
        raise AssertionError

    doc_names = list(set(data["doc"]))
    # frequency of each motif per document
    counts = (
        pd.DataFrame(data.groupby("doc")["token"].value_counts())
        .pivot_table(columns="doc", index="token", values="count")
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
    tfidf.columns = ["token", "doc", "tfidf"]

    return tfidf


def build_token_freq(data, freq_filter=2, n_motifs=None) -> pd.DataFrame():
    if not set(["token", "doc"]).issubset(set(data.columns)):
        LOGGER.error(
            "Wrong columns names: ['token', 'doc'] are not in the data"
        )
        raise AssertionError

    # Denumbering
    data = (
        data.groupby("doc")["token"]
        .value_counts(sort=True, ascending=False)
        .reset_index()
    )
    # Filtering low freq motifs
    if n_motifs is not None:
        # Select n first motifs for each text
        data = data.groupby("doc").head(10)
    else:
        assert freq_filter is not None, "You must give a freq_filter!"
        # Select motifs that appear at least freq_filter times
        data = data[data["count"] >= freq_filter]
    data.rename({"count": "freq"}, axis=1, inplace=True)

    return data


def build_specificity(ngrams, u: float = 0.5):
    """
    CF Lafon, P. “Sur la variabilité de la fréquence des formes dans un
    corpus.” Mots, no. 1 (1980): 127-165. http://www.persee.fr/web/revues/home/
    prescript/article/mots_0243-6450_1980_num_1_1_1008

    cf: TXM at https://txm.gitpages.huma-num.fr/textometrie/files/documentation
    /Manuel%20de%20TXM%200.7%20FR.pdf

    :param ngrams:
    :param sort:
    :return:
    """
    assert 0 <= u <= 1
    # f: the frequence of a token in the part
    corpus_grams = build_token_freq(ngrams, freq_filter=0)
    corpus_grams = corpus_grams.set_index("doc")
    corpus_grams.rename({"freq": "f"}, axis=1, inplace=True)

    # Format it as table to ease computation of F, t, T
    corpus_grams = corpus_grams.pivot_table(
        index="token", columns="doc", values="f"
    )
    # F: the total frequence of a token in the corpus
    F = np.sum(corpus_grams, axis=1)  # Sum by row (token)
    # t: the length of the part
    t = np.sum(corpus_grams, axis=0)  # Sum by column (doc or part)
    # T : the total length of the corpus
    T = sum(t)

    # Reformat as a tidy dataframe and add F, t, T
    corpus_grams = corpus_grams.stack().reset_index().rename({0: "f"}, axis=1)

    corpus_grams = corpus_grams.set_index("token")
    corpus_grams["F"] = F
    corpus_grams = corpus_grams.reset_index().set_index("doc")
    corpus_grams["t"] = t
    corpus_grams["T"] = T
    corpus_grams = corpus_grams.astype(
        {"f": int, "F": int, "T": int, "t": int}
    )

    # Mode: Equation 7.25 in TXM
    corpus_grams["mod"] = (
        (corpus_grams["F"] + 1) * (corpus_grams["f"] + 1)
    ) / (corpus_grams["T"] + 2)

    # Now compute probas
    corpus_grams["probas"] = 0.0
    corpus_grams = corpus_grams.reset_index().set_index("token")
    spec_neg = corpus_grams["f"] < corpus_grams["mod"]
    spec_pos = corpus_grams["f"] >= corpus_grams["mod"]

    # Positive spec
    corpus_grams.loc[spec_pos, "probas"] = corpus_grams.loc[spec_pos].apply(
        lambda row: hypergeom.cdf(row["f"] - 1, row["T"], row["F"], row["t"]),
        axis=1,
    )
    # Negative spec
    corpus_grams.loc[spec_neg, "probas"] = corpus_grams.loc[spec_neg].apply(
        lambda row: hypergeom.cdf(row["f"], row["T"], row["F"], row["t"]),
        axis=1,
    )

    # Build spec index
    corpus_grams["spec"] = 0.0
    mask = corpus_grams["probas"] < u
    corpus_grams.loc[mask, "spec"] = np.log10(corpus_grams.loc[mask, "probas"])
    mask = corpus_grams["probas"] > u
    corpus_grams.loc[mask, "spec"] = np.abs(
        np.log10(1 - corpus_grams.loc[mask, "probas"])
    )

    # Format data to return
    spec = corpus_grams.pivot_table(
        columns="doc", index="token", values="spec"
    ).fillna(0)
    spec = spec.join(
        pd.DataFrame(corpus_grams["f"] / corpus_grams["t"]).rename(
            {0: "ref_f"}, axis=1
        )
    )
    spec = spec.join(corpus_grams[["f", "t", "doc"]])

    return spec
