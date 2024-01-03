import math
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_array
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

    ngrams = pd.DataFrame()
    for f in data.doc.unique():
        temp = transform_token_to_ngrams(data[data["doc"] == f], n)
        # Add first word of each n-gram
        temp["word"] = data.loc[data["doc"] == f, "text"].values[: -n + 1]
        ngrams = pd.concat([ngrams, temp], ignore_index=True)

    ngrams = ngrams.drop("text", axis=1).rename(
        {"ngram_text": "text", "ngram_token": "token"}, axis=1
    )

    return ngrams[["word", "text", "token", "doc"]]


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

    # Cache to efficiently compute hypergeom cdf for every row
    cdf = {}

    def hypergeom_cdf(f, t, T, F):
        key = (f, t, T, F)
        if key not in cdf:
            val = hypergeom.cdf(k=f, M=T, n=F, N=t)
            cdf[key] = val
        else:
            val = cdf[key]
        return val

    # Positive spec
    corpus_grams.loc[spec_pos, "probas"] = corpus_grams.loc[spec_pos].apply(
        lambda row: hypergeom_cdf(
            f=row["f"] - 1, t=row["t"], T=row["T"], F=row["F"]
        ),
        axis=1,
    )
    # Negative spec
    corpus_grams.loc[spec_neg, "probas"] = corpus_grams.loc[spec_neg].apply(
        lambda row: hypergeom_cdf(
            f=row["f"], t=row["t"], T=row["T"], F=row["F"]
        ),
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
    spec = spec.join(corpus_grams, how="left")
    spec["rel_f"] = spec["f"] / spec["t"]
    spec.drop(["F", "T", "mod", "probas"], axis=1, inplace=True)

    return spec


def build_cooccurrence_matrix(
    data: pd.DataFrame, by: str = "window"
) -> Tuple[csr_array, list[str], list[any], list[int], list[int]]:
    """
    Compute the cooccurrence matrix of the tokens within each `by` value.
    The input DataFrame must have a columns "token" (the variable of
    interest) and a column named by the `by` parameter which defines the id of
    each part within a text.
    For example, we wish to compute the cooccurrence of the words at the
    sentence level within a text. The `by` will be the sentence id (
    "sent_id") and the tokens will correspond to each word in each sentence.

    The cooccurrence of two tokens, A and B, in a text within a context (for
    example a sentence) is defined as the number of pairs (A,B) that exists
    in a sentence within the text. A way to compute it, is to first
    calculate the number of pairs within each sentence of the text and sum
    them.
    To calculate, the number of pairs (A,B) within a sentence, we compute
    the frequences of A and B in the sentence, to obtain $f_A$, $f_B$. The
    number of pairs is then simply $f_A\times f_B$.

    :param data: DataFrame with columns ["token", by]
    :param by: Name of the variable defining the window on which the
    cooccurrence is computed
    :return:
    """
    if not set(["token", by]).issubset(data.columns):
        LOGGER.error(
            "Missing columns in data DataFrame with columns: "
            f"{data.columns}"
        )
        raise ValueError
    data = data.loc[:, ["token", by]]
    data = data.groupby(by)["token"].value_counts().reset_index()

    rows, row_pos = np.unique(data["window"], return_inverse=True)
    cols, col_pos = np.unique(data["token"], return_inverse=True)
    occu = csr_array(
        (data["count"], (row_pos, col_pos)),
        shape=(len(rows), len(cols)),
    )
    cooc = occu.T.dot(occu)

    # Diagonal: cooccurrence of a token with itself, number of pair in a set
    # of f_i elements where f_i is the frequence of the token i within each
    # sentence.
    data["diag"] = data["count"].apply(lambda x: math.comb(x, 2))
    cooc.setdiag(data.groupby("token")["diag"].sum().loc[cols], k=0)

    return cooc, rows, cols, row_pos, col_pos
