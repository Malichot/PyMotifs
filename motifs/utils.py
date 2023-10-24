import numpy as np
import pandas as pd


def ngrams(tokens: list, n: int) -> list[tuple]:
    return list(zip(*[tokens[i:] for i in range(n)]))


def ngrams_to_text(tokens: list, n: int):
    tokens = ngrams(tokens, n)
    return [" ".join(t) for t in tokens]


def transform_token_to_ngrams(data: pd.DataFrame, n: int):
    df_ngrams = pd.DataFrame(
        [
            ngrams_to_text(data["text"].values.tolist(), n),
            ngrams_to_text(data["motif"].values.tolist(), n),
        ],
        index=["ngram_text", "ngram_motif"],
    ).T
    assert len(data) - len(df_ngrams) == n - 1

    df_ngrams["text"] = data["text"].values[: -(n - 1)]
    df_ngrams["filename"] = data["filename"].values[: -(n - 1)]

    # simple test: first token in n_gram is token
    assert all(
        df_ngrams["ngram_text"].str.split(" ").apply(lambda x: x[0])
        == df_ngrams["text"]
    )
    assert all(
        df_ngrams["ngram_motif"].str.split(" ").apply(lambda x: x[0]).values
        == data["motif"].values[: -(n - 1)]
    )

    return df_ngrams


def motif_tfidf(data, log=False):
    doc_names = list(set(data["filename"]))
    # frequency of each motif per document
    counts = (
        pd.DataFrame(data.groupby("filename")["motif"].value_counts())
        .pivot_table(columns="filename", index="motif", values="count")
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
    tfidf.columns = ["motif", "filename", "tfidf"]

    return tfidf
