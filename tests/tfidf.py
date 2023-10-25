def tfidf_from_corpus(corpus: dict):
    """
    Source: http://www.sefidian.com/2022/07/28/understanding-tf-idf-with-python-example/

    :param corpus:
    :return:
    """
    words_set = set()

    for f, words in corpus.items():
        words_set = words_set.union(set(words))

    n_docs = len(corpus)  # ·Number of documents in the corpus
    n_words_set = len(words_set)  # ·Number of unique words in the

    df_tf = pd.DataFrame(
        np.zeros((n_docs, n_words_set)),
        columns=list(words_set),
        index=corpus.keys(),
    )

    # Compute Term Frequency (TF)
    for f in corpus.keys():
        words = corpus[f]  # Words in the document
        for w in words:
            df_tf[w][f] = df_tf[w][f] + (1 / len(words))

    idf = {}
    for w in words_set:
        k = 0  # number of documents in the corpus that contain this word

        for f in corpus.keys():
            if w in corpus[f]:
                k += 1

        idf[w] = np.log(n_docs / k)

    df_tf_idf = df_tf.copy()
    for w in words_set:
        for f in corpus.keys():
            df_tf_idf[w][f] = df_tf[w][f] * idf[w]

    return df_tf, idf, df_tf_idf


if __name__ == "__main__":
    import time

    import numpy as np
    import pandas as pd

    from motifs.config import PKG_DATA_PATH
    from motifs.tokenizer import MotifTokenizer
    from motifs.utils import motif_tfidf, transform_token_to_ngrams

    CORPUS_PATH = PKG_DATA_PATH.joinpath("corpus_test")
    pipe = MotifTokenizer(CORPUS_PATH)
    data = pipe.transform()

    filenames = list(set(data["filename"]))

    n = 4
    data_ngrams = pd.concat(
        [
            transform_token_to_ngrams(data[data["filename"] == f], n)
            for f in filenames
        ],
        ignore_index=True,
    )
    data_ngrams = data_ngrams.drop("text", axis=1).rename(
        {"ngram_text": "text", "ngram_motif": "motif"}, axis=1
    )

    # tf idf
    corpus = {
        f: data_ngrams[data_ngrams["filename"] == f]["motif"].values.tolist()
        for f in filenames
    }

    t1 = time.time()
    df_tf, idf, df_tf_idf = tfidf_from_corpus(corpus)
    t2 = time.time()

    df_tf_idf.sort_index(axis=0, inplace=True)
    df_tf_idf.sort_index(axis=1, inplace=True)

    t1 = time.time()
    tf_idf = motif_tfidf(data_ngrams, log=True)
    t2 = time.time()
    my_df = tf_idf.pivot_table(
        index="filename", columns="motif", values="tfidf"
    )

    assert all((my_df == df_tf_idf).all())
