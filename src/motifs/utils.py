import os

import numpy as np
import pandas as pd


def load_tokens_from_directory(dir_):
    files = [f for f in os.listdir(dir_) if f.endswith(".csv")]
    tokens = pd.concat(
        [pd.read_csv(f"{dir_}/{f}") for f in files], ignore_index=True
    )
    return tokens


def build_window_corpus(ngrams: pd.DataFrame, seq_length: int):
    window_ngram = pd.DataFrame()
    for doc in ngrams.doc.unique():
        window = build_window_data(ngrams[ngrams["doc"] == doc], seq_length)
        window["doc"] = doc
        window_ngram = pd.concat([window_ngram, window], ignore_index=True)

    return window_ngram


def build_window_data(ngrams: pd.DataFrame, seq_length: int):
    p = ngrams[["text", "token"]].values
    window = np.zeros((p.shape[0], 3), dtype=object)
    for i in range(0, len(ngrams) - 1, seq_length):
        window[i : i + seq_length, 0] = str(i)
        window[i : i + seq_length, 1] = p[i : i + seq_length, 0]
        window[i : i + seq_length, 2] = p[i : i + seq_length, 1]

    window = pd.DataFrame(window, columns=["window", "text", "token"])
    window["window"] = window["window"].astype(int)

    return window
