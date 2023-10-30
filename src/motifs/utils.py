import os

import pandas as pd


def load_tokens_from_directory(dir_):
    files = [f for f in os.listdir(dir_) if f.endswith(".csv")]
    tokens = pd.concat(
        [pd.read_csv(f"{dir_}/{f}") for f in files], ignore_index=True
    )
    return tokens


def build_window_data(ngrams, seq_length):
    window_ngram = pd.DataFrame()
    for doc in ngrams.doc.unique():
        piece = ngrams[ngrams["doc"] == doc]
        seq = []
        for i in range(0, len(piece) - 1, seq_length):
            seq.append(piece["token"].values[i : i + seq_length].tolist())
        window = pd.DataFrame(seq).stack().droplevel(1).reset_index()
        window.columns = ["window", "token"]
        window["doc"] = doc
        window_ngram = pd.concat([window_ngram, window], ignore_index=True)

    return window_ngram
