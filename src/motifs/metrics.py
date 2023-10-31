import numpy as np
import pandas as pd
from scipy.sparse import find, triu

from motifs.config import LOGGER
from motifs.features import build_cooccurrence_matrix


def find_top_n_cooccurrence(data: pd.DataFrame, n, by: str = "window"):
    temp, _, cols, _, _ = build_cooccurrence_matrix(data, by=by)
    # Extract upper triangular matrix
    temp = triu(temp, k=0)
    r, c, v = find(temp)
    non_zeros = np.array([cols[r], cols[c], v])

    return np.sort(non_zeros, axis=-1)[:, -n:]


def corpus_top_n_cooccurence(data: pd.DataFrame, n, by: str = "window"):
    cooc = pd.DataFrame()
    for doc in data.doc.unique():
        LOGGER.debug(f"Build cooccurrence matrix for {doc}...")
        temp = data.loc[data["doc"] == doc, ["token", by]]
        temp = find_top_n_cooccurrence(temp, n, by=by)
        temp = pd.DataFrame(temp).T
        temp.columns = ["token_1", "token_2", "count"]

        temp["token"] = temp.apply(
            lambda x: '"' + x["token_1"] + '" ' + '"' + x["token_2"] + '"',
            axis=1,
        )

        temp["doc"] = doc
        cooc = pd.concat([cooc, temp], ignore_index=True)

    return cooc


def find_cooccurrent_tokens(
    token: str, data: pd.DataFrame, n, by: str = "window"
):
    cooc, _, cols, _, _ = build_cooccurrence_matrix(data, by=by)
    token_id_mapper = {c: i for i, c in enumerate(cols)}

    _, c, v = find(cooc.getrow(token_id_mapper[token]))

    non_zeros = np.array([cols[c], v])

    return np.sort(non_zeros, axis=-1)[:, -n:]
