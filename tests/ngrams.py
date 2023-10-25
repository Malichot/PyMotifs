import pandas as pd

from motifs.config import LOGGER
from motifs.utils import ngrams_to_text


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

    # To this in unit tests
    # simple test: first token in n_gram is token
    assert all(
        df_ngrams["ngram_text"].str.split(" ").apply(lambda x: x[0])
        == df_ngrams["text"]
    )
    assert all(
        df_ngrams["ngram_token"].str.split(" ").apply(lambda x: x[0]).values
        == data["token"].values[: -(n - 1)]
    )

    return df_ngrams
