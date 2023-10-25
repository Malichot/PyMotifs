import datetime as dt
import os
from typing import Optional

import pandas as pd

from motifs.config import LOGGER
from motifs.constants import AVAILABLE_FEATURES, AVAILABLE_METHODS
from motifs.plots import (
    pca_variable_plot,
    plot_explained_variance_ratio,
    plot_pca_projection,
    plot_tf_idf,
)
from motifs.tokenizer import Tokenizer
from motifs.utils import (
    build_tfidf,
    build_token_freq,
    transform_corpus_to_ngrams,
)

"""
feature = [{
    "name": "freq",
    "params": {}
}]
"""


def verify_feature(features: list[dict]):
    assert isinstance(features, list)

    if len(features) > 1:
        LOGGER.error(
            "You passed more than one feature. This is not " "implemented."
        )
        raise NotImplementedError

    assert all([isinstance(f, dict) for f in features])
    assert all([f.get("name") is not None for f in features])

    for f in features:
        if f["name"] not in AVAILABLE_FEATURES:
            LOGGER.error(
                f"This feature is not implemented! Available features are"
                f" {AVAILABLE_FEATURES}"
            )
            raise NotImplementedError


def load_tokens_from_directory(dir_):
    files = [f for f in os.listdir(dir_) if f.endswith(".csv")]
    tokens = pd.concat(
        [pd.read_csv(f"{dir_}/{f}") for f in files], ignore_index=True
    )
    return tokens


class Pipeline:
    """

    :param token_type: type of the token to use for the analysis. Should be
    one of ["text", "lemma", "pos", "motif"]
    :param features: list of features configuration. For now only a list of
    a single feature is implemented.
    :param tokens_dir: The folder where the tokens for each text is located.
    The tokens should be stored in a csv file obtained from `transform_corpus`
    of `motifs.tokenizer.Tokenizer`. This is used by default.
    :param corpus_dir: If the tokens_dir is not provided, then the Pipeline
    will perform tokenization on corpus_dir (cf motifs.tokenizer.Tokenizer)
    :param save:
    :param kwargs:
    """

    def __init__(
        self,
        token_type: str,
        features: list[dict],
        tokens_dir: Optional[str] = None,
        corpus_dir: Optional[str] = None,
        save: bool = True,
        **kwargs,
    ):
        self.token_type = token_type
        verify_feature(features)
        self.features = features
        if save:
            self.output_dir = (
                f"{os.getcwd()}/"
                + f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_pipeline"
            )
            if not os.path.isdir(self.output_dir):
                LOGGER.debug(
                    f"Creating output destination at {self.output_dir}"
                )
                os.makedirs(self.output_dir)
            else:
                LOGGER.debug(
                    f"The destination folder {self.output_dir} already "
                    f"exists, outputs will be overwritten!"
                )
            tokenizer_dir = os.path.join(self.output_dir, "tokens")

        else:
            self.output_dir = None
            tokenizer_dir = None

        if tokens_dir is not None:
            self.__tokens = load_tokens_from_directory(tokens_dir)
        else:
            if corpus_dir is None:
                LOGGER.error("You must pass tokens_dir or corpus_dir!")
                raise ValueError
            self.tokenizer = Tokenizer(
                corpus_dir=corpus_dir,
                token_type=token_type,
                output_dir=tokenizer_dir,
                **kwargs,
            )
            self.__tokens = self.tokenizer.transform(save=save)

        self.__tokens.rename({self.token_type: "token"}, axis=1, inplace=True)

        self.__features_data = None
        self.__ngrams = None
        self.__transformer = None

    def execute(self, method: str, n: int, plot: bool = False, **kwargs):
        """

        :param method:
        :param n: n-gram length
        :param plot:
        :param kwargs:
        :return:
        """
        assert method in AVAILABLE_METHODS
        self.__ngrams = transform_corpus_to_ngrams(self.tokens, n)

        # Remove empty cells (just in case)
        empty_cells = self.__ngrams.apply(lambda x: x.apply(len)) != 0
        self.__ngrams = self.__ngrams[empty_cells.all(axis=1)]

        for feature in self.features:
            if feature["name"] == "tfidf":
                self.__features_data = build_tfidf(
                    self.__ngrams, **feature.get("params", {"log": True})
                )
                if plot:
                    plot_tf_idf(self.__features_data, **kwargs)
            elif feature["name"] == "freq":
                self.__features_data = build_token_freq(
                    self.__ngrams,
                    **feature.get(
                        "params", {"freq_filter": 2, "n_motifs": None}
                    ),
                )
            else:
                raise NotImplementedError

        if method == "pca":
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            temp = self.features_data.pivot_table(
                index="token", columns=["piece"], values=feature["name"]
            )
            pieces = temp.columns

            # Normalize the input data
            scaler = StandardScaler()
            temp = scaler.fit_transform(temp)
            temp = pd.DataFrame(temp, columns=pieces)

            pca = PCA(n_components=temp.shape[-1])
            pca.fit(temp)

            if plot:
                plot_explained_variance_ratio(pca)
                plot_pca_projection(pca, pieces)
                pca_variable_plot(temp, pca, colwrap=4)

            self.__transformer = pca

    @property
    def tokens(self):
        return self.__tokens

    @property
    def features_data(self):
        return self.__features_data

    @property
    def ngrams(self):
        return self.__ngrams

    @property
    def transformer(self):
        return self.__transformer
