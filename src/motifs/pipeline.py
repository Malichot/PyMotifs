import datetime as dt
import os
from typing import Optional

import seaborn as sns

from motifs.config import LOGGER
from motifs.constants import AVAILABLE_FEATURES, AVAILABLE_METHODS
from motifs.features import (
    build_tfidf,
    build_token_freq,
    transform_corpus_to_ngrams,
)
from motifs.pca import pca_transform
from motifs.plots import plot_motif_histogram, plot_tf_idf
from motifs.tokenizer import Tokenizer
from motifs.utils import load_tokens_from_directory

"""
feature = [{
    "name": "freq",
    "params": {}
}]
"""


def verify_feature(feature: dict):
    assert isinstance(feature, dict)
    assert feature.get("name") is not None

    if feature["name"] not in AVAILABLE_FEATURES:
        LOGGER.error(
            f"The feature {feature['name']} is not implemented! Available "
            f"features are {AVAILABLE_FEATURES}"
        )
        raise NotImplementedError


class Pipeline:
    """

    :param token_type: type of the token to use for the analysis. Should be
    one of ["text", "lemma", "pos", "motif"]
    :param feature: Feature's configuration.
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
        feature: dict,
        n: int,
        tokens_dir: Optional[str] = None,
        corpus_dir: Optional[str] = None,
        save: bool = True,
        **kwargs,
    ):
        self.token_type = token_type
        verify_feature(feature)
        self.feature = feature
        self.n = n
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

    def transform_to_ngrams(self):
        self.__ngrams = transform_corpus_to_ngrams(self.tokens, self.n)
        # Remove empty cells (just in case)
        empty_cells = self.__ngrams.apply(lambda x: x.apply(len)) != 0
        self.__ngrams = self.__ngrams[empty_cells.all(axis=1)]

    def get_features(self):
        """

        :param method:
        :param n: n-gram length
        :param plot:
        :param kwargs:
        :return:
        """
        if self.__ngrams is None:
            LOGGER.error(
                "You must first call `transform_to_ngrams` to get ngrams!"
            )
            raise AssertionError
        # Remove empty cells (just in case)
        empty_cells = self.__ngrams.apply(lambda x: x.apply(len)) != 0
        self.__ngrams = self.__ngrams[empty_cells.all(axis=1)]

        if self.feature["name"] == "tfidf":
            features_data, _ = build_tfidf(self.__ngrams)
        elif self.feature["name"] == "freq":
            features_data = build_token_freq(
                self.__ngrams,
                **self.feature.get(
                    "params", {"freq_filter": 2, "n_motifs": None}
                ),
            )
        else:
            raise NotImplementedError

        return features_data

    def execute(self, method: str, plot: bool = False, **kwargs):
        """

        :param method:
        :param n: n-gram length
        :param plot:
        :param kwargs:
        :return:
        """
        assert method in AVAILABLE_METHODS
        self.transform_to_ngrams()
        if plot:
            # Plot the count of tokens for each document within the corpus
            sns.countplot(self.ngrams, x="doc")
            # Plot distribution of tokens
            plot_motif_histogram(self.ngrams, **kwargs)

        self.__features_data = self.get_features()
        if plot:
            if self.feature["name"] == "tfidf":
                plot_tf_idf(
                    self.features_data,
                    n_tokens=kwargs.get("n_tokens", 20),
                    plot_type=kwargs.get("plot_type", "sep"),
                    col_wrap=kwargs.get("col_wrap", 3),
                )

        if method == "pca":
            pca = pca_transform(
                self.features_data.pivot_table(
                    index="token", columns=["doc"], values=self.feature["name"]
                ),
                plot=plot,
            )
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
