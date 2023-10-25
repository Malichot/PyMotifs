from statsmodels.multivariate.pca import PCA

from motifs.config import LOGGER
from motifs.constants import AVAILABLE_FEATURES, AVAILABLE_METHODS
from motifs.plots import (
    pca_variable_plot,
    plot_explained_variance_ratio,
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


class Pipeline:
    """

    :param target: name of the target variable of interest, should be one of
    AVAILABLE_TARGET
    :param features: list of features configuration
    :param  **kwargs: extra keyword arguments of tokenizer.Tokenizer
    """

    def __init__(
        self, corpus_dir: str, token_type: str, features: list[dict], **kwargs
    ):
        self.token_type = token_type
        verify_feature(features)

        self.features = features

        self.tokenizer = Tokenizer(
            corpus_dir=corpus_dir, token_type=token_type, **kwargs
        )
        self.__tokens = self.tokenizer.transform()
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
        self.__ngrams = transform_corpus_to_ngrams(self.token, n)

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
            pca = PCA(
                self.features_data.pivot_table(
                    index="token", columns=["piece"], values=feature["name"]
                ),
                standardize=True,
            )
            if plot:
                plot_explained_variance_ratio(pca)
                pca_variable_plot(pca)
            self.__transformer = pca

    @property
    def token(self):
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
