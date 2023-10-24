import json
import os
from typing import Any, Optional

import pandas as pd
import spacy_udpipe
from spacy.matcher import Matcher
from spacy_udpipe.utils import LANGUAGES as UDPIPE_LANGUAGES
from spacy_udpipe.utils import MODELS_DIR as UDPIPE_MODELS_DIR

from motifs.config import LOGGER, PKG_DATA_PATH

BASE_MOTIFS = json.load(open(f"{PKG_DATA_PATH}/fr_motifs.json", "r"))


class MotifTokenizer:
    """
    This pipeline transforms a corpus of documents to tokens with linguistic
    informations and motifs.

    :param corpus_dir: The folder where the corpus is located. The corpus
    must contain at least one document as a .txt file.
    :param motifs: A dictionary of motif with the following structure
    dict[list[dict[any]], that is for example: {"motif1": pattern1,
    "motif2": pattern2}, where each pattern is a rule for the token Matcher
    (see https://spacy.io/usage/rule-based-matching for more details).
    A simple example with one motif "ADJ" would be:
    `motif = {"ADJ": [{"POS": "ADJ"}]}`. Each token with `pos` attribute
    equal to "ADJ" will be annotated with the motif "ADJ".
    You can check the `BASE_MOTIFS` for more examples and spacy Matcher to
    create your own motif.
    :param output_dir: The folder where to save the outputs.
    :param lang: language for the udpipe model (default is "fr")

    :Example:

    >>> pipe = Pipeline(path, motifs=BASE_MOTIFS, output_dir="output_pipeline")
    >>> data = pipe.transform(save=True)
    """

    def __init__(
        self,
        corpus_dir: str,
        motifs: dict[list[dict[Any]]] = BASE_MOTIFS,
        output_dir: Optional[str] = None,
        lang: str = "fr",
    ):
        self.corpus_dir = corpus_dir
        self.output_dir = output_dir
        if self.output_dir is not None:
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

        self.corpus_path = {
            f: os.path.join(self.corpus_dir, f)
            for f in filter(
                lambda p: p.endswith("txt"), os.listdir(self.corpus_dir)
            )
        }
        self.motifs = motifs

        if not UDPIPE_LANGUAGES[lang] in os.listdir(UDPIPE_MODELS_DIR):
            spacy_udpipe.download(lang)
        self.nlp = spacy_udpipe.load(lang)

    @staticmethod
    def preprocessing(text: str) -> str:
        return text.replace("’", "'").replace("'", "'")

    @staticmethod
    def load_txt(path) -> str:
        """
        :param path: file path
        :return: content of file
        """
        try:
            with open(path, mode="r") as f:
                content = f.read()
                if len(content) > 0:
                    return content
                else:
                    LOGGER.warning(f"{path} seems to be empty! Ignoring it")
        except Exception as exc:
            LOGGER.exception(f"Error while loading {path}...")
            raise exc

    def transform_text(self, text: str, validate: bool = False):
        """
        Transform a text to tokens with linguistic informations and motifs
        :param text:
        :param validate: Validate Matcher pattern, see Spacy
        :return: data, a DataFrame with columns ["text", "lemma", "pos",
        "morph", "dep", "n_lefts", "n_rights", "motif"]. See token Spacy
        documentation for more information.
        """
        text = self.preprocessing(text)
        doc = self.nlp(text)

        # Initialized dataframe
        # Initialized motif column with lemma
        data = pd.DataFrame(
            (
                (
                    token.text,
                    token.lemma_,
                    token.pos_,
                    token.morph,
                    token.dep_,
                    token.n_lefts,
                    token.n_rights,
                    token.lemma_,
                )
                for token in self.nlp(text)
            ),
            columns=[
                "text",
                "lemma",
                "pos",
                "morph",
                "dep",
                "n_lefts",
                "n_rights",
                "motif",
            ],
        )
        # Initialize matcher
        matcher = Matcher(self.nlp.vocab, validate=validate)
        for m in self.motifs:
            matcher.add(m, [self.motifs[m]])
        # Apply it to the doc
        matches = matcher(doc)
        # Get the motif for each match
        motif_match = [
            [start, end, self.nlp.vocab.strings[match_id]]
            for match_id, start, end in matches
        ]
        motif_match = pd.DataFrame(
            motif_match, columns=["start", "end", "motif"]
        )
        # We must be sure that the matches correspond to one token
        matches_length = motif_match["end"] - motif_match["start"]
        errors = motif_match[matches_length > 1]
        if len(errors) > 0:
            to_print = [
                f"span: ({start}, {end}), text: {doc[start:end]}, motif: "
                f"{m}"
                for _, (start, end, m) in errors.iterrows()
            ]
            "\n".join(to_print)
            raise AssertionError(
                "There is a problem with the motif matches. The matchers "
                "returned more than one token at the following spans"
                f"{to_print}!"
            )
        motif_match.set_index("start", inplace=True)
        # Modify motif column if we found a match
        data.loc[motif_match.index, "motif"] = motif_match.loc[:, "motif"]

        return data

    def transform_corpus(self, save: bool = False, **kwargs):
        for file in self.corpus_path:
            data = self.transform_text(
                self.load_txt(self.corpus_path[file]), **kwargs
            )
            # Add filename columns
            data["filename"] = file
            if save:
                assert self.output_dir is not None
                filename = file.split(".txt")[0]
                data.to_csv(f"{self.output_dir}/{filename}.csv", index=False)
            yield data

    def transform(self, save: bool = False, **kwargs):
        return pd.concat(
            [d for d in self.transform_corpus(save=save, **kwargs)],
            ignore_index=True,
        )
