import json
import os
from typing import Optional

import pandas as pd
import spacy_udpipe
from spacy.matcher import Matcher
from spacy_udpipe.utils import LANGUAGES as UDPIPE_LANGUAGES
from spacy_udpipe.utils import MODELS_DIR as UDPIPE_MODELS_DIR

from motifs.config import LOGGER, PKG_DATA_PATH

BASE_MOTIFS = json.load(open(f"{PKG_DATA_PATH}/fr_motifs.json", "r"))


# WORDS_PATH = os.path.join(PKG_DATA_PATH, "french")
#
# MORPH_BASED = {
#     "INF": {"VerbForm": "Inf"},
#     "PPAS": {"Tense": "Past", "VerbForm": "Part"},
#     "PPRES": {"Tense": "Pres", "VerbForm": "Part"},
#     "VSUBP": {"Mood": "Sub", "Tense": "Pres"},
#     "VSUBI": {"Mood": "Sub", "Tense": "Imp"},
#     "IMP": {"Mood": "Imp", "Tense": "Pres"},
#     "VCOND": {"Mood": "Cnd", "Tense": "Pres"},
#     "PRES": {"Mood": "Ind", "Tense": "Pres"},
#     "VIMP": {"Mood": "Ind", "Tense": "Imp"},
#     "VPS": {"Mood": "Ind", "Tense": "Past"},
#     "VF": {"Mood": "Ind", "Tense": "Fut"},
#     "DETPOSS": {"Poss": "Yes", "PronType": "Prs"},
# }
# POS_BASED = {
#     "ADV": "ADV",
#     "ADJ": "ADJ",
#     "NUM": "NUM",
#     "DETPOSS": "DETPOSS",
#     "NOUN": "NC",
#     "PROPN": "PROPN",
#     "INTJ": "INTJ",
# }
# PRONOUNS = {
#     # 1st
#     "je": "je",
#     "Je": "je",
#     "j": "je",  # TODO: J -> J' -> je
#     "J": "je",  # TODO: J -> J' -> je
#     "me": "je",  # TODO: me -> je or me -> me comme te -> te
#     "Me": "je",
#     # 2nd
#     "tu": "tu",
#     "Tu": "tu",
#     "te": "te",
#     "Te": "te",
#     # 3rd
#     "Il": "il",
#     "il": "il",
#     "Elle": "elle",
#     "elle": "elle",
#     "Se": "se",
#     "se": "se",
#     # 4th & 5th
#     "Nous": "nous",
#     "nous": "nous",
#     "Vous": "vous",
#     "vous": "vous",
#     # 6th
#     "Ils": "ils",
#     "ils": "ils",
#     "Elles": "elles",
#     "elles": "elles",
# }
# UNINFLECTED_WORDS = read_txt(os.path.join(WORDS_PATH, "mots_invariables.txt"))
#
# WORD_GROUPS = {
#     "ADVTOT": read_txt(os.path.join(WORDS_PATH, "adverbes_tot.txt")),
#     "ADVPHA": read_txt(os.path.join(WORDS_PATH, "adverbes_phase.txt")),
#     "ADVFRE": read_txt(os.path.join(WORDS_PATH, "adverbes_freq.txt")),
#     "ADVINT": read_txt(os.path.join(WORDS_PATH, "adverbes_intensite.txt")),
#     "ADVHAB": read_txt(os.path.join(WORDS_PATH, "adverbes_habitude.txt")),
#     "ADVMOD": read_txt(os.path.join(WORDS_PATH, "adverbes_modaux.txt")),
#     "ADVMAN": read_txt(os.path.join(WORDS_PATH, "adverbes_maniere.txt")),
#     "NCCOR": read_txt(os.path.join(WORDS_PATH, "parties_corps.txt")),
#     "NCABS": read_txt(os.path.join(WORDS_PATH, "noms_abstraits.txt")),
# }
#
# POS_DEP_BASED = {
#     "NCs": {"pos": "NOUN", "dep": "nsubj"},
#     "NCspas": {"pos": "NOUN", "dep": "nsubj:pass"},
#     "NCo": {"pos": "NOUN", "dep": "obj"},
#     "NCmod": {"pos": "NOUN", "dep": "nmod"},
# }


class Pipeline:
    """

    :param corpus_dir:
    :param output_dir:
    :param lang:
    :param motifs:

    :Example:

    >>> pipe = Pipeline(path, output_dir="output_pipeline")
    >>> data = pipe.transform(save=True)
    """

    def __init__(
        self,
        corpus_dir: str,
        output_dir: Optional[str] = None,
        lang: str = "fr",
        motifs: list[dict] = BASE_MOTIFS,
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

    def transform_text(self, text: str):
        """
        Transform a text to tokens with linguistic informations and motifs
        :param text:
        :return: data, a DataFrame with columns ["word", "lemma", "pos",
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
                "word",
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
        matcher = Matcher(self.nlp.vocab, validate=True)
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

    def transform_corpus(self, save: bool = False):
        for file in self.corpus_path:
            data = self.transform_text(self.load_txt(self.corpus_path[file]))
            # Add filename columns
            data["filename"] = file
            if save:
                assert self.output_dir is not None
                filename = file.split(".txt")[0]
                data.to_csv(f"{self.output_dir}/{filename}.csv", index=False)
            yield data

    def transform(self, save: bool = False):
        return pd.concat(
            [d for d in self.transform_corpus(save=save)], ignore_index=True
        )

    # @staticmethod
    # def transform_token(token) -> str:
    #     motif = token.lemma_
    #     # Remplacement des pos auxiliaires pour les conserver dans les motifs :
    #     # Preprocessing
    #     if token.lemma_ == "être":
    #         return "être"
    #     elif token.lemma_ == "avoir":
    #         return "avoir"
    #     # Guillemets anglais :
    #     elif token.lemma_ == "«":
    #         return '"'
    #     elif token.lemma_ == "»":
    #         return '"'
    #     else:
    #         # Pour les mots invariables, le motif est le mot:
    #         if token.text in UNINFLECTED_WORDS:
    #             return token.text
    #
    #         # S/O : prise en compte des fonctions grammaticales.
    #         for m in POS_DEP_BASED:
    #             m_cond = POS_DEP_BASED[m]
    #             if token.pos_ == m_cond["pos"] and token.dep_ == m_cond[
    #                 "dep"]:
    #                 return m
    #
    #         # Conservation des autres étiquettes morphosyntaxiques restantes :
    #         # pos
    #         for p in POS_BASED:
    #             if token.pos_ == p:
    #                 return POS_BASED[p]
    #
    #         morph = token.morph.to_dict()
    #         for m in MORPH_BASED:
    #             m_cond = MORPH_BASED[m]
    #             if all([morph.get(c) == m_cond[c] for c in m_cond]):
    #                 return m
    #
    #         # TODO: assert that intersection of words lists is empty: no overlaps
    #         for m in WORD_GROUPS:
    #             if token.lemma_ in WORD_GROUPS[m]:
    #                 return m
    #
    #         # Conservation des pronoms personnels :
    #         for w in PRONOUNS:
    #             if token.text == w:
    #                 return PRONOUNS[w]
    #
    #     return motif
