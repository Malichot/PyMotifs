import json
import os
from typing import Optional

import pandas as pd
import spacy_udpipe

from motifs.config import LOGGER, PKG_DATA_PATH
from utils import read_txt

BASE_MOTIFS = json.load(open(f"{PKG_DATA_PATH}/motifs.json", "r"))

WORDS_PATH = os.path.join(PKG_DATA_PATH, "french")


class Pipeline:
    def __init__(
        self,
        corpus_dir: str,
        output_dir: Optional[str] = None,
        lang: str = "fr",
        motifs: list[dict] = BASE_MOTIFS,
    ):
        self.corpus_dir = corpus_dir
        self.output_dir = output_dir
        self.corpus_path = {
            f: os.path.join(self.corpus_dir, f)
            for f in filter(
                lambda p: p.endswith("txt"), os.listdir(self.corpus_dir)
            )
        }
        self.motifs = motifs

        spacy_udpipe.download(lang)
        self.nlp = spacy_udpipe.load(lang)
        if lang == "fr":
            self._uninflected_words = read_txt(
                os.path.join(WORDS_PATH, "mots_invariables.txt")
            )
            self._advtot = read_txt(
                os.path.join(WORDS_PATH, "adverbes_tot.txt")
            )
            self._advphase = read_txt(
                os.path.join(WORDS_PATH, "adverbes_phase.txt")
            )
            self._advfreq = read_txt(
                os.path.join(WORDS_PATH, "adverbes_freq.txt")
            )
            self._advintensite = read_txt(
                os.path.join(WORDS_PATH, "adverbes_intensite.txt")
            )
            self._advhabitude = read_txt(
                os.path.join(WORDS_PATH, "adverbes_habitude.txt")
            )
            self._advmodaux = read_txt(
                os.path.join(WORDS_PATH, "adverbes_modaux.txt")
            )
            self._advmaniere = read_txt(
                os.path.join(WORDS_PATH, "adverbes_maniere.txt")
            )
            self._nccorps = read_txt(
                os.path.join(WORDS_PATH, "parties_corps.txt")
            )
            self._nccabs = read_txt(
                os.path.join(WORDS_PATH, "noms_abstraits.txt")
            )

    @staticmethod
    def load_corpus(path) -> str:
        """
        :param path: file path
        :return: content of file
        """
        try:
            with open(path, mode="r") as f:
                content = f.read()
                # some cleaning
                content = content.replace("’", "'").replace("'", "'")
                if len(content) > 0:
                    return content
                else:
                    LOGGER.warning(f"{path} seems to be empty! Ignoring it")
        except Exception as exc:
            LOGGER.exception(f"Error while loading {path}...")
            raise exc

    def transform(self, text):
        return pd.DataFrame(
            (
                (
                    token.text,
                    token.lemma_,
                    token.pos_,
                    token.morph,
                    token.dep_,
                    token.n_lefts,
                    token.n_rights,
                    self.transform_token(token),
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

    def transform_token(self, token) -> str:
        motif = token.lemma_
        # Remplacement des pos auxiliaires pour les conserver dans les motifs :
        # Preprocessing
        if token.lemma_ == "être":
            return "être"
        if token.lemma_ == "avoir":
            return "avoir"

        morph = token.morph.to_dict()
        # Infinitifs :
        if morph.get("VerbForm") == "Inf":
            return "INF"
        # Participe Passé :
        elif morph.get("Tense") == "Past" and morph.get("VerbForm") == "Part":
            return "PPAS"
        # Participe Présent :
        elif morph.get("Tense") == "Pres" and morph.get("VerbForm") == "Part":
            return "PPRES"
        # Subjonctif présent :
        elif morph.get("Mood") == "Sub" and morph.get("Tense") == "Pres":
            return "VSUBP"
        # Subjonctif imparfait :
        elif morph.get("Mood") == "Sub" and morph.get("Tense") == "Imp":
            return "VSUBI"
        # Impératif présent :
        elif morph.get("Mood") == "Imp" and morph.get("Tense") == "Pres":
            return "IMP"
        # Conditionnel:
        elif morph.get("Mood") == "Cnd" and morph.get("Tense") == "Pres":
            return "VCOND"
        # Indicatif présent :
        elif morph.get("Mood") == "Ind" and morph.get("Tense") == "Pres":
            return "PRES"
        # Imparfait :
        elif morph.get("Mood") == "Ind" and morph.get("Tense") == "Imp":
            return "VIMP"
        # Passé simple :
        elif morph.get("Mood") == "Ind" and morph.get("Tense") == "Past":
            return "VPS"
        # Futur :
        elif morph.get("Mood") == "Ind" and morph.get("Tense") == "Fut":
            return "VF"
        # Determinants possessifs :
        elif morph.get("Poss") == "Yes" and morph.get("PronType") == "Prs":
            return "DETPOSS"
        else:
            pass

        # TODO: assert that intersection of words lists is empty: no overlaps
        # Pour les mots invariables, le motif est le mot:
        if token.text in self._uninflected_words:
            return token.text

        word_groups = {
            "ADVTOT": self._advtot,
            "ADVPHA": self._advphase,
            "ADVFRE": self._advfreq,
            "ADVINT": self._advintensite,
            "ADVHAB": self._advhabitude,
            "ADVMOD": self._advmodaux,
            "ADVMAN": self._advmaniere,
            "NCCOR": self._nccorps,
            "NCABS": self._nccabs,
        }
        for m in word_groups:
            if token.lemma_ in word_groups[m]:
                return m
        # if token.lemma_ in self._advtot:
        #     return "ADVTOT"
        # elif token.lemma_ in self._advphase:
        #     return "ADVPHA"
        # elif token.lemma_ in self._advfreq:
        #     return "ADVFRE"
        # elif token.lemma_ in self._advintensite:
        #     return "ADVINT"
        # elif token.lemma_ in self._advhabitude:
        #     return "ADVHAB"
        # elif token.lemma_ in self._advmodaux:
        #     return "ADVMOD"
        # elif token.lemma_ in self._advmaniere:
        #     return "ADVMAN"
        # # Noms communs abstraits, parties du corps :
        # # Parties du corps :
        # elif token.lemma_ in self._nccorps:
        #     return "NCCOR"
        # # Noms abstraits :
        # elif token.lemma_ in self._nccabs:
        #     return "NCABS"

        # Conservation des autres étiquettes morphosyntaxiques restantes :
        # pos
        pos = {
            "ADV": "ADV",
            "ADJ": "ADJ",
            "NUM": "NUM",
            "DETPOSS": "DETPOSS",
            "NOUN": "NC",
            "PROPN": "PROPN",
            "INTJ": "INTJ",
        }
        for p in pos:
            if token.text == p:
                return pos[p]

        # Conservation des pronoms personnels :
        # 1st:
        words = {
            # 1st
            "je": "je",
            "Je": "je",
            "j": "je",
            "J": "je",
            "me": "je",  # TODO: me -> je or me -> me comme te -> te
            "Me": "je",
            # 2nd
            "tu": "tu",
            "Tu": "tu",
            "te": "te",
            "Te": "te",
            # 3rd
            "Il": "il",
            "il": "il",
            "Elle": "elle",
            "elle": "elle",
            "Se": "se",
            "se": "se",
            # 4th & 5th
            "Nous": "nous",
            "nous": "nous",
            "Vous": "vous",
            "vous": "vous",
            # 6th
            "Ils": "ils",
            "ils": "ils",
            "Elles": "elles",
            "elles": "elles",
        }
        for w in words:
            if token.text == w:
                return words[w]

        # Guillemets anglais :
        if token.lemma_ == "«":
            return '"'
        elif token.lemma_ == "»":
            return '"'

        return motif


def is_vsubp(token):
    morph = token.morph.to_dict()
    if morph.get("Mood") == "Sub" and morph.get("Tense") == "Pres":
        return "VSUBP"


"""
could do a for loop with a break or while
"""
