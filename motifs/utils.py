"""
- full preprocessing, motifs pipeline per doc
- save output in local file per doc
- La pipeline prend comme paramètre le json des definitions des motifs:
{"PPAS": {"morph": {"Tense":"Past", "VerbForm":"Part"}}}

{"NCs": {"POS": "NOUN", "dep": "nsubj"}}

{"être": {"value": "être"}}



mutate(lemmes = replace(lemmes, POS == "NOUN"& fonction=="nsubj", "NCs")) %>%
mutate(lemmes = replace(lemmes, POS == "NOUN"& fonction=="nsubj:pass", "NCspas")) %>%
mutate(lemmes = replace(lemmes, POS == "NOUN"& fonction=="obj", "NCo")) %>%
mutate(lemmes = replace(lemmes, POS == "NOUN"& fonction=="nmod", "NCmod")) %>%

"""

import os
from typing import Optional

import numpy as np
import pandas as pd
import spacy_udpipe

from motifs.config import LOGGER, PKG_DATA_PATH

WORDS_PATH = os.path.join(PKG_DATA_PATH, "french")


def read_txt(path, sep=",\n"):
    try:
        with open(path, mode="r") as f:
            content = f.read()
    except Exception as exc:
        LOGGER.exception(f"Error while loading {path}...")
        raise exc

    # Handle case where string ends with sep to avoid having an empty
    # element in final list
    if content[-len(sep) :] == sep:
        content = content[: -len(sep)]

    return content.split(sep)


def load_corpus(path: str) -> dict:
    """
    Read all texts inside the folder path and put them into a dictionary.
    :param path: a path to a directory
    :return: corpus as a dictionnary with txt filename keys
    """
    corpus = {}  # create empty list to save content

    for filename in filter(lambda p: p.endswith("txt"), os.listdir(path)):
        try:
            with open(os.path.join(path, filename), mode="r") as f:
                content = f.read()
                # some cleaning
                content = content.replace("’", "'").replace("'", "'")
                if len(content) > 0:
                    corpus[filename] = content
                else:
                    LOGGER.warning(
                        f"{filename} seems to be empty! Ignoring it"
                    )
        except Exception as exc:
            LOGGER.exception(f"Error while loading {filename}...")
            raise exc

    LOGGER.debug(f"Corpus length: {len(corpus)}.")

    return corpus


def annotate(path: str, save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Annotation udpipe of the texts in folder path
    :param path: path to the input directory
    :param save_path: path to the output directory
    :return: data, a DataFrame with columns ["filename", "words", "lemma",
    "pos", "morph", "dep", "n_lefts", "n_rights"]
    """
    data = load_corpus(path)
    spacy_udpipe.download("fr")  # download French model
    nlp = spacy_udpipe.load("fr")

    data = pd.concat(
        [
            pd.DataFrame(
                (
                    f,
                    token.text,
                    token.lemma_,
                    token.pos_,
                    token.morph,
                    token.dep_,
                    token.n_lefts,
                    token.n_rights,
                )
                for token in nlp(data[f])
            )
            for f in data.keys()
        ],
        ignore_index=True,
    )
    data.columns = [
        "filename",
        "words",
        "lemma",
        "pos",
        "morph",
        "dep",
        "n_lefts",
        "n_rights",
    ]

    # Exportation csv :
    if save_path:
        data.to_csv(f"{save_path}/udpipe_corpus_complet.csv", index=False)

    return data


def transform(
    df: pd.DataFrame, save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Transform the annotated texts in df into motifs

    :param df: DataFrame with annotated texts obtain with `annotate`
    :param save_path: path to the output directory
    :return: a DataFrame with two columns ["filename", "motifs"]
    """

    # Remplacement des pos auxiliaires pour les conserver dans les motifs :
    # Preprocessing
    df["pos"][df.lemma == "être"] = "être"
    df["pos"][df.lemma == "avoir"] = "avoir"

    # Infinitifs :
    # Condition: VerbForm=Inf
    df["lemma"][df.morph == "Typo=Yes|VerbForm=Inf"] = "INF"
    df["lemma"][df.morph == "VerbForm=Inf"] = "INF"
    df["lemma"][df.morph == "Typo=No|VerbForm=Inf"] = "INF"

    # Participes :
    # Passé
    # Condition: Tense=Past and VerbForm=Part
    df["lemma"][
        df.morph == "Gender=Masc|Number=Sing|Tense=Past|Typo=Yes|VerbForm=Part"
    ] = "PPAS"  # masc sg
    df["lemma"][
        df.morph == "Gender=Masc|Number=Sing|Tense=Past|VerbForm=Part"
    ] = "PPAS"  # masc sg
    df["lemma"][
        df.morph == "Gender=Fem|Number=Sing|Tense=Past|Typo=Yes|VerbForm=Part"
    ] = "PPAS"  # fem sing
    df["lemma"][
        df.morph == "Gender=Masc|Number=Plur|Tense=Past|Typo=Yes|VerbForm=Part"
    ] = "PPAS"  # masc plu
    df["lemma"][
        df.morph == "Gender=Fem|Number=Plur|Tense=Past|VerbForm=Part"
    ] = "PPAS"  # fem plu
    df["lemma"][
        df.morph == "Gender=Fem|Number=Plur|Tense=Past|VerbForm=Part"
    ] = "PPAS"  # Ppsé fem sing
    df["lemma"][
        df.morph == "Gender=Masc|Number=Plur|Tense=Past|VerbForm=Part"
    ] = "PPAS"  # Ppsé masc plu

    # Présent
    # Condition: Tense=Pres and VerbForm=Part
    df["lemma"][df.morph == "Tense=Pres|VerbForm=Part"] = "PPRES"  # Pürésent

    # Verbs :
    # Subjonctif présent :
    # Condition: Mood=Sub and Tense=Pres (and VerbForm=Fin)
    df["lemma"][
        df.morph == "Mood=Sub|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin"
    ] = "VSUBP"
    df["lemma"][
        df.morph == "Mood=Sub|Number=Sing|Person=2|Tense=Pres|VerbForm=Fin"
    ] = "VSUBP"
    df["lemma"][
        df.morph == "Mood=Sub|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin"
    ] = "VSUBP"
    df["lemma"][
        df.morph == "Mood=Sub|Number=Plur|Person=1|Tense=Pres|VerbForm=Fin"
    ] = "VSUBP"
    df["lemma"][
        df.morph == "Mood=Sub|Number=Plur|Person=2|Tense=Pres|VerbForm=Fin"
    ] = "VSUBP"
    df["lemma"][
        df.morph == "Mood=Sub|Number=Plur|Person=3|Tense=Pres|VerbForm=Fin"
    ] = "VSUBP"

    # Subjonctif imparfait :
    # Condition: Mood=Sub and Tense=Imp (and VerbForm=Fin)
    df["lemma"][
        df.morph == "Mood=Sub|Number=Sing|Person=1|Tense=Imp|VerbForm=Fin"
    ] = "VSUBI"
    df["lemma"][
        df.morph == "Mood=Sub|Number=Sing|Person=2|Tense=Imp|VerbForm=Fin"
    ] = "VSUBI"
    df["lemma"][
        df.morph == "Mood=Sub|Number=Sing|Person=3|Tense=Imp|VerbForm=Fin"
    ] = "VSUBI"
    df["lemma"][
        df.morph == "Mood=Sub|Number=Plur|Person=1|Tense=Imp|VerbForm=Fin"
    ] = "VSUBI"
    df["lemma"][
        df.morph == "Mood=Sub|Number=Plur|Person=2|Tense=Imp|VerbForm=Fin"
    ] = "VSUBI"
    df["lemma"][
        df.morph == "Mood=Sub|Number=Plur|Person=3|Tense=Imp|VerbForm=Fin"
    ] = "VSUBI"

    # Impératif présent :
    # Condition: Mood=Imp and Tense=Pres (and VerbForm=Fin)
    df["lemma"][
        df.morph == "Mood=Imp|Number=Sing|Person=2|Tense=Pres|VerbForm=Fin"
    ] = "IMP"
    df["lemma"][
        df.morph == "Mood=Imp|Number=Plur|Person=1|Tense=Pres|VerbForm=Fin"
    ] = "IMP"
    df["lemma"][
        df.morph == "Mood=Imp|Number=Plur|Person=2|Tense=Pres|VerbForm=Fin"
    ] = "IMP"

    # Conditionnel:
    # Condition: Mood=Cnd and Tense=Pres (and VerbForm=Fin)
    df["lemma"][
        df.morph == "Mood=Cnd|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin"
    ] = "VCOND"
    df["lemma"][
        df.morph == "Mood=Cnd|Number=Sing|Person=2|Tense=Pres|VerbForm=Fin"
    ] = "VCOND"
    df["lemma"][
        df.morph == "Mood=Cnd|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin"
    ] = "VCOND"
    df["lemma"][
        df.morph == "Mood=Cnd|Number=Plur|Person=1|Tense=Pres|VerbForm=Fin"
    ] = "VCOND"
    df["lemma"][
        df.morph == "Mood=Cnd|Number=Plur|Person=2|Tense=Pres|VerbForm=Fin"
    ] = "VCOND"
    df["lemma"][
        df.morph == "Mood=Cnd|Number=Plur|Person=3|Tense=Pres|VerbForm=Fin"
    ] = "VCOND"

    # Indicatif présent :
    # Condition: Mood=Ind and Tense=Pres (and VerbForm=Fin)
    df["lemma"][
        df.morph == "Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin"
    ] = "PRES"
    df["lemma"][
        df.morph == "Mood=Ind|Number=Sing|Person=2|Tense=Pres|VerbForm=Fin"
    ] = "PRES"
    df["lemma"][
        df.morph == "Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin"
    ] = "PRES"
    df["lemma"][
        df.morph == "Mood=Ind|Number=Plur|Person=1|Tense=Pres|VerbForm=Fin"
    ] = "PRES"
    df["lemma"][
        df.morph == "Mood=Ind|Number=Plur|Person=2|Tense=Pres|VerbForm=Fin"
    ] = "PRES"
    df["lemma"][
        df.morph == "Mood=Ind|Number=Plur|Person=3|Tense=Pres|VerbForm=Fin"
    ] = "PRES"

    # Imparfait :
    # Condition: Mood=Ind and Tense=Imp (and VerbForm=Fin)
    df["lemma"][
        df.morph == "Mood=Ind|Number=Sing|Person=1|Tense=Imp|VerbForm=Fin"
    ] = "VIMP"
    df["lemma"][
        df.morph == "Mood=Ind|Number=Sing|Person=2|Tense=Imp|VerbForm=Fin"
    ] = "VIMP"
    df["lemma"][
        df.morph == "Mood=Ind|Number=Sing|Person=3|Tense=Imp|VerbForm=Fin"
    ] = "VIMP"
    df["lemma"][
        df.morph == "Mood=Ind|Number=Plur|Person=1|Tense=Imp|VerbForm=Fin"
    ] = "VIMP"
    df["lemma"][
        df.morph == "Mood=Ind|Number=Plur|Person=2|Tense=Imp|VerbForm=Fin"
    ] = "VIMP"
    df["lemma"][
        df.morph == "Mood=Ind|Number=Plur|Person=3|Tense=Imp|VerbForm=Fin"
    ] = "VIMP"

    # Passé simple :
    # Condition: Mood=Ind and Tense=Past
    df["lemma"][
        df.morph == "Mood=Ind|Number=Sing|Person=1|Tense=Past|VerbForm=Fin"
    ] = "VPS"
    df["lemma"][
        df.morph == "Mood=Ind|Number=Sing|Person=2|Tense=Past|VerbForm=Fin"
    ] = "VPS"
    df["lemma"][
        df.morph == "Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin"
    ] = "VPS"
    df["lemma"][
        df.morph == "Mood=Ind|Number=Plur|Person=1|Tense=Past|VerbForm=Fin"
    ] = "VPS"
    df["lemma"][
        df.morph == "Mood=Ind|Number=Plur|Person=2|Tense=Past|VerbForm=Fin"
    ] = "VPS"
    df["lemma"][
        df.morph == "Mood=Ind|Number=Plur|Person=3|Tense=Past|VerbForm=Fin"
    ] = "VPS"

    # Futur :
    # Condition: Mood=Ind and Tense=Fut
    df["lemma"][
        df.morph == "Mood=Ind|Number=Sing|Person=1|Tense=Fut|VerbForm=Fin"
    ] = "VF"
    df["lemma"][
        df.morph == "Mood=Ind|Number=Sing|Person=2|Tense=Fut|VerbForm=Fin"
    ] = "VF"
    df["lemma"][
        df.morph == "Mood=Ind|Number=Sing|Person=3|Tense=Fut|VerbForm=Fin"
    ] = "VF"
    df["lemma"][
        df.morph == "Mood=Ind|Number=Plur|Person=1|Tense=Fut|VerbForm=Fin"
    ] = "VF"
    df["lemma"][
        df.morph == "Mood=Ind|Number=Plur|Person=2|Tense=Fut|VerbForm=Fin"
    ] = "VF"
    df["lemma"][
        df.morph == "Mood=Ind|Number=Plur|Person=3|Tense=Fut|VerbForm=Fin"
    ] = "VF"

    # Determinants possessifs :
    # Condition: Poss=Yes (and PronType=Prs)
    df["lemma"][
        df.morph == "Gender=Masc|Number=Sing|Poss=Yes|PronType=Prs"
    ] = "DETPOSS"
    df["lemma"][
        df.morph == "Gender=Fem|Number=Sing|Poss=Yes|PronType=Prs"
    ] = "DETPOSS"

    df["lemma"][
        df.morph == "Gender=Masc|Number=Plur|Poss=Yes|PronType=Prs"
    ] = "DETPOSS"
    df["lemma"][
        df.morph == "Gender=Fem|Number=Plur|Poss=Yes|PronType=Prs"
    ] = "DETPOSS"

    # Mots invariables, locutions adverbiales, verbes courants, titres :
    mots_inv = pd.read_csv(
        os.path.join(WORDS_PATH, "mots_invariables.txt"),
        sep=",",
        header=None,
    )[0].tolist()

    locutions = pd.read_csv(
        os.path.join(WORDS_PATH, "locutions_adverbiales.txt"),
        sep=",",
        header=None,
    )[0].tolist()

    verbes_courants = pd.read_csv(
        os.path.join(WORDS_PATH, "verbes_courants.txt"), sep=",", header=None
    )[0].tolist()

    titres = pd.read_csv(
        os.path.join(WORDS_PATH, "titres.txt"), sep=",", header=None
    )[0].tolist()

    # On remplace les POS par les mots pour conserver ces mots invariables
    # dans les conservations futures des POS pour les motifs :
    idminv = df["words"].isin(mots_inv)
    # Check wich is True ie which POS are to be replaced by their lemma value
    pos_to_replace = np.where(idminv)[0]
    # Replace them :
    df["pos"][pos_to_replace] = df["words"][pos_to_replace]

    # Idem avec les locutions :
    id_locutions = df["lemma"].isin(locutions)
    pos_to_replace = np.where(id_locutions)[0]
    df["pos"][pos_to_replace] = df["lemma"][pos_to_replace]

    # Idem avec les verbes courants (lemma) :
    id_vb_courants = df["lemma"].isin(verbes_courants)
    pos_to_replace = np.where(id_vb_courants)[0]
    df["pos"][pos_to_replace] = df["lemma"][pos_to_replace]

    # Idem avec les titres (lemma) :
    id_titres = df["lemma"].isin(titres)
    pos_to_replace = np.where(id_titres)[0]
    df["pos"][pos_to_replace] = df["lemma"][pos_to_replace]

    # Adverbes :
    # Adverbes de totalité, phase, fréquence, intensité, habitude, modaux,
    # manière
    advtot = pd.read_csv(
        os.path.join(WORDS_PATH, "adverbes_tot.txt"), sep=",", header=None
    )[0].tolist()
    advphase = pd.read_csv(
        os.path.join(WORDS_PATH, "adverbes_phase.txt"), sep=",", header=None
    )[0].tolist()
    advfreq = pd.read_csv(
        os.path.join(WORDS_PATH, "adverbes_freq.txt"), sep=",", header=None
    )[0].tolist()
    advintensite = pd.read_csv(
        os.path.join(WORDS_PATH, "adverbes_intensite.txt"),
        sep=",",
        header=None,
    )[0].tolist()
    advhabitude = pd.read_csv(
        os.path.join(WORDS_PATH, "adverbes_habitude.txt"),
        sep=",",
        header=None,
    )[0].tolist()
    advmodaux = pd.read_csv(
        os.path.join(WORDS_PATH, "adverbes_modaux.txt"), sep=",", header=None
    )[0].tolist()
    advmaniere = pd.read_csv(
        os.path.join(WORDS_PATH, "adverbes_maniere.txt"),
        sep=",",
        header=None,
    )[0].tolist()

    # Totalité :
    id_advtot = df["lemma"].isin(advtot)
    to_replace = np.where(id_advtot)[0]
    df["lemma"][to_replace] = "ADVTOT"
    df["pos"][to_replace] = "ADVTOT"

    # Phase :
    id_advphase = df["lemma"].isin(advphase)
    to_replace = np.where(id_advphase)[0]
    df["lemma"][to_replace] = "ADVPHA"
    df["pos"][to_replace] = "ADVPHA"

    # Fréquence :
    id_advfreq = df["lemma"].isin(advfreq)
    to_replace = np.where(id_advfreq)[0]
    df["lemma"][to_replace] = "ADVFRE"
    df["pos"][to_replace] = "ADVFRE"

    # Intensité :
    id_advint = df["lemma"].isin(advintensite)
    to_replace = np.where(id_advint)[0]
    df["lemma"][to_replace] = "ADVINT"
    df["pos"][to_replace] = "ADVINT"

    # Habitude :
    id_advhab = df["lemma"].isin(advhabitude)
    to_replace = np.where(id_advhab)[0]
    df["lemma"][to_replace] = "ADVHAB"
    df["pos"][to_replace] = "ADVHAB"

    # Modaux :
    id_advmod = df["lemma"].isin(advmodaux)
    to_replace = np.where(id_advmod)[0]
    df["lemma"][to_replace] = "ADVMOD"
    df["pos"][to_replace] = "ADVMOD"

    # Manière :
    id_advman = df["lemma"].isin(advmaniere)
    to_replace = np.where(id_advman)[0]
    df["lemma"][to_replace] = "ADVMAN"
    df["pos"][to_replace] = "ADVMAN"

    # Noms communs abstraits, parties du corps :
    # Parties du corps :
    nccorps = pd.read_csv(
        os.path.join(WORDS_PATH, "parties_corps.txt"), sep=",", header=None
    )[0].tolist()
    id_nccorps = df["lemma"].isin(nccorps)
    to_replace = np.where(id_nccorps)[0]
    df["lemma"][to_replace] = "NCCOR"
    df["pos"][to_replace] = "NCCOR"

    # Noms abstraits :
    nccabs = pd.read_csv(
        os.path.join(WORDS_PATH, "noms_abstraits.txt"), sep=",", header=None
    )[0].tolist()
    id_nccabs = df["lemma"].isin(nccabs)
    to_replace = np.where(id_nccabs)
    df["lemma"][to_replace] = "NCABS"
    df["pos"][to_replace] = "NCABS"

    # Conservation des autres étiquettes morphosyntaxiques restantes :
    # ADV, ADJ, NUM, DETPOSS, NC, PROPN, INTJ
    df["lemma"][df.pos == "ADV"] = "ADV"
    df["lemma"][df.pos == "ADJ"] = "ADJ"
    df["lemma"][df.pos == "NUM"] = "NUM"
    df["lemma"][df.pos == "DETPOSS"] = "DETPOSS"
    df["lemma"][df.pos == "NOUN"] = "NC"
    df["lemma"][df.pos == "PROPN"] = "PROPN"
    df["lemma"][df.pos == "INTJ"] = "INTJ"

    # Conservation des pronoms personnels :
    # 1st:
    df["lemma"][df.words == "je"] = "je"
    df["lemma"][df.words == "Je"] = "je"
    df["lemma"][df.words == "j"] = "je"
    df["lemma"][df.words == "J"] = "je"
    df["lemma"][df.words == "me"] = "je"
    df["lemma"][df.words == "Me"] = "je"

    # 2n:
    df["lemma"][df.words == "tu"] = "tu"
    df["lemma"][df.words == "Tu"] = "tu"
    df["lemma"][df.words == "te"] = "te"
    df["lemma"][df.words == "Te"] = "te"

    # 3rd :
    df["lemma"][df.words == "Il"] = "il"
    df["lemma"][df.words == "il"] = "il"
    df["lemma"][df.words == "Elle"] = "elle"
    df["lemma"][df.words == "elle"] = "elle"
    df["lemma"][df.words == "Se"] = "se"
    df["lemma"][df.words == "se"] = "se"

    # 4th & 5th
    df["lemma"][df.words == "Nous"] = "nous"
    df["lemma"][df.words == "nous"] = "nous"
    df["lemma"][df.words == "Vous"] = "vous"
    df["lemma"][df.words == "vous"] = "vous"

    # 6th:
    df["lemma"][df.words == "Ils"] = "ils"
    df["lemma"][df.words == "ils"] = "ils"
    df["lemma"][df.words == "Elles"] = "elles"
    df["lemma"][df.words == "elles"] = "elles"

    # Last corrections :
    # Guillemets anglais :

    df["lemma"][df.lemma == "«"] = '"'
    df["lemma"][df.lemma == "»"] = '"'

    # Lignes vides :
    df = df.dropna(subset=["lemma"])

    # # Save results :
    # ## Rename column before saving:
    # df.rename(columns={"lemma": "motifs"}, inplace=True)
    # df_to_export = df[["filename", "motifs"]]
    #
    # df_to_export.to_csv(path_or_buf=save_csv, encoding="utf-8")
    return df
