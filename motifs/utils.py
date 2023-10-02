import os

from motifs.config import LOGGER


def load_corpus(path: str) -> dict:
    """
    Read all txts inside a folder and put them into a dictionary.
    :param path: input path
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
