from motifs.config import PKG_DATA_PATH
from motifs.tokenizer import MotifTokenizer

CORPUS_PATH = PKG_DATA_PATH.joinpath("corpus_test")
pipe = MotifTokenizer(CORPUS_PATH)
data = pipe.transform()
print(data.head())
