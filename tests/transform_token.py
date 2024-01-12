from motifs.features import Motif

motif = {"motif": {"morph": "morph"}}


class Token:
    def __init__(self, morph=None, pos_=None, dep_=None):
        self.morph = morph
        self.pos_ = pos_
        self.dep_ = dep_


token = Token(morph="morph")

name = list(motif.keys())[0]
params = motif[name]
m = Motif(name, **params)
