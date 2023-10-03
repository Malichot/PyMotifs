from typing import Optional

import spacy.tokens


class Motif:
    def __init__(
        self,
        name: str,
        value: Optional[str] = None,
        morph: Optional[dict] = None,
        pos: Optional[str] = None,
        dep: Optional[str] = None,
        operator: str = "and",
    ):
        self.name = name
        self.value = value
        self.morph = morph
        self.pos = pos
        self.dep = dep
        self.operator = operator
        if self.value:
            assert (
                self.morph is None and self.pos is None and self.dep is None
            ), "You cannot provide value and other parameters"
        else:
            assert (
                self.morph or self.pos or self.dep
            ), "You must providevalue or morph, pos, and or dep"

    def transform_token(self, token: spacy.tokens.Token):
        if self.value:
            if token.lemma_ == self.value:
                return self.name
            else:
                return token.lemma_
        else:
            equal_pos = self.pos == token.pos_
            equal_dep = self.dep == token.dep_
            if self.morph:
                token_morph = token.morph  # .to_dict()
                equal_morph = all(
                    [self.morph[k] == token_morph.get(k) for k in self.morph]
                )
            else:
                equal_morph = False

            if self.morph and self.pos and self.dep:
                to_transform = equal_morph and equal_pos and equal_dep
            else:
                if self.morph and self.pos:
                    to_transform = equal_morph and equal_pos
                elif self.morph and self.dep:
                    to_transform = equal_morph and equal_dep
                elif self.pos and self.dep:
                    to_transform = equal_pos and equal_dep
                elif self.morph:
                    to_transform = equal_morph
                elif self.pos:
                    to_transform = equal_pos
                elif self.dep:
                    to_transform = equal_dep
                else:
                    raise Exception

            if to_transform:
                return self.name
            else:
                return token.lemma_
