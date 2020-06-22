import spacy


class AdjNounExtractor:
    def __init__(self):
        self.nlp = spacy.load('en')

    def extract(self, text):
        doc = self.nlp(text)

        adj_noun_pairs = []
        for token in doc:
            adjs = []
            nouns = []
            adv = None

            if token.pos_ in ('NOUN', 'PROPN'):
                for child in token.children:
                    if child.dep_ == 'amod':
                        adjs = self.get_conj(child)
                        nouns = self.get_conj(token)
                    if child.dep_ == 'neg':
                        adv = child.text

                if token.dep_ == 'attr':
                    for ancestor in token.ancestors:
                        if ancestor.pos_ == 'VERB':
                            for child in ancestor.children:
                                if child.dep_ == 'neg':
                                    adv = child.text

            if token.pos_ == 'VERB':
                for child in token.children:
                    if child.dep_ == 'acomp':
                        adjs = self.get_conj(child)
                    if child.dep_ == 'nsubj':
                        nouns = self.get_conj(child)
                    if child.dep_ == 'neg':
                        adv = child.text

            for adj in adjs:
                for noun in nouns:
                    lefts = [left.text for left in adj.lefts if left.pos_ == 'ADV']
                    if lefts:
                        adv = " ".join(lefts)

                    if adv is not None:
                        adj_noun_pairs.append((adv + ' ' + adj.text, noun.text, noun.lemma_))
                    else:
                        adj_noun_pairs.append((adj.text, noun.text, noun.lemma_))

        return adj_noun_pairs

    def get_conj(self, token):
        conjs = [token]
        for child in token.children:
            if child.dep_ == 'conj':
                conjs.extend(self.get_conj(child))
        return conjs
    
    