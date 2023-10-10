# import nltk
# nltk.download("wordnet")
from nltk.corpus import wordnet as wn

# Distractors from Wordnet


class Distractors:
    def __init__(self, word: str):
        self.word = word
        self.syn = self._get_syn_to_use()

    def _get_syn_to_use(self):
        return wn.synsets(self.word, "n")

    def get_distractors_wordnet(self) -> list:
        # lower_word = self.word.lower()
        distractors = []
        for syn in self.syn:
            hypernym = syn.hypernyms()

            if len(hypernym) == 0:
                continue

            for item in hypernym[0].hyponyms():
                name = item.lemmas()[0].name()

                if name == self.word:
                    continue

                if name is not None and name not in distractors:
                    distractors.append(name)

        return distractors


word = "green"
distract = Distractors(word)

distractors = distract.get_distractors_wordnet()
print(distractors)
