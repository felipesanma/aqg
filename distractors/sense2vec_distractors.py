from sense2vec import Sense2Vec
from collections import OrderedDict


class Sense2VecDistractors:
    def __init__(self, word: str):
        self.word = word
        self.s2v = Sense2Vec().from_disk("distractors/s2v_old")

    def get_distractors(self, k: int = 10):
        distractor_list = []
        sense = self.s2v.get_best_sense(self.word)
        most_similar = self.s2v.most_similar(sense, n=k)

        for each_word in most_similar:
            append_word = each_word[0].split("|")[0].replace("_", " ").lower()
            if append_word.lower() != self.word:
                distractor_list.append(append_word.title())

        out = list(OrderedDict.fromkeys(distractor_list))
        return out


"""
word = "lion"
disctract = Sense2VecDistractors(word)
distractors = disctract.get_distractors()
print(distractors)
"""
