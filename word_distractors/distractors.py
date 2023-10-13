from conceptnet import ConcepnetDistractors
from wordnet import WordnetDistractors
from sense2vec_distractors import Sense2VecDistractors
from sentence_transformers_filter import TransformersFilter


class WordDistractors:
    def __init__(self, word: str, distractor_type: str = "sense2vec"):
        self.word = word
        self.distractor_type = distractor_type
        self.distractors = self.get()
        # self.filter = TransformersFilter(self.word, self.distractors)

    def get_conceptnet_distractor(self):
        c_distractor = ConcepnetDistractors(self.word)
        return c_distractor.get_distractors()

    def get_wordnet_distractor(self):
        c_distractor = WordnetDistractors(self.word)
        return c_distractor.get_distractors()

    def get_sense2vec_distractor(self):
        c_distractor = Sense2VecDistractors(self.word)
        return c_distractor.get_distractors()

    def get(self):
        if self.distractor_type == "sense2vec":
            return self.get_sense2vec_distractor()

        elif self.distractor_type == "wordnet":
            return self.get_wordnet_distractor()

        elif self.distractor_type == "conceptnet":
            return self.get_conceptnet_distractor()

        else:
            raise TypeError(f"Only 'conceptnet', 'wordnet' and 'sense2vec' are valid")

    def get_best(self, top_n: int = 3) -> list:
        filter = TransformersFilter(self.word, self.distractors)

        return filter.get_best_distractors(top_n)

    def get_all(self):
        return (
            self.get_conceptnet_distractor()
            + self.get_wordnet_distractor()
            + self.get_sense2vec_distractor()
        )


"""
word = "lion"
distractors = WordDistractors(word)
print(distractors.get_best(5))
"""
