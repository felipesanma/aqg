from sentence_transformers import SentenceTransformer
from mmr import mmr


class TransformersFilter:
    def __init__(self, answer: str, distractors: list):
        self.answer = answer
        self.distractors = [answer] + distractors
        self.model = SentenceTransformer("all-MiniLM-L12-v2")
        self.answer_embedding = self._get_embedding([self.answer])
        self.distractors_embedding = self._get_embedding(self.distractors)

    def _get_embedding(self, word: list):
        return self.model.encode(word)

    def get_best_distractors(self, top_n: int = 3):
        final_distractors = mmr(
            self.answer_embedding, self.distractors_embedding, self.distractors, top_n
        )
        filtered_distractors = [dist[0] for dist in final_distractors]
        filter_distractors = filtered_distractors[1:]

        return filter_distractors


"""
word = "Lion"

distractors = [
    "Bear",
    "Wolf",
    "Lioness",
    "Crocodile",
    "Hippo",
    "Tiger",
    "Croc",
    "Tusk",
    "Sheep",
    "Wildebeest",
]

filter = TransformersFilter(word, distractors)

best_distractors = filter.get_best_distractors(5)
print(best_distractors)
"""
