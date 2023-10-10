import requests
import json
import re
import random


class ConcepnetDistractors:
    def __init__(self, word: str):
        self.word = word

    def get_distractors(self):
        word = self.word.lower()
        distractor_list = []
        url = f"http://api.conceptnet.io/query?node=/c/en/{word}/n&rel=/r/PartOf&start=/c/en/{word}&limit=5"

        obj = requests.get(url).json()

        for edge in obj["edges"]:
            link = edge["end"]["term"]

            url2 = f"http://api.conceptnet.io/query?node={link}&rel=/r/PartOf&end={link}&limit=10"
            obj2 = requests.get(url2).json()
            for edge in obj2["edges"]:
                word2 = edge["start"]["label"]
                if word2 not in distractor_list and word not in word2.lower():
                    distractor_list.append(word2)

        return distractor_list


word = "lion"
disctract = ConcepnetDistractors(word)
distractors = disctract.get_distractors()
print(distractors)
