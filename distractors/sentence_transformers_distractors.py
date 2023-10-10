from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L12-v2")

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
