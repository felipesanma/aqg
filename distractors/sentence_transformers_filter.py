from sentence_transformers import SentenceTransformer
from mmr import mmr

model = SentenceTransformer("all-MiniLM-L12-v2")

word = "Lion"

distractors = [
    word,
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


def get_answer_and_distractor_embeddings(answer, candidate_distractors):
    answer_embedding = model.encode([answer])
    distractor_embeddings = model.encode(candidate_distractors)
    return answer_embedding, distractor_embeddings


answer_embedd, distractor_embedds = get_answer_and_distractor_embeddings(
    word, distractors
)


final_distractors = mmr(answer_embedd, distractor_embedds, distractors, 5)
filtered_distractors = []
for dist in final_distractors:
    filtered_distractors.append(dist[0])


answer = filtered_distractors[0]
filter_distractors = filtered_distractors[1:]

print(answer)
print("------------------->")
for k in filter_distractors:
    print(k)
