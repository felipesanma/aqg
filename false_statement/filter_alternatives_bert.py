from sentence_transformers import SentenceTransformer
import scipy


def bert_filter_false_sentences(
    original_sentence: str, possible_false_sentences: list, top_n: int
):
    BERT_model = SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")

    false_sentences_embeddings = BERT_model.encode(possible_false_sentences)
    original_sentence_embedding = BERT_model.encode([original_sentence])

    distances = scipy.spatial.distance.cdist(
        original_sentence_embedding, false_sentences_embeddings, "cosine"
    )[0]
    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])
    dissimilar_sentences = []
    for idx, distance in results:
        dissimilar_sentences.append(possible_false_sentences[idx])

    dissimilar_sentences = reversed(dissimilar_sentences)
    false_sentences_list_final = []
    for sent in dissimilar_sentences:
        if len(false_sentences_list_final) > top_n:
            break
        false_sentences_list_final.append(sent)

    return false_sentences_list_final
