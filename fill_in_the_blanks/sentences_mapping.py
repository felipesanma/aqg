import textwrap
import string
import pke
from nltk.corpus import stopwords
import traceback
from nltk.tokenize import sent_tokenize
from flashtext import KeywordProcessor

"""
# The 1st time run this
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")
"""


class SentenceMapping:
    def __init__(self, text: str):
        self.text = text
        self.sentences = self.tokenize_sentences()
        self.key_words = self.get_noun_adj_verb()
        self.sentence_mapping = self.get_sentences_for_keyword()

    def _print_wrapper_text(self):
        wrapper = textwrap.TextWrapper(width=150)
        word_list = wrapper.wrap(text=self.text)
        for element in word_list:
            print(element)

    def tokenize_sentences(self):
        sentences = sent_tokenize(self.text)
        sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
        return sentences

    def get_noun_adj_verb(self):
        out = []
        try:
            extractor = pke.unsupervised.MultipartiteRank()
            # Run this before: python -m spacy download en_core_web_sm
            extractor.load_document(input=self.text, language="en")
            #    not contain punctuation marks or stopwords as candidates.
            pos = {"VERB", "ADJ", "NOUN"}
            stoplist = list(string.punctuation)
            stoplist += ["-lrb-", "-rrb-", "-lcb-", "-rcb-", "-lsb-", "-rsb-"]
            stoplist += stopwords.words("english")
            # extractor.candidate_selection(pos=pos, stoplist=stoplist)
            extractor.candidate_selection(pos=pos)
            extractor.candidate_weighting(alpha=1.1, threshold=0.75, method="average")
            keyphrases = extractor.get_n_best(n=30)

            for val in keyphrases:
                out.append(val[0])
        except:
            out = []
            traceback.print_exc()

        return out

    def get_sentences_for_keyword(self):
        keyword_processor = KeywordProcessor()
        keyword_sentences = {}
        for word in self.key_words:
            keyword_sentences[word] = []
            keyword_processor.add_keyword(word)
        for sentence in self.sentences:
            keywords_found = keyword_processor.extract_keywords(sentence)
            for key in keywords_found:
                keyword_sentences[key].append(sentence)

        for key in keyword_sentences.keys():
            values = keyword_sentences[key]
            values = sorted(values, key=len, reverse=True)
            keyword_sentences[key] = values
        return keyword_sentences
