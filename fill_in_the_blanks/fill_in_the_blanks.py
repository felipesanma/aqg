import textwrap
import string
import re
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


class FillBlanks:
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

    def get_fill_in_the_blanks(self):
        out = {"mapping": {}}
        blank_sentences = []
        processed = []
        keys = []
        for key in self.sentence_mapping:
            if len(self.sentence_mapping[key]) > 0:
                sent = self.sentence_mapping[key][0]
                # Compile a regular expression pattern into a regular expression object, which can be used for matching and other methods
                insensitive_sent = re.compile(re.escape(key), re.IGNORECASE)
                no_of_replacements = len(
                    re.findall(re.escape(key), sent, re.IGNORECASE)
                )
                line = insensitive_sent.sub(" _________ ", sent)
                if (
                    self.sentence_mapping[key][0] not in processed
                ) and no_of_replacements < 2:
                    print(self.sentence_mapping[key][0], key)
                    blank_sentences.append(line)
                    processed.append(self.sentence_mapping[key][0])
                    keys.append(key)
                    out["mapping"][key] = line
        out["sentences"] = blank_sentences[:10]
        out["keys"] = keys[:10]
        return out


"""
text = "There is a lot of volcanic activity at divergent plate boundaries in the oceans. For example, many undersea volcanoes are found along the Mid-Atlantic Ridge. This is a divergent plate boundary that runs north-south through the middle of the Atlantic Ocean. As tectonic plates pull away from each other at a divergent plate boundary, they create deep fissures, or cracks, in the crust. Molten rock, called magma, erupts through these cracks onto Earth’s surface. At the surface, the molten rock is called lava. It cools and hardens, forming rock. Divergent plate boundaries also occur in the continental crust. Volcanoes form at these boundaries, but less often than in ocean crust. That’s because continental crust is thicker than oceanic crust. This makes it more difficult for molten rock to push up through the crust. Many volcanoes form along convergent plate boundaries where one tectonic plate is pulled down beneath another at a subduction zone. The leading edge of the plate melts as it is pulled into the mantle, forming magma that erupts as volcanoes. When a line of volcanoes forms along a subduction zone, they make up a volcanic arc. The edges of the Pacific plate are long subduction zones lined with volcanoes. This is why the Pacific rim is called the “Pacific Ring of Fire.”"

fill_blanks = FillBlanks(text)
fill_in_the_blanks = fill_blanks.get_fill_in_the_blanks()
print(fill_in_the_blanks)
"""
