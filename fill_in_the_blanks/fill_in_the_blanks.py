import re
from sentences_mapping import SentenceMapping


class FillBlanks:
    def __init__(self, text: str):
        self.text = text
        self.sentences_and_keywords = SentenceMapping(self.text)
        self.sentences = self.sentences_and_keywords.tokenize_sentences()
        self.key_words = self.sentences_and_keywords.get_noun_adj_verb()
        self.sentence_mapping = self.sentences_and_keywords.get_sentences_for_keyword()

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
print(fill_in_the_blanks["keys"])
"""
