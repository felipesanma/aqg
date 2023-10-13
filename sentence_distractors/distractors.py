from alternative_endings_gpt3 import chatGPT
from filter_alternatives_bert import bert_filter_false_sentences
from constituency_parsing import ConstituencyParser


class SentenceDistractors:
    def __init__(self, sentence: str):
        self.sentence = sentence
        self.parser = ConstituencyParser(self.sentence)
        self.starting_phrase, self.ending_phrase = self.parser.get_true_statement()
        self.prompt = f"The original phrase is: '{self.sentence}'. Generate 10 alternative endings starting with the phrase: '{self.starting_phrase}'"

    def get(self, top_n: int = 3):
        alternative_endings_davinci = chatGPT(self.prompt)
        return bert_filter_false_sentences(
            self.sentence, alternative_endings_davinci, top_n
        )


"""
sentence = "The old woman was sitting under a tree and sipping coffee."
distractors = SentenceDistractors(sentence)

print(distractors.get())
"""
