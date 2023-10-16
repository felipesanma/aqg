from fill_in_the_blanks.sentences_mapping import SentenceMapping
from question_generator.question_gen import QuestionGenerator
from summarization.t5 import summarizer
from key_words_generation.key_words_and_phrases import get_keywords


class QGen:
    def __init__(self, text: str):
        self.text = text
        self.sentence_mapping = SentenceMapping(self.text)
        self.summary = summarizer(self.text)
        self.key_words = get_keywords(self.text, self.summary)

        # self.question = QuestionGenerator()

    def get_questions(self):
        qa = []
        for answer in self.key_words:
            question = QuestionGenerator(self.summary, answer, 2, "v1.0").generate()
            tmp = {"answer": answer, "question": question}
            qa.append(tmp)
            print(tmp)
        return qa

    def get_questions_v2(self):
        key_words = self.sentence_mapping.get_sentences_for_keyword()
        qa = []
        for answer in key_words:
            question = QuestionGenerator(
                key_words[answer], answer, 2, "v1.0"
            ).generate()
            tmp = {"answer": answer, "question": question}
            qa.append(tmp)
            print(tmp)
        return qa


text = """Elon Musk has shown again he can influence the digital currency market with just his tweets. After saying that his electric vehicle-making company
Tesla will not accept payments in Bitcoin because of environmental concerns, he tweeted that he was working with developers of Dogecoin to improve
system transaction efficiency. Following the two distinct statements from him, the world's largest cryptocurrency hit a two-month low, while Dogecoin
rallied by about 20 percent. The SpaceX CEO has in recent months often tweeted in support of Dogecoin, but rarely for Bitcoin.  In a recent tweet,
Musk put out a statement from Tesla that it was “concerned” about the rapidly increasing use of fossil fuels for Bitcoin (price in India) mining and
transaction, and hence was suspending vehicle purchases using the cryptocurrency.  A day later he again tweeted saying, “To be clear, I strongly
believe in crypto, but it can't drive a massive increase in fossil fuel use, especially coal”.  It triggered a downward spiral for Bitcoin value but
the cryptocurrency has stabilised since.   A number of Twitter users welcomed Musk's statement. One of them said it's time people started realising
that Dogecoin “is here to stay” and another referred to Musk's previous assertion that crypto could become the world's future currency."""
qgen = QGen(text)
print(text)
# print(qgen.summary)
# questions = qgen.get_questions()
print(qgen.get_questions_v2())
