import stanza
from nltk.tree import Tree
import re
from alternative_endings_gpt3 import chatGPT


class ConstituencyParser:
    def __init__(
        self, text: str, lang: str = "en", processors: str = "tokenize,pos,constituency"
    ):
        self.text = text.rstrip("?:!.,;")
        self.nlp = stanza.Pipeline(lang=lang, processors=processors)
        self.tree = self._get_tree_from_text()

    def _get_tree_from_text(self):
        doc = self.nlp(self.text)
        tree_string = str(doc.sentences[0].constituency)
        return Tree.fromstring(tree_string)

    def _get_flattened(self, t):
        sent_str_final = None
        if t is not None:
            sent_str = [" ".join(x.leaves()) for x in list(t)]
            sent_str_final = [" ".join(sent_str)]
            sent_str_final = sent_str_final[0]
        return sent_str_final

    def _get_right_most_VP_or_NP(self, last_subtree, last_NP=None, last_VP=None):
        if len(last_subtree.leaves()) == 1:
            return last_NP, last_VP
        last_subtree = last_subtree[-1]
        if last_subtree.label() == "NP":
            last_NP = last_subtree
        elif last_subtree.label() == "VP":
            last_VP = last_subtree

        return self._get_right_most_VP_or_NP(last_subtree, last_NP, last_VP)

    def _get_termination_portion(self, main_string, sub_string):
        combined_sub_string = sub_string.replace(" ", "")
        main_string_list = main_string.split()
        last_index = len(main_string_list)
        for i in range(last_index):
            check_string_list = main_string_list[i:]
            check_string = "".join(check_string_list)
            check_string = check_string.replace(" ", "")
            if check_string == combined_sub_string:
                return " ".join(main_string_list[:i])

        return None

    def get_true_statement(self):
        last_nounphrase, last_verbphrase = self._get_right_most_VP_or_NP(self.tree)
        last_nounphrase_flattened = self._get_flattened(last_nounphrase)
        last_verbphrase_flattened = self._get_flattened(last_verbphrase)

        longest_phrase_to_use = max(
            last_nounphrase_flattened, last_verbphrase_flattened, key=len
        )
        longest_phrase_to_use = re.sub(r"-LRB- ", "(", longest_phrase_to_use)
        longest_phrase_to_use = re.sub(r" -RRB-", ")", longest_phrase_to_use)

        split_sentence = self._get_termination_portion(self.text, longest_phrase_to_use)
        print("Original sentence : ", self.text)
        print("Original sentence after splitting at ending phrase: ", split_sentence)
        print("Ending phrase: ", longest_phrase_to_use)

        return split_sentence, longest_phrase_to_use


test_sentence = "The old woman was sitting under a tree and sipping coffee."
phrase_parser = ConstituencyParser(test_sentence)
starting_phrase, ending_phrase = phrase_parser.get_true_statement()
prompt = f"The original phrase is: '{test_sentence}'. Generate 10 alternative endings starting with the phrase: '{starting_phrase}'"

alternative_endings_davinci = chatGPT(prompt)
print(alternative_endings_davinci)
