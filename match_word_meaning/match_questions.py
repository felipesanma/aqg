import json
import requests
import string
import re
import nltk
import string
import itertools
import textwrap
import os
import zipfile
import torch
import math
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer
import torch
import re
import time
import torch
import csv
import os
from collections import namedtuple
import statistics
from statistics import mode
import re

import nltk
from nltk.corpus import wordnet as wn
from tabulate import tabulate
from torch.nn.functional import softmax
from tqdm import tqdm
from transformers import BertTokenizer

"""
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")
nltk.download('omw-1.4')
"""

import pke
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import traceback
from nltk.tokenize import sent_tokenize
from flashtext import KeywordProcessor


def tokenize_sentences(text):
    sentences = sent_tokenize(text)
    sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
    return sentences


# Story source - https://byjus.com/kids-learning/moral-stories-the-lion-and-the-mouse/

text = """ Once upon a time, there lived a lion in the dense Amazon rainforest. While he was sleeping by resting his big head on his paws, a tiny little mouse unexpectedly crossed by and ran across the lion’s nose in haste. This woke up the lion and he laid his huge paw angrily on the tiny mouse to kill her.

The poor mouse begged the lion to spare her this time and she would pay him back on some other day. Hearing this, the lion was amused and wondered how could such a tiny creature ever help him. But he was in a good mood and in his generosity he finally let the mouse go.

A few days later, a hunter set a trap for the lion while the big animal was stalking for prey in the forest. Caught in the toils of a hunter’s net, the lion found it difficult to free himself and roared loudly in anger.

As the mouse was passing by, she heard the roar and found the lion struggling hard to free himself from the hunter’s net. The little creature quickly ran towards the lion’s trap that bound him and she gnawed the net with her sharp teeth until the net tore apart. Slowly she made a big hole in the net and soon the lion was able to free himself from the hunter’s trap.

The lion thanked the little mouse for her help and the mouse reminded him that she had finally repaid the lion for sparing her life before. Thereafter, the lion and the mouse became good friends and lived happily in the forest. """

wrapper = textwrap.TextWrapper(width=150)
word_list = wrapper.wrap(text=text)
for element in word_list:
    # print(element)
    pass


def get_keywords(text):
    out = []
    try:
        # extractor = pke.unsupervised.MultipartiteRank()
        extractor = pke.unsupervised.YAKE()
        extractor.load_document(input=text, language="en")
        grammar = r"""
                NP:
                    {<NOUN|PROPN>+}
            """
        extractor.ngram_selection(n=1)
        extractor.grammar_selection(grammar=grammar)
        # pos = {'VERB', 'ADJ', 'NOUN'}
        # pos ={'NOUN'}
        # stoplist = list(string.punctuation)
        # stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        # stoplist += stopwords.words('english')
        # extractor.candidate_selection(n=1,pos=pos, stoplist=stoplist)
        extractor.candidate_selection(n=1)

        extractor.candidate_weighting(window=3, use_stems=False)

        keyphrases = extractor.get_n_best(n=30)

        for val in keyphrases:
            out.append(val[0])
    except:
        out = []
        traceback.print_exc()

    return out


keywords = get_keywords(text)[:8]
print("keywords: ", keywords)

sentences = tokenize_sentences(text)
print(sentences)


def get_sentences_for_keyword(keywords, sentences):
    keyword_processor = KeywordProcessor()
    keyword_sentences = {}
    for word in keywords:
        keyword_sentences[word] = []
        keyword_processor.add_keyword(word)
    for sentence in sentences:
        keywords_found = keyword_processor.extract_keywords(sentence)
        for key in keywords_found:
            keyword_sentences[key].append(sentence)

    for key in keyword_sentences.keys():
        values = keyword_sentences[key]
        values = sorted(values, key=len, reverse=False)
        keyword_sentences[key] = values
    return keyword_sentences


keyword_sentence_mapping = get_sentences_for_keyword(keywords, sentences)
print(keyword_sentence_mapping)


class BertWSD(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        self.ranking_linear = torch.nn.Linear(config.hidden_size, 1)

        self.init_weights()


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = "match_word_meaning/bert_base-augmented-batch_size=128-lr=2e-5-max_gloss=6"


model = BertWSD.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained(model_dir)
tokenizer.added_tokens_encoder["[TGT]"] = 100
# add new special token
if "[TGT]" not in tokenizer.additional_special_tokens:
    tokenizer.add_special_tokens({"additional_special_tokens": ["[TGT]"]})
    assert "[TGT]" in tokenizer.additional_special_tokens
    model.resize_token_embeddings(len(tokenizer))

model.to(DEVICE)
model.eval()

GlossSelectionRecord = namedtuple(
    "GlossSelectionRecord", ["guid", "sentence", "sense_keys", "glosses", "targets"]
)
BertInput = namedtuple(
    "BertInput", ["input_ids", "input_mask", "segment_ids", "label_id"]
)

MAX_SEQ_LENGTH = 128


def _create_features_from_records(
    records,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    pad_on_left=False,
    cls_token="[CLS]",
    sep_token="[SEP]",
    pad_token=0,
    sequence_a_segment_id=0,
    sequence_b_segment_id=1,
    cls_token_segment_id=1,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
    disable_progress_bar=False,
):
    """Convert records to list of features. Each feature is a list of sub-features where the first element is
    always the feature created from context-gloss pair while the rest of the elements are features created from
    context-example pairs (if available)
    `cls_token_at_end` define the location of the CLS token:
        - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
        - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    features = []
    for record in tqdm(records, disable=disable_progress_bar):
        tokens_a = tokenizer.tokenize(record.sentence)

        sequences = [
            (gloss, 1 if i in record.targets else 0)
            for i, gloss in enumerate(record.glosses)
        ]

        pairs = []
        for seq, label in sequences:
            tokens_b = tokenizer.tokenize(seq)

            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = tokens_a + [sep_token]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

            if cls_token_at_end:
                tokens = tokens + [cls_token]
                segment_ids = segment_ids + [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = (
                    [0 if mask_padding_with_zero else 1] * padding_length
                ) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                input_mask = input_mask + (
                    [0 if mask_padding_with_zero else 1] * padding_length
                )
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            pairs.append(
                BertInput(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_id=label,
                )
            )

        features.append(pairs)

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


sentence = "Mark's favourite game is **Cricket**."

sentence_for_bert = sentence.replace("**", " [TGT] ")
sentence_for_bert = " ".join(sentence_for_bert.split())

print(sentence_for_bert)

re_result = re.search(r"\[TGT\](.*)\[TGT\]", sentence_for_bert)
if re_result is None:
    print("\nIncorrect input format. Please try again.")

ambiguous_word = re_result.group(1).strip()

print("Word: ", ambiguous_word)


results = dict()

# wn_pos = wn.NOUN
# for i, synset in enumerate(set(wn.synsets(ambiguous_word, pos=wn_pos))):
for i, synset in enumerate(set(wn.synsets(ambiguous_word))):
    results[synset] = synset.definition()

print(results)

sense_keys = []
definitions = []
for sense_key, definition in results.items():
    sense_keys.append(sense_key)
    definitions.append(definition)


print(sense_keys)
print(definitions)


record = GlossSelectionRecord("test", sentence_for_bert, sense_keys, definitions, [-1])

print(model.config.vocab_size)

features = _create_features_from_records(
    [record],
    MAX_SEQ_LENGTH,
    tokenizer,
    cls_token=tokenizer.cls_token,
    sep_token=tokenizer.sep_token,
    cls_token_segment_id=1,
    pad_token_segment_id=0,
    disable_progress_bar=True,
)[0]

print(len(features))

for ftr in features:
    print(tokenizer.convert_ids_to_tokens(ftr.input_ids))


with torch.no_grad():
    logits = torch.zeros(len(definitions), dtype=torch.double).to(DEVICE)
    for i, bert_input in list(enumerate(features)):
        print(i)
        logits[i] = model.ranking_linear(
            model.bert(
                input_ids=torch.tensor(bert_input.input_ids, dtype=torch.long)
                .unsqueeze(0)
                .to(DEVICE),
                attention_mask=torch.tensor(bert_input.input_mask, dtype=torch.long)
                .unsqueeze(0)
                .to(DEVICE),
                token_type_ids=torch.tensor(bert_input.segment_ids, dtype=torch.long)
                .unsqueeze(0)
                .to(DEVICE),
            )[1]
        )
    scores = softmax(logits, dim=0)

    preds = sorted(
        zip(sense_keys, definitions, scores), key=lambda x: x[-1], reverse=True
    )

print("\n")
for pred in preds:
    print(pred)
sense = preds[0][0]
meaning = preds[0][1]

print("\nMost appropriate sense: ", sense)
print("Most appropriate meaning: ", meaning)


def get_sense(sent):
    re_result = re.search(r"\[TGT\](.*)\[TGT\]", sent)
    if re_result is None:
        print("\nIncorrect input format. Please try again.")

    ambiguous_word = re_result.group(1).strip()
    results = dict()

    for i, synset in enumerate(set(wn.synsets(ambiguous_word))):
        results[synset] = synset.definition()

    if len(results) == 0:
        return None

    sense_keys = []
    definitions = []
    for sense_key, definition in results.items():
        sense_keys.append(sense_key)
        definitions.append(definition)

    record = GlossSelectionRecord("test", sent, sense_keys, definitions, [-1])

    features = _create_features_from_records(
        [record],
        MAX_SEQ_LENGTH,
        tokenizer,
        cls_token=tokenizer.cls_token,
        sep_token=tokenizer.sep_token,
        cls_token_segment_id=1,
        pad_token_segment_id=0,
        disable_progress_bar=True,
    )[0]

    with torch.no_grad():
        logits = torch.zeros(len(definitions), dtype=torch.double).to(DEVICE)
        for i, bert_input in list(enumerate(features)):
            logits[i] = model.ranking_linear(
                model.bert(
                    input_ids=torch.tensor(bert_input.input_ids, dtype=torch.long)
                    .unsqueeze(0)
                    .to(DEVICE),
                    attention_mask=torch.tensor(bert_input.input_mask, dtype=torch.long)
                    .unsqueeze(0)
                    .to(DEVICE),
                    token_type_ids=torch.tensor(
                        bert_input.segment_ids, dtype=torch.long
                    )
                    .unsqueeze(0)
                    .to(DEVICE),
                )[1]
            )
        scores = softmax(logits, dim=0)

        preds = sorted(
            zip(sense_keys, definitions, scores), key=lambda x: x[-1], reverse=True
        )

    sense = preds[0][0]
    meaning = preds[0][1]
    return sense


def get_synsets_for_word(word):
    return set(wn.synsets(word))


keyword_best_sense = {}

for keyword in keyword_sentence_mapping:
    print("\n\n")
    print("Original word: ", keyword)
    try:
        identified_synsets = get_synsets_for_word(keyword)
    except:
        continue
    for synset in identified_synsets:
        print(synset, "   ", synset.definition())
    top_3_sentences = keyword_sentence_mapping[keyword][:3]
    best_senses = []
    for sent in top_3_sentences:
        insensitive_keyword = re.compile(re.escape(keyword), re.IGNORECASE)
        modified_sentence = insensitive_keyword.sub(
            " [TGT] " + keyword + " [TGT] ", sent, count=1
        )
        modified_sentence = " ".join(modified_sentence.split())
        print("modified sentence ", modified_sentence)
        best_sense = get_sense(modified_sentence)
        best_senses.append(best_sense)
    best_sense = mode(best_senses)
    print("Best sense: ", best_sense)
    defn = best_sense.definition()
    print(defn)
    keyword_best_sense[keyword] = defn


print(keyword_best_sense)
