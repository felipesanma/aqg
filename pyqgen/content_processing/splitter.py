import pandas as pd
import random


class ContentSplitter:
    def __init__(self) -> None:
        pass

    def clean_text_v2(self, *, s: str) -> str:
        return s.replace("\n", " ")

    def get_segments(self, *, txt: str):
        segments = txt.split(".")
        # Put the . back in
        segments = [segment + "." for segment in segments]
        # Further split by comma
        segments = [segment.split(",") for segment in segments]
        # Flatten
        segments = [item for sublist in segments for item in sublist]
        return segments

    def create_sentences(self, *, segments, MIN_WORDS, MAX_WORDS):
        # Combine the non-sentences together
        sentences = []

        is_new_sentence = True
        sentence_length = 0
        sentence_num = 0
        sentence_segments = []

        for i in range(len(segments)):
            if is_new_sentence == True:
                is_new_sentence = False
            # Append the segment
            sentence_segments.append(segments[i])
            segment_words = segments[i].split(" ")
            sentence_length += len(segment_words)

            # If exceed MAX_WORDS, then stop at the end of the segment
            # Only consider it a sentence if the length is at least MIN_WORDS
            if (
                sentence_length >= MIN_WORDS and segments[i][-1] == "."
            ) or sentence_length >= MAX_WORDS:
                sentence = " ".join(sentence_segments)
                sentences.append(
                    {
                        "sentence_num": sentence_num,
                        "text": sentence,
                        "sentence_length": sentence_length,
                    }
                )
                # Reset
                is_new_sentence = True
                sentence_length = 0
                sentence_segments = []
                sentence_num += 1

        return sentences

    def create_chunks(self, *, sentences, CHUNK_LENGTH, STRIDE):
        sentences_df = pd.DataFrame(sentences)

        chunks = []
        for i in range(0, len(sentences_df), (CHUNK_LENGTH - STRIDE)):
            chunk = sentences_df.iloc[i : i + CHUNK_LENGTH]
            chunk_text = " ".join(chunk["text"].tolist())

            chunks.append(
                {
                    "start_sentence_num": chunk["sentence_num"].iloc[0],
                    "end_sentence_num": chunk["sentence_num"].iloc[-1],
                    "text": chunk_text,
                    "num_words": len(chunk_text.split(" ")),
                }
            )

        chunks_df = pd.DataFrame(chunks)
        return chunks_df.to_dict("records")

    def generate_chunks_text(
        self,
        *,
        content,
        min_words: int = 35,
        max_words: int = 100,
        chunk_lenght: int = 8,
        stride: int = 1,
    ):
        all_content = self.clean_text_v2(s=content)
        segments = self.get_segments(txt=all_content)
        sentences = self.create_sentences(
            segments=segments, MIN_WORDS=min_words, MAX_WORDS=max_words
        )
        chunks = self.create_chunks(
            sentences=sentences, CHUNK_LENGTH=chunk_lenght, STRIDE=stride
        )
        chunks = [chunk["text"] for chunk in chunks]
        while "" in chunks:
            chunks.remove("")
        return chunks

    def get_random_chunks(self, *, chunks: list, n_choices: int = 5):
        n_characters = sum(map(len, chunks)) / len(chunks)

        choices = []
        while len(choices) < n_choices:
            sample = random.choice(chunks)
            if sample not in choices and len(sample) >= n_characters:
                choices.append(sample)
        return choices

    def get_largest_chunks(self, *, chunks: list, n_choices: int = 5):
        sorted_list = sorted(chunks, key=len, reverse=True)

        return sorted_list[:n_choices]

    def get_custom_chunks(self, *, chunks: list, index_list: list, n_choices: int = 5):
        n_characters = sum(map(len, chunks)) / len(chunks)
        index_choices = []
        row = 0
        column = 0
        while len(index_choices) < n_choices:
            if len(chunks[index_list[column][row]]) >= n_characters:
                index_choices.append(index_list[column][row])
            print(column, row)
            if column == len(index_list):
                row += 1
                column = 0
            else:
                column += 1
        return [chunks[index] for index in index_choices]
