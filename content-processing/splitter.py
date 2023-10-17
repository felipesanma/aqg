import pandas as pd


class ContentSplitter:
    def __init__(
        self,
        content: str,
    ):
        self.content = content

    def clean_text_v2(self, s: str) -> str:
        return s.replace("\n", " ")

    def get_segments(self, txt):
        segments = txt.split(".")
        # Put the . back in
        segments = [segment + "." for segment in segments]
        # Further split by comma
        segments = [segment.split(",") for segment in segments]
        # Flatten
        segments = [item for sublist in segments for item in sublist]
        return segments

    def create_sentences(self, segments, MIN_WORDS, MAX_WORDS):
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

    def create_chunks(self, sentences, CHUNK_LENGTH, STRIDE):
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
        min_words: int = 35,
        max_words: int = 100,
        chunk_lenght: int = 8,
        stride: int = 1,
    ):
        all_content = self.clean_text_v2(self.content)
        segments = self.get_segments(all_content)
        sentences = self.create_sentences(
            segments, MIN_WORDS=min_words, MAX_WORDS=max_words
        )
        chunks = self.create_chunks(sentences, CHUNK_LENGTH=chunk_lenght, STRIDE=stride)
        chunks = [chunk["text"] for chunk in chunks]
        while "" in chunks:
            chunks.remove("")
        return chunks
