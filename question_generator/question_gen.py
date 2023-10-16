from typing import Any
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


class QuestionGenerator:
    def __init__(
        self,
        context: str,
        answer: str,
        n_questions: int = 1,
        model_version: str = "v1.2",
    ):
        self.context = context
        self.answer = answer
        self.n_questions = n_questions
        self.model_version = model_version
        self.model_name = "pipesanma/chasquilla-question-generator"

    def _question_parser(self, question: str) -> str:
        return " ".join(question.split(":")[1].split())

    def generate(self):
        model = T5ForConditionalGeneration.from_pretrained(
            self.model_name, revision=self.model_version
        )
        tokenizer = T5Tokenizer.from_pretrained(
            self.model_name, revision=self.model_version
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print("device ", device)
        model = model.to(device)

        text = f"context: {self.context} answer: {self.answer}"

        encoding = tokenizer.encode_plus(
            text, max_length=512, padding=True, return_tensors="pt"
        )
        input_ids, attention_mask = encoding["input_ids"].to(device), encoding[
            "attention_mask"
        ].to(device)

        model.eval()
        beam_outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=72,
            early_stopping=True,
            num_beams=5,
            num_return_sequences=self.n_questions,
        )

        questions = []

        for beam_output in beam_outputs:
            sent = tokenizer.decode(
                beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

            questions.append(self._question_parser(sent))

        return questions


"""
context = "President Donald Trump said and predicted that some states would reopen this month."
answer = "Donald Trump"

questions = QuestionGenerator(context, answer, 2)
print(questions.generate())
print(questions.answer)
"""
