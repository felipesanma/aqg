import openai
import json
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


class QGenOpenai:
    def __init__(self) -> None:
        pass

    def _create_mcq_function(self, questions_number: int = 1):
        return [
            {
                "name": "create_mcq",
                "description": f"Create {questions_number} multiple choice questions from the input text with four candidate options. Three options are incorrect and one is correct. Indicate the correct option after each question.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "questions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "question": {
                                        "type": "string",
                                        "description": "A multiple choice question extracted from the input text.",
                                    },
                                    "options": {
                                        "type": "array",
                                        "items": {
                                            "type": "string",
                                            "description": "Candidate option for the extracted multiple choice question.",
                                        },
                                    },
                                    "answer": {
                                        "type": "string",
                                        "description": "Correct option for the multiple choice question.",
                                    },
                                },
                            },
                        }
                    },
                },
                "required": ["questions"],
            }
        ]

    def _output_parser(self, openai_repsonse):
        return json.loads(
            openai_repsonse["choices"][0]["message"]["function_call"]["arguments"]
        )

    def generate(
        self, *, content: str, questions_number: int = 1, model: str = "gpt-3.5-turbo"
    ):
        mcq_function = self._create_mcq_function(questions_number=questions_number)
        response = openai.ChatCompletion.create(
            model=model,  # "gpt-4",
            messages=[{"role": "user", "content": content}],
            functions=mcq_function,
            function_call={"name": "create_mcq"},
        )
        return self._output_parser(response)


""" text = Elon Musk has shown again he can influence the digital currency market with just his tweets. After saying that his electric vehicle-making company
Tesla will not accept payments in Bitcoin because of environmental concerns, he tweeted that he was working with developers of Dogecoin to improve
system transaction efficiency. Following the two distinct statements from him, the world's largest cryptocurrency hit a two-month low, while Dogecoin
rallied by about 20 percent. The SpaceX CEO has in recent months often tweeted in support of Dogecoin, but rarely for Bitcoin.  In a recent tweet,
Musk put out a statement from Tesla that it was “concerned” about the rapidly increasing use of fossil fuels for Bitcoin (price in India) mining and
transaction, and hence was suspending vehicle purchases using the cryptocurrency.  A day later he again tweeted saying, “To be clear, I strongly
believe in crypto, but it can't drive a massive increase in fossil fuel use, especially coal”.  It triggered a downward spiral for Bitcoin value but
the cryptocurrency has stabilised since.   A number of Twitter users welcomed Musk's statement. One of them said it's time people started realising
that Dogecoin “is here to stay” and another referred to Musk's previous assertion that crypto could become the world's future currency."""

"""
questions_number = 5
questions_generator = QGenOpenai(text, questions_number)
questions = questions_generator.generate()
print(questions)
print(type(questions))
print(len(questions))
"""
