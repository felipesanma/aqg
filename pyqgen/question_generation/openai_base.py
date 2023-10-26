import openai
import json
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


class BaseOpenAI:
    """
    Clase base para todos los demás módulos de subdomains
    """

    def __init__(self):
        pass

    def _output_parser(self, openai_repsonse):
        return json.loads(
            openai_repsonse["choices"][0]["message"]["function_call"]["arguments"]
        )

    def generate(
        self,
        *,
        content: str,
        openai_function: list,
        model: str = "gpt-3.5-turbo",
        language: str = "Spanish",
    ):
        response = openai.ChatCompletion.create(
            model=model,  # "gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": f"You always answer in {language}.",
                },
                {"role": "user", "content": content},
            ],
            functions=openai_function,
            function_call={"name": openai_function[0]["name"]},
        )
        return self._output_parser(response)
