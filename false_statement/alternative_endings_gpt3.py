import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def chatGPT(text):
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=text,
            temperature=0,
            max_tokens=2000,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
    except Exception as err:
        print(err)
        response = {"choices": [{"text": ""}]}
    return response["choices"][0]["text"]
