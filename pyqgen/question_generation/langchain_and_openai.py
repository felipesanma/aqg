import re
import json

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.chat_models import ChatOpenAI

from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from dotenv import load_dotenv

load_dotenv()


class QGenLangChain:
    def __init__(self, content, questions_number: int = 5) -> None:
        self.content = content
        self.questions_number = questions_number
        self.response_schemas = self._get_response_schemas()
        self.prompt = self._get_prompt_template()

    def _get_response_schemas(self):
        response_schemas = [
            ResponseSchema(
                name="question",
                description="A multiple choice question generated from input text snippet.",
            ),
            ResponseSchema(
                name="options",
                description="Possible choices for the multiple choice question.",
            ),
            ResponseSchema(
                name="answer", description="Correct answer for the question."
            ),
        ]

        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

        return output_parser.get_format_instructions()

    def _get_prompt_template(self):
        return ChatPromptTemplate(
            messages=[
                HumanMessagePromptTemplate.from_template(
                    """Given a text input, generate {questions_number} multiple choice questions 
            from it along with the correct answer. 
            \n{format_instructions}\n{user_prompt}"""
                )
            ],
            input_variables=["user_prompt"],
            partial_variables={
                "format_instructions": self.response_schemas,
                "questions_number": self.questions_number,
            },
        )

    def _parser_output(self, user_query_output):
        markdown_text = user_query_output.content

        json_string = re.search(r"```json\n(.*?)```", markdown_text, re.DOTALL).group(1)

        return json.loads(f"[{json_string}]")

    def generate(self):
        chat_model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        user_query = self.prompt.format_prompt(user_prompt=self.content)
        user_query_output = chat_model(user_query.to_messages())
        return self._parser_output(user_query_output)


text = """Elon Musk has shown again he can influence the digital currency market with just his tweets. After saying that his electric vehicle-making company
Tesla will not accept payments in Bitcoin because of environmental concerns, he tweeted that he was working with developers of Dogecoin to improve
system transaction efficiency. Following the two distinct statements from him, the world's largest cryptocurrency hit a two-month low, while Dogecoin
rallied by about 20 percent. The SpaceX CEO has in recent months often tweeted in support of Dogecoin, but rarely for Bitcoin.  In a recent tweet,
Musk put out a statement from Tesla that it was “concerned” about the rapidly increasing use of fossil fuels for Bitcoin (price in India) mining and
transaction, and hence was suspending vehicle purchases using the cryptocurrency.  A day later he again tweeted saying, “To be clear, I strongly
believe in crypto, but it can't drive a massive increase in fossil fuel use, especially coal”.  It triggered a downward spiral for Bitcoin value but
the cryptocurrency has stabilised since.   A number of Twitter users welcomed Musk's statement. One of them said it's time people started realising
that Dogecoin “is here to stay” and another referred to Musk's previous assertion that crypto could become the world's future currency."""
"""
qgen = QGenLangChain(text)
print(qgen.generate())
"""
