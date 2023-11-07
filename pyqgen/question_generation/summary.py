from .openai_base import BaseOpenAI


class Summary(BaseOpenAI):
    def create_summary_function(self, max_words: int = 250):
        return [
            {
                "name": "create_summary",
                "description": f"Create an executive summary of the input text. Use about {max_words}.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "A summary from the input text.",
                        }
                    },
                },
                "required": ["summaries"],
            }
        ]

    def generate_summary(
        self,
        *,
        content: str,
        max_words: int = 80,
        language: str = "Spanish",
        model: str = "gpt-3.5-turbo",
    ):
        function = self.create_summary_function(max_words=max_words)
        return self.generate(
            content=content, openai_function=function, language=language, model=model
        )
