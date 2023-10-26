from .openai_base import BaseOpenAI


class Summary(BaseOpenAI):
    def create_summary_function(self, max_words: int = 80):
        return [
            {
                "name": "create_summary",
                "description": f"Create an executive summary of the input text. Use about {max_words}. At the end provide 3 interesting questions that can be answered with the input text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summaries": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "summary": {
                                        "type": "string",
                                        "description": "A summary from the input text.",
                                    },
                                    "questions": {
                                        "type": "array",
                                        "items": {
                                            "type": "string",
                                            "description": "An interesting question extracted from the input text.",
                                        },
                                    },
                                },
                            },
                        }
                    },
                },
                "required": ["summaries"],
            }
        ]

    def generate_summary(
        self, *, content: str, max_words: int = 80, language: str = "Spanish"
    ):
        function = self.create_summary_function(max_words=max_words)
        return self.generate(
            content=content, openai_function=function, language=language
        )
