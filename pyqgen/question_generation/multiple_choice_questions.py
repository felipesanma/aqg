from .openai_base import BaseOpenAI


class MCQ(BaseOpenAI):
    def create_mcq_function(self, questions_number: int = 1):
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

    def generate_mcq_questions(
        self, *, content: str, questions_number: int = 4, language: str = "Spanish"
    ):
        function = self.create_mcq_function(questions_number=questions_number)
        return self.generate(
            content=content, openai_function=function, language=language
        )
