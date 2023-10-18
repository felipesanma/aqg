from .content_processing import ContentSplitter, PDF2text, TopicsDetection
from .question_generation import QGenOpenai


class PyQGen:
    def __init__(self) -> None:
        self.splitter = ContentSplitter()
        self.pdf_to_text = PDF2text()
        self.questions = QGenOpenai()
        self.topics = TopicsDetection()
