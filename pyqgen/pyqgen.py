from .content_processing import (
    ContentSplitter,
    PDF2text,
    TopicsDetection,
    YT2text,
    LangDetect,
    Audio2text,
    Web2text,
)
from .question_generation import MCQ, Summary


class PyQGen:
    def __init__(self) -> None:
        self.splitter = ContentSplitter()
        self.pdf_to_text = PDF2text()
        self.youtube_video_to_text = YT2text()
        self.mcq = MCQ()
        self.summary = Summary()
        self.topics = TopicsDetection()
        self.lang_detector = LangDetect()
        self.audio_to_text = Audio2text()
        self.web_to_text = Web2text()

    def get_pages_mapping_per_chunk(
        self, *, chunks_text: list, content_per_page: dict
    ) -> list:
        pages = list(content_per_page.keys())
        pages_ready = []
        mapping = []

        for chunk in chunks_text:
            chunk_length = len(chunk)
            acum_length = 0
            tmp_pages = []

            while acum_length < chunk_length:
                if len(pages_ready) == len(pages):
                    page = pages[-1]
                    acum_length += chunk_length
                else:
                    page = pages[len(pages_ready)]
                acum_length += len(content_per_page[page])
                if page not in pages_ready:
                    pages_ready.append(page)
                tmp_pages.append(page)
            tmp_pages = list(dict.fromkeys(tmp_pages))
            mapping.append(tmp_pages)
            diff = acum_length - chunk_length
            content_per_page[page] = content_per_page[page][:diff]
            pages_ready.pop()

        return mapping
