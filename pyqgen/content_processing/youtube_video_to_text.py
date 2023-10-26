from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound
from youtubesearchpython import Video, ResultMode


class YT2text:
    def __init__(self) -> None:
        pass

    def generate_transcript(self, *, id, lan="en"):
        try:
            transcript = YouTubeTranscriptApi.get_transcript(id, languages=[lan])
            return transcript
        except NoTranscriptFound:
            return None, None

    def get_content_from_transcript(self, *, transcript):
        content = "".join(
            [text["text"] + " " for text in transcript if text["text"] != "[Music]"]
        )
        return content

    def mapping_content_and_timestamp_from_transcript(
        self, *, transcript, max_length: int = 2000
    ):
        mapping = []
        tmp_text = ""
        max_length = max_length
        start_time = 0
        for text in transcript:
            tmp_text += f'{text["text"]} '
            if len(tmp_text) >= max_length:
                tmp_element = {
                    "content": tmp_text,
                    "start_time": start_time,
                    "end_time": text["start"],
                }
                print(tmp_element)
                mapping.append(tmp_element)
                tmp_text = ""
                start_time = text["start"]
        return mapping

    def extract(self, *, video_id: str, language: str = "es", max_length: int = 2000):
        transcript = self.generate_transcript(id=video_id, lan=language)
        content = self.get_content_from_transcript(transcript=transcript)
        mapping = self.mapping_content_and_timestamp_from_transcript(
            transcript=transcript, max_length=max_length
        )
        video_info = Video.getInfo(video_id, mode=ResultMode.json)
        video_mapping = {
            "id": video_info["id"],
            "title": video_info["title"],
            "description": video_info["description"],
            "transcription": content,
            "timestamp_and_content_mapping": mapping,
        }

        return video_mapping
