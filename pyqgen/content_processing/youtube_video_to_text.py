from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound


def generate_transcript(id, lan="en"):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(id, languages=[lan])
        return transcript
    except NoTranscriptFound:
        return None, None


def get_content_from_transcript(transcript):
    content = [text["text"] + " " for text in transcript if text["text"] != "[Music]"]
    return content


def mapping_content_and_timestamp_from_transcript(transcript):
    return ""


video_id = "CZn236ZHw6A"
transcript = generate_transcript(video_id, "es")
content = get_content_from_transcript(transcript)
for text in transcript[:5]:
    print(f"{len(text['text'])}")
    print(text["text"])
# print(script)
