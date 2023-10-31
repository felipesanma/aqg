from pyqgen import PyQGen

pyqgen = PyQGen()

video_id = "CZn236ZHw6A"

video_info = pyqgen.youtube_video_to_text.extract(video_id=video_id)
print(video_info["transcription"])
print(len(video_info["timestamp_and_content_mapping"]))
for content in video_info["timestamp_and_content_mapping"]:
    print(content)
# mcq_questions = pyqgen.mcq.generate_mcq_questions(content=content["content"])
# print(mcq_questions)
# print(len(video_info["timestamp_and_content_mapping"]))
