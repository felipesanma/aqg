from pyqgen import PyQGen
import glob

pyqgen = PyQGen()

files = glob.glob("*.mp3")

audio_content = pyqgen.audio_to_text.extract(audio_file=files[0])

model = "gpt-4-1106-preview"
summary = pyqgen.summary.generate_summary(
    content=audio_content["transcription"], model=model
)

print(summary)

# print(len(audio_content["timestamp_and_content_mapping"]))
# for content in audio_content["timestamp_and_content_mapping"]:
#    print(content["content"])
#    print(pyqgen.lang_detector.detect_language_from_text(content["content"]))
#    mcq_questions = pyqgen.mcq.generate_mcq_questions(content=content["content"])
#    print(mcq_questions)
