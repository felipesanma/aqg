from pyqgen import PyQGen

pyqgen = PyQGen()

video_id = "CZn236ZHw6A"

video_info = pyqgen.youtube_video_to_text.extract(video_id=video_id)
print(video_info)
