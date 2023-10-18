from pyqgen import PyQGen
import glob


pyqgen = PyQGen()

files = glob.glob("*.pdf")

content = pyqgen.pdf_to_text.extract(pdf_file=files[0])

chunks_text = pyqgen.splitter.generate_chunks_text(content=content)

choices = 5
chunks_choices = pyqgen.splitter.get_random_chunks(
    chunks=chunks_text, n_choices=choices
)

for i, chunk in enumerate(chunks_choices):
    print(len(chunk))

print(len(chunks_choices))
