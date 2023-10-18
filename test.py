from pyqgen import PyQGen
import glob


pyqgen = PyQGen()

files = glob.glob("*.pdf")

content = pyqgen.pdf_to_text.extract(pdf_file=files[0])

chunks_text = pyqgen.splitter.generate_chunks_text(content=content)

choices = 5
random_chunks_choices = pyqgen.splitter.get_random_chunks(
    chunks=chunks_text, n_choices=choices
)


def log_chunks(chunks: list):
    print(" ")
    for chunk in chunks:
        print(" ")
        print(len(chunk))
        print(" ")
        print(chunk)
    print(" ")
    print(len(chunks))
    print(" ")


log_chunks(random_chunks_choices)

largest_chunks = pyqgen.splitter.get_largest_chunks(chunks=chunks_text)

log_chunks(largest_chunks)
