from pyqgen import PyQGen
import glob


pyqgen = PyQGen()

files = glob.glob("*.pdf")

(
    document,
    all_content,
    content_per_page,
) = pyqgen.pdf_to_text.extract_text_with_langchain_pdf(pdf_file=files[0])

chunks_text = pyqgen.splitter.generate_chunks_text(all_content)

choices = 5
chunks_choices = pyqgen.splitter.get_random_chunks(chunks_text, choices)

print(chunks_choices)
