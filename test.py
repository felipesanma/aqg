from pyqgen import PyQGen
import glob


pyqgen = PyQGen()

files = glob.glob("*.pdf")

content = pyqgen.pdf_to_text.extract(pdf_file=files[0])

chunks_text = pyqgen.splitter.generate_chunks_text(content=content)

print(len(chunks_text))

topics = pyqgen.topics.classify(chunks_text=chunks_text)


topics_cluster = topics["topics"]
print(topics_cluster)
sorted_topics_cluster = pyqgen.topics.get_sorted_candidates_by_topic(
    chunks_text=chunks_text, topics=topics_cluster
)

print(sorted_topics_cluster)

custom_chunks = pyqgen.splitter.get_custom_chunks(
    chunks=chunks_text, index_list=sorted_topics_cluster
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


log_chunks(custom_chunks)

"""
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

choices = 5
random_chunks_choices = pyqgen.splitter.get_random_chunks(
    chunks=chunks_text, n_choices=choices
)


log_chunks(random_chunks_choices)

largest_chunks = pyqgen.splitter.get_largest_chunks(chunks=chunks_text)

log_chunks(largest_chunks)
"""
