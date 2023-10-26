from pyqgen import PyQGen
import glob

pyqgen = PyQGen()

files = glob.glob("*.pdf")

content, content_per_page = pyqgen.pdf_to_text.extract(pdf_file=files[0])


chunks_text = pyqgen.splitter.generate_chunks_text(content=content)

mapping = pyqgen.get_pages_mapping_per_chunk(
    chunks_text=chunks_text, content_per_page=content_per_page
)

print(mapping)
topics = pyqgen.topics.classify(chunks_text=chunks_text)


topics_cluster = topics["topics"]
print(topics_cluster)
sorted_topics_cluster = pyqgen.topics.get_sorted_candidates_by_topic(
    chunks_text=chunks_text,  # topics=topics_cluster
)

print(sorted_topics_cluster)
n_choices = 8
index_custom_chunks = pyqgen.splitter.get_custom_chunks(
    chunks=chunks_text, index_list=sorted_topics_cluster, n_choices=n_choices
)

print(index_custom_chunks)
for i in index_custom_chunks:
    print("")
    print(mapping[i])
    print(chunks_text[i])
    print(len(chunks_text[i]))
    mcq_questions = pyqgen.mcq.generate_mcq_questions(content=chunks_text[i])
    print(mcq_questions)
    print("")

"""
if len(custom_chunks) < 11:
    summaries = []
    questions = []
    for chunk in custom_chunks:
        summary = pyqgen.questions.generate(content=chunk, function="summary")
        print(summary)
        summaries.append(summary["summaries"][0]["summary"])
        questions.append(summary["summaries"][0]["questions"])


print(summaries)
print(questions)
final_content = " ".join([summary for summary in summaries])
print(final_content)
final_summary = pyqgen.questions.generate(content=final_content, function="summary")
print(final_summary)
"""
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
