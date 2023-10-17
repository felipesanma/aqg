import glob
from langchain.document_loaders import UnstructuredFileLoader
import re
import numpy as np
from custom_splitter import generate_chunks_text
from topics_detection import (
    get_embedding,
    create_similarity_matrix,
    get_topics,
)


def group_content_per_page(documents: list) -> dict:
    content_per_page = {}
    for doc in documents:
        if doc.metadata["page_number"] in content_per_page:
            content_per_page[doc.metadata["page_number"]] += f" {doc.page_content}"
        else:
            content_per_page[doc.metadata["page_number"]] = doc.page_content
    return content_per_page


def clean_text(s: str) -> str:
    clean = re.compile("<.*?>")
    s = re.sub(clean, "", s)
    s = s.replace("\r", " ")
    # s = s.replace("\n", " ").replace("\r", " ")
    # s = s.replace(":selected:", "").replace(":unselected:", "")
    # s = s.replace('\"', '')
    # s = s.replace(".", "")
    return s


def extract_text_with_langchain_pdf(pdf_file):
    loader = UnstructuredFileLoader(pdf_file, strategy="fast", mode="elements")
    documents = loader.load()

    pdf_pages_content = "\n".join(doc.page_content for doc in documents)
    clean_content = clean_text(pdf_pages_content)

    content_per_page = group_content_per_page(documents)
    for k, v in content_per_page.items():
        content_per_page[k] = clean_text(v)

    return documents, clean_content, content_per_page


files = glob.glob("content-processing/*.pdf")

document, all_content, content_per_page = extract_text_with_langchain_pdf(files[0])

chunks_text = generate_chunks_text(all_content)

embeddings = []
for chunk in chunks_text:
    embeddings.append(get_embedding(chunk))

embeddings_array = np.array(embeddings)

similarity_matrix = create_similarity_matrix(embeddings_array)

num_topics = min(int(len(chunks_text) / 4), 8)
topics_out = get_topics(similarity_matrix, num_topics=num_topics, bonus_constant=0.2)
print(topics_out)
chunk_topics = topics_out["chunk_topics"]
topics = topics_out["topics"]
"""
all_lens = []
for i, _ in enumerate(chunks_text):
    print(f"chunk # {i}, size: {len(chunks_text[i])}")
    print("--------------------")
    print("\n")
    print(chunks_text[i])
    print("\n")
    all_lens.append(len(chunks_text[i]))

print(max(all_lens))
"""
