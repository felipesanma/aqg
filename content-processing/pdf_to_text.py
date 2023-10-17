import glob
from langchain.document_loaders import UnstructuredFileLoader
import re
import numpy as np
from custom_splitter import generate_chunks_text
from summarize_langchain import (
    summarize_stage_1,
    create_documents_embeds,
    summarize_stage_2,
)
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


"""
files = glob.glob("content-processing/*.pdf")

document, all_content, content_per_page = extract_text_with_langchain_pdf(files[0])

chunks_text = generate_chunks_text(all_content)
# Run Stage 1 Summarizing
stage_1_outputs = summarize_stage_1(chunks_text)
# Split the titles and summaries
stage_1_summaries = [e["summary"] for e in stage_1_outputs]
stage_1_titles = [e["title"] for e in stage_1_outputs]
num_1_chunks = len(stage_1_summaries)
# print(stage_1_summaries)
# print(stage_1_titles)
# print(num_1_chunks)


summary_embeds = np.array(create_documents_embeds(stage_1_summaries))
title_embeds = np.array(create_documents_embeds(stage_1_titles))



embeddings = []
for chunk in chunks_text:
    embeddings.append(get_embedding(chunk))

summary_similarity_matrix = create_similarity_matrix(summary_embeds)

num_topics = min(int(len(chunks_text) / 4), 8)
topics_out = get_topics(
    summary_similarity_matrix, num_topics=num_topics, bonus_constant=0.2
)

chunk_topics = topics_out["chunk_topics"]
topics = topics_out["topics"]
print(topics_out)

out = summarize_stage_2(stage_1_outputs, topics, summary_num_words=250)
stage_2_outputs = out["stage_2_outputs"]
stage_2_titles = [e["title"] for e in stage_2_outputs]
stage_2_summaries = [e["summary"] for e in stage_2_outputs]
final_summary = out["final_summary"]

print(stage_2_outputs)

print(final_summary)


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
