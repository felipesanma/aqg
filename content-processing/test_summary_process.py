import glob
import numpy as np
from splitter import ContentSplitter
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
from pdf_to_text import extract_text_with_langchain_pdf


files = glob.glob("content-processing/*.pdf")

document, all_content, content_per_page = extract_text_with_langchain_pdf(files[0])

chunks_text = ContentSplitter(all_content).generate_chunks_text()
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
