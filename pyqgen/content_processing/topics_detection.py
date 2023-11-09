from scipy.spatial.distance import cosine
import networkx as nx
from networkx.algorithms import community
from sentence_transformers import SentenceTransformer
import numpy as np
import torch


class TopicsDetection:
    def __init__(self) -> None:
        pass

    def get_embeddings(self, *, sentences: str):
        # set device to GPU if available
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        # load the retriever model from huggingface model hub
        retriever = SentenceTransformer(
            "sentence-transformers/distiluse-base-multilingual-cased-v1", device=device
        )

        # Create the embeddings for our sentences
        return retriever.encode(sentences, convert_to_tensor=True)

    def create_similarity_matrix(self, *, content_embeds):
        num_1_chunks = len(content_embeds)
        # Get similarity matrix between the embeddings of the chunk summaries
        content_similarity_matrix = np.zeros((num_1_chunks, num_1_chunks))
        content_similarity_matrix[:] = np.nan

        for row in range(num_1_chunks):
            for col in range(row, num_1_chunks):
                # Calculate cosine similarity between the two vectors
                similarity = 1 - cosine(content_embeds[row], content_embeds[col])
                content_similarity_matrix[row, col] = similarity
                content_similarity_matrix[col, row] = similarity

        return content_similarity_matrix

    # Run the community detection algorithm
    def get_topics(
        self, *, title_similarity, num_topics=8, bonus_constant=0.25, min_size=3
    ):
        proximity_bonus_arr = np.zeros_like(title_similarity)
        for row in range(proximity_bonus_arr.shape[0]):
            for col in range(proximity_bonus_arr.shape[1]):
                if row == col:
                    proximity_bonus_arr[row, col] = 0
                else:
                    proximity_bonus_arr[row, col] = (
                        1 / (abs(row - col)) * bonus_constant
                    )

        title_similarity += proximity_bonus_arr

        title_nx_graph = nx.from_numpy_array(title_similarity)

        desired_num_topics = num_topics
        # Store the accepted partitionings
        topics_title_accepted = []

        resolution = 0.85
        resolution_step = 0.01
        iterations = 40

        # Find the resolution that gives the desired number of topics
        topics_title = []
        while len(topics_title) not in [
            desired_num_topics,
            desired_num_topics + 1,
            desired_num_topics + 2,
        ]:
            topics_title = community.louvain_communities(
                title_nx_graph, weight="weight", resolution=resolution
            )
            resolution += resolution_step

        topic_sizes = [len(c) for c in topics_title]
        sizes_sd = np.std(topic_sizes)

        lowest_sd_iteration = 0
        # Set lowest sd to inf
        lowest_sd = float("inf")

        for i in range(iterations):
            topics_title = community.louvain_communities(
                title_nx_graph, weight="weight", resolution=resolution
            )

            # Check SD
            topic_sizes = [len(c) for c in topics_title]
            sizes_sd = np.std(topic_sizes)

            topics_title_accepted.append(topics_title)

            if sizes_sd < lowest_sd and min(topic_sizes) >= min_size:
                lowest_sd_iteration = i
                lowest_sd = sizes_sd

        # Set the chosen partitioning to be the one with highest modularity
        topics_title = topics_title_accepted[lowest_sd_iteration]
        # print(f"Best SD: {lowest_sd}, Best iteration: {lowest_sd_iteration}")

        topic_id_means = [sum(e) / len(e) for e in topics_title]
        # Arrange title_topics in order of topic_id_means
        topics_title = [
            list(c)
            for _, c in sorted(
                zip(topic_id_means, topics_title), key=lambda pair: pair[0]
            )
        ]
        # Create an array denoting which topic each chunk belongs to
        chunk_topics = [None] * title_similarity.shape[0]
        for i, c in enumerate(topics_title):
            for j in c:
                chunk_topics[j] = i

        return {"chunk_topics": chunk_topics, "topics": topics_title}

    def classify(self, *, chunks_text, num_topics: int = 8):
        embeddings = self.get_embeddings(sentences=chunks_text)

        similarity_matrix = self.create_similarity_matrix(content_embeds=embeddings)

        num_topics = min(int(len(chunks_text) / 4), num_topics)
        return self.get_topics(
            title_similarity=similarity_matrix,
            num_topics=num_topics,
            bonus_constant=0.2,
        )

    def get_sorted_candidates_by_topic(
        self, *, chunks_text, topics=None, dsc: bool = True
    ):
        if topics is None:
            topics = self.classify(chunks_text=chunks_text)["topics"]
        sorted_topics_cluster = []

        for topic in topics:
            tmp_topic = {}
            for index in topic:
                tmp_topic[index] = len(chunks_text[int(index)])
            tmp_topic = sorted(tmp_topic.items(), key=lambda x: x[1], reverse=dsc)
            # characters = [index[1] for index in tmp_topic]
            tmp_topic = [index[0] for index in tmp_topic]
            sorted_topics_cluster.append(tmp_topic)
        return sorted_topics_cluster
