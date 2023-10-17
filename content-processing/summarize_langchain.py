from datetime import datetime
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv

load_dotenv()


def create_documents_embeds(content):
    openai_embed = OpenAIEmbeddings()

    return openai_embed.embed_documents(content)


def parse_title_summary_results(results):
    out = []
    for e in results:
        e = e.replace("\n", "")
        if "|" in e:
            processed = {"title": e.split("|")[0], "summary": e.split("|")[1][1:]}
        elif ":" in e:
            processed = {"title": e.split(":")[0], "summary": e.split(":")[1][1:]}
        elif "-" in e:
            processed = {"title": e.split("-")[0], "summary": e.split("-")[1][1:]}
        else:
            processed = {"title": "", "summary": e}
        out.append(processed)
    return out


def summarize_stage_1(chunks_text, language="Spanish"):
    print(f"Start time: {datetime.now()}")

    # Prompt to get title and summary for each chunk
    map_prompt_template = """Firstly, give the following text an informative title in {language}. Then, on a new line, write a 75-100 word summary in {language} of the following text:
    {text}

    Return your answer in the following format:
    Title | Summary...
    e.g. 
    Why Artificial Intelligence is Good | AI can make humans more productive by automating many repetitive processes.

    TITLE AND CONCISE SUMMARY:"""

    map_prompt = PromptTemplate(
        template=map_prompt_template, input_variables=["text", "language"]
    )

    # Define the LLMs
    map_llm = OpenAI(temperature=0, model_name="text-davinci-003")
    map_llm_chain = LLMChain(llm=map_llm, prompt=map_prompt)
    map_llm_chain_input = [{"text": t, "language": language} for t in chunks_text]
    # Run the input through the LLM chain (works in parallel)
    map_llm_chain_results = map_llm_chain.apply(map_llm_chain_input)

    stage_1_outputs = parse_title_summary_results(
        [e["text"] for e in map_llm_chain_results]
    )

    print(f"Stage 1 done time {datetime.now()}")

    return stage_1_outputs


def summarize_stage_2(
    stage_1_outputs, topics, summary_num_words=250, language="Spanish"
):
    print(f"Stage 2 start time {datetime.now()}")

    # Prompt that passes in all the titles of a topic, and asks for an overall title of the topic
    title_prompt_template = """Write in {language} an informative title that summarizes each of the following groups of titles. Make sure that the titles capture as much information as possible, 
    and are different from each other:
    {text}
    
    Return your answer in a numbered list, with new line separating each title: 
    1. Title 1
    2. Title 2
    3. Title 3

    TITLES:
    """

    map_prompt_template = """Wite in {language} a 75-100 word summary of the following text:
    {text}

    CONCISE SUMMARY:"""

    combine_prompt_template = (
        "Write in {language} a "
        + str(summary_num_words)
        + """-word summary of the following, removing irrelevant information. Finish your answer:
    {text}
    """
        + str(summary_num_words)
        + """-WORD SUMMARY:"""
    )

    title_prompt = PromptTemplate(
        template=title_prompt_template, input_variables=["text", "language"]
    )
    map_prompt = PromptTemplate(
        template=map_prompt_template, input_variables=["text", "language"]
    )
    combine_prompt = PromptTemplate(
        template=combine_prompt_template, input_variables=["text", "language"]
    )

    topics_data = []
    for c in topics:
        topic_data = {
            "summaries": [stage_1_outputs[chunk_id]["summary"] for chunk_id in c],
            "titles": [stage_1_outputs[chunk_id]["title"] for chunk_id in c],
        }
        topic_data["summaries_concat"] = " ".join(topic_data["summaries"])
        topic_data["titles_concat"] = ", ".join(topic_data["titles"])
        topics_data.append(topic_data)

    # Get a list of each community's summaries (concatenated)
    topics_summary_concat = [c["summaries_concat"] for c in topics_data]
    topics_titles_concat = [c["titles_concat"] for c in topics_data]

    # Concat into one long string to do the topic title creation
    topics_titles_concat_all = """"""
    for i, c in enumerate(topics_titles_concat):
        topics_titles_concat_all += f"""{i+1}. {c}
    """

    # print('topics_titles_concat_all', topics_titles_concat_all)

    title_llm = OpenAI(temperature=0, model_name="text-davinci-003")
    title_llm_chain = LLMChain(llm=title_llm, prompt=title_prompt)
    title_llm_chain_input = [{"text": topics_titles_concat_all, "language": language}]
    title_llm_chain_results = title_llm_chain.apply(title_llm_chain_input)

    # Split by new line
    titles = title_llm_chain_results[0]["text"].split("\n")
    # Remove any empty titles
    titles = [t for t in titles if t != ""]
    # Remove spaces at start or end of each title
    titles = [t.strip() for t in titles]

    map_llm = OpenAI(temperature=0, model_name="text-davinci-003")
    reduce_llm = OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=-1)

    # Run the map-reduce chain
    docs = [Document(page_content=t) for t in topics_summary_concat]
    chain = load_summarize_chain(
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        return_intermediate_steps=True,
        llm=map_llm,
        reduce_llm=reduce_llm,
    )

    output = chain({"input_documents": docs}, return_only_outputs=True)
    summaries = output["intermediate_steps"]
    stage_2_outputs = [{"title": t, "summary": s} for t, s in zip(titles, summaries)]
    final_summary = output["output_text"]

    # Return: stage_1_outputs (title and summary), stage_2_outputs (title and summary), final_summary, chunk_allocations
    out = {"stage_2_outputs": stage_2_outputs, "final_summary": final_summary}
    print(f"Stage 2 done time {datetime.now()}")

    return out
