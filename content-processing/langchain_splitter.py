from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_text(text):
    rec_text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        separators=["\n\n", "\n"],
        chunk_size=4000,
        chunk_overlap=0,
        length_function=len,
    )

    return rec_text_splitter.split_text(text)
