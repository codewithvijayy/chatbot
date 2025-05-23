from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextChunk():
    def chunks(self,texts):
        text_spliter = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap = 50)
        text_chunk = text_spliter.split_documents(texts)
        return text_chunk

chunks = TextChunk()