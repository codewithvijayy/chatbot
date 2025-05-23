from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os


class ChunkEmbeddings:
    def __init__(self):
        self.embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.getenv("GEMINI_API_KEY"))
    
    def get_embedding_model(self):
        return self.embedding_model

    def embed_texts(self, documents):
        
        texts = [doc.page_content for doc in documents]
        return self.embedding_model.embed_documents(texts)

chunkss_embeddding = ChunkEmbeddings()
