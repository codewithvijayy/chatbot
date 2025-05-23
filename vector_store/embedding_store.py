from pinecone import Pinecone ,ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import os


class EmbeddingStore():


    def upload_embeddings_to_database(self,ind_name,text_chunk,embeddings,batch_size = 50):
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        try:
            if ind_name not in pc.list_indexes():
                pc.create_index(
                name=ind_name,
                dimension=768, 
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                ) 
                )
        except Exception as e:
            if "ALREADY_SOURCE" in str(e):
                print(f"index_already created {ind_name} contineous...")
            else:
                raise
        index = pc.Index(ind_name)
        for i in range(0, len(text_chunk), batch_size):
            batch_docs = text_chunk[i:i+batch_size]
            batch_embeds = embeddings[i:i+batch_size]
            vectors = [
                (f"id_{i + idx}", emb, {"text": batch_docs[idx].page_content})
                for idx, emb in enumerate(batch_embeds)
            ]

            index.upsert(vectors=vectors)
            print(f"Uploaded batch {i // batch_size + 1}")
embedding_store = EmbeddingStore()

