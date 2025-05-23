from chunks.text_chunk import chunks
from pdf_load.pdf_loader import extract_data_fromPDf
from text_embedding.text_embeddings import chunkss_embeddding
from vector_store.embedding_store import embedding_store
from dotenv import load_dotenv
import os 

load_dotenv()


index_name = "clgdata"

current_dir = os.getcwd()
file_path = os.path.join(current_dir,"documents")


loaded_pdf  = extract_data_fromPDf.load_pdf(file=file_path)
texts_chunks = chunks.chunks(loaded_pdf)
embed = chunkss_embeddding.embed_texts(texts_chunks)

vectore_store = embedding_store.upload_embeddings_to_database(embeddings=embed,ind_name=index_name,text_chunk=texts_chunks)







            






