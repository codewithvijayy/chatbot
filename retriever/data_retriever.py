import re
import random
from langchain_pinecone import PineconeVectorStore
from text_embedding.text_embeddings import chunkss_embeddding
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

class Retrieve():
    def __init__(self):
        
        self.dev_keywords = [
            "who developed", "who made", "who created", "who built",
            "developer of", "made this", "built this", "creator of", "created by",
            "designed by", "developed by", "made by","developed"
        ]

    def _normalize_text(self, text):
        # Lowercase and remove punctuation for better matching
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
        return text

    def data_retrieve(self, query):
        normalized_query = self._normalize_text(query)
        
        # Check if query contains any developer-related phrase
        if any(keyword in normalized_query for keyword in self.dev_keywords):
            # Return developer info immediately
            return "This website and assistant were developed by Vijay. You can reach out to him on Instagram @1251_vijay."
        
        # Else, proceed with regular retrieval from vector store
        vector_store = PineconeVectorStore.from_existing_index(
            index_name="clgdata",
            embedding=chunkss_embeddding.get_embedding_model()
        )

        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.3}
        )
        docs = retriever.invoke(query)
        
        if not docs:
            return "This information is not trained, Vijay.."

        context = "\n\n".join(doc.page_content for doc in docs)

        fallback_messages = [
            "Sorry, this information is not available yet. For further details, please reach out to Vijay on Instagram: @1251_vijay.",
            "I don't have that data right now. You can contact Vijay directly on Instagram at @1251_vijay for assistance.",
            "This information is currently not part of my knowledge base. Feel free to message Vijay on Instagram @1251_vijay for more info.",
            "I'm unable to find this information. Please connect with Vijay via Instagram @1251_vijay to get help.",
            "That data isn't available here. You can contact Vijay on Instagram: @1251_vijay for more details.",
            "I’m sorry, I don’t have that info. For inquiries, please contact Vijay on Instagram @1251_vijay.",
            "Unfortunately, I don’t have this information. You may reach out to Vijay through Instagram @1251_vijay.",
            "This information is not currently in the system. For further assistance, please contact Vijay on Instagram: @1251_vijay."
        ]

        selected_fallback = random.choice(fallback_messages)

        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

        prompt_template = ChatPromptTemplate.from_template(
            """
            You are an assistant answering questions based on college data.

            Context:
            {context}

            Question:
            {question}

            If the context does not contain relevant information, reply with:
            "{selected_fallback}"
            """
        )

        chain = prompt_template | model | StrOutputParser()

        result = chain.invoke({
            "context": context,
            "question": query,
            "selected_fallback": selected_fallback
        })

        return result

# Example usage:
retrieve_data = Retrieve()