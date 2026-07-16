import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from logger import logger

load_dotenv()


class Retriever:

    _retrievers = {}

    def get_retriever(self, scheme_id: str):
        try:
            if scheme_id in Retriever._retrievers:
                logger.info(f"Using Existing Retriever for '{scheme_id}'...")
                return Retriever._retrievers[scheme_id]

            persist_directory = os.path.join("chroma_db", scheme_id)
            if not os.path.isdir(persist_directory):
                raise ValueError(
                    f"No vector store found for scheme '{scheme_id}'. "
                    f"Upload PDFs for this scheme first."
                )

            logger.info("Loading Embedding Model...")

            embedding = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001",
                google_api_key=os.environ["GOOGLE_API_KEY"]
            )

            logger.info(f"Loading ChromaDB for '{scheme_id}'...")

            vector_db = Chroma(
                persist_directory=persist_directory,
                embedding_function=embedding
            )

            logger.info("Creating Retriever...")

            Retriever._retrievers[scheme_id] = vector_db.as_retriever(
                search_kwargs={"k": 3}
            )

            logger.info(f"Retriever for '{scheme_id}' Created Successfully")

            return Retriever._retrievers[scheme_id]

        except Exception as e:
            logger.error(f"Retriever Error: {e}")
            raise