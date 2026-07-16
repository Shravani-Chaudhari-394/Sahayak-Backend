import os
import time
from dotenv import load_dotenv
from logger import logger
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()


class VectorStore():

    # Small batches + a pause between them keeps us comfortably under
    # the free-tier embed_content rate limit (100 requests/minute).
    BATCH_SIZE = 25
    PAUSE_BETWEEN_BATCHES = 3        # seconds, paced even on success
    MAX_RETRIES_PER_BATCH = 5
    RATE_LIMIT_WAIT_SECONDS = 65     # a bit over Google's stated 53s cooldown

    def create_vector_store(self, chunks, scheme_id: str):
        try:
            logger.info(f"creating Embeddings for scheme '{scheme_id}'...")

            embedding = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001",
                google_api_key=os.environ["GOOGLE_API_KEY"],
            )

            persist_directory = os.path.join("chroma_db", scheme_id)
            logger.info(f"Creating ChromaDB Vector Store at {persist_directory}...")

            vector_db = Chroma(
                embedding_function=embedding,
                persist_directory=persist_directory,
            )

            total = len(chunks)
            total_batches = (total + self.BATCH_SIZE - 1) // self.BATCH_SIZE

            for i in range(total_batches):
                start = i * self.BATCH_SIZE
                batch = chunks[start:start + self.BATCH_SIZE]

                for attempt in range(1, self.MAX_RETRIES_PER_BATCH + 1):
                    try:
                        vector_db.add_documents(batch)
                        logger.info(
                            f"Embedded batch {i + 1}/{total_batches} "
                            f"({len(batch)} chunks) ✅"
                        )
                        break
                    except Exception as e:
                        is_rate_limit = "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e)
                        if is_rate_limit and attempt < self.MAX_RETRIES_PER_BATCH:
                            logger.info(
                                f"Rate limit hit on batch {i + 1}/{total_batches}. "
                                f"Waiting {self.RATE_LIMIT_WAIT_SECONDS}s "
                                f"(retry {attempt}/{self.MAX_RETRIES_PER_BATCH})..."
                            )
                            time.sleep(self.RATE_LIMIT_WAIT_SECONDS)
                        else:
                            raise

                time.sleep(self.PAUSE_BETWEEN_BATCHES)

            logger.info(f"Vector Store for '{scheme_id}' Created Successfully 🥹🥹")

            return vector_db

        except Exception as e:
            logger.error(f"Error creating vector store for '{scheme_id}': {e} 😖😖")
            raise
