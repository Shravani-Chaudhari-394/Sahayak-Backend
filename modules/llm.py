from langchain_groq import ChatGroq
from logger import logger
from dotenv import load_dotenv

load_dotenv()


class LLM:

    def get_llm(self):

        logger.info("Loading LLM...")

        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0
        )