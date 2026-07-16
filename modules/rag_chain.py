from langchain_core.output_parsers import StrOutputParser

from modules.llm import LLM
from modules.prompt import Prompt


class RAGChain:

    def get_chain(self):

        # Load Prompt
        prompt = Prompt().get_prompt()

        # Load LLM
        llm = LLM().get_llm()

        # Create Chain
        chain = (
            prompt | llm | StrOutputParser()
        )

        return chain