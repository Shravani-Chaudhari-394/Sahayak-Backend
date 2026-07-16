
from langchain_community.document_loaders import PyPDFLoader

from logger import logger


class DocumentLoader:
    def load_pdf(self, file_path : str):
        try:
            logger.info(f"loading pdf : {file_path}")
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Clean unpaired surrogate characters to prevent Hugging Face tokenizer crash
            for doc in documents:
                doc.page_content = doc.page_content.encode('utf-8', 'ignore').decode('utf-8')
                
            logger.info(f"loaded {len(documents)} pages 😎😎")

            return documents

        except Exception as e:
 
            logger.error(f"Error loading PDF : {e} 😭😭")

            raise

