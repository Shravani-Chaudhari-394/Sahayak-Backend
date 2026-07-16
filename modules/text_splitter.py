
from langchain_text_splitters import RecursiveCharacterTextSplitter
from logger import logger



class TextSplitter:
    def split_documents(self, documents ):
        try :
            
            logger.info("spitting documents into chunks...🥳🍾")
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150,
            )
            chunks = splitter.split_documents(documents)
            
            logger.info(f"Created {len(chunks)} nos of chunks ")

            return chunks

        except Exception as e:
            logger.error(f"effor spitting document 😔")
            raise




