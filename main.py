from fastapi import UploadFile, File, Form, FastAPI, HTTPException
from pydantic import BaseModel
import os
import re
import shutil

from modules.document_loader import DocumentLoader
from modules.text_splitter import TextSplitter
from modules.vector_store import VectorStore
from modules.retriever import Retriever
from modules.rag_chain import RAGChain

from fastapi.middleware.cors import CORSMiddleware
from logger import logger


app = FastAPI()      

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "https://sahayak-frontend-phi.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# document loader  

loader = DocumentLoader()

# text splitter 

splitter = TextSplitter()

# creating vector Store 

vector_store = VectorStore()

# retriever

retriever_obj = Retriever()



chain = RAGChain().get_chain()

SCHEME_ID_PATTERN = re.compile(r"^[a-z0-9_]+$")


def validate_scheme_id(scheme_id: str) -> str:
    scheme_id = scheme_id.strip().lower()
    if not SCHEME_ID_PATTERN.match(scheme_id):
        raise HTTPException(
            status_code=400,
            detail="scheme_id must contain only lowercase letters, numbers, and underscores (e.g. 'pm_kisan')."
        )
    return scheme_id


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), scheme_id: str = Form(...)):

    scheme_id = validate_scheme_id(scheme_id)

    upload_dir = os.path.join("uploaded_files", scheme_id)

    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # document loding
    documents = loader.load_pdf(file_path)

    #creating chunks
    chunks = splitter.split_documents(documents)

    # creating vector Store
    vector_store.create_vector_store(chunks, scheme_id)



    return {
        "message": f"Vector Store Updated Successfully for '{scheme_id}'",
        "scheme_id": scheme_id,
        "pages": len(documents),
        "chunks": len(chunks)
    }


# # -----------------------------
# # Chat Endpoint
# # -----------------------------




class ChatRequest(BaseModel):
    question : str
    scheme_id : str

@app.post("/chat")

async def chat(request: ChatRequest):
    scheme_id = validate_scheme_id(request.scheme_id)

    try:
        retriever = retriever_obj.get_retriever(scheme_id)
        docs = retriever.invoke(request.question)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
            logger.error(f"Gemini embedding quota exceeded: {e}")
            raise HTTPException(
                status_code=429,
                detail="Daily embedding quota exceeded. Please try again after the quota resets (~24h), or upgrade your Gemini API plan."
            )
        raise

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    answer = chain.invoke({
        "context" : context,
         "question" : request.question
    })

    return{
        "question":request.question,
        "answer": answer
    }