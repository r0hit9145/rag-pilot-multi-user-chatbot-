#  /chat endpoint
from fastapi  import FastAPI, UploadFile, File, HTTPException
import uuid
from contextlib import asynccontextmanager
import os
import shutil
from app.schemas import QueryInput, QueryResponse, DeleteFileRequest, DocumentInfo, UploadResponse
from app.db_utils import create_application_log, get_chat_history, insert_application_log, create_document_store, insert_document, get_all_documents, delete_document_record
from app.chroma_utils import add_document_to_chroma, delete_document_from_chroma
from app.langchain_utils import get_rag_chain

app = FastAPI()

DOCS_PATH = "RAG_Docs"
os.makedirs(DOCS_PATH, exist_ok=True)

@asynccontextmanager
async def lifespan():
    try:
        create_application_log()
        create_document_store()
        print("Startup successfully!")
    except Exception as e:
        print(f"Startup error: {e}")
        raise

    yield # app runs here

app = FastAPI()


@app.get("/")
def root():
    return {"message": "RAG Chatbot API is running"}


@app.post("/chat", response_model=QueryResponse)
def chat(query: QueryInput):
    session_id = query.session_id or str(uuid.uuid4())
    chat_history = get_chat_history(session_id)
    rag_chain = get_rag_chain(query.model)

    result = rag_chain.invoke({
        "input": query.question,
        "chat_history": chat_history
    })

    answer = result["answer"] if isinstance(result, dict) else str(result)

    insert_application_log(
        session_id = session_id,
        user_query = query.question,
        gpt_response = answer,
        model  = query.model
    )

    return QueryResponse(
        answer=answer,
        session_id=session_id,
        model=query.model
    )

@app.post("/upload-doc", response_model=UploadResponse)
def upload_doc(file: UploadFile = File(...)):
    file_path = os.path.join(DOCS_PATH, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file_id = insert_document(file.filename)
    add_document_to_chroma(file_path, file_id)

    return UploadResponse(
        message="Document uploaded and indexed successfully",
        filename=file.filename
    )

@app.get("/list-docs", response_model=list[DocumentInfo])
def list_docs():
    rows = get_all_documents()
    return [{"id": row[0], "filename": row[1]} for row in rows]

@app.post("/delete-doc")
def delete_doc(request: DeleteFileRequest):
    delete_document_from_chroma(request.file_id)
    delete_document_record(request.file_id)
    return {"message": f"Document with id {request.file_id} deleted successfully"}