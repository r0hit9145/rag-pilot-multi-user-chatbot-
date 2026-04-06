import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma


DOCS_PATH = "RAG_Docs"
CHROMA_PATH = "basic_code/chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def get_embedding_function():
    return SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

def get_vectorstore():
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

def load_single_document(file_path: str):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".xlsx"):
        loader = UnstructuredExcelLoader(file_path)
    else:
        raise ValueError("Unsupported file type")
    return loader.load()


def add_document_to_chroma(file_path: str, file_id: int):
    documents = load_single_document(file_path)
    splitter = get_text_splitter()
    splits = splitter.split_documents(documents)

    for split in splits:
        split.metadata["file_id"] = file_id
        split.metadata["source"] = os.path.basename(file_path)

    vectorstore = get_vectorstore()
    vectorstore.add_documents(splits)
    vectorstore.persist()


def delete_document_from_chroma(file_id: int):
    vectorstore = get_vectorstore()
    data = vectorstore.get(where={"file_id": file_id})
    print("Delete check:", data)

    ids = data.get("ids", [])
    if ids:
        vectorstore.delete(ids=ids)
        vectorstore.persist()
        print("deleted ids", ids)
    else:
        print("no ids or mismatch for the file id:", file_id)



