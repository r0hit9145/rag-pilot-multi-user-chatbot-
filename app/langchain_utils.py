# the RAG chain
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

CHROMA_PATH = "basic_code/chroma_db"   # adjust if needed
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_llm(model_name="llama-3.3-70b-versatile"):
    return ChatGroq(
        model=model_name,
        temperature=0,
        groq_api_key=os.environ.get("GROQ_API_KEY"),
    )


def get_vectorstore():
    embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function,
    )
    return vectorstore


def get_rag_chain(model_name="llama-3.3-70b-versatile"):
    llm = get_llm(model_name)
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful AI assistant. "
            "Use the following context to answer the user's question. "
            "If the answer is not in the context, say you do not know."
        ),
        ("system", "Context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain