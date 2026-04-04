# import langchain
# print(langchain.__version__)

from dotenv import load_dotenv
import os
load_dotenv()

# Call llm without parser
from langchain_groq import ChatGroq

llm = ChatGroq(
    model = "llama-3.3-70b-versatile", #free and fast
    temperature = 0,
    groq_api_key= os.environ.get("GROQ_API_KEY"),
)


# Parsing output
from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()

# Simple chain
chain = llm | output_parser
result_chain = chain.invoke("Tell me a joke?")




# Structured output
from typing import List
from pydantic import BaseModel, Field

class MobileReview(BaseModel):
    phone_model: str = Field(description = "Name and model of the phone")
    rating: float = Field(description= "Overall rating out of 5")
    pros: List[str] = Field(description = "List of positive aspects")
    cons: List[str] = Field(description = "List of negative aspects")
    summary: str = Field(description = "Brief summary of the review topic")


review_text = """
The Oppo Reno 11 5G comes with a sleek and premium design that feels comfortable in hand. It has a bright and colorful display, making it great for watching videos and browsing. The performance is smooth for everyday use like social media, apps, and moderate gaming. The camera quality stands out, especially for portraits and selfies, giving sharp and clear images. Battery life is reliable and can easily last a full day on normal usage. It also supports fast charging, which is very convenient. Overall rating for this phone is 4.2 out of 5. The main pros are its stylish design, good camera, fast charging, and decent performance. The cons include a slightly higher price, average performance for heavy gaming, and some pre-installed apps. In summary, it is a good choice for users who want a balanced smartphone with strong camera features and modern design.
"""

structured_llm = llm.with_structured_output(MobileReview)
output = structured_llm.invoke(review_text)



#Prompt Template
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("tell me a joke in short about {topic}")
result_prompt = prompt.invoke({"topic": "doremoon"})
# print(result_prompt)

# compose to chain
chain = prompt | llm | output_parser

#chain result
result_chainned = chain.invoke({"topic": "programming"})

# LLM Messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

system_message = SystemMessage(content = "You are a helpful assistant that tells jokes.")
human_message  = HumanMessage(content= "Tell me about programming")

result_message = llm.invoke([system_message, human_message])

# template
template = ChatPromptTemplate([
    ("system", "You are a helpful assistant that tells jokes"),
    ("human", "Tell me about {topic}")
])

prompt_value = template.invoke({
    "topic": "programming"
})

result = llm.invoke(prompt_value)
# print(result)


# RAG-Based -> load fictional data and split
from langchain_community.document_loaders import UnstructuredExcelLoader, PyPDFLoader #JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
    length_function = len
)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"  # lightweight & fast
)

# print(embeddings)

xlsx_loader = UnstructuredExcelLoader(
    "/home/rohit/projects/rag-chatbot/RAG_Docs/05_budget_forecast.xlsx",
    mode="elements"  # loads each cell/row as separate element
)
documents = xlsx_loader.load()
splits = text_splitter.split_documents(documents)
# print((documents))
# print(splits[0].metadata)
# print(splits[1])


# Function to load documents from a folder
def load_documents(folder_path: str):
    # print("Files in folder:", os.listdir(folder_path))
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        loader = None
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif filename.endswith('.xlsx'):
            loader = UnstructuredExcelLoader(file_path)
        
        # elif filename.endswith('.json'):
        #     loader = JSONLoader(
        #         file_path = file_path, 
        #         jq_schema = ".documents[] | .content")

        else:
            print(f"Unsupported file type: {filename}")
            continue 

        
        # print(f"{filename} -> {len(documents)} documents")
        documents.extend(loader.load())

    return documents


# Load documents from folder
folder_path = "/home/rohit/projects/rag-chatbot/RAG_Docs"
documents1 = load_documents(folder_path)

# print(f" loaded {len(documents1)} documents from the folder")
splits = text_splitter.split_documents(documents1)
# print(f"split the documents into {len(splits)} chunks.")

# 🧠 Real-world analogy

# Think of it like this:

# Simple embeddings → keyword matching
# SentenceTransformers → understanding meaning


# Embedding Documents in simple basic
document_embeddings = embeddings.embed_documents([split.page_content for split in splits])
# print(f"Created embeddings for {len(document_embeddings)} document chunks.")

# print(document_embeddings[0])

# Embedding based on more accurate using sentence_transformers_embeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
embedding_function = SentenceTransformerEmbeddings(model_name = "all-MiniLM-L6-v2")
docs_embeddings = embedding_function.embed_documents([split.page_content for split in splits])


# Create and persist Chroma vector store -> ChromaDB
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# create DB
db = Chroma.from_documents(
    documents1,
    embedding_function,
    persist_directory="./chroma_db"
)

query = "GTA game is what?"

# results = db.similarity_search(query, k=2)

# for i, result in enumerate(results, 1):
#     print(f"Result {i}")
#     print(f"Source: {result.metadata.get('source', 'Unknown')}")
#     print(f"content: {result.page_content}")
#     print()

# ✅ 1. Use score threshold (BEST FIX)
# 🧠 Final intuition (SUPER SIMPLE)

# Think of threshold like:

# 🎓 Exam cutoff
# 90% cutoff → only toppers pass
# 50% cutoff → more students pass

results = db.similarity_search_with_relevance_scores(query, k=2, score_threshold=0.3)


# if not results:
    # print("not relevant search...........")
# for doc, rel in results:
    # print(rel, "relevance...")
    # print(doc.page_content[:200])


# as same job to do -> retriever 
retriever = db.as_retriever(search_kwargs= {"k": 2})
result = retriever.invoke("What is warehouse cooling incident report?")
# print(result)


# again to give in prompt
template = """
Answer the question based only on the following content:
{context}
Question: {question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

from langchain_core.runnables import RunnablePassthrough

rag_chain=(
    {"context": retriever, "question": RunnablePassthrough()} | prompt
)

result_runnable = rag_chain.invoke("What is warehouse cooling incident report?")


# seperate by "\n\n"
def doc2str(docs):
    return "\n".join(doc.page_content for doc in docs)


rag_chain=(
    {"context": retriever | doc2str, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

question = "what is Aster App Launch Meeting Minutes?"
response = rag_chain.invoke(question)
# print(end="\n\n")
# print(response)


# Conversational RAG (continue or remember previous chat history to ans / Handling follow up Ques.)
from langchain_core.messages import HumanMessage, AIMessage
chat_history = []
chat_history.extend([
    HumanMessage(content= question),
    AIMessage(content= response)
])

# print(chatHistory)

from langchain_core.prompts import MessagesPlaceholder

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_system_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

contextualize_chain = contextualize_q_system_prompt | llm | StrOutputParser()

contextualize_result = contextualize_chain.invoke({"input": "decision notes of it?", "chat_history": chat_history})
# print(contextualize_result)


# contextualize chain or history remember
from langchain_classic.chains import create_history_aware_retriever

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_system_prompt
)

result_history_aware_retriever = history_aware_retriever.invoke({"input": "decision notes of it?", "chat_history": chat_history})
# print(result_history_aware_retriever)

# now we're going to question or answering chain because it is capable enough to understand or based on previous chat history

# Purpose: Combines retrieved docs + prompt to generate answers
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the following context to answer the user's question."),
    ("system", "Context: {context}"),
    MessagesPlaceholder(variable_name = "chat_history"),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# actually rag chain or final RAG CHAIN
result = rag_chain.invoke({"input": "decision notes of it?", "chat_history": chat_history})
# print(result)



# DATABASE CONNTECTIVITY with backend
import sqlite3

def get_db_connection():
    conn = sqlite3.connect("rag_app.db")
    return conn


def create_application_log():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        create table if not exists application_log(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id text not null,
            user_query text not null,
            gpt_response text not null,
            model text,
            create_at timestamp default current_timestamp
        )
    """)
    conn.commit()
    conn.close()


# DB functions
from langchain_core.messages import HumanMessage, AIMessage

def insert_application_log(session_id, user_query, gpt_response, model):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        insert into application_log(
            session_id, user_query, gpt_response, model
        )values
        (?, ?, ?, ?)
""", (session_id, user_query, gpt_response, model))
    conn.commit()
    conn.close()


def get_chat_history(session_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        select user_query, gpt_response 
        from application_log
        where session_id = ?
        order by id ASC
""",(session_id,))
    rows = cursor.fetchall()
    conn.close()

    history = []
    for user_query, gpt_response in rows:
        history.append(HumanMessage(content = user_query))
        history.append(AIMessage(content = gpt_response))
    
    return history


import uuid

create_application_log()

session_id = str(uuid.uuid4()) # for first conversation only
question = "what is Aster App Launch Meeting Minutes?"

chat_history = get_chat_history(session_id)

result = rag_chain.invoke({
    "input": question,
    "chat_history": chat_history
})

answer = result["answer"] if isinstance(result, dict) else result

insert_application_log(
    session_id = session_id,
    user_query = question,
    gpt_response = answer,
    model = "llama-3.3-70b-versatile"
)

print("Human Ask: ", question)
print("SESSION: ", session_id)
print("AI ANSWER: ", answer)



# follow up question - answering

followup_question = "decision note of it?"

chat_history = get_chat_history(session_id)

result2 = rag_chain.invoke({
    "input": followup_question,
    "chat_history": chat_history
})

answer2 = result2["answer"] if isinstance(result2, dict) else result2


insert_application_log(
    session_id = session_id,
    user_query = followup_question,
    gpt_response = answer2,
    model = "llama-3.3-70b-versatile",
)

print("SESSION ID: ", session_id)
print("Human Ask: ", followup_question)
print("AI ANSWER 2: ", answer2)



# next move this is all about convert into API (fastAPI ) fro the code, then connect or pipeline with frontend 