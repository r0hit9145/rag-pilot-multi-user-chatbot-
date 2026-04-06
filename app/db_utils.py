# chat-history storage
# DATABASE CONNTECTIVITY with backend
import sqlite3

def get_db_connection():
    conn = sqlite3.connect("basic_code/rag_app.db")
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


def create_document_store():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        create table if not exists document_store(
            id integer primary key autoincrement,
            filename text not null,
            created_at timestamp default current_timestamp
    )""")

    conn.commit()
    conn.close()


def insert_document(filename):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        insert into document_store(filename)
        values(?)
""", (filename,)
)
    doc_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return doc_id


def get_all_documents():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, filename
        FROM document_store
        ORDER BY id ASC
    """)

    rows = cursor.fetchall()
    conn.close()
    return rows


def delete_document_record(file_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM document_store WHERE id = ?", (file_id,))
    conn.commit()
    conn.close()

