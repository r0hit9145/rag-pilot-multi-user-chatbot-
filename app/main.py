#  /chat endpoint
from fastapi  import FastAPI
import uuid

from app.schemas import QueryInput, QueryResponse
from app.db_utils import create_application_log, get_chat_history, insert_application_log
from app.langchain_utils import get_rag_chain

app = FastAPI()

create_application_log()

@app.post("/chat", response_model=QueryResponse)
def chat(query: QueryInput):
    session_id = query.session_id or str(uuid.uudi4())
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