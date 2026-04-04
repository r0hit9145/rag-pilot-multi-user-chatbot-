# request/response models
from typing import Optional, Literal
from pydantic import BaseModel

class QueryInput(BaseModel):
    question: str
    session_id: Optional[str] = None
    model: Literal["llama-3.3-70b-versatile"] = "llama-3.3-70b-versatile"


class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: str


class DeleteFileRequest(BaseModel):
    file_id: int


class DocumentInfo(BaseModel):
    id: int 
    filename: str


class UploadResponse(BaseModel):
    message: str
    filename: str
