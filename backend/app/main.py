from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .rag import answer_question


app = FastAPI(title="PWC RAG API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]


@app.get("/")
async def read_root():
    return {"message": "PWC RAG API is running"}


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    answer, citations = answer_question(request.question)
    return QueryResponse(answer=answer, citations=citations)
