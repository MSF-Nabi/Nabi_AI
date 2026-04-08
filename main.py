from datetime import datetime, timedelta, timezone
import os
import re
import logging

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel
from typing import List
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

openai_api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()
KST = timezone(timedelta(hours=9))


# --- Models ---

class DiarysEntrys(BaseModel):
    userId: str
    diarys: str

class DiaryEntry(BaseModel):
    userId: str
    content: str
    date: str

class UpdateDiaryEntry(BaseModel):
    id: str
    userId: str
    content: str
    date: str

class Query(BaseModel):
    userId: str
    question: str
    chatHistory: List[str]


# --- ChromaDB ---

class OpenAIEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: str):
        self.openai_embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    def __call__(self, input: Documents) -> Embeddings:
        return self.openai_embeddings.embed_documents(input)


try:
    logger.info("Initializing Chroma database")
    chroma_client = chromadb.PersistentClient(path="db")
    embedding_function = OpenAIEmbeddingFunction(openai_api_key)
    collection = chroma_client.get_or_create_collection(
        name="diary_entries",
        embedding_function=embedding_function,
    )
    logger.info("Chroma database initialized successfully")
except Exception as e:
    logger.error(f"Error during initialization: {e}", exc_info=True)
    raise


# --- LLM ---

chat_model = ChatOpenAI(model="gpt-5-nano", api_key=openai_api_key)

PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "previous_log", "new_question"],
    template="""
    지금 너는 사용자와 대화 중이야.

    [네가 기억하고 있는 사용자 이야기]
    {context}

    [지금까지의 대화 흐름]
    {previous_log}

    [사용자가 방금 한 말]
    {new_question}

    위 상황에서,
    사용자의 감정을 먼저 공감하고
    친한 친구처럼 자연스럽게 답해줘.
    """,
)


# --- Helpers ---

def get_kst_now() -> str:
    return datetime.now(KST).strftime('%B %d, %Y')


def build_system_message() -> str:
    today = get_kst_now()
    return f"""
너는 사용자의 아주 친한 친구야.
AI나 상담사처럼 말하지 말고 사람처럼 반말로 대화해.
정답을 주려 하지 말고 감정을 먼저 공감해.
일기 내용은 이미 알고 있는 기억처럼 자연스럽게 써.
모르는 건 아는 척하지 말고 자연스럽게 다시 물어봐.
오늘 날짜는 {today}야.

같은 의미의 문장이나 표현을 반복하지 마.
이전 대화에서 쓴 말투·표현·구조는 최대한 피해서 말해.
공감 표현은 매번 다른 방식으로 짧게 말해.

질문이 필요하지 않다면 공감만 하고 대화를 끝내도 돼.
""".strip()


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != openai_api_key:
        raise HTTPException(status_code=403, detail="Invalid authentication")
    return credentials.credentials


# --- Routes ---

@app.get("/")
def read_root():
    return {"test": "success"}


@app.post("/diarys")
async def add_diarys(entry: DiarysEntrys, _: str = Depends(verify_token)):
    try:
        logger.info(f"Adding diary entries for user: {entry.userId}")
        existing = collection.get(where={"userId": entry.userId})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])

        diary_entries = re.findall(r"(\d{4}-\d{2}-\d{2}): ([^\n]+)", entry.diarys)
        for date, content in diary_entries:
            collection.add(
                documents=[content],
                metadatas=[{"userId": entry.userId, "date": date}],
                ids=[f"{entry.userId}_{date}"],
            )
        logger.info("Diary entries successfully added")
        return {"message": "Diary entries have been successfully embedded and saved."}
    except Exception as e:
        logger.error(f"Error adding diary entries: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/diary")
async def add_diary(entry: DiaryEntry, _: str = Depends(verify_token)):
    try:
        logger.info(f"Adding diary entry for user: {entry.userId}")
        entry_id = f"{entry.userId}_{entry.date}"
        collection.add(
            documents=[entry.content],
            metadatas=[{"userId": entry.userId, "date": entry.date}],
            ids=[entry_id],
        )
        logger.info("Diary entry successfully added")
        return {"id": entry_id}
    except Exception as e:
        logger.error(f"Error adding diary entry: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/diary")
async def update_diary(entry: UpdateDiaryEntry, _: str = Depends(verify_token)):
    try:
        logger.info(f"Updating diary entry for user: {entry.userId}")
        collection.update(
            ids=[entry.id],
            documents=[entry.content],
            metadatas=[{"userId": entry.userId, "date": entry.date}],
        )
        logger.info("Diary entry successfully updated")
        return {"id": entry.id}
    except Exception as e:
        logger.error(f"Error updating diary entry: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/diary")
async def delete_diary(id: str, _: str = Depends(verify_token)):
    try:
        logger.info(f"Deleting diary entry: {id}")
        collection.delete(ids=[id])
        logger.info("Diary entry successfully deleted")
        return {"message": "Diary entry has been successfully deleted."}
    except Exception as e:
        logger.error(f"Error deleting diary entry: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/diaries")
async def delete_diarys(id: str, _: str = Depends(verify_token)):
    try:
        logger.info(f"Deleting all diary entries for user: {id}")
        collection.delete(where={"userId": id})
        logger.info("All diary entries successfully deleted")
        return {"message": "All Diary entries have been successfully deleted."}
    except Exception as e:
        logger.error(f"Error deleting diary entries: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/diary")
async def get_diary(id: str, _: str = Depends(verify_token)):
    try:
        diary_entry = collection.get(ids=[id])
        if not diary_entry["documents"]:
            raise HTTPException(status_code=404, detail="Diary entry not found")
        return diary_entry
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching diary entry: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query_api(query: Query, _: str = Depends(verify_token)):
    try:
        logger.info(f"Received query for user: {query.userId} | Question: {query.question}")

        history = query.chatHistory + [f"user: {query.question}"]

        results = collection.query(
            query_texts=[query.question],
            n_results=3,
            where={"userId": str(query.userId)},
        )
        logger.info(f"Documents found in DB: {results.get('documents')}")

        docs = ((results or {}).get("documents") or [[]])[0]
        retrieved_context = "\n\n".join(
            doc if isinstance(doc, str)
            else f"{(doc.get('metadata') or {}).get('date', 'unknown date')}: {doc.get('document', '')}"
            for doc in docs
        )

        prompt = PROMPT_TEMPLATE.format(
            context=retrieved_context.strip(),
            previous_log="\n".join(history).strip(),
            new_question=query.question.strip(),
        )
        messages = [
            SystemMessage(content=build_system_message()),
            HumanMessage(content=prompt),
        ]

        response = await chat_model.ainvoke(messages)
        answer = response.content

        logger.info(f"AI Response: {answer}")
        logger.info("Query processed successfully")

        return {"message": answer}
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
