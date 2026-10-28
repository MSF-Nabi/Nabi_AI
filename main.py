from datetime import datetime, timedelta, timezone
import os
import re
import logging

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import List, Optional
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

class DiaryItem(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)  # 숫자 PK(예: 15)도 "15"로 받음

    id: str = Field(pattern=r"\S")  # 호출 측 일기 PK, RUD에 그대로 사용
    date: str
    content: str = Field(min_length=1)

    @field_validator("date")
    @classmethod
    def normalize_date(cls, v: str) -> str:
        return datetime.strptime(v, "%Y-%m-%d").strftime("%Y-%m-%d")

class DiarysEntrys(BaseModel):
    userId: str = Field(pattern=r"\S")
    diarys: List[DiaryItem] = Field(min_length=1)

class DiaryEntry(BaseModel):
    userId: str
    content: str
    date: str

class UpdateDiaryEntry(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)  # /diarys와 같이 숫자 ID도 허용

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

    [네가 기억하고 있는 사용자 일기 (날짜: 내용)]
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

WEEKDAYS_KO = "월화수목금토일"


def get_kst_now() -> str:
    now = datetime.now(KST)
    return f"{now:%Y-%m-%d} ({WEEKDAYS_KO[now.weekday()]}요일)"


def build_system_message() -> str:
    today = get_kst_now()
    return f"""
너는 사용자의 아주 친한 친구야.
AI나 상담사처럼 말하지 말고 사람처럼 반말로 대화해.
정답을 주려 하지 말고 감정을 먼저 공감해.
일기 내용은 이미 알고 있는 기억처럼 자연스럽게 써.
모르는 건 아는 척하지 말고 자연스럽게 다시 물어봐.
오늘 날짜는 {today}야.

기억하는 일기에는 '2026-04-24: 내용'처럼 쓴 날짜가 붙어 있어.
오늘 날짜와 비교해서 지난 일은 지난 일로 말해. 예전 일기를 오늘이나 어제 일처럼 말하지 마.
사용자가 특정 날짜나 달에 뭐 했는지 물으면 그 날짜 일기 내용으로 구체적으로 답해.
그 날짜 일기가 없다고 적혀 있으면 지어내지 말고 기억나는 일기가 없다고 말해.

같은 의미의 문장이나 표현을 반복하지 마.
이전 대화에서 쓴 말투·표현·구조는 최대한 피해서 말해.
공감 표현은 매번 다른 방식으로 짧게 말해.

질문이 필요하지 않다면 공감만 하고 대화를 끝내도 돼.
""".strip()


FULL_DATE_RE = re.compile(r"(\d{4})\s*(?:년|[-./])\s*(\d{1,2})\s*(?:월|[-./])\s*(\d{1,2})")
MONTH_DAY_RE = re.compile(r"(?<!\d)(\d{1,2})\s*월\s*(\d{1,2})\s*일")
SLASH_DATE_RE = re.compile(r"(?<![\d/])(\d{1,2})/(\d{1,2})(?![\d/])")
YEAR_MONTH_RE = re.compile(r"(\d{4})\s*년\s*(\d{1,2})\s*월")
MONTH_RE = re.compile(r"(?<!\d)(\d{1,2})\s*월")
RELATIVE_DAYS = {"그저께": 2, "그제": 2, "어제": 1, "오늘": 0}


def extract_question_dates(question: str, today) -> tuple[set, set]:
    # 질문 속 날짜(YYYY-MM-DD)와 달(YYYY-MM). 연도가 없으면 오늘 기준 가장 최근 과거로 본다
    days, months = set(), set()

    def add_day(y, m, d):
        try:
            days.add(datetime(y, m, d).strftime("%Y-%m-%d"))
        except ValueError:
            pass

    def add_recent_day(m, d):
        year = today.year
        try:
            if datetime(year, m, d).date() > today:
                year -= 1
        except ValueError:
            return
        add_day(year, m, d)

    def full(match):
        add_day(*map(int, match.groups()))
        return " "

    def month_day(match):
        add_recent_day(*map(int, match.groups()))
        return " "

    def year_month(match):
        y, m = map(int, match.groups())
        if 1 <= m <= 12:
            months.add(f"{y:04d}-{m:02d}")
        return " "

    def month(match):
        m = int(match.group(1))
        if 1 <= m <= 12:
            y = today.year if m <= today.month else today.year - 1
            months.add(f"{y:04d}-{m:02d}")
        return " "

    text = FULL_DATE_RE.sub(full, question)
    text = MONTH_DAY_RE.sub(month_day, text)
    text = SLASH_DATE_RE.sub(month_day, text)
    text = YEAR_MONTH_RE.sub(year_month, text)
    MONTH_RE.sub(month, text)
    for word, offset in RELATIVE_DAYS.items():
        if word in question:
            days.add((today - timedelta(days=offset)).strftime("%Y-%m-%d"))
    return days, months


def find_diaries_on_dates(user_id: str, days: set, months: set, limit: int = 10) -> list:
    # 해당 날짜·달의 일기를 날짜순으로 (id, date, document)
    if not days and not months:
        return []
    found = collection.get(where={"userId": user_id})
    hits = [
        (diary_id, (meta or {}).get("date", ""), doc)
        for diary_id, meta, doc in zip(found["ids"], found["metadatas"], found["documents"])
        if (meta or {}).get("date") in days or (meta or {}).get("date", "")[:7] in months
    ]
    return sorted(hits, key=lambda hit: hit[1])[:limit]


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != openai_api_key:
        raise HTTPException(status_code=403, detail="Invalid authentication")
    return credentials.credentials


def ensure_owned_diary(diary_id: str, user_id: Optional[str]) -> None:
    # 없는 ID는 404, userId가 주어졌는데 주인이 다르면 403
    found = collection.get(ids=[diary_id])
    if not found["ids"]:
        raise HTTPException(status_code=404, detail="Diary entry not found")
    if user_id is not None and (found["metadatas"][0] or {}).get("userId") != user_id:
        raise HTTPException(status_code=403, detail="Diary entry belongs to another user")


# --- Routes ---

@app.get("/")
def read_root():
    return {"test": "success"}


@app.get("/health")
def health_check():
    checks = {}

    # ChromaDB 연결 확인
    try:
        chroma_client.heartbeat()
        checks["chromadb"] = "ok"
    except Exception as e:
        logger.warning(f"ChromaDB health check failed: {e}")
        checks["chromadb"] = "error"

    # OpenAI API 키 설정 확인
    checks["openai_key"] = "ok" if openai_api_key else "missing"

    is_healthy = all(v == "ok" for v in checks.values())
    status_code = 200 if is_healthy else 503

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ok" if is_healthy else "degraded",
            "timestamp": datetime.now(KST).isoformat(),
            "checks": checks,
        },
    )


@app.post("/diarys")
async def add_diarys(entry: DiarysEntrys, _: str = Depends(verify_token)):
    # 사용자의 일기 전체를 요청 목록으로 교체. ids는 요청 순서 그대로, deleted는 빠져서 지워진 일기
    ids = [item.id for item in entry.diarys]
    if len(set(ids)) != len(ids):
        raise HTTPException(status_code=422, detail="Duplicate diary ids in request")
    if len(ids) > chroma_client.get_max_batch_size():
        raise HTTPException(status_code=413, detail=f"Too many diaries in one request (max {chroma_client.get_max_batch_size()})")

    try:
        logger.info(f"Replacing diary entries for user: {entry.userId} ({len(ids)} entries)")
        documents = [item.content for item in entry.diarys]
        # 임베딩은 이벤트 루프를 막지 않게 비동기로 먼저 계산 (실패 시 기존 일기는 그대로)
        embeddings = await embedding_function.openai_embeddings.aembed_documents(documents)

        # 아래는 await 없이 한 번에 실행: 소유권 확인 → upsert → 빠진 일기 삭제
        taken = collection.get(ids=ids)
        foreign = [
            diary_id for diary_id, meta in zip(taken["ids"], taken["metadatas"])
            if (meta or {}).get("userId") != entry.userId
        ]
        if foreign:
            raise HTTPException(status_code=409, detail=f"Diary ids already used by another user: {foreign}")

        existing = collection.get(where={"userId": entry.userId})
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=[{"userId": entry.userId, "date": item.date} for item in entry.diarys],
        )
        new_ids = set(ids)
        deleted = [diary_id for diary_id in existing["ids"] if diary_id not in new_ids]
        if deleted:
            collection.delete(ids=deleted)

        logger.info(f"Diary entries successfully saved ({len(deleted)} removed)")
        return {"message": "Diary entries have been successfully embedded and saved.", "ids": ids, "deleted": deleted}
    except HTTPException:
        raise
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
        ensure_owned_diary(entry.id, entry.userId)
        collection.update(
            ids=[entry.id],
            documents=[entry.content],
            metadatas=[{"userId": entry.userId, "date": entry.date}],
        )
        logger.info("Diary entry successfully updated")
        return {"id": entry.id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating diary entry: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/diary")
async def delete_diary(id: str, userId: Optional[str] = None, _: str = Depends(verify_token)):
    try:
        logger.info(f"Deleting diary entry: {id}")
        ensure_owned_diary(id, userId)
        collection.delete(ids=[id])
        logger.info("Diary entry successfully deleted")
        return {"message": "Diary entry has been successfully deleted."}
    except HTTPException:
        raise
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

        user_id = str(query.userId)
        # 질문에 날짜가 있으면 그 날짜의 일기를 먼저 넣고, 그 뒤에 의미가 비슷한 일기를 붙인다
        days, months = extract_question_dates(query.question, datetime.now(KST).date())
        diaries = find_diaries_on_dates(user_id, days, months)

        results = collection.query(
            query_texts=[query.question],
            n_results=3,
            where={"userId": user_id},
        )
        seen = {diary_id for diary_id, _, _ in diaries}
        for diary_id, meta, doc in zip(results["ids"][0], results["metadatas"][0], results["documents"][0]):
            if diary_id not in seen:
                diaries.append((diary_id, (meta or {}).get("date", "날짜 모름"), doc))
                seen.add(diary_id)

        context_lines = [f"{date}: {doc}" for _, date, doc in diaries]
        missing_days = sorted(days - {date for _, date, _ in diaries})
        context_lines += [f"({day} 날짜의 일기는 없음)" for day in missing_days]
        logger.info(f"Dates in question: {sorted(days)} months: {sorted(months)}")
        logger.info(f"Documents found in DB: {context_lines}")
        retrieved_context = "\n\n".join(context_lines)

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
