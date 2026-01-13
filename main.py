from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from typing import List
import os
import re
from dotenv import load_dotenv
import logging
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from langchain_openai import OpenAIEmbeddings

app = FastAPI()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DiarysEntrys(BaseModel):
    userId: str
    diarys: str


class UpdateDiaryEntry(BaseModel):
    id: str
    userId: str
    content: str
    date: str


class DiaryEntry(BaseModel):
    userId: str
    content: str
    date: str


class Query(BaseModel):
    userId: str
    question: str
    chatHistory: List[str]


load_dotenv()
security = HTTPBearer()
openai_api_key = os.getenv("OPENAI_API_KEY")


class OpenAIEmbeddingFunction(EmbeddingFunction):
    def __init__(self, openai_api_key):
        self.openai_embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    def __call__(self, input: Documents) -> Embeddings:
        return self.openai_embeddings.embed_documents(input)


# ChromaDB 초기화
try:
    logger.info("Initializing Chroma database")
    client = chromadb.PersistentClient(path='db')
    embedding_function = OpenAIEmbeddingFunction(openai_api_key)
    collection = client.get_or_create_collection(
        name="diary_entries",
        embedding_function=embedding_function
    )
    logger.info("Chroma database initialized successfully")
except Exception as e:
    logger.error(f"Error during initialization: {str(e)}", exc_info=True)
    raise


def get_kst_now():
    utc_now = datetime.now(timezone.utc)
    kst_now = utc_now + timedelta(hours=9)
    return kst_now.strftime('%B %d, %Y')


def create_prompt(conversation: List[str], new_question: str, retrieved_context: str) -> dict:
    today = get_kst_now()

    system_message = f"""
    너는 사용자의 아주 친한 친구야.
    AI나 상담사처럼 말하지 말고 사람처럼 반말로 대화해.
    정답을 주려 하지 말고 감정을 먼저 공감해.
    질문은 감정을 이해하는 데 필요할 때만 한 번만 해.
    모르는 건 아는 척하지 말고 자연스럽게 다시 물어봐.
    일기 내용은 이미 알고 있는 기억처럼 자연스럽게 써.
    질문이 감정을 더 깊이 이해하는 데 도움이 될 때만 질문해.
    질문이 필요 없다면 공감만 해.
    오늘 날짜는 {today}야.
    """

    previous_log = "\n".join(conversation)

    return {
        "system_message": system_message.strip(),
        "context": retrieved_context.strip(),
        "new_question": new_question.strip(),
        "previous_log": previous_log.strip()
    }



def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials or credentials.scheme != "Bearer" or credentials.credentials != openai_api_key:
        raise HTTPException(status_code=403, detail="Invalid authentication")
    return credentials.credentials


@app.post("/diarys")
async def add_diarys(entry: DiarysEntrys, credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        logger.info(f"Adding diary entries for user: {entry.userId}")
        existing_entries = collection.get(where={"userId": entry.userId})
        if existing_entries["ids"]:
            collection.delete(ids=existing_entries["ids"])

        diary_entries = re.findall(r"(\d{4}-\d{2}-\d{2}): ([^\n]+)", entry.diarys)
        for date, content in diary_entries:
            collection.add(
                documents=[content],
                metadatas=[{"userId": entry.userId, "date": date}],
                ids=[f"{entry.userId}_{date}"]
            )
        logger.info("Diary entries successfully added")
        return {"message": "Diary entries have been successfully embedded and saved."}
    except Exception as e:
        logger.error(f"Error adding diary entries: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/diary")
async def add_diary(entry: DiaryEntry, credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        logger.info(f"Adding diary entry for user: {entry.userId}")
        collection.add(
            documents=[entry.content],
            metadatas=[{"userId": entry.userId, "date": entry.date}],
            ids=[f"{entry.userId}_{entry.date}"]
        )
        logger.info("Diary entry successfully added")
        return {"id": f"{entry.userId}_{entry.date}"}
    except Exception as e:
        logger.error(f"Error adding diary entry: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/diary")
async def update_diary(entry: UpdateDiaryEntry, credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        logger.info(f"Updating diary entry for user: {entry.userId}")
        collection.update(
            ids=[entry.id],
            documents=[entry.content],
            metadatas=[{"userId": entry.userId, "date": entry.date}]
        )
        logger.info("Diary entry successfully updated")
        return {"id": entry.id}
    except Exception as e:
        logger.error(f"Error updating diary entry: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/diary")
async def delete_diary(id: str, credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        logger.info(f"Deleting diary entry: {id}")
        collection.delete(ids=[id])
        logger.info("Diary entry successfully deleted")
        return {"message": "Diary entry has been successfully deleted."}
    except Exception as e:
        logger.error(f"Error deleting diary entry: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
@app.delete("/diaries")
async def delete_diarys(id: str, credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        logger.info(f"Deleting all diarys: {id}")
        collection.delete(
            where={"userId": id},
        )
        logger.info("Diary entry successfully deleted")
        return {"message": "Diary entry has been successfully deleted."}
    except Exception as e:
        logger.error(f"Error deleting diary entry: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/diary")
async def get_diary(id: str, credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        diary_entry = collection.get(ids=[id])
        if not diary_entry["documents"]:
            raise HTTPException(status_code=404, detail="Diary entry not found")
        return diary_entry
    except Exception as e:
        logger.error(f"Error fetching diary entry: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query_api(query: Query, credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        logger.info(f"Received query for user: {query.userId}")
        user_conversation = query.chatHistory
        user_conversation.append(f"user: {query.question}")

        results = collection.query(
            query_texts=[query.question],
            n_results=3,
            where={"userId": str(query.userId)}
        )
        print(results)
        logger.info(f"Documents in results: {results.get('documents')}")

        # 안전 처리: 검색 결과가 비었을 때
        docs_batches = (results or {}).get("documents") or []
        docs = docs_batches[0] if docs_batches else []
        retrieved_context = "\n\n".join(
            doc if isinstance(doc, str)
            else f"{(doc.get('metadata') or {}).get('date', 'unknown date')}: {doc.get('document', '')}"
            for doc in docs
        )

        prompt_dict = create_prompt(user_conversation, query.question, retrieved_context)
        prompt_template = PromptTemplate(
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
            """
            )
        prompt = prompt_template.format(**prompt_dict)
        logger.info(f"prompt: {prompt}")

        # LangChain OpenAI 통합: model 인자 사용
        from langchain_openai import ChatOpenAI
        chat_model = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key, temperature=0)

        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content=prompt_dict["system_message"]),
            HumanMessage(content=prompt),
        ]

        # 단일 호출은 ainvoke 사용 -> AIMessage 반환
        response_msg = await chat_model.ainvoke(messages)

        assistant_text = response_msg.content
        user_conversation.append(f"assistant: {assistant_text}")
        logger.info("Response generated successfully")

        return {"message": assistant_text}
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return {"test": "success"}

# 서버 실행: uvicorn main:app --reload
# 8f9451a8-f15d-4ec5-8213-97f587982164
