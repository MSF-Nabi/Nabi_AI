from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
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
    너는 user와 매우 친한 친구야. 그리고 아는 것이 매우 많아.
    만약, 사용자가 질문을 그만해달라고 하면 질문하지말고 공감만 해.
    대화 맥락이 끝나면 이전 대화랑 관련된 주제를 먼저 꺼내.
    사용자와 비슷한 횟수로 질문 빈도를 조절해. 대답 한번에 최대 질문은 1개 까지만.
    사용자의 일기(User's Diary)에서 아는 게 있으면 이를 바탕으로 대답하고, 모를 경우 아는 척은 하지 마.
    질문보다는 감정적인 공감을 많이 해줘. 너는 공감을 잘해주는 사람이야.
    만약, 일기 데이터에서 검색이 필요 없는 질문이면 이전 답변들을 참조해서 답변해주고, 그렇지 않으면 일기 데이터를 참조해서 답변해줘.
    모르는 정보가 나오면 차라리 다시 물어봐.
    너는 반말을 항상 써 그리고 한국인이야.
    오늘 날짜는 {today}야. 필요하면 자연스럽게 대화 중에 날짜를 언급해줘.
    다시 말하지만 너무 빈번한 질문은 자제해줘.
    답변은 sns 채팅처럼 간결히 부탁해.
    """
    previous_log = "\n".join(conversation)
    return {
        "system_message": system_message,
        "context": retrieved_context,
        "new_question": new_question,
        "previous_log": previous_log
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

        retrieved_context = "\n\n".join(
            [f"{result['metadata']['date']}: {result['document']}" for result in results['documents'][0]])

        prompt_dict = create_prompt(user_conversation, query.question, retrieved_context)
        prompt_template = PromptTemplate(
            input_variables=["system_message", "previous_log", "context", "new_question"],
            template="""
                {system_message}

                User's Diary:
                {context}

                User Query:
                {new_question}

                Previous_log:
                {previous_log}

                assistant Response:
                """
        )
        prompt = prompt_template.format(**prompt_dict)

        gpt_model = ChatOpenAI(model="gpt-4-1106-preview", openai_api_key=openai_api_key)
        response = gpt_model.invoke([HumanMessage(content=prompt)])

        user_conversation.append(f"assistant: {response.content}")
        logger.info("Response generated successfully")

        return {"message": response.content}
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return {"test": "success"}

# 서버 실행: uvicorn main:app --reload
# 8f9451a8-f15d-4ec5-8213-97f587982164
