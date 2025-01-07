from datetime import datetime, timedelta, timezone
from fastapi import HTTPException, security
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import HumanMessage
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from typing import List
import os
import re
from dotenv import load_dotenv
import logging
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import FastAPI, Depends
from fastapi.security import OAuth2PasswordBearer

app = FastAPI()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요한 도메인으로 제한하는 것이 좋음
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 데이터 모델 정의

class DiarysEntrys(BaseModel):
    userId: str
    diarys: str
class UpdateDiaryEntry(BaseModel):
    id:str
    userId: str
    content: str
    date: str
class DiaryEntry(BaseModel):
    userId: str
    content: str
    date: str
class DeleteDiaryEntry(BaseModel):
    id:str

class Query(BaseModel):
    userId: str
    question: str
    chatHistory: List[str]

# 환경 변수 또는 직접 설정으로부터 OpenAI API 키 가져오기
load_dotenv()
security = HTTPBearer()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Embeddings와 ChromaDB 초기화
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
persist_directory = 'db'

# ChromaDB 로드 또는 초기화
try:
    logger.info("Initializing OpenAIEmbeddings")
    embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
    logger.info("OpenAIEmbeddings initialized successfully")

    logger.info("Initializing Chroma database")
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )
    logger.info("Chroma database initialized successfully")
except Exception as e:
    logger.error(f"Error during initialization: {str(e)}", exc_info=True)
    raise

# 한국 시간으로 현재 날짜와 시간 가져오기 함수
def get_kst_now():
    utc_now = datetime.now(timezone.utc)
    kst_now = utc_now + timedelta(hours=9)
    return kst_now.strftime('%B %d, %Y')

# 일상 대화 및 공감 챗봇을 위한 프롬프트 생성 함수
def create_prompt(conversation: List[str], new_question: str, retrieved_context: str) -> dict:
    today = get_kst_now()  # 현재 날짜를 가져옴
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
    inputs = {
        "system_message": system_message,
        "context": retrieved_context,
        "new_question": new_question,
        "previous_log": previous_log
    }
    return inputs


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials:
        # "Bearer" 스키마인지 확인
        if credentials.scheme != "Bearer":
            raise HTTPException(status_code=403, detail="잘못된 인증 스키마")

        # 실제 토큰 검증 (openai_api_key와 비교)
        openai_api_key = "your_openai_api_key"  # 여기에 실제 OpenAI API 키를 설정
        if not credentials.credentials or credentials.credentials != openai_api_key:
            raise HTTPException(status_code=403, detail="잘못된 토큰")

        return credentials.credentials
    else:
        raise HTTPException(status_code=403, detail="인증 정보가 없습니다")


# 일기 항목을 추가하는 엔드포인트
@app.post("/diarys")
async def add_diarys(entry: DiarysEntrys, credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        logger.info(f"Adding diary entry for user: {entry.userId}")
        results = vectordb.get(where={"userId": entry.userId})
        if "ids" in results:
            ids_to_delete = results["ids"]

            # ids_to_delete를 출력하여 확인 (디버깅용)
            #print("IDs to delete:", ids_to_delete)

            # 가져온 ID들로 삭제를 수행
            if ids_to_delete:
                vectordb.delete(ids=ids_to_delete)
        # 날짜 패턴에 맞춰 일기를 분리하고, 정규식에서 "날짜: 내용"을 추출
        diary_entries = re.findall(r"(\d{4}-\d{2}-\d{2}): ([^\n]+)", entry.diarys)
        # 각 날짜별 일기를 처리
        for date, content in diary_entries:
            # 일기를 벡터화
            embedding_vector = embedding.embed_query(content)
            # 벡터DB에 저장
            vectordb.add_texts([content], metadatas=[{"userId": entry.userId, "date": date}], embedding=[embedding_vector])
        logger.info("Diary entries successfully added")
        return {"message": "Diary entries have been successfully embedded and saved."}
    except Exception as e:
        logger.error(f"Error adding diary entry: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/diary")
async def add_diary(entry: DiaryEntry, credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        logger.info(f"Adding diary entry for user: {entry.userId}")
        content = entry.content
        # 각 날짜별 일기를 처리
        embedding_vector = embedding.embed_query(content)
        # 벡터DB에 저장
        ids = vectordb.add_texts(texts=[entry.content],metadatas=[{"userId": entry.userId, "date": entry.date}], embedding=[embedding_vector])
        logger.info("Diary entries successfully added")
        return {"id": ids[0]}
    except Exception as e:
        logger.error(f"Error adding diary entry: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/diary")
async def update_diary(entry: UpdateDiaryEntry, credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        logger.info(f"Adding diary entry for user: {entry.userId}")
        content = entry.content
        # 각 날짜별 일기를 처리
        embedding_vector = embedding.embed_query(content)
        # 벡터DB에 저장
        vectordb.delete(ids=entry.id)
        ids = vectordb.add_texts(texts=[entry.content],metadatas=[{"userId": entry.userId, "date": entry.date}], embedding=[embedding_vector])

        logger.info("Diary entries successfully updated")
        return {"id": ids[0]}
    except Exception as e:
        logger.error(f"Error updating diary entry: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/diary")
async def delete_diary(id: str, credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        logger.info(f"Deleting diary entry for user: {id}")

        vectordb.delete(ids=id)
        logger.info("Diary entries successfully deleted")
        return {"message": "Diary entries have been successfully deleted."}
    except Exception as e:
        logger.error(f"Error deleting diary entry: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/diary")
async def get_diary(id: str, credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        # 벡터 DB에서 유저 ID와 날짜를 기준으로 다이어리 조회
        diary_entry = vectordb.get(ids=id)
        print(diary_entry)
        if not diary_entry:
            raise HTTPException(status_code=404, detail="Diary entry not found")

        # 찾은 다이어리 반환
        return diary_entry  # 다이어리 항목을 그대로 반환

    except Exception as e:
        logger.error(f"Error fetching diary entry: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# 챗봇 쿼리를 처리하는 엔드포인트
@app.post("/query")
async def query_api(query: Query, credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        logger.info(f"Received query for user: {query.userId}")
        logger.info(openai_api_key)
        user_id = query.userId
        # 사용자로부터 받은 채팅 내역 사용
        user_conversation = query.chatHistory

        # 새로운 질문을 대화 기록에 추가
        user_conversation.append(f"user: {query.question}")
        retriever_with_user_id = vectordb.as_retriever(search_kwargs={"k": 3, "filter": {"userId": user_id}})

        # 최신 메서드 invoke 사용
        logger.info("Retrieving documents from Chroma")
        retrieved_docs = retriever_with_user_id.invoke(query.question)
        logger.info(f"Retrieved {len(retrieved_docs)} documents")

        # 중복 제거 및 날짜와 함께 내용 결합
        unique_docs = {}
        for doc in retrieved_docs:
            date = doc.metadata.get('date', 'Unknown date')
            content = doc.page_content.strip()

            if date not in unique_docs:
                unique_docs[date] = content
            else:
                if unique_docs[date] != content:
                    unique_docs[date] += f"\n{content}"

        # 최종 문자열로 결합
        retrieved_context = "\n\n".join([f"{date}: {content}" for date, content in unique_docs.items()])

        # 프롬프트 생성
        prompt_dict = create_prompt(user_conversation, query.question, retrieved_context)

        # PromptTemplate을 이용해 실제 프롬프트 문자열 생성
        prompt_template = PromptTemplate(
            input_variables=["system_message", "previous_log", "context", "new_question"],
            template=f"""
                {{system_message}}

                User's Diary:
                {{context}}

                User Query:
                {{new_question}}

                Previous_log:
                {{previous_log}}

                assistant Response:
                """
        )
        prompt = prompt_template.format(**prompt_dict)

        # GPT 모델 초기화
        gpt_model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)

        # 모델에 프롬프트 전달
        response = gpt_model.invoke([HumanMessage(content=prompt)])

        # 모델의 응답을 대화 기록에 추가
        user_conversation.append(f"assistant: {response.content}")
        logger.info("Generating response from GPT model")
        response = gpt_model.invoke([HumanMessage(content=prompt)])
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
