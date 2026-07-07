"""
LLM 백엔드 추상화 레이어

설계 의도 (경험8 ⑥ 항목):
- 공개 Streamlit 데모에서는 API 모델(Claude)을 사용 (배포 환경에 GPU 불필요, 안정적 응답 품질)
- 로컬/사내 환경에서는 동일한 인터페이스로 sLLM(Ollama 기반)으로 즉시 교체 가능
  → 보안상 외부 API 호출이 제한되는 반도체 제조 현장(폐쇄망)을 가정한 설계
- 애플리케이션 코드(그래프, 툴, 리트리버)는 어떤 백엔드가 쓰이는지 전혀 알 필요 없음
  → LangChain의 BaseChatModel 인터페이스로 통일되어 있기 때문에 가능
"""

import os
from functools import lru_cache
from langchain_core.language_models.chat_models import BaseChatModel

# 환경변수로 백엔드 선택: "api" (기본값, Claude) | "local" (Ollama sLLM)
LLM_BACKEND = os.getenv("LLM_BACKEND", "api")

# API 백엔드 설정
API_MODEL_NAME = os.getenv("API_MODEL_NAME", "claude-sonnet-4-6")

# 로컬 sLLM 백엔드 설정 (Ollama)
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "qwen2.5:7b-instruct")
LOCAL_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# 임베딩 모델 (한/영 혼용 도메인 문서를 고려해 multilingual 모델 사용)
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-small"
)


@lru_cache(maxsize=2)
def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """
    현재 설정된 백엔드에 맞는 Chat 모델 인스턴스를 반환.
    호출부(그래프/툴)는 이 함수가 반환하는 객체가 API 모델인지 로컬 sLLM인지 몰라도 됨
    → LangGraph 노드에서는 항상 get_llm()만 호출.
    """
    if LLM_BACKEND == "local":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=LOCAL_MODEL_NAME,
            base_url=LOCAL_BASE_URL,
            temperature=temperature,
        )

    elif LLM_BACKEND == "api":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=API_MODEL_NAME,
            temperature=temperature,
            max_tokens=2048,
        )

    else:
        raise ValueError(
            f"알 수 없는 LLM_BACKEND: {LLM_BACKEND} (api 또는 local만 지원)"
        )


@lru_cache(maxsize=1)
def get_embeddings():
    """
    Sentence-Transformers 기반 임베딩 모델.
    임베딩은 백엔드(API/local) 전환과 무관하게 항상 로컬에서 계산
    → 문서 임베딩 비용이 API 호출량에 영향을 주지 않도록 분리한 설계.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def backend_info() -> dict:
    """현재 백엔드 설정을 딕셔너리로 반환 (UI에 표시용, LangSmith 메타데이터용)"""
    if LLM_BACKEND == "local":
        return {"backend": "local", "model": LOCAL_MODEL_NAME}
    return {"backend": "api", "model": API_MODEL_NAME}
