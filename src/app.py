"""
Streamlit 데모 앱 - LangGraph Agent 전체 파이프라인 진입점

경험8 서술 매핑:
  ① 문서 코퍼스 + 오버랩 청킹     -> ingest_corpus.py (앱 시작시 1회 인덱싱)
  ② 임베딩 + Chroma               -> retriever.py (build_vectorstore)
  ③ BM25 하이브리드                -> retriever.py (build_hybrid_retriever)
  ④ 베어링/웨이퍼 Tool 연동        -> tools.py
  ⑤ LangGraph 조건부 라우팅        -> graph.py
  ⑥ LLM 백엔드 추상화              -> config.py (사이드바에서 API/local 선택)
  ⑦ LangSmith 트레이싱             -> observability.py
"""

import streamlit as st
import os
import tempfile

st.set_page_config(page_title="제조 설비 Agentic AI 데모", page_icon="🏭", layout="wide")

# --- 사이드바: 백엔드 및 트레이싱 설정 ---
with st.sidebar:
    st.header("⚙️ 설정")

    backend = st.radio(
        "LLM 백엔드",
        ["api", "local"],
        format_func=lambda x: "Claude API (공개 데모)" if x == "api" else "로컬 sLLM (Ollama)",
    )
    os.environ["LLM_BACKEND"] = backend

    # 배포 환경(Streamlit Cloud)에서는 Secrets에 미리 넣어둔 키를 자동 사용 -> 방문자가
    # API 키를 입력할 필요 없음. secrets.toml 자체가 없는 로컬 환경에서는 st.secrets 접근이
    # 예외를 던지므로 try/except로 안전하게 처리하고 사이드바 직접 입력으로 폴백.
    try:
        preset_api_key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        preset_api_key = None

    if preset_api_key:
        os.environ["ANTHROPIC_API_KEY"] = preset_api_key
        st.caption("✅ API 키 설정됨 (관리자 제공)")
    elif backend == "api":
        api_key = st.text_input("Anthropic API Key", type="password")
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
    else:
        st.text_input("Ollama Base URL", value="http://localhost:11434", key="ollama_url")
        os.environ["OLLAMA_BASE_URL"] = st.session_state.get("ollama_url", "http://localhost:11434")

    st.divider()
    langsmith_key = st.text_input("LangSmith API Key (선택)", type="password")
    if langsmith_key:
        os.environ["LANGSMITH_API_KEY"] = langsmith_key

    st.divider()
    st.caption(
        "아키텍처: 사용자 질의 → LangGraph Router → "
        "(RAG 하이브리드 검색 | 베어링 Tool | 웨이퍼 Tool | 일반대화) → 답변 생성"
    )

st.title("🏭 제조 설비 Agentic AI 데모")
st.caption("ML(베어링 예지보전) + DL(웨이퍼 결함) 모델을 Tool로 감싼 LangGraph RAG Agent")

# --- Agent 초기화 (캐시) ---
@st.cache_resource
def get_app():
    from observability import setup_langsmith
    from graph import build_graph

    setup_langsmith()
    return build_graph()


# --- 채팅 UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

col1, col2 = st.columns([3, 1])
with col2:
    with st.expander("🔧 베어링 특징값 입력 (선택)"):
        h_rms = st.number_input("h_rms (RMS 진폭)", value=0.0, step=0.1)
        h_kurt = st.number_input("h_kurt (첨도)", value=0.0, step=0.1)
        h_skew = st.number_input("h_skew (왜도)", value=0.0, step=0.1)
        h_crest = st.number_input("h_crest (Crest factor)", value=0.0, step=0.1)

    with st.expander("🖼️ 웨이퍼 맵 업로드 (선택)"):
        wafer_file = st.file_uploader(
            "웨이퍼 맵 (.npy, 값 0=밖/1=정상/2=불량인 2D 배열)", type=["npy"]
        )

if question := st.chat_input("질문을 입력하세요 (예: GroupKFold를 왜 썼어?)"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Agent가 답변을 생성 중..."):
            if not os.getenv("ANTHROPIC_API_KEY") and backend == "api":
                st.error("사이드바에 Anthropic API Key를 입력해주세요.")
            else:
                app = get_app()

                state = {"question": question}
                if h_rms or h_kurt or h_skew or h_crest:
                    state["bearing_features"] = {
                        "h_rms": h_rms,
                        "h_kurt": h_kurt,
                        "h_skew": h_skew,
                        "h_crest": h_crest,
                    }
                if wafer_file:
                    tmp_path = os.path.join(tempfile.gettempdir(), wafer_file.name)
                    with open(tmp_path, "wb") as f:
                        f.write(wafer_file.getbuffer())
                    state["wafer_map_path"] = tmp_path

                result = app.invoke(state)

                answer = result.get("answer", "답변을 생성하지 못했습니다.")
                route = result.get("route", "unknown")

                st.write(answer)
                st.caption(f"라우팅 경로: `{route}`")

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
