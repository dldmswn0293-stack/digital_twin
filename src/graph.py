r"""
⑤ LangGraph 조건부 Edge 라우팅

아키텍처:

        [START]
           |
       [router]  <- LLM이 질의 의도를 분류 (rag / bearing / wafer / general)
        /  |  \  \
   [rag] [bearing] [wafer] [general]
        \  |  \  /
       [generate]  <- Tool 결과(or 없음)를 종합해 최종 답변 생성
           |
         [END]

설계 의도:
- 모든 질의를 무조건 RAG나 특정 Tool로 흘려보내지 않고, 먼저 router 노드에서
  "이 질의가 지식 검색이 필요한지 / 베어링 예측이 필요한지 / 웨이퍼 예측이 필요한지 /
  둘 다 필요없는 일반 대화인지"를 LLM으로 분류한다.
  -> 조건부 Edge(add_conditional_edges)로 다음 노드를 동적으로 결정.
- state는 TypedDict로 정의, 각 노드는 state의 일부 필드만 갱신하고 반환한다
  (LangGraph의 표준 상태 축적 패턴).
- generate 노드는 router가 어떤 경로를 탔든 마지막에 항상 거쳐가는 합류점이라
  "RAG로 찾은 배경지식 + Tool 예측 결과"를 함께 참조해 자연어 답변을 만들 수 있다.
  -> 이게 "Agentic AI - Legacy System 통합" 의 핵심: 레거시 ML/DL 모델의 숫자 출력을
     LLM이 사람이 이해할 수 있는 설명으로 감싸는 구조.
"""

from typing import TypedDict, Optional, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage

from config import get_llm
from tools import predict_bearing_condition, predict_wafer_defect, search_manufacturing_docs


# ---------------------------------------------------------------------------
# State 정의
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    question: str
    route: Optional[str]              # router가 결정한 경로
    retrieved_docs: Optional[str]     # RAG 결과
    tool_result: Optional[str]        # 베어링/웨이퍼 Tool 결과
    bearing_features: Optional[dict]  # 사용자 입력에서 추출된 베어링 특징값 (있을 경우)
    wafer_map_path: Optional[str]     # 사용자 제공 웨이퍼 맵 .npy 경로 (있을 경우)
    answer: Optional[str]


# ---------------------------------------------------------------------------
# 노드 1: Router - 질의 의도 분류
# ---------------------------------------------------------------------------
ROUTER_SYSTEM_PROMPT = """너는 반도체 제조 설비 도메인 AI Agent의 라우터다.
사용자 질의를 아래 4가지 중 하나로 정확히 분류해서 그 라벨 하나만 출력해라. 다른 말은 하지 마.

- rag: 프로젝트 방법론, 기술 배경, 도메인 지식(ESC, FDC, GroupKFold, Focal Loss 등)에 대한 질문
- bearing: 베어링 진동 신호 특징값(RMS, Kurtosis 등)을 주고 설비 상태 판정을 요청하는 질문
- wafer: 웨이퍼 맵(.npy) 데이터를 주고 결함 유형 판정을 요청하는 질문
- general: 위 세 가지에 해당하지 않는 일반 대화

라벨만 출력: rag, bearing, wafer, general 중 하나."""


def router_node(state: AgentState) -> dict:
    # 데이터가 실제로 첨부된 경우, 질문 문구와 무관하게 해당 Tool로 확정 라우팅.
    # (사용자가 "이거 분석해줘"처럼 애매하게 말해도 첨부 여부로 의도가 이미 명확하기 때문 -
    #  LLM 분류보다 결정론적 규칙이 더 정확하고, LLM 호출도 줄어 비용 절감됨)
    if state.get("wafer_map_path"):
        return {"route": "wafer"}
    if state.get("bearing_features"):
        return {"route": "bearing"}

    llm = get_llm(temperature=0.0)
    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=state["question"]),
    ]
    response = llm.invoke(messages)
    route = response.content.strip().lower()

    if route not in {"rag", "bearing", "wafer", "general"}:
        route = "general"  # 안전한 폴백

    return {"route": route}


def route_decision(state: AgentState) -> Literal["rag", "bearing", "wafer", "general"]:
    """조건부 Edge에서 사용할 라우팅 함수. router_node가 정한 값을 그대로 사용."""
    return state["route"]


# ---------------------------------------------------------------------------
# 노드 2-a: RAG 검색
# ---------------------------------------------------------------------------
def rag_node(state: AgentState) -> dict:
    docs = search_manufacturing_docs.invoke({"query": state["question"]})
    return {"retrieved_docs": docs}


# ---------------------------------------------------------------------------
# 노드 2-b: 베어링 예측 Tool 호출
# ---------------------------------------------------------------------------
def bearing_node(state: AgentState) -> dict:
    features = state.get("bearing_features")
    if not features:
        return {
            "tool_result": (
                "베어링 상태 판정을 위해 RMS, Kurtosis, Skewness, "
                "주파수대역 에너지 값이 필요합니다. 값을 함께 알려주세요."
            )
        }
    result = predict_bearing_condition.invoke(features)
    return {"tool_result": result}


# ---------------------------------------------------------------------------
# 노드 2-c: 웨이퍼 예측 Tool 호출
# ---------------------------------------------------------------------------
def wafer_node(state: AgentState) -> dict:
    wafer_map_path = state.get("wafer_map_path")
    if not wafer_map_path:
        return {
            "tool_result": "웨이퍼 결함 판정을 위해 웨이퍼 맵 .npy 파일이 필요합니다."
        }
    result = predict_wafer_defect.invoke({"wafer_map_path": wafer_map_path})
    return {"tool_result": result}


# ---------------------------------------------------------------------------
# 노드 2-d: 일반 대화 (Tool/RAG 불필요)
# ---------------------------------------------------------------------------
def general_node(state: AgentState) -> dict:
    return {}  # 아무 것도 조회하지 않고 바로 generate로 진행


# ---------------------------------------------------------------------------
# 노드 3: 최종 답변 생성 (합류점)
# ---------------------------------------------------------------------------
GENERATE_SYSTEM_PROMPT = """너는 반도체 제조 설비 도메인 AI Agent다.
아래 컨텍스트(검색된 문서 또는 예측 모델 결과)가 있다면 반드시 근거로 활용해서
정확하고 간결하게 한국어로 답변해라. 컨텍스트가 없다면 일반적인 지식으로 답변해라.
숫자/판정 결과가 있다면 그 의미를 엔지니어가 이해하기 쉽게 풀어서 설명해라."""


def generate_node(state: AgentState) -> dict:
    llm = get_llm(temperature=0.3)

    context_parts = []
    if state.get("retrieved_docs"):
        context_parts.append(f"[검색된 문서]\n{state['retrieved_docs']}")
    if state.get("tool_result"):
        context_parts.append(f"[예측 모델 결과]\n{state['tool_result']}")

    context = "\n\n".join(context_parts) if context_parts else "(참고할 컨텍스트 없음)"

    messages = [
        SystemMessage(content=GENERATE_SYSTEM_PROMPT),
        HumanMessage(
            content=f"질문: {state['question']}\n\n컨텍스트:\n{context}"
        ),
    ]
    response = llm.invoke(messages)
    return {"answer": response.content}


# ---------------------------------------------------------------------------
# 그래프 조립
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("rag", rag_node)
    graph.add_node("bearing", bearing_node)
    graph.add_node("wafer", wafer_node)
    graph.add_node("general", general_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("router")

    # 조건부 Edge: router_node가 정한 route 값에 따라 다음 노드를 동적으로 선택
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "rag": "rag",
            "bearing": "bearing",
            "wafer": "wafer",
            "general": "general",
        },
    )

    # 4갈래 모두 generate로 합류
    graph.add_edge("rag", "generate")
    graph.add_edge("bearing", "generate")
    graph.add_edge("wafer", "generate")
    graph.add_edge("general", "generate")

    graph.add_edge("generate", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()

    print("=== 그래프 구조 ===")
    print(app.get_graph().draw_ascii())
    
    print("\n=== 실제 실행 테스트 (LangSmith 트레이싱) ===")
    result = app.invoke({"question": "GroupKFold를 왜 썼는지 설명해줘"})
    print("라우팅 경로:", result.get("route"))
    print("답변:", result.get("answer")[:300])
