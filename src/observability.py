"""
⑦ LangSmith 트레이싱 및 평가

설계 의도:
- LangGraph/LangChain은 아래 환경변수만 설정하면 코드 수정 없이 모든 노드 실행,
  Tool 호출, LLM 요청/응답이 LangSmith 대시보드에 자동으로 트레이싱된다.
  -> "왜 router가 이 질의를 bearing으로 분류했는지", "RAG가 어떤 청크를 가져왔는지"를
     실행 후에도 추적 가능 -> 실서비스에서 오분류/오검색 디버깅에 필수.
- 평가(evaluate)는 별도로 QA 데이터셋(질문-기대답변 쌍)을 만들어 LangSmith의
  evaluate() 함수로 라우팅 정확도, 답변 품질을 정량 측정한다.
"""

import os


def setup_langsmith(project_name: str = "manufacturing-agent-ej"):
    """
    LangSmith 트레이싱 활성화.
    필요 환경변수:
      LANGSMITH_API_KEY   - LangSmith API 키 (smith.langchain.com에서 발급)
      LANGSMITH_TRACING   - "true"로 설정하면 자동 트레이싱 시작
      LANGSMITH_PROJECT   - 대시보드에 표시될 프로젝트명
    """
    if not os.getenv("LANGSMITH_API_KEY"):
        print(
            "[observability] LANGSMITH_API_KEY 미설정 -> 트레이싱 비활성화 상태로 진행. "
            "https://smith.langchain.com 에서 API 키 발급 후 환경변수 설정 필요."
        )
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT", project_name)
    print(f"[observability] LangSmith 트레이싱 활성화됨 (project={project_name})")
    return True


# ---------------------------------------------------------------------------
# 평가용 예시 데이터셋: (질문, 기대 라우트) 쌍
# 실제로는 LangSmith Dataset으로 업로드해서 evaluate()에 사용
# ---------------------------------------------------------------------------
ROUTING_EVAL_EXAMPLES = [
    {"question": "GroupKFold를 왜 썼는지 설명해줘", "expected_route": "rag"},
    {"question": "ESC O-ring 소재 개선 사례가 뭐야", "expected_route": "rag"},
    {
        "question": "RMS 2.8, Kurtosis 6.5인 베어링 상태가 어때?",
        "expected_route": "bearing",
    },
    {"question": "이 웨이퍼 맵 이미지 결함이 뭔지 봐줘", "expected_route": "wafer"},
    {"question": "오늘 날씨 어때?", "expected_route": "general"},
]


def evaluate_routing_accuracy(app, examples=None) -> dict:
    """
    router 노드의 분류 정확도를 간단 측정.
    LangSmith 대시보드 없이도 로컬에서 라우팅 로직 자체를 검증할 수 있는 스모크 테스트.
    """
    examples = examples or ROUTING_EVAL_EXAMPLES
    correct = 0
    details = []

    for ex in examples:
        result = app.invoke({"question": ex["question"]})
        actual_route = result.get("route")
        is_correct = actual_route == ex["expected_route"]
        correct += int(is_correct)
        details.append(
            {
                "question": ex["question"],
                "expected": ex["expected_route"],
                "actual": actual_route,
                "correct": is_correct,
            }
        )

    accuracy = correct / len(examples)
    return {"accuracy": accuracy, "details": details}


if __name__ == "__main__":
    setup_langsmith()
