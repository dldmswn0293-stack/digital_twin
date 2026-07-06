"""
④ 베어링·웨이퍼 모델 Tool 연동

설계 의도:
- 기존에 완성한 ML(베어링, RandomForest+GroupKFold)과 DL(웨이퍼 결함, CNN+Focal Loss)
  모델을 "새로 만드는" 게 아니라 LangGraph Agent가 호출하는 Tool로 감싸는 것.
  → "Agentic AI - Legacy System integration" 포지셔닝의 핵심 구현부.
- 각 Tool은 (1) 모델 로드 (2) 입력 전처리 (3) 예측 (4) 사람이 읽을 수 있는 요약 문자열 반환
  까지 책임진다. LangGraph 노드는 Tool을 호출하고 결과를 상태에 저장하기만 하면 됨.
- 실제 배포시 model_path는 각자 학습한 .pkl(RandomForest) / .pt or .h5(CNN) 아티팩트 경로로 교체.
  지금은 두 모델이 아직 로드되지 않은 상태를 가정해 그레이스풀 폴백(모델 부재 시 안내 메시지)을 포함.
"""

import os
import json
from pathlib import Path
from typing import Optional
from langchain_core.tools import tool

MODEL_DIR = Path(__file__).parent.parent / "data" / "models"
BEARING_MODEL_PATH = MODEL_DIR / "bearing_rf_model.pkl"

# 웨이퍼 CNN: wafer_detection6.ipynb / streamlit_wafer.py 와 동일 아티팩트 규격
WAFER_MODEL_PATH = MODEL_DIR / "wafer_cnn_model.h5"
WAFER_LABEL_ENCODER_PATH = MODEL_DIR / "wafer_label_encoder.pkl"
WAFER_IMG_SIZE = 128  # 노트북 cell 3: IMG_SIZE = 128 (96->128, Loc/Edge-Loc 디테일 확보)

# streamlit_wafer.py의 PATTERN_INFO를 그대로 이식 (LLM 답변에서 "형태 + 추정 공정 원인" 근거로 사용)
WAFER_PATTERN_INFO = {
    "Center": ("웨이퍼 한가운데에 불량이 원형으로 뭉쳐 나타남",
               "척(chuck) 중심부 온도·압력 편차, 스핀 코팅 중심 불균일, 중심부 가스 흐름 정체 의심"),
    "Donut": ("중심은 비우고 중간 반경에 고리(도넛) 모양으로 나타남",
              "반경 방향 공정 편차 — 가스 흐름·플라즈마 분포가 특정 반경대에서 불균일할 때 의심"),
    "Edge-Ring": ("가장자리 전체를 따라 링처럼 빙 둘러 나타남",
                  "에지 척킹·클램프 접촉 이슈, 에지 비드 제거 불량, 가장자리 식각·증착 균일도 저하 의심"),
    "Edge-Loc": ("가장자리 중 일부 구간에만 호(arc) 형태로 나타남",
                 "에지 핸들링 중 특정 위치 접촉 손상, 클램프/핀 자국, 국부 에지 세정 불량 의심"),
    "Loc": ("위치와 무관하게 한 구역에 작게 뭉쳐 나타남",
            "파티클 낙하, 국부적 디펙트 소스 존재 의심 (가장자리에 국한되지 않음)"),
    "Scratch": ("가늘고 길게 선(줄) 형태로 이어져 나타남",
                "핸들링·이송 중 로봇 암/캐리어 기계적 긁힘, CMP·세정 중 스크래치 의심"),
    "Random": ("전면에 규칙 없이 흩뿌려져 나타남",
               "전반적 파티클 오염, 클린룸·장비 청정도 저하, 재료 자체 결함 의심"),
    "Near-full": ("웨이퍼 거의 전체가 불량으로 덮여 나타남 (정상이 드묾)",
                  "심각한 공정/장비 이상 또는 레시피 오류 — 즉시 장비 점검 필요"),
    "none": ("뚜렷한 패턴 없이 산발적 점만 있음",
             "정상 또는 산발적 불량 — 특이 공정 이슈 신호 아님"),
}


# ---------------------------------------------------------------------------
# ④-1. 베어링 예지보전 Tool
# ---------------------------------------------------------------------------
# 실제 학습 모델(bearing_model.pkl) 기준: RandomForestClassifier, 이진분류(0=정상/1=이상,열화)
# feature_names_in_ = ['h_rms', 'h_kurt', 'h_skew', 'h_crest']
# feature_importances_: h_rms 0.82(압도적) > h_skew 0.12 > h_crest 0.05 > h_kurt 0.02
BEARING_LABEL_MAP = {0: "정상", 1: "이상(열화)"}


@tool
def predict_bearing_condition(
    h_rms: float,
    h_kurt: float,
    h_skew: float,
    h_crest: float,
) -> str:
    """
    베어링 진동 신호의 시간 영역 특징을 입력받아 예지보전 분류 모델(RandomForest,
    GroupKFold 검증, 이진분류)로 정상/이상(열화) 상태를 예측한다.
    사용 시점: 사용자가 베어링 진동 데이터, RMS, Kurtosis, Skewness, Crest factor 등
    진동 특징값을 언급하며 설비 상태 판정을 요청할 때 호출한다.

    Args:
        h_rms: RMS(제곱평균제곱근) 진동 진폭 - 가장 중요한 특징(모델 기여도 약 82%)
        h_kurt: 첨도 (충격성 이상 진동의 지표)
        h_skew: 왜도
        h_crest: Crest factor (피크값/RMS 비율, 충격성 지표)
    """
    if not BEARING_MODEL_PATH.exists():
        # 모델 아티팩트가 아직 이 환경에 없는 경우: h_rms가 피처 중요도 82%로 압도적이므로
        # h_rms 단일 기준의 근사 룰로 폴백 (실제 임계값은 학습 데이터 분포 기반 재보정 필요)
        if h_rms > 2.5:
            status, confidence = "이상(열화)", 0.70
        else:
            status, confidence = "정상", 0.75

        return json.dumps(
            {
                "status": status,
                "confidence": confidence,
                "note": (
                    "실제 학습 모델(bearing_rf_model.pkl) 미탑재 상태 -> "
                    "h_rms(피처 중요도 82%) 기준 근사 룰로 대체 판정됨. "
                    "배포시 data/models/에 실제 아티팩트 배치 필요."
                ),
                "input_features": {
                    "h_rms": h_rms,
                    "h_kurt": h_kurt,
                    "h_skew": h_skew,
                    "h_crest": h_crest,
                },
            },
            ensure_ascii=False,
        )

    import joblib
    import pandas as pd

    model = joblib.load(BEARING_MODEL_PATH)
    # 학습시 DataFrame(컬럼명 포함)으로 fit되어 feature_names_in_이 저장돼 있으므로
    # 동일 컬럼명의 DataFrame으로 넣어야 경고 없이 정확히 매핑됨
    X = pd.DataFrame(
        [[h_rms, h_kurt, h_skew, h_crest]],
        columns=["h_rms", "h_kurt", "h_skew", "h_crest"],
    )
    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][pred])
    status = BEARING_LABEL_MAP.get(pred, str(pred))

    return json.dumps(
        {
            "status": status,
            "confidence": proba,
            "input_features": {
                "h_rms": h_rms,
                "h_kurt": h_kurt,
                "h_skew": h_skew,
                "h_crest": h_crest,
            },
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# ④-2. 웨이퍼 결함 분류 Tool
# ---------------------------------------------------------------------------
def _load_wafer_assets():
    """모델/라벨인코더 로드를 프로세스당 1회만 수행 (Streamlit @st.cache_resource와 동일 목적)"""
    if not hasattr(_load_wafer_assets, "_cache"):
        from tensorflow.keras.models import load_model
        import joblib

        model = load_model(str(WAFER_MODEL_PATH))
        le = joblib.load(str(WAFER_LABEL_ENCODER_PATH))
        _load_wafer_assets._cache = (model, le)
    return _load_wafer_assets._cache


@tool
def predict_wafer_defect(wafer_map_path: str) -> str:
    """
    웨이퍼 맵 데이터(.npy, 값 0=웨이퍼 밖/1=정상 die/2=불량 die인 2D 배열)를 입력받아
    CNN 결함 분류 모델(Focal Loss gamma=3.0, 128px, WM-811K 학습)로 9개 결함 클래스
    (Center/Donut/Edge-Loc/Edge-Ring/Loc/Random/Scratch/Near-full/none) 중 하나를 예측하고,
    해당 패턴의 전형적 형태와 추정 공정 원인을 함께 반환한다.
    사용 시점: 사용자가 웨이퍼 맵(.npy) 파일을 제시하며 결함 유형 판정을 요청할 때 호출한다.

    Args:
        wafer_map_path: 웨이퍼 맵 .npy 파일 경로 (2D 배열, shape=(H, W))
    """
    if not WAFER_MODEL_PATH.exists() or not WAFER_LABEL_ENCODER_PATH.exists():
        return json.dumps(
            {
                "error": True,
                "note": (
                    "실제 학습 모델(wafer_cnn_model.h5) 또는 라벨인코더(wafer_label_encoder.pkl) "
                    f"미탑재 상태. {MODEL_DIR} 경로에 두 파일을 배치해야 함. "
                    "노트북(wafer_detection6.ipynb) cell 27에서 model.save() / joblib.dump()로 생성됨."
                ),
            },
            ensure_ascii=False,
        )

    if not Path(wafer_map_path).exists():
        return json.dumps(
            {"error": True, "note": f"웨이퍼 맵 파일을 찾을 수 없음: {wafer_map_path}"},
            ensure_ascii=False,
        )

    import numpy as np
    import cv2

    model, le = _load_wafer_assets()

    wafer_map = np.load(wafer_map_path)
    arr = np.asarray(wafer_map, dtype="float32")
    if arr.shape != (WAFER_IMG_SIZE, WAFER_IMG_SIZE):
        # 노트북과 동일하게 값(0/1/2) 보존을 위해 최근접 보간 사용
        arr = cv2.resize(
            arr, (WAFER_IMG_SIZE, WAFER_IMG_SIZE), interpolation=cv2.INTER_NEAREST
        )

    proba = model.predict(arr[np.newaxis, ..., np.newaxis], verbose=0)[0]
    pred_idx = int(proba.argmax())
    pred_cls = str(le.classes_[pred_idx])
    confidence = float(proba[pred_idx])

    appearance, cause = WAFER_PATTERN_INFO.get(pred_cls, ("정보 없음", "정보 없음"))

    return json.dumps(
        {
            "predicted_class": pred_cls,
            "confidence": confidence,
            "class_probabilities": {
                cls: float(p) for cls, p in zip(le.classes_, proba)
            },
            "appearance": appearance,
            "suspected_cause": cause,
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# RAG 검색 Tool (문서 코퍼스 - retriever.py의 하이브리드 리트리버를 감싼 것)
# ---------------------------------------------------------------------------
@tool
def search_manufacturing_docs(query: str) -> str:
    """
    ML 기술보고서, DL 발표자료, 공개 반도체/설비 참고 문서를 대상으로 하이브리드
    검색(Dense+BM25)을 수행하여 관련 문서 조각을 반환한다.
    사용 시점: 사용자가 프로젝트의 기술적 배경, 방법론 선택 이유, 도메인 지식(ESC, FDC 등)을
    질문할 때 호출한다.

    Args:
        query: 검색할 질의문
    """
    from retriever import build_hybrid_retriever

    retriever = build_hybrid_retriever()
    results = retriever.invoke(query)

    formatted = []
    for r in results:
        formatted.append(
            f"[출처: {r.metadata.get('source')} / {r.metadata.get('category')}]\n"
            f"{r.page_content}"
        )
    return "\n\n---\n\n".join(formatted) if formatted else "관련 문서를 찾지 못함."


ALL_TOOLS = [predict_bearing_condition, predict_wafer_defect, search_manufacturing_docs]


if __name__ == "__main__":
    print("=== 베어링 Tool 테스트 ===")
    print(
        predict_bearing_condition.invoke(
            {"h_rms": 2.8, "h_kurt": 6.5, "h_skew": 0.3, "h_crest": 4.0}
        )
    )

    print("\n=== 웨이퍼 Tool 테스트 (모델 미탑재 -> 안내) ===")
    print(predict_wafer_defect.invoke({"wafer_map_path": "nonexistent.npy"}))

    print("\n=== Tool 목록 (LangGraph bind용) ===")
    for t in ALL_TOOLS:
        print(f"- {t.name}: {t.description[:60]}...")
