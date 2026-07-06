# 제조 설비 Agentic AI 데모
LangGraph 기반 RAG Agent — ML/DL 예지보전 모델을 Tool로 감싸 Legacy System과 통합

## 경험8 항목 ↔ 코드 매핑

| # | 항목 | 파일 |
|---|------|------|
| ① | 문서 코퍼스(ML 기술보고서 + DL 발표자료 + 공개 문서) + 오버랩 청킹 | `corpus/`, `src/ingest_corpus.py` |
| ② | Sentence-Transformers 임베딩 + Chroma | `src/config.py` (`get_embeddings`), `src/retriever.py` (`build_vectorstore`) |
| ③ | BM25 하이브리드 (EnsembleRetriever) | `src/retriever.py` (`build_hybrid_retriever`) |
| ④ | 베어링·웨이퍼 모델 Tool 연동 | `src/tools.py` |
| ⑤ | LangGraph 조건부 Edge 라우팅 | `src/graph.py` |
| ⑥ | LLM 백엔드 추상화(API ↔ 로컬 sLLM) | `src/config.py` (`get_llm`) |
| ⑦ | LangSmith 트레이싱·평가 | `src/observability.py` |

## 아키텍처

```
질문 입력 → [Router: LLM이 의도 분류]
              ├─ rag     → 하이브리드 검색(Dense+BM25) → generate
              ├─ bearing → predict_bearing_condition Tool → generate
              ├─ wafer   → predict_wafer_defect Tool → generate
              └─ general → generate
                              ↓
                          최종 답변 (Claude API 또는 로컬 sLLM)
```

## 실행 방법

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 실제 문서로 교체 (중요)
`corpus/` 폴더의 3개 `.txt` 파일은 **더미 샘플**입니다.
실제 ML 기술보고서 / DL 발표자료 / 공개 반도체 문서로 교체하세요.
- 파일명 접두사 규칙: `ml_*` / `dl_*` / `public_*` (카테고리 자동 태깅에 사용됨)
- 지원 포맷: `.txt`, `.pdf`, `.pptx`, `.docx`

### 3. 환경변수 설정 (`.env` 또는 셸)
```bash
export ANTHROPIC_API_KEY="sk-ant-..."      # API 백엔드 사용시
export LANGSMITH_API_KEY="lsv2_..."        # 트레이싱 사용시 (선택)
export LLM_BACKEND="api"                   # api | local
```

### 4. 파이프라인 개별 테스트
```bash
cd src
python3 ingest_corpus.py    # ① 청킹 확인
python3 retriever.py        # ②③ 하이브리드 검색 확인
python3 tools.py            # ④ Tool 동작 확인
python3 graph.py            # ⑤ 그래프 구조 확인
```

### 5. 데모 실행
```bash
cd src
streamlit run app.py
```

## 로컬 sLLM으로 전환하기 (⑥)
사내 폐쇄망 등 API 호출이 제한된 환경에서는 `LLM_BACKEND=local`로 전환하면
동일한 그래프/Tool 코드를 그대로 재사용하면서 Ollama 기반 sLLM(예: qwen2.5:7b)으로
백엔드만 교체됩니다. 애플리케이션 로직(`graph.py`, `tools.py`)은 전혀 수정할 필요 없습니다.

```bash
export LLM_BACKEND="local"
export LOCAL_MODEL_NAME="qwen2.5:7b-instruct"
ollama serve &
ollama pull qwen2.5:7b-instruct
```

## 실제 모델 아티팩트 연결 (④)
`data/models/`에 아래 파일을 배치하면 `tools.py`가 폴백/에러 대신 실제 모델을 사용합니다.
- `bearing_rf_model.pkl` — RandomForestClassifier, 이진분류(0=정상/1=이상,열화).
  입력 피처는 반드시 `h_rms`, `h_kurt`, `h_skew`, `h_crest`(Crest factor) 순서/이름으로 맞춰야 함
  (모델이 `feature_names_in_`로 컬럼명을 기억하고 있어 DataFrame으로 넣음).
  피처 중요도: h_rms 82% > h_skew 12% > h_crest 5% > h_kurt 2%
- `wafer_cnn_model.h5` — `wafer_detection6.ipynb` cell 27 `model.save('model/wafer_cnn_model.h5')` 결과물
- `wafer_label_encoder.pkl` — 같은 셀의 `joblib.dump(le, 'model/wafer_label_encoder.pkl')` 결과물

**웨이퍼 Tool 입력 형식 주의**: 이미지 파일이 아니라 `.npy` 2D 배열(값 0=웨이퍼 밖 / 1=정상 die / 2=불량 die,
IMG_SIZE=128)입니다. `streamlit_wafer.py`의 웨이퍼맵 업로드 규격과 동일합니다.

로컬(F드라이브)에 있는 두 파일을 그대로 복사:
```bash
cp "F:\AI-projects\DL\model\wafer_cnn_model.h5" data/models/
cp "F:\AI-projects\DL\model\wafer_label_encoder.pkl" data/models/
```

## 알려진 제약사항 (샌드박스 개발 환경)
이 코드는 Anthropic 개발 샌드박스(제한된 네트워크)에서 작성되어 huggingface.co
접근이 차단된 상태였습니다. 따라서 임베딩 모델(`intfloat/multilingual-e5-small`)
다운로드는 로컬/Streamlit Cloud 배포 환경에서 최초 실행시 이루어집니다.
파이프라인 배선(Chroma+BM25+EnsembleRetriever 결합) 자체는 더미 임베딩으로
검증 완료했습니다.
