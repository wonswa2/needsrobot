import os
import json
from collections import defaultdict
from typing import Any, List, Tuple, Dict
from llama_cpp import Llama

# =====================================
# 0. LLM 모델 로드 (GPU: n_gpu_layers=-1 로 전체 계층 GPU)
#    * CUDA 빌드 설치 필요: pip install --force-reinstall --upgrade "llama-cpp-python[cuda]"
# =====================================
LLM_PATH = "llama-3-Korean-Bllossom-8B-Q4_K_M.gguf"
llm = Llama(
    model_path=LLM_PATH,
    n_ctx=4096,
    n_threads=os.cpu_count() or 8,  # CPU 스레드 (GPU와 병행)
    n_gpu_layers=-1                 # 가능한 모든 레이어를 GPU로


# =====================================
# 1. 키워드 추출기 (모든 JSON 형태 대응 → (word, score)로 정규화)
# =====================================
def extract_keywords(data: Any) -> List[Tuple[str, float]]:
    """
    다양한 구조의 JSON에서 키워드와 점수를 뽑아 (word, score) 리스트로 반환.
    점수가 없으면 1.0을 기본값으로 부여.
    지원 예:
      - {"keywords":[{"word":"체중","score":4.587}, ...]}
      - {"word":"체중","score":4.587} (단건)
      - [{"word":"체중","score":4.587}, ...]
      - {"체중": 4.587, "분석": 3.473}  (dict 매핑)
      - ["체중","분석","데이터"]        (리스트 문자열)
    """
    out: List[Tuple[str, float]] = []

    if data is None:
        return out

    # 1) dict
    if isinstance(data, dict):
        # (a) 대표 케이스: {"keywords":[{word, score}, ...]}
        if "keywords" in data and isinstance(data["keywords"], list):
            out.extend(extract_keywords(data["keywords"]))

        # (b) 단건: {"word": "...", "score": ...}
        if "word" in data:
            w = str(data["word"])
            s = data.get("score", 1.0)
            try:
                s = float(s)
            except Exception:
                s = 1.0
            out.append((w, s))

        # (c) 키→점수 매핑 dict
        # 값이 숫자/숫자형 문자열이면 키워드로 간주
        for k, v in data.items():
            if k in ("keywords", "word", "score", "filename", "case"):
                continue
            if isinstance(k, str):
                if isinstance(v, (int, float)):
                    out.append((k, float(v)))
                elif isinstance(v, str):
                    try:
                        out.append((k, float(v)))
                    except Exception:
                        # 점수로 못바꾸면 키워드만 등록
                        out.append((k, 1.0))
                else:
                    # 중첩 구조 재귀
                    out.extend(extract_keywords(v))

    # 2) list
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                out.extend(extract_keywords(item))
            elif isinstance(item, list):
                out.extend(extract_keywords(item))
            elif isinstance(item, str):
                out.append((item, 1.0))
            else:
                # 그 외 타입은 무시
                pass

    # 3) 그 외 타입은 무시
    return out

# =====================================
# 2. 폴더 스캔 & 그룹화 (A_, B_, ... 접두사 제거해 같은 파일군 묶기)
# =====================================
FOLDER = "output/KR_WordRank_case/json"
groups: Dict[str, List[str]] = defaultdict(list)

for fname in os.listdir(FOLDER):
    if fname.lower().endswith(".json"):
        base = "_".join(fname.split("_")[1:])  # "G_ai_x.json" → "ai_x.json"
        groups[base].append(os.path.join(FOLDER, fname))

if not groups:
    print("⚠️ 스캔 결과: JSON 파일을 찾지 못했습니다.")
    exit(0)

# =====================================
# 3. 그룹별 키워드 통합 → (평균 점수) 정렬 → LLM 요약(2~3문장, 전문가 톤)
# =====================================
results: Dict[str, str] = {}

for group_name, files in groups.items():
    merged: List[Tuple[str, float]] = []

    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            merged.extend(extract_keywords(data))
        except Exception as e:
            print(f"⚠️ {os.path.basename(path)} 읽기 실패: {e}")

    # (word → 점수목록)
    score_map: Dict[str, List[float]] = defaultdict(list)
    for w, s in merged:
        if not w:
            continue
        try:
            score_map[w].append(float(s))
        except Exception:
            score_map[w].append(1.0)

    if not score_map:
        print(f"⚠️ {group_name} → 키워드 없음, 건너뜀")
        continue

    # 평균 점수 계산 & 정렬
    avg_scores = {w: (sum(v) / len(v)) for w, v in score_map.items()}
    sorted_keywords = sorted(avg_scores.keys(), key=lambda k: avg_scores[k], reverse=True)

    # 상위 N개만 사용 (과도한 토큰 방지)
    TOP_K = 30
    top_keywords = sorted_keywords[:TOP_K]

    # ===== 프롬프트: 기술전문가 톤, 2~3문장, 설계 전 코멘트 느낌 =====
    user_prompt = (
        "당신은 해당 분야의 기술전문가입니다.\n"
        "아래 키워드를 바탕으로, 설계 도안 준비 전에 참고할 수 있는 전문가 코멘트를 2~3문장으로 작성하세요.\n"
        "핵심 구성 요소와 동작 원리를 간결하고 전문적인 어휘로 설명하되, 제품/시스템의 목적과 차별점이 드러나게 하세요.\n"
        f"키워드: {', '.join(top_keywords)}"
    )

    # llama_cpp의 chat 템플릿 사용 (모델 메타의 chat_template 따라감)
    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "모든 응답은 한국어로, 간결하고 전문적인 톤으로 작성합니다."},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.5,
        max_tokens=300,
        stop=["<|eot_id|>"]
    )

    text = resp["choices"][0]["message"]["content"].strip()
    results[group_name] = text
    print(f"[OK] {group_name} → {text[:60].replace('/n',' ')}...")

# =====================================
# 4. 결과 저장
# =====================================
os.makedirs("output", exist_ok=True)
out_path = "output/tech_overview.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"✅ 기술 개요 생성 완료! → {out_path}")
