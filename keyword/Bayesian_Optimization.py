import os, json, csv, warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skopt import gp_minimize
from skopt.space import Categorical, Real
from skopt.utils import use_named_args

from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util

# -------------------- 경로 & 폴더 --------------------
RESULT_DIR = "output/KeyBERT_Soynlp_case/json"  # 예측 키워드 저장할 곳 (임시)
GOLD_FILE  = "data/gold/gold_keywords.json"     # 정답 키워드
VAL_DIR    = "output/val"
os.makedirs(VAL_DIR, exist_ok=True)

# -------------------- 한글 폰트(윈도우) --------------------
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# -------------------- 정답 로드 --------------------
with open(GOLD_FILE, encoding="utf-8") as f:
    GOLD = json.load(f)

# -------------------- Soft-matching --------------------
embed_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def soft_match_count(pred_list, gold_list, thr=0.8):
    """pred_list 중 gold_list와 의미 유사(코사인>=thr)한 개수 반환"""
    if not pred_list or not gold_list:
        return 0
    pred_emb = embed_model.encode(pred_list, convert_to_tensor=True)
    gold_emb = embed_model.encode(gold_list, convert_to_tensor=True)
    sims = util.cos_sim(pred_emb, gold_emb).cpu().numpy()  # (P x G)
    # 각 pred는 gold 중 최대 유사도 1개와만 매칭(중복 방지)
    matched = 0
    gold_used = set()
    for i in range(sims.shape[0]):
        j = int(np.argmax(sims[i]))
        if sims[i, j] >= thr and j not in gold_used:
            matched += 1
            gold_used.add(j)
    return matched

def evaluate_predictions(pred_keywords, gold_keywords, thr=0.8):
    tp = soft_match_count(pred_keywords, gold_keywords, thr=thr)
    fp = max(len(pred_keywords) - tp, 0)
    fn = max(len(gold_keywords) - tp, 0)
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2*prec*rec / (prec+rec + 1e-8)
    return round(prec,3), round(rec,3), round(f1,3)

# -------------------- 키워드 추출 유틸 --------------------
STOPWORDS = set(["개발","기술","이용","수행","위해","통한","내용","삼육대학교"])
def clean_word(w):
    import re
    return re.sub(r"[^\w\s]", "", w).strip()

def is_valid(word):
    w = clean_word(word)
    return len(w) >= 2 and w not in STOPWORDS

def extract_for_doc(text, kw_model, ngram, use_mmr, diversity, top_n, topk=10):
    raw = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=ngram,
        stop_words=None,
        use_mmr=use_mmr,
        diversity=(diversity if use_mmr else None),
        top_n=top_n
    )
    filtered, seen = [], set()
    for w, s in raw:
        w = clean_word(w)
        if w in seen or not is_valid(w):
            continue
        seen.add(w)
        filtered.append(w)
        if len(filtered) >= topk:
            break
    return filtered  # 단어 리스트만 리턴

# -------------------- 데이터 로딩(문서 텍스트) --------------------
# 이미 결과 json(A_* 등)을 생성하는 파이프라인이 있다면, 거기서 텍스트를 직접 받아오셔도 됩니다.
# 여기선 골드 파일 key를 파일명으로 삼고, data/parsed에서 텍스트를 읽어오는 예시를 사용합니다.
DOC_DIR = "data/parsed"  # 기존 당신 파이프라인 기준
def load_doc_text(fname):
    path = os.path.join(DOC_DIR, fname)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return "\n".join(data.get("sections", {}).values())

DOCS = {fn: load_doc_text(fn) for fn in GOLD.keys()}
DOCS = {k:v for k,v in DOCS.items() if v}  # 텍스트 있는 것만

# -------------------- 탐색공간 정의 --------------------
space = [
    Categorical([(1,1), (1,3), (2,3)], name="ngram"),
    Categorical([True], name="use_mmr"),             # 필요 시 [True, False]
    Real(0.3, 0.9, name="diversity"),
    Categorical([20, 40, 60], name="top_n"),
    Real(0.75, 0.9, name="soft_thr")                 # 소프트매칭 임계치도 함께 최적화(선택)
]

# -------------------- 목표함수 --------------------
#  - 여러 문서에 대해 평균 F1을 계산하여 최대화(= 최소화 문제에선 -F1)
@use_named_args(space)
def objective(ngram, use_mmr, diversity, top_n, soft_thr):
    # 모델은 고정(멀티링구얼) 또는 조건에 따라 바꾸려면 여기서 if 문으로 변경 가능
    kw_model = KeyBERT(SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'))
    f1s = []
    for fname, text in DOCS.items():
        preds = extract_for_doc(text, kw_model, ngram, use_mmr, diversity, top_n, topk=10)
        gold  = GOLD.get(fname, [])
        _, _, f1 = evaluate_predictions(preds, gold, thr=soft_thr)
        f1s.append(f1)
    mean_f1 = float(np.mean(f1s)) if f1s else 0.0
    # 로그 저장
    trial = {
        "ngram": ngram, "use_mmr": use_mmr, "diversity": round(diversity,3),
        "top_n": int(top_n), "soft_thr": round(soft_thr,3),
        "mean_f1": round(mean_f1,3)
    }
    TRIALS.append(trial)
    return -mean_f1

TRIALS = []

# -------------------- 베이지안 최적화 실행 --------------------
warnings.filterwarnings("ignore")
res = gp_minimize(
    objective,
    dimensions=space,
    acq_func="EI",          # Expected Improvement
    n_calls=25,             # 전체 시도 횟수 (10~30 추천)
    n_initial_points=6,     # 초기 랜덤 샘플 수
    random_state=42
)

# -------------------- 결과 정리/저장 --------------------
# 1) trial 로그
trial_csv = os.path.join(VAL_DIR, "bo_trials.csv")
with open(trial_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(TRIALS[0].keys()))
    w.writeheader()
    w.writerows(TRIALS)

# 2) 최적 파라미터
best = {
    "best_params": {
        "ngram": res.x[0],
        "use_mmr": res.x[1],
        "diversity": round(res.x[2], 3),
        "top_n": int(res.x[3]),
        "soft_thr": round(res.x[4], 3)
    },
    "best_mean_f1": round(-res.fun, 3),
    "n_calls": len(TRIALS)
}
with open(os.path.join(VAL_DIR, "bo_results.json"), "w", encoding="utf-8") as f:
    json.dump(best, f, indent=2, ensure_ascii=False)

# 3) 수렴 그래프
ys = [t["mean_f1"] for t in TRIALS]
best_so_far = np.maximum.accumulate(ys)
plt.figure(figsize=(8,4))
plt.plot(range(1, len(ys)+1), ys, marker="o", label="trial F1")
plt.plot(range(1, len(ys)+1), best_so_far, linestyle="--", label="best so far")
plt.xlabel("Trial")
plt.ylabel("F1")
plt.title("Bayesian Optimization Convergence (mean F1)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(VAL_DIR, "bo_convergence.png"))
plt.close()

print("[DONE] Saved:",
      trial_csv,
      os.path.join(VAL_DIR, "bo_results.json"),
      os.path.join(VAL_DIR, "bo_convergence.png"))
