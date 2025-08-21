import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# 🔧 경로 설정
RESULT_DIR = "output/KR_WordRank_case/json"
GOLD_FILE = "data/gold/gold_keywords.json"
VAL_OUT_DIR = "output/val"
os.makedirs(VAL_OUT_DIR, exist_ok=True)

# 🔠 폰트 설정 (윈도우 한글 깨짐 방지)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 🔍 임베딩 모델 (Soft Matching용)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 📥 정답 키워드 불러오기
with open(GOLD_FILE, encoding="utf-8") as f:
    gold_data = json.load(f)

# 🧠 Soft Matching 기준
def is_match(pred, gold, threshold=0.6):
    pred_emb = model.encode(pred, convert_to_tensor=True)
    for label in gold:
        label_emb = model.encode(label, convert_to_tensor=True)
        sim = util.cos_sim(pred_emb, label_emb).item()
        if sim >= threshold:
            return True
    return False

# 📊 성능 저장
scores = {}

# 📦 평가 시작
for fname in tqdm(os.listdir(RESULT_DIR)):
    if not fname.endswith(".json"):
        continue

    # 예측 키워드 불러오기
    with open(os.path.join(RESULT_DIR, fname), encoding="utf-8") as f:
        pred_data = json.load(f)
    pred_keywords = [kw["word"] for kw in pred_data["keywords"]]

    # 정답 키워드
    base_name = fname.split("_", 1)[1]
    gold_keywords = gold_data.get(base_name, [])
    if not gold_keywords:
        continue

    # TP / FP / FN 계산
    tp = sum([1 for pred in pred_keywords if is_match(pred, gold_keywords)])
    fp = len(pred_keywords) - tp
    fn = len(gold_keywords) - tp

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    scores[fname] = {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3)
    }

# 💾 JSON 저장
with open(os.path.join(VAL_OUT_DIR, "KR_WordRank_performance_scores.json"), "w", encoding="utf-8") as f:
    json.dump(scores, f, indent=2, ensure_ascii=False)

# 📊 A~H 성능 집계
from collections import defaultdict
import csv

agg = defaultdict(lambda: {"precision": [], "recall": [], "f1": []})

for fname, prf in scores.items():
    case = fname[0]
    for k in ["precision", "recall", "f1"]:
        agg[case][k].append(prf[k])

# 평균 점수 계산
summary = {}
for case in sorted(agg.keys()):
    summary[case] = {
        "precision": round(sum(agg[case]["precision"]) / len(agg[case]["precision"]), 3),
        "recall": round(sum(agg[case]["recall"]) / len(agg[case]["recall"]), 3),
        "f1": round(sum(agg[case]["f1"]) / len(agg[case]["f1"]), 3)
    }

# CSV 저장
csv_path = os.path.join(VAL_OUT_DIR, "KR-WordRank_prf_table.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["case", "precision", "recall", "f1"])
    for case, prf in summary.items():
        writer.writerow([case, prf["precision"], prf["recall"], prf["f1"]])

# 📊 시각화 (막대 그래프)
labels = list(summary.keys())
prec = [summary[c]["precision"] for c in labels]
rec = [summary[c]["recall"] for c in labels]

x = range(len(labels))
plt.figure(figsize=(10, 6))
plt.bar(x, prec, width=0.25, label='Precision')
plt.bar([i+0.25 for i in x], rec, width=0.25, label='Recall')
plt.bar([i+0.5 for i in x], f1, width=0.25, label='F1-score')

plt.xticks([i+0.25 for i in x], labels)
plt.ylabel("Score")
plt.ylim(0, 1)
plt.title("A~H 실험별 키워드 추출 성능 (Soft Matching)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(VAL_OUT_DIR, "KR-WordRank_prf_barplot.png"))
plt.close()
