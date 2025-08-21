import os
import json
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
from krwordrank.word import summarize_with_keywords

# 🔧 경로 설정
INPUT_DIR = "data/parsed"
JSON_OUT_DIR = "output/KR_WordRank_case/json"
IMG_OUT_DIR = "output/KR_WordRank_case/png"
os.makedirs(JSON_OUT_DIR, exist_ok=True)
os.makedirs(IMG_OUT_DIR, exist_ok=True)

# 🔠 폰트 설정 (윈도우 한글 깨짐 방지)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 🔤 불용어 및 필터 세트
STOPWORDS = set(["개발", "기술", "이용", "수행", "위해", "통한", "내용", "삼육대학교"])

def clean_word(w):
    return re.sub(r'[^\w\s]', '', w).strip()

def is_valid(word):
    word = clean_word(word)
    return len(word) >= 2 and word not in STOPWORDS

# 📚 텍스트 로딩
def load_text(fname):
    with open(os.path.join(INPUT_DIR, fname), encoding="utf-8") as f:
        data = json.load(f)
    return "\n".join(data.get("sections", {}).values())

# 📊 시각화
def visualize_keywords(keywords, save_path):
    if not keywords:
        return
    words, scores = zip(*keywords)
    total = sum(scores)
    probs = [s / total for s in scores]

    plt.figure(figsize=(10, 6))
    plt.barh(words[::-1], probs[::-1], color='#ffc000')
    for i, (w, p) in enumerate(zip(words[::-1], probs[::-1])):
        plt.text(p + 0.001, i, f"{w} ({p:.1%})", va='center')
    plt.title("Top Keywords (Normalized)")
    plt.xlabel("Score")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 🧪 krwordrank 실험 설정
EXPERIMENTS = {
    "A": {"min_count": 2, "max_length": 10, "beta": 0.85, "max_iter": 10, "topk": 10},
    "B": {"min_count": 2, "max_length": 8,  "beta": 0.9,  "max_iter": 10, "topk": 10},
    "C": {"min_count": 3, "max_length": 10, "beta": 0.85, "max_iter": 20, "topk": 10},
    "D": {"min_count": 1, "max_length": 15, "beta": 0.9,  "max_iter": 20, "topk": 15},
    "E": {"min_count": 2, "max_length": 10, "beta": 0.95, "max_iter": 10, "topk": 10},
    "F": {"min_count": 2, "max_length": 7,  "beta": 0.85, "max_iter": 15, "topk": 10},
    "G": {"min_count": 3, "max_length": 12, "beta": 0.8,  "max_iter": 10, "topk": 20},
    "H": {"min_count": 2, "max_length": 10, "beta": 0.7,  "max_iter": 5,  "topk": 10}
}

# 🔍 키워드 추출 함수
def extract_keywords(text, params):
    lines = text.split('\n')
    lines = [line.strip() for line in lines if len(line.strip()) > 10]

    keywords = summarize_with_keywords(
        lines,
        min_count=params["min_count"],
        max_length=params["max_length"],
        beta=params["beta"],
        max_iter=params["max_iter"],
        verbose=False
    )

    filtered = []
    for word, score in sorted(keywords.items(), key=lambda x: -x[1]):
        word = clean_word(word)
        if is_valid(word):
            filtered.append((word, round(score, 3)))
        if len(filtered) >= params["topk"]:
            break
    return filtered

# 📦 전체 처리 루프
def process_all():
    for case_id, params in EXPERIMENTS.items():
        print(f"\n🚀 실행 중: 실험 {case_id}")
        for fname in tqdm(os.listdir(INPUT_DIR), desc=f"Case {case_id}"):
            if not fname.endswith(".json"):
                continue

            text = load_text(fname)
            keywords = extract_keywords(text, params)

            if not keywords:
                print(f"⚠️ 키워드 없음: {fname}")
                continue

            json_path = os.path.join(JSON_OUT_DIR, f"{case_id}_{fname}")
            img_path = os.path.join(IMG_OUT_DIR, f"{case_id}_{fname.replace('.json', '.png')}")

            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump({
                    "filename": fname,
                    "case": case_id,
                    "keywords": [{"word": w, "score": s} for w, s in keywords]
                }, jf, indent=2, ensure_ascii=False)

            visualize_keywords(keywords, img_path)

# ▶ 실행
if __name__ == "__main__":
    process_all()
