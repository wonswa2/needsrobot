import os
import json
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
from krwordrank.word import summarize_with_keywords

# 🔧 경로 설정
INPUT_DIR = "data/parsed"
JSON_OUT_DIR = "output/krwordrank/json"
IMG_OUT_DIR = "output/krwordrank/png"
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

# 📚 텍스트 로딩 (soynlp 학습용은 제거)
def load_texts():
    texts = []
    for fname in os.listdir(INPUT_DIR):
        if fname.endswith(".json"):
            with open(os.path.join(INPUT_DIR, fname), encoding="utf-8") as f:
                data = json.load(f)
                texts.append("\n".join(data.get("sections", {}).values()))
    return texts

# 🔍 krwordrank 기반 키워드 추출 함수
def extract_keywords(text, topk=10):
    # 문장 단위로 분할
    lines = text.split('\n')
    lines = [line.strip() for line in lines if len(line.strip()) > 10]

    keywords = summarize_with_keywords(
        lines,
        min_count=2,
        max_length=10,
        beta=0.85,
        max_iter=10,
        verbose=False
    )

    filtered = []
    for word, score in sorted(keywords.items(), key=lambda x: -x[1]):
        word = clean_word(word)
        if is_valid(word):
            filtered.append((word, round(score, 3)))
        if len(filtered) >= topk:
            break
    return filtered

# 📊 시각화
def visualize_keywords(keywords, save_path):
    if not keywords:
        return
    words, scores = zip(*keywords)
    total = sum(scores)
    probs = [s / total for s in scores]

    plt.figure(figsize=(10, 6))
    plt.barh(words[::-1], probs[::-1], color='skyblue')
    for i, (w, p) in enumerate(zip(words[::-1], probs[::-1])):
        plt.text(p + 0.001, i, f"{w} ({p:.1%})", va='center')
    plt.title("Top Keywords (Normalized)")
    plt.xlabel("Score")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 📦 전체 처리 루프
def process_all():
    for fname in tqdm(os.listdir(INPUT_DIR)):
        if not fname.endswith(".json"):
            continue

        with open(os.path.join(INPUT_DIR, fname), encoding="utf-8") as f:
            data = json.load(f)

        text = "\n".join(data.get("sections", {}).values())
        keywords = extract_keywords(text, topk=10)

        if not keywords:
            print(f"⚠️ 키워드 없음 (시각화 생략): {fname}")
            continue

        json_path = os.path.join(JSON_OUT_DIR, fname)
        img_path = os.path.join(IMG_OUT_DIR, fname.replace(".json", ".png"))

        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump({
                "filename": fname,
                "keywords": [{"word": w, "score": s} for w, s in keywords]
            }, jf, indent=2, ensure_ascii=False)

        visualize_keywords(keywords, img_path)
        print(f"✅ 완료: {fname}")

# ▶ 실행
if __name__ == "__main__":
    process_all()
