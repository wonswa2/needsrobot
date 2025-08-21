import os
import json
from tqdm import tqdm
from openai import OpenAI
import matplotlib.pyplot as plt

# ✅ OpenAI 클라이언트 (신버전)
client = OpenAI(api_key="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")  # ← 실제 API 키 입력

# ✔️ 디렉토리
INPUT_DIR = "data/parsed"
OUTPUT_JSON_DIR = "output/gpt/json"
OUTPUT_PNG_DIR = "output/gpt/png"
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
os.makedirs(OUTPUT_PNG_DIR, exist_ok=True)

# ✔️ 시스템 프롬프트
SYSTEM_MSG = "당신은 기술 사업계획서를 요약하고 분석하는 전문 평가자입니다."

# ✔️ 프롬프트 생성
def build_prompt(text):
    return f"""
아래 기술 내용에서 다음을 생성해줘:

1. 핵심 키워드 10개 (중복 없이, 중요한 단어 위주)
2. 중요 문장 2개 (기술적 강점이나 차별성 중심)
3. 요약 1문장 (기술 요약)
4. 슬로건 1문장 (간결하게 어필할 수 있도록)

❗ '삼육대학교'는 반드시 제외하고, '기술', '내용' 같은 일반 단어는 키워드에서 제거해줘.

내용:
{text}
"""

# ✔️ 시각화 함수
def visualize_keywords(keywords, fname):
    if not keywords:
        print(f"⚠️ 키워드 없음 (시각화 생략): {fname}")
        return

    words = [k.strip().split()[0] for k in keywords if k.strip()]
    counts = [1 for _ in words]  # 단순 빈도 1씩 부여

    plt.figure(figsize=(10, 6))
    plt.barh(words[::-1], counts[::-1], color='skyblue')
    plt.title("Top 10 Keywords")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PNG_DIR, fname.replace(".json", ".png")))
    plt.close()

# ✔️ GPT 처리 함수
def summarize_with_gpt(text):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": build_prompt(text)}
            ],
            temperature=0.5,
        )
        return response.choices[0].message.content
    except Exception as e:
        print("❌ GPT 호출 실패:", e)
        return None

# ✔️ 전체 루프
def process_all():
    for fname in tqdm(os.listdir(INPUT_DIR)):
        if not fname.endswith(".json"):
            continue

        with open(os.path.join(INPUT_DIR, fname), encoding="utf-8") as f:
            data = json.load(f)
        text = "\n".join(data.get("sections", {}).values())

        result = summarize_with_gpt(text)
        if not result:
            continue

        out_path = os.path.join(OUTPUT_JSON_DIR, fname)
        with open(out_path, "w", encoding="utf-8") as out_f:
            json.dump({
                "filename": fname,
                "result": result
            }, out_f, ensure_ascii=False, indent=2)

        # 키워드 추출 (단순 패턴 기반)
        lines = result.split("\n")
        keywords = []
        for line in lines:
            if line.strip().startswith("1."):
                keyword_section = line.split(":", 1)[-1] if ":" in line else line
                keywords = [w.strip(" ,•-") for w in keyword_section.split(",") if w.strip()]
                break

        visualize_keywords(keywords, fname)
        print(f"✅ 완료: {fname}")

# ▶ 실행
if __name__ == "__main__":
    process_all()
