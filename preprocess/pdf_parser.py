import pdfplumber
import re
import json
import os

# ------------------------------
# 1. PDF 텍스트 추출
# ------------------------------
def extract_text_from_pdf(file_path):
    """
    PDF 파일에서 전체 텍스트 추출
    """
    full_text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    return full_text

# ------------------------------
# 2. 텍스트 전처리
# ------------------------------
def clean_text(text):
    """
    불필요한 줄바꿈, 공백 제거 등 전처리
    """
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\xa0', ' ', text)
    return text.strip()

# ------------------------------
# 3. 섹션 분리
# ------------------------------
def split_sections(text):
    section_titles = [
        "기술명", "기술 내용", "시작품 제작 목적", "시작품 제작 유형",
        "시작품 제작 필요성", "시작품 제작 내용", "시작품 가능여부"
    ]
    lines = text.split("\n")
    section_points = []

    for i, line in enumerate(lines):
        for title in section_titles:
            if title in line and title not in [s[0] for s in section_points]:
                section_points.append((title, i))
                break

    if not section_points:
        return {"전체": text}

    section_points.sort(key=lambda x: x[1])
    sections = {}
    for idx in range(len(section_points)):
        title, start_idx = section_points[idx]
        end_idx = section_points[idx + 1][1] if idx + 1 < len(section_points) else len(lines)
        content = "\n".join(lines[start_idx + 1:end_idx]).strip()
        sections[title] = content

    return sections

# ------------------------------
# 4. JSON 저장
# ------------------------------
def save_parsed_json(file_path, sections, save_dir="data/parsed"):
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.basename(file_path).replace(".pdf", ".json")
    json_path = os.path.join(save_dir, filename)

    result = {
        "filename": os.path.basename(file_path),
        "sections": sections
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return json_path

# ------------------------------
# 5. 시각화 이미지 저장 (텍스트 박스 표시)
# ------------------------------
def save_visualization(file_path, save_dir="data/visual"):
    os.makedirs(save_dir, exist_ok=True)

    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            words = page.extract_words()
            rects = [
                {
                    "x0": float(w["x0"]),
                    "top": float(w["top"]),
                    "x1": float(w["x1"]),
                    "bottom": float(w["bottom"])
                }
                for w in words
            ]
            img = page.to_image(resolution=150)
            img.draw_rects(rects, stroke="red", stroke_width=1)

            img_path = os.path.join(
                save_dir,
                f"{os.path.basename(file_path).replace('.pdf','')}_p{i+1}.png"
            )
            img.save(img_path)

# ------------------------------
# 6. 실행 파트 (폴더 내 전체 PDF 처리)
# ------------------------------
if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    pdf_dir = os.path.abspath(os.path.join(base_dir, "..", "data/test"))

    for fname in os.listdir(pdf_dir):
        if fname.endswith(".pdf"):
            file_path = os.path.join(pdf_dir, fname)
            print(f"\n📄 파일 처리 중: {file_path}")

            try:
                raw_text = extract_text_from_pdf(file_path)
                cleaned = clean_text(raw_text)
                sections = split_sections(cleaned)
                json_path = save_parsed_json(file_path, sections)
                save_visualization(file_path)  # 텍스트 박스 시각화

                print(f"저장 완료: {json_path}")
                print("섹션 미리보기:")
                for k, v in sections.items():
                    print(f"\n🟦 [{k}]\n{v[:200]}...\n")

            except Exception as e:
                print(f"에러 발생: {e}")
