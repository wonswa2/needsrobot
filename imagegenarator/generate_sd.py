import json, os, torch, re
from PIL import Image
from typing import List, Union, Dict, Any
import numpy as np
import cv2
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel, DPMSolverMultistepScheduler

# =====================
# 0) 프롬프트 구성 유틸
# =====================

# 한국어 키워드 -> 영어 압축 매핑 (필요 시 계속 추가)
KO_EN_MAP = {
    "체중": "weight",
    "분석": "analysis",
    "데이터": "data",
    "건강": "health",
    "측정": "measurement",
    "성분": "composition",
    "장치": "device",
    "반려": "pet",
    "시스템": "system",
    "근골격계": "musculoskeletal",
    "분포": "distribution",
    "보행": "gait",
    "다리에": "leg",
    "활용": "application",
    "평가하는": "assessment",
    "필요": "requirement",
    "동물": "animal",
    "연구": "research",
    "상태": "status",
    "시작품": "prototype",
}

# 간단한 구문 합치기 규칙 (상위 키워드 몇 개를 짧은 구문으로 묶어 토큰 절약)
def compose_compact_phrases(words: List[str]) -> List[str]:
    wset = set(words)
    phrases = []

    # 자주 쓰는 조합 우선 배치
    if "weight" in wset and "measurement" in wset:
        phrases.append("weight measurement")
        wset.discard("weight"); wset.discard("measurement")
    if "gait" in wset and "analysis" in wset:
        phrases.append("gait analysis")
        wset.discard("gait"); wset.discard("analysis")
    if "musculoskeletal" in wset and ("assessment" in wset or "analysis" in wset):
        phrases.append("musculoskeletal assessment")
        wset.discard("musculoskeletal"); wset.discard("assessment"); wset.discard("analysis")
    if "health" in wset and "pet" in wset:
        phrases.append("pet health device")
        wset.discard("health"); wset.discard("pet"); wset.discard("device")
    if "data" in wset and "system" in wset:
        phrases.append("data-driven system")
        wset.discard("data"); wset.discard("system")

    # 남은 단어는 단독으로 추가
    phrases.extend(sorted(wset))
    return phrases

def to_english_keywords(raw_keywords: List[Union[str, Dict[str, Any]]]) -> List[str]:
    # 입력이 [{"word": "...", "score": ...}, ...] 또는 ["...", "..."] 둘 다 지원
    if raw_keywords and isinstance(raw_keywords[0], dict):
        # 점수 내림차순으로 핵심 먼저 배치
        raw_keywords = sorted(raw_keywords, key=lambda x: x.get("score", 0), reverse=True)
        words = [KO_EN_MAP.get(x.get("word", "").strip(), x.get("word", "").strip()) for x in raw_keywords]
    else:
        words = [KO_EN_MAP.get(str(w).strip(), str(w).strip()) for w in raw_keywords]

    # 중복 제거(순서 보존)
    seen = set()
    uniq = []
    for w in words:
        if not w or w in seen:
            continue
        seen.add(w)
        uniq.append(w)
    return uniq

def clean_tokens(text: str) -> str:
    # 불필요한 연속 공백/구두점 축소
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" ,.;:|-")
    return text

def build_compact_prompt_from_keywords(raw_keywords: List[Union[str, Dict[str, Any]]],
                                       style: str = "industrial design sketch",
                                       max_words: int = 55) -> str:
    """
    - 한국어 키워드를 영어로 압축
    - 핵심(점수 상위) 먼저
    - 짧은 구문으로 결합
    - 스타일 꼬리표 최소화
    - 대략 단어 수 기반 길이 캡(77 토큰 대비 보수적으로 55 단어 정도로 컷)
    """
    en_words = to_english_keywords(raw_keywords)
    phrases = compose_compact_phrases(en_words)

    # 단어 수 제한 (대략적으로 토큰 77 이내를 노리는 보수 컷)
    # 각 구문을 공백 분할해서 누적 단어 수를 세며 자름
    kept = []
    count = 0
    for p in phrases:
        add = len(p.split())
        if count + add > max_words:
            break
        kept.append(p)
        count += add

    prompt_core = ", ".join(kept)
    prompt = f"{prompt_core}, {style}" if prompt_core else style
    return clean_tokens(prompt)

# =====================
# 1) Canny 생성
# =====================
def make_canny_condition(img: Image.Image, low_threshold=100, high_threshold=200) -> Image.Image:
    np_img = np.array(img.convert("RGB"))
    edges = cv2.Canny(np_img, low_threshold, high_threshold)
    edges_3 = np.stack([edges]*3, axis=-1)
    return Image.fromarray(edges_3)

# =====================
# 2) 파이프라인 로더
# =====================
def load_base_pipeline(device="cuda"):
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    return pipe

def load_canny_pipeline(device="cuda"):
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    return pipe

# =====================
# 3) 생성 함수
# =====================
@torch.inference_mode()
def generate_base(pipe, prompt, seed=42, guidance_scale=7.5, steps=28):
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    image = pipe(
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        generator=generator
    ).images[0]
    return image

@torch.inference_mode()
def generate_canny(pipe, prompt, canny_img, seed=42, guidance_scale=7.5, steps=28, cond_scale=1.0):
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    image = pipe(
        prompt=prompt,
        image=canny_img,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        controlnet_conditioning_scale=cond_scale,
        generator=generator
    ).images[0]
    return image

# =====================
# 4) 메인 실행
# =====================
if __name__ == "__main__":
    # 1) JSON에서 키워드 로드
    with open("output/KR_WordRank_case/json/G_smart_pet_machine.json", encoding="utf-8") as f:
        data = json.load(f)
    raw_keywords = data["keywords"]  # [{"word":..., "score":...}, ...] 형태

    # 2) 압축 프롬프트 생성 (핵심 먼저, 영어 압축, 구두점 최소화, 스타일 압축)
    prompt = build_compact_prompt_from_keywords(raw_keywords, style="industrial design sketch", max_words=55)
    print("PROMPT:", prompt)
    print("~ rough word count:", len(prompt.split()))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 3) 베이스 이미지
    base_pipe = load_base_pipeline(device=device)
    base_img = generate_base(base_pipe, prompt, seed=123, guidance_scale=7.5, steps=28)
    os.makedirs("sd_output", exist_ok=True)
    base_path = "sd_output/base.png"
    base_img.save(base_path)
    print("Saved:", base_path)

    # 4) Canny 컨디션
    canny_img = make_canny_condition(base_img)
    canny_path = "sd_output/canny_condition.png"
    canny_img.save(canny_path)

    # 5) ControlNet-Canny 이미지
    canny_pipe = load_canny_pipeline(device=device)
    sketch_img = generate_canny(canny_pipe, prompt, canny_img, seed=123, guidance_scale=7.5, steps=28, cond_scale=1.0)
    sketch_path = "sd_output/sketch.png"
    sketch_img.save(sketch_path)
    print("Saved:", sketch_path)
