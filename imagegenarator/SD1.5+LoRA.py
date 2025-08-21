# -*- coding: utf-8 -*-
"""
Stable Diffusion 1.5 + 경량 LoRA 자동 파이프라인 (4GB VRAM 친화)
- 문맥 문장 + 키워드(가중치) → SD 친화 프롬프트 자동 생성
- 공통 네거티브 프롬프트 자동 적용
- 선택: ControlNet(Canny)로 구조만 살짝 고정
- 결과 위에 제목/요약 bullets를 Pillow로 자동 오버레이

입력:
  - 문장:  output/tech_overview.json   # { "filename.json": "한국어 문장", ... }
  - 키워드: output/keywords/{basename}.json  # [{ "word": "...", "score": ... }, ...]  (없어도 OK)

출력:
  - sd_output/{basename}/image.png (+ meta.json)

필요 패키지:
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # (GPU 환경)
  pip install diffusers transformers accelerate xformers safetensors opencv-python pillow
"""

import os, re, json, math, random
from typing import List, Dict, Any, Tuple, Optional

import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
)

# =========================
# 0) 전역 설정 (4GB VRAM 안정값)
# =========================
SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"

# LoRA 파일 경로와 가중치(원하면 비워두기)
# 예시는 두 개만 권장: 스타일 1 + 라인아트 1
LORAS: Dict[str, float] = {
    # "loras/product_design_sketch.safetensors": 0.8,
    # "loras/lineart_clean.safetensors": 0.7,
}

# 생성 모드: "base" | "canny"
MODE = "base"       # 먼저 base로 확인, 필요 시 "canny"
SEED = 123
SIZE = (640, 480)   # 4GB 권장 해상도
STEPS = 24
GUIDE = 6.0

# 입력 경로
SENTENCE_JSON = "output/tech_overview.json"   # 파일명 → 한국어 문장
KEYWORD_DIR   = "output/KR_WordRank_case/json/G_smart_pet_machine.json"             # 파일별 키워드 json이 있으면 사용

# 출력 경로
OUT_DIR = "sd_output"

# =========================
# 1) 유틸
# =========================
def clean_tokens(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"[ ,;|]+$", "", t)
    return t

def safe_name(s: str) -> str:
    s = os.path.splitext(os.path.basename(s))[0]
    s = re.sub(r"[^0-9a-zA-Z가-힣_-]+", "_", s).strip("_")
    return s or "item"

def make_canny(img: Image.Image, low: int = 100, high: int = 200) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    edges = cv2.Canny(arr, low, high)
    edges3 = np.stack([edges] * 3, axis=-1)
    return Image.fromarray(edges3)

def split_sentences_korean(s: str) -> List[str]:
    # 발표 bullets로 간단 분할
    parts = re.split(r"[.!?。…\n]+", s)
    parts = [p.strip(" \"“)”)[]") for p in parts if len(p.strip()) >= 5]
    return parts

# =========================
# 2) 프롬프트 자동 생성
# =========================
STYLE_TAGS = "product design sketch, industrial concept render, clean white background, isometric view, lineart emphasis, soft shadow"
NEG_PROMPT_COMMON = (
    "text, letters, words, numbers, watermark, logo, caption, signature, "
    "human, person, people, face, hands"
    "anime, cartoon, comic, blurry, low quality, noisy, cluttered background, presentation slide, document"
)

BAN_SET = {"사람", "인체",}  # 그림에 불필요한 실물 등장 방지

# 한국어 키워드 → 영문(간단 맵)
KO_EN = {
    "체중": "weight",
    "분석": "analysis",
    "데이터": "data",
    "건강": "health",
    "측정": "measurement",
    "성분": "composition",
    "장치": "device",
    "시스템": "system",
    "근골격계": "musculoskeletal",
    "분포": "distribution",
    "보행": "gait",
    "다리에": "leg",
    "평가하는": "assessment",
    "상태": "status",
    "시작품": "prototype",
    "필요": "requirement",
    "연구": "research",
    "활용": "application",
}

def load_keywords_for_file(basename: str) -> Optional[List[Dict[str, Any]]]:
    """
    output/KR_WordRank_case/json/G_smart_pet_machine.json 이 존재하면 로드
    형태: [{"word": "체중", "score": 4.58}, ...]
    """
    path = os.path.join(KEYWORD_DIR, f"{basename}.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return None

def simple_keywords_from_sentence(sentence: str, max_n: int = 8) -> List[Dict[str, Any]]:
    # 키워드 파일이 없을 때, 아주 단순 추출(명사/숫자/영문 토큰 느낌)
    tokens = re.findall(r"[A-Za-z0-9가-힣]{2,}", sentence)
    # 빈도 기반 상위 n
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    items = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:max_n]
    return [{"word": w, "score": float(c)} for w, c in items]

def build_prompts(sentence_ko: str, keywords_scored: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    - 문맥 문장(한국어)은 영문 주체 문장으로 고정(장치 중심)
    - 키워드 상위 3개 가중치 + 나머지 5개 정도
    - 공통 스타일/네거티브
    """
    # 1) 키워드 정리(스코어 내림차순)
    kk = []
    for d in sorted(keywords_scored, key=lambda x: x.get("score", 0.0), reverse=True):
        w = str(d.get("word", "")).strip()
        if not w or w in BAN_SET:
            continue
        kk.append(KO_EN.get(w, w))  # 간단 영문화

    boosted = []
    for i, w in enumerate(kk[:3]):
        boosted.append(f"({w}:1.2)")
    rest = [w for w in kk[3:8] if w]  # 나머지는 5개 제한
    kw_block = ", ".join(boosted + rest) if (boosted or rest) else ""

    # 2) 한국어 문맥 → 영문 주체 문장(도메인 중립, 장치 중심)
    sentence_en = (
        "A veterinary health assessment device that measures weight in real time, "
        "analyzes musculoskeletal and neural condition, and evaluates gait patterns and leg load distribution"
    )

    pos = f"{sentence_en}"
    if kw_block:
        pos += f", {kw_block}"
    pos += f", {STYLE_TAGS}"
    return clean_tokens(pos), NEG_PROMPT_COMMON

# =========================
# 3) 파이프라인 로더 + LoRA/ControlNet
# =========================
def load_sd15(device: str = "cuda"):
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        SD_MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    return pipe

def load_sd15_canny(device: str = "cuda"):
    dtype = torch.float16 if device == "cuda" else torch.float32
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=dtype
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SD_MODEL_ID,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    return pipe

def load_loras(pipe, loras: Dict[str, float]):
    """
    Diffusers 최신 기준 로딩. 버전에 따라 adapter API가 없을 수 있어 예외 처리.
    """
    if not loras:
        return pipe
    adapter_names, weights = [], []
    for i, (path, w) in enumerate(loras.items()):
        if not os.path.exists(path):
            print(f"[WARN] LoRA not found: {path}")
            continue
        name = f"lora_{i}"
        try:
            pipe.load_lora_weights(path, adapter_name=name)
            adapter_names.append(name); weights.append(float(w))
        except Exception as e:
            # 구버전 호환(단일 weight 로딩)
            try:
                pipe.load_lora_weights(path, weight=float(w))
                adapter_names.append(name); weights.append(float(w))
            except Exception as e2:
                print(f"[WARN] LoRA load failed: {path} ({e2})")
    try:
        if adapter_names:
            pipe.set_adapters(adapter_names, adapter_weights=weights)
    except Exception:
        pass
    return pipe

# =========================
# 4) 생성 함수
# =========================
@torch.inference_mode()
def gen_base(pipe, prompt, neg, seed=SEED, steps=STEPS, guidance=GUIDE, size=SIZE) -> Image.Image:
    g = torch.Generator(device=pipe.device).manual_seed(seed)
    out = pipe(
        prompt=prompt,
        negative_prompt=neg,
        width=size[0], height=size[1],
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=g
    )
    return out.images[0]

@torch.inference_mode()
def gen_canny(
    pipe, prompt, canny_img, neg,
    seed=SEED, steps=STEPS, guidance=GUIDE, size=SIZE, cond=0.55,
    c_start=0.0, c_end=0.8
) -> Image.Image:
    g = torch.Generator(device=pipe.device).manual_seed(seed)
    kwargs = dict(
        prompt=prompt, negative_prompt=neg, image=canny_img,
        width=size[0], height=size[1],
        num_inference_steps=steps, guidance_scale=guidance,
        controlnet_conditioning_scale=cond, generator=g
    )
    # 일부 버전만 지원됨: control_guidance_start/end
    try:
        kwargs["control_guidance_start"] = c_start
        kwargs["control_guidance_end"] = c_end
    except Exception:
        pass
    out = pipe(**kwargs)
    return out.images[0]

# =========================
# 5) 텍스트 오버레이 (발표용)
# =========================
def overlay_text(img: Image.Image, title: str, bullets: List[str]) -> Image.Image:
    img = img.convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)
    # 폰트
    try:
        font_title = ImageFont.truetype("arial.ttf", size=max(20, W // 18))
        font_body  = ImageFont.truetype("arial.ttf", size=max(14, W // 34))
    except Exception:
        font_title = ImageFont.load_default()
        font_body  = ImageFont.load_default()

    pad = max(20, W // 40)
    panel_w = int(W * 0.44)
    panel = Image.new("RGBA", (panel_w, H), (20, 28, 40, 180))
    img.paste(panel, (W - panel_w, 0), panel)

    x = W - panel_w + pad
    y = pad
    draw.text((x, y), title, fill=(255, 255, 255), font=font_title)
    y += int(font_title.size * 1.4)

    for b in bullets[:6]:
        draw.text((x, y), f"• {b}", fill=(220, 230, 240), font=font_body)
        y += int(font_body.size * 1.35)

    return img

# =========================
# 6) 메인
# =========================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT_DIR, exist_ok=True)

    # (1) 입력 문장 맵 로드
    if not os.path.exists(SENTENCE_JSON):
        raise FileNotFoundError(f"Not found: {SENTENCE_JSON}")
    with open(SENTENCE_JSON, "r", encoding="utf-8") as f:
        sentence_map: Dict[str, str] = json.load(f)

    # (2) 파이프라인 1회 로드 + LoRA
    base_pipe = load_sd15(device)
    canny_pipe = load_sd15_canny(device) if MODE == "canny" else None

    if LORAS:
        base_pipe = load_loras(base_pipe, LORAS)
        if canny_pipe:
            canny_pipe = load_loras(canny_pipe, LORAS)

    # (3) 항목별 생성
    for fname, sentence_ko in sentence_map.items():
        base = safe_name(fname)
        out_dir = os.path.join(OUT_DIR, base)
        os.makedirs(out_dir, exist_ok=True)
        out_img = os.path.join(out_dir, "image.png")

        # 키워드가 있으면 사용
        kw = load_keywords_for_file(base)
        if kw is None:
            kw = simple_keywords_from_sentence(sentence_ko, max_n=8)

        pos_prompt, neg_prompt = build_prompts(sentence_ko, kw)

        # 생성
        if MODE == "base":
            img = gen_base(base_pipe, pos_prompt, neg=neg_prompt)
        elif MODE == "canny":
            # 먼저 베이스 1장 → Canny 조건으로 구조 가이드 (초중반만)
            tmp = gen_base(base_pipe, pos_prompt, neg=neg_prompt)
            canny_cond = make_canny(tmp)
            img = gen_canny(canny_pipe, pos_prompt, canny_cond, neg=neg_prompt,
                            cond=0.55, c_start=0.0, c_end=0.8)
        else:
            raise ValueError("MODE must be 'base' or 'canny'.")

        # (선택) 텍스트 오버레이
        bullets = split_sentences_korean(sentence_ko)[:5]
        title = base.replace("_", " ").upper()
        img2 = overlay_text(img, title, bullets)
        img2.save(out_img, quality=95)

        # 메타
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as mf:
            json.dump({
                "file": fname,
                "mode": MODE,
                "seed": SEED,
                "size": SIZE,
                "steps": STEPS,
                "guidance": GUIDE,
                "pos_prompt": pos_prompt[:500],
                "neg_prompt": neg_prompt[:500],
                "loras": LORAS
            }, mf, ensure_ascii=False, indent=2)

        print(f"[OK] Saved: {out_img}")

if __name__ == "__main__":
    main()
