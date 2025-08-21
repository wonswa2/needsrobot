import os, json, re, torch
import numpy as np, cv2
from typing import List, Union, Dict, Any
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
)

# ---------------------------
# 0) 프롬프트 압축 유틸 (원본 그대로)
# ---------------------------
KO_EN_MAP = {
    "체중":"weight","분석":"analysis","데이터":"data","건강":"health","측정":"measurement",
    "성분":"composition","장치":"device","반려":"pet","시스템":"system","근골격계":"musculoskeletal",
    "분포":"distribution","보행":"gait","다리에":"leg","활용":"application","평가하는":"assessment",
    "필요":"requirement","동물":"animal","연구":"research","상태":"status","시작품":"prototype",
}

def to_english_keywords(raw_keywords: List[Union[str, Dict[str, Any]]]) -> List[str]:
    if raw_keywords and isinstance(raw_keywords[0], dict):
        raw_keywords = sorted(raw_keywords, key=lambda x: x.get("score",0), reverse=True)
        words = [KO_EN_MAP.get(x.get("word","").strip(), x.get("word","").strip()) for x in raw_keywords]
    else:
        words = [KO_EN_MAP.get(str(w).strip(), str(w).strip()) for w in raw_keywords]
    seen, uniq = set(), []
    for w in words:
        if w and w not in seen:
            seen.add(w); uniq.append(w)
    return uniq

def compose_phrases(words: List[str]) -> List[str]:
    w = set(words); p=[]
    if "weight" in w and "measurement" in w: p.append("weight measurement"); w -= {"weight","measurement"}
    if "gait" in w and "analysis" in w: p.append("gait analysis"); w -= {"gait","analysis"}
    if "musculoskeletal" in w and "assessment" in w: p.append("musculoskeletal assessment"); w -= {"musculoskeletal","assessment"}
    if "pet" in w and "health" in w and "device" in w: p.append("pet health device"); w -= {"pet","health","device"}
    if "data" in w and "system" in w: p.append("data-driven system"); w -= {"data","system"}
    p.extend(sorted(w))
    return p

def clean_tokens(t:str)->str:
    t = re.sub(r"\s+"," ",t)
    return t.strip(" ,.;:|-")

def build_prompt(raw_keywords, style="industrial design sketch, product render, white background", max_words=50):
    en = to_english_keywords(raw_keywords)
    phrases = compose_phrases(en)
    kept, cnt = [], 0
    for ph in phrases:
        add = len(ph.split())
        if cnt + add > max_words: break
        kept.append(ph); cnt += add
    core = ", ".join(kept)
    prompt = f"{core}, {style}" if core else style
    return clean_tokens(prompt)

NEG_PROMPT = (
    "human, person, face, portrait, selfie, hands, body, anime, cartoon, "
    "text, watermark, logo, blurry, low quality, nsfw"
)

# ---------------------------
# 1) ControlNet Canny
# ---------------------------
def make_canny(img: Image.Image, low=100, high=200) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    edges = cv2.Canny(arr, low, high)
    edges3 = np.stack([edges]*3, axis=-1)
    return Image.fromarray(edges3)

# ---------------------------
# 2) 파이프라인 로더
# ---------------------------
def load_sd15(device="cuda"):
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device=="cuda" else torch.float32,
        safety_checker=None
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    try: pipe.enable_xformers_memory_efficient_attention()
    except Exception: pass
    return pipe

def load_sd15_canny(device="cuda"):
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch.float16 if device=="cuda" else torch.float32
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16 if device=="cuda" else torch.float32,
        safety_checker=None
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    try: pipe.enable_xformers_memory_efficient_attention()
    except Exception: pass
    return pipe

# (옵션) 제품 디자인 LoRA가 있다면 경로 지정해서 로드
def maybe_load_lora(pipe, lora_path:str|None, weight:float=0.8):
    if lora_path and os.path.exists(lora_path):
        pipe.load_lora_weights(lora_path, weight=weight)
    return pipe

# ---------------------------
# 3) 생성 함수
# ---------------------------
@torch.inference_mode()
def gen_base(pipe, prompt, neg=NEG_PROMPT, seed=123, steps=28, guidance=7.5, size=(768,768)):
    g = torch.Generator(device=pipe.device).manual_seed(seed)
    img = pipe(
        prompt=prompt, negative_prompt=neg,
        width=size[0], height=size[1],
        num_inference_steps=steps, guidance_scale=guidance,
        generator=g
    ).images[0]
    return img

@torch.inference_mode()
def gen_canny(pipe, prompt, canny_img, neg=NEG_PROMPT, seed=123, steps=28, guidance=7.5, cond=1.0, size=(768,768)):
    g = torch.Generator(device=pipe.device).manual_seed(seed)
    img = pipe(
        prompt=prompt, image=canny_img, negative_prompt=neg,
        width=size[0], height=size[1],
        num_inference_steps=steps, guidance_scale=guidance,
        controlnet_conditioning_scale=cond, generator=g
    ).images[0]
    return img

# ---------------------------
# 4) 메인 (tech_overview.json 배치)
# ---------------------------
if __name__ == "__main__":
    # (1) tech_overview.json 로드 (파일명 -> 설명 텍스트)
    TECH_PATH = "output/tech_overview.json"
    with open(TECH_PATH, "r", encoding="utf-8") as f:
        prompt_map: Dict[str, str] = json.load(f)

    # (2) 공통 설정
    MODE  = "base"   # 'base' (항목당 1장) | 'canny' (ControlNet-Canny로 1장)
    STYLE = "industrial design sketch, product render, white background"
    SEED  = 123
    SIZE  = (768, 768)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # (3) 파이프라인 1회 로드
    base_pipe = load_sd15(device)
    canny_pipe = load_sd15_canny(device) if MODE == "canny" else None

    # (4) 안전 폴더명 유틸
    def safe_name(s: str) -> str:
        s = os.path.splitext(os.path.basename(s))[0]
        return re.sub(r"[^0-9a-zA-Z가-힣_-]+", "_", s).strip("_")

    os.makedirs("sd_output", exist_ok=True)

    # (5) 항목 하나 = 이미지 한 장, 항목별 폴더 저장
    for fname, text in prompt_map.items():
        # 설명문 + 스타일 꼬리표 → 최종 프롬프트
        prompt = clean_tokens(f"{text}\n{STYLE}")

        out_dir = os.path.join("sd_output", safe_name(fname))
        os.makedirs(out_dir, exist_ok=True)
        out_img = os.path.join(out_dir, "image.png")

        if MODE == "base":
            img = gen_base(base_pipe, prompt, seed=SEED, size=SIZE)
            img.save(out_img)
        elif MODE == "canny":
            # 내부에서만 베이스 한 장 생성 → 캐니 조건으로 사용(최종 저장은 한 장)
            tmp = gen_base(base_pipe, prompt, seed=SEED, size=SIZE)
            canny_cond = make_canny(tmp)
            img = gen_canny(canny_pipe, prompt, canny_cond, seed=SEED, size=SIZE, cond=1.1)
            img.save(out_img)
        else:
            raise ValueError("MODE must be 'base' or 'canny'.")

        # (선택) 메타 저장
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as mf:
            json.dump({
                "name": fname, "mode": MODE, "seed": SEED, "size": SIZE,
                "style": STYLE, "prompt_preview": prompt[:200]
            }, mf, ensure_ascii=False, indent=2)

        print(f"Saved: {out_img}")
