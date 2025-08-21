# Stable Diffusion 1.5 + ControlNet(Canny) — CPU (token-aware, compact prompts)
import os, torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import CannyDetector

DEVICE = "cpu"
OUTDIR = "output/SD1.5_ControlNet_CPU"
os.makedirs(OUTDIR, exist_ok=True)

# CPU 스레드(원하면 조정)
try:
    torch.set_num_threads(max(1, (os.cpu_count() or 8) // 2))
except Exception:
    pass

# 1) 경량 모델 로드 (fp16로 메모리 절약)
base = "runwayml/stable-diffusion-v1-5"
ctrl = "lllyasviel/sd-controlnet-canny"

controlnet = ControlNetModel.from_pretrained(ctrl, torch_dtype=torch.float32)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base,
    controlnet=controlnet,
    torch_dtype=torch.float32,   # CPU에서도 메모리 절약 (느리면 그대로)
    safety_checker=None,
    use_safetensors=True,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(DEVICE)
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

# 2) 조건맵: 흰 캔버스 → Canny (원본 스케치 있으면 그걸 넣기)
W, H = 512, 512  # 느리면 448/384
canny = CannyDetector()
cond = canny(Image.new("RGB", (W, H), "white"), low_threshold=100, high_threshold=200)

# 3) 압축 프롬프트 (77토큰 이하)
# 3) 초압축 프롬프트 (77 토큰 이하)
STYLE = "technical blueprint, black line art, orthographic views, exploded, block diagram, annotations"
FOCUS = "connections, sensors, actuators, data arrows, labels"
NEG = "color, gradient, texture, heavy shading, photorealistic, blur, noise"

briefs = {
    "smart_pet_machine.json":
        "real-time weight sensing, musculoskeletal state, neural state, "
        "gait pattern, leg load distribution, weight variation, health status",
}

STEPS = 16
GUIDE  = 6.0
CSCALE = 0.85

for name, brief in briefs.items():
    prompt = f"{brief}, {STYLE}, {FOCUS}"  # ← 대괄호/콜론 없이, 키워드만
    img = pipe(
        prompt=prompt,
        negative_prompt=NEG,
        image=cond,
        controlnet_conditioning_scale=CSCALE,
        num_inference_steps=STEPS,
        guidance_scale=GUIDE,
        width=W, height=H,
    ).images[0]
    img.save(os.path.join(OUTDIR, f"{name}.png"))
