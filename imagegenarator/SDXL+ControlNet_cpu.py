# SDXL + ControlNet (Canny) — CPU 버전 전체 스크립트
# 느리지만 VRAM 없이 동작. 기본 해상도/스텝은 CPU 친화적으로 낮춤.

import os, torch
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from controlnet_aux import CannyDetector

# -----------------------
# 설정
# -----------------------
DEVICE = "cpu"  # ★ CPU 고정
OUTDIR = "output/SDXL_ControlNet_CPU"
os.makedirs(OUTDIR, exist_ok=True)

# CPU 성능(스레드) 조정: 필요시 숫자 변경
try:
    torch.set_num_threads(max(1, os.cpu_count() // 2))
except Exception:
    pass

# -----------------------
# 1) SDXL + ControlNet(Canny) 로드 (CPU/FP32)
# -----------------------
cn = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float32
)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=cn,
    torch_dtype=torch.float32,
    use_safetensors=True,
    add_watermarker=False,
    # safety_checker=None,  # 일부 버전에선 허용. 가능하면 켜서 메모리/속도 조금 절약
)
# CPU로 이동
pipe.to(DEVICE)

# CPU에서도 약간의 메모리 절감에 도움 되는 옵션들 (효과 제한적)
try:
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
except Exception:
    pass

# -----------------------
# 2) 조건맵: 스케치가 없을 때 흰 배경 → Canny
#    (CPU에서 1024는 매우 느림 → 기본 768 권장)
# -----------------------
BASE_W, BASE_H = 640, 512  # ★ CPU 권장 해상도 (느리면 640/512로 낮추기)
canny = CannyDetector()
blank = Image.new("RGB", (BASE_W, BASE_H), "white")
cond = canny(blank, low_threshold=100, high_threshold=200)

# -----------------------
# 3) 설계 브리프(네가 준 문장들 그대로)
# -----------------------
briefs = {
  "ai_som_devices.json": "\"IF (인공지능 기반의 인공신경망) 기술을 활용한 뉴로모픽 디바이스는 인체의 신경계와 유사한 구조를 가진 인공 신경망을 사용하여, 인간의 뇌 기능을 모방하는 목적을 가지고 있습니다. 이 디바이스는 Surface Mount Technology(SMT) 방식을 통해 소자를 최소화하고, JCR(Journal of Clinical Research) 기준을 준수하여 안전성을 보장합니다.\"\n\n\"인공 신경계 디바이스는 컴퓨팅 파워를 활용하여 AI를 적용하여, 인지 기능을 향상시키는 데 중점을 둡니다. SOM(Self-Organizing Map) 알고리즘을 통해 데이터를 분석하고, JC(Journal of Computing) 연구 결과를 바탕으로 최적화된 디바이스를 제작합니다.\"\n\n\"뉴로모픽 디바이스는 실험실 연구에서 시작하여, 창업 및 산업 적용을 목표로 합니다. AI 활용을 통해 다양한 활용 사례를 모색하고, 필요에 맞는 구현 방안을 모색하여 제품화에 도전합니다.\"",
  "info_total_service.json": "설계 도안 준비 전에 참고할 수 있는 전문가 코멘트:\n\"이 솔루션은 스마트 안전을 중심으로 한 인프라를 구축하여, 현장에서 실시간으로 데이터를 수집하고 분석하여 위험 상황을 사전에 모니터링하고 대응할 수 있는 DX(Digital Transformation)를 가능하게 합니다. 이를 위해 Twin 기술과 BIM을 활용하여, 구조물의 레이아웃과 형상을 정밀하게 모듈화하고, 센서를 통해 실시간 정보를 수집하여 안전한 인프라 기반을 마련합니다. 또한, 이를 통해 사업 효율성을 크게 향상시키는 서비스를 제공할 수 있습니다.\"",
  "smart_pet_machine.json": "체중 분석을 위한 시스템은 동물의 건강 상태를 평가하는 데 중요한 역할을 합니다. 센싱지표를 통해 체중 데이터를 실시간으로 측정하고, 근골격계와 신경의 상태를 분석하여 건강한 보행 패턴을 파악합니다. 이를 통해 다리 하중의 분포를 평가하고, 체중 변화에 따른 건강 상태를 정확히 분석할 수 있습니다.",
  "smart_skincare_machine.json": "\"플라즈마를 이용한 미용 디바이스는 피부와의 상호작용을 통해 안티에이징 효과를 극대화합니다. 2013년 연구 결과를 바탕으로 제작된 이 디바이스는 10년 넘게 안면 시술에 널리 사용되어 왔으며, 연평균 20%의 피부 개선 효과를 보입니다. 이는 기존 기타 미용 장치와 구분되는 차별점입니다.\"",
  "smart_tree_system.json": "\"이 시스템은 순화된 농장 환경을 목표로 하고 있으며, 퍼머컬쳐 원칙을 적용하여 최적의 생육 상태를 촉진합니다. 실내 환경에서 부정근을 최소화하고, AI를 활용한 배양 시스템을 통해 식물의 상태를 조절하고 필요에 맞게 배양액을 제공합니다. 이를 통해 도시 농장에서도 높은 생산성을 확보할 수 있습니다.\""
}

STYLE = (
    "technical drawing, blueprint sheet, clean black lineart on white, "
    "orthographic projection (front, side, top), exploded view and block diagram, "
    "annotations and callouts, minimal shading, vector-like edges"
)
FOCUS = (
    "show functional blocks and connections, sensors and actuators placement, "
    "data flow arrows, component labels, scale bars, dimension placeholders"
)
NEG = "color, gradient, texture, heavy shading, photorealistic, blur, noise"

# -----------------------
# 4) 생성 파라미터 (CPU 친화값)
# -----------------------
NUM_STEPS = 12          # ★ 10~16 권장 (느리면 더 낮추기)
GUIDANCE = 4.5
COND_SCALE = 0.85

for name, brief in briefs.items():
    prompt = f"[DESIGN BRIEF] {brief}\n[DRAW AS] {STYLE}\n[FOCUS] {FOCUS}"
    image = pipe(
        prompt=prompt,
        negative_prompt=NEG,
        image=cond,
        controlnet_conditioning_scale=COND_SCALE,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE,
        width=BASE_W, height=BASE_H
    ).images[0]
    image.save(os.path.join(OUTDIR, f"{name}.png"))
