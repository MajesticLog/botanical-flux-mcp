"""Botanical Flux MCP server: drive ComfyUI (FLUX.1 [dev] + per-species LoKr / LoRA) from an AI assistant.

Tools:
  - species()          list the species, their stages / seasons and the vocabulary the models were trained on
  - generate(...)      one image of a plant, from scratch (txt2img)
  - plant(...)         grow a species on your own photo, inside the white area of a mask (inpainting)
  - make_autumn(...)   Boston ivy facade: summer -> autumn, same building (foliage mask computed automatically)

Requires a running ComfyUI server (core nodes only) with FLUX.1 [dev] and the models from
https://huggingface.co/collections/Potiron/botanical-lora-lokr-6adf7e67b920a405a2ed33ee
Settings: environment variables, see README.md.
"""
import os

os.environ.pop("SSLKEYLOGFILE", None)  # some antivirus tools inject it; it crashes OpenSSL in Python on Windows

import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import numpy as np
from PIL import Image as PILImage, ImageFilter
from mcp.server.mcpserver import Image, MCPServer

# --- Settings (environment variables) -------------------------------------------------------------
COMFY = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
LORA_DIR = os.environ.get("BOTANICAL_LORA_SUBFOLDER", "botanical")  # subfolder of ComfyUI/models/loras
OUTPUT_DIR = Path(os.environ.get("BOTANICAL_OUTPUT_DIR", Path(__file__).parent / "outputs"))
UNET = os.environ.get("FLUX_UNET", "flux1-dev.safetensors")
CLIP_L = os.environ.get("FLUX_CLIP_L", "clip_l.safetensors")
T5 = os.environ.get("FLUX_T5", "t5xxl_fp8_e4m3fn.safetensors")
VAE = os.environ.get("FLUX_VAE", "ae.safetensors")

# --- Species ----------------------------------------------------------------------------------------
# Prompts follow the order and vocabulary of the training captions: a prompt that reuses the caption
# words works much better than free text. One template for the climbing plants (facades), and each
# species can bring its own template.
CLIMBER = ("{trigger}, {name} {stage}. {coverage}, mature growth, {posture}. "
           "Setting: {setting}. {light}, {framing}. No people.")
BUSH = ("{trigger}, {name} {stage}. "
           "Background: {setting}. {light}, {framing}. No people.")

SPECIES = {
    "lavender": {
        "trigger": "lvnd_zc", "name": "English lavender (Lavandula angustifolia)", "file": "lavender_lvnd_zc",
        "template": BUSH,
        "stages": {
            "bloom": "in bloom, with many purple flower spikes and tight flower buds",
            "foliage": "in leaf, without flowers, with silvery-grey narrow leaves and woody stems",
        },
        "defaults": {"framing": "close-up", "light": "Sunny", "size": 768},
    },
    "wisteria": {
        "trigger": "glcn_zc", "name": "a wisteria vine", "file": "wisteria_glcn_zc",
        "template": CLIMBER,
        "stages": {
            "bloom": "in bloom, with lavender flowers, hanging flower clusters, long racemes, "
                     "yellow-green foliage and pinnate leaves, twisted woody trunk",
            "foliage": "in leaf, with green foliage, pinnate leaves and twining tendrils",
        },
        "defaults": {"posture": "climbing upward and cascading downward", "framing": "medium shot"},
    },
    "clematis": {
        "trigger": "clmts_zc", "name": "an evergreen clematis (Clematis armandii)", "file": "clematis_armandii_clmts_zc",
        "template": CLIMBER,
        "stages": {
            "bloom": "in bloom, with white flowers, star-shaped flowers, flower clusters and long narrow leaves",
            "foliage": "in leaf, with dark green foliage, long narrow leaves and leathery leaves",
        },
        "defaults": {"posture": "cascading downward", "coverage": "Full coverage", "framing": "medium shot"},
    },
    "boston_ivy": {
        "trigger": "vgn3vrs_zc", "name": "a Boston ivy vine (Parthenocissus)", "file": "boston_ivy_vgn3vrs_zc",
        "template": CLIMBER,
        "stages": {
            "summer": "in summer, with dark green foliage",
            "autumn": "in autumn, with red foliage and green foliage",
        },
    },
    "english_ivy": {
        "trigger": "lrre_zc", "name": "English ivy (Hedera helix)", "file": "english_ivy_lrre_zc",
        "template": CLIMBER,
        "stages": {"evergreen": "in leaf, with dark green foliage, lobed leaves and thick woody trunk"},
        # tested: on wide facades the LoRA climbs around doors, the LoKr tends to draw a tree trunk
        "model": "lora",
    },
}
CLIMBER_DEFAULTS = {"coverage": "Medium coverage", "posture": "climbing upward, clinging directly and framing windows",
                    "framing": "wide shot", "light": "Overcast", "size": 1024}
ALIASES = {"glycine": "wisteria", "clematite": "clematis", "clématite": "clematis", "vigne vierge": "boston_ivy",
           "vigne_vierge": "boston_ivy", "boston ivy": "boston_ivy", "lierre": "english_ivy", "ivy": "english_ivy",
           "english ivy": "english_ivy", "flowers": "bloom", "flowering": "bloom", "leaf": "foliage",
           "leaves": "foliage", "fall": "autumn",
           "fleurs": "bloom", "floraison": "bloom", "feuilles": "foliage", "feuillage": "foliage",
           "été": "summer", "ete": "summer", "automne": "autumn", "lavande":"lavender"}

server = MCPServer(
    "botanical-flux",
    instructions="Generates plants by species and season / growth stage with FLUX.1 [dev] and one fine-tuned "
                 "LoKr or LoRA per species, through ComfyUI. Call species() first for the vocabulary. "
                 "FLUX.1 [dev] is non-commercial: demo images only.",
)


def _resolve(species: str, stage: str | None):
    key = species.lower().strip()
    key = ALIASES.get(key, key).replace(" ", "_")
    if key not in SPECIES:
        raise ValueError(f"Unknown species: {species}. Available: {', '.join(SPECIES)}")
    sp = SPECIES[key]
    st = (stage or "").lower().strip()
    st = ALIASES.get(st, st) or next(iter(sp["stages"]))
    if st not in sp["stages"]:
        raise ValueError(f"Unknown stage for {key}: {stage}. Available: {', '.join(sp['stages'])}")
    return key, sp, st


def _setting(sp, name):
    """Species value, else climbing-plant default (the climbers share the same defaults)."""
    return sp.get("defaults", {}).get(name, CLIMBER_DEFAULTS.get(name))


def _prompt(sp, stage, setting, **override):
    fields = {k: override.get(k) or _setting(sp, k) for k in ("coverage", "posture", "framing", "light")}
    return sp["template"].format(trigger=sp["trigger"], name=sp["name"], stage=sp["stages"][stage],
                                 setting=setting, **fields)


def _model_file(sp, model):
    kind = (model or sp.get("model", "lokr")).lower()
    if kind not in ("lokr", "lora"):
        raise ValueError("model must be 'lokr' or 'lora'")
    return kind, f"{LORA_DIR}/{sp['file']}_{kind}.safetensors" if LORA_DIR else f"{sp['file']}_{kind}.safetensors"


def _base(lora_name, weight, prompt, guidance):
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "fp8_e4m3fn"}},
        "2": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": CLIP_L, "clip_name2": T5,
              "type": "flux", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "4": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": lora_name, "strength_model": weight,
              "model": ["1", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "7": {"class_type": "FluxGuidance", "inputs": {"guidance": guidance, "conditioning": ["6", 0]}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
    }


def _ksampler(latent, seed, denoise, model=("4", 0)):
    # settings validated in the training tests: 28 steps, euler / simple, CFG 1 (Flux uses FluxGuidance instead)
    return {"class_type": "KSampler", "inputs": {"seed": seed, "steps": 28, "cfg": 1.0, "sampler_name": "euler",
            "scheduler": "simple", "denoise": denoise, "model": list(model), "positive": ["7", 0],
            "negative": ["8", 0], "latent_image": latent}}


def _fix_lora_name(wf):
    """ComfyUI lists models with the OS path separator: use the exact name it knows."""
    wanted = wf["4"]["inputs"]["lora_name"]
    try:
        info = json.load(urllib.request.urlopen(f"{COMFY}/object_info/LoraLoaderModelOnly", timeout=30))
        names = info["LoraLoaderModelOnly"]["input"]["required"]["lora_name"][0]
    except Exception:
        return
    for n in names:
        if n.replace("\\", "/") == wanted:
            wf["4"]["inputs"]["lora_name"] = n
            return
    raise FileNotFoundError(f"ComfyUI does not see the model {wanted}. Download it from Hugging Face into "
                            f"ComfyUI/models/loras/{LORA_DIR}/ (see README) and refresh ComfyUI.")


def _upload(img: PILImage.Image, name: str) -> str:
    """Send an image to ComfyUI's input folder through its API (no shared disk needed)."""
    buf = io.BytesIO()
    img.save(buf, "PNG")
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            f"Content-Type: image/png\r\n\r\n").encode() + buf.getvalue() + (
            f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n"
            f"--{boundary}--\r\n").encode()
    req = urllib.request.Request(f"{COMFY}/upload/image", body,
                                 {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    rep = json.load(urllib.request.urlopen(req, timeout=60))
    return f"{rep['subfolder']}/{rep['name']}" if rep.get("subfolder") else rep["name"]


def _run(wf: dict, out_name: str) -> Path:
    _fix_lora_name(wf)
    req = urllib.request.Request(f"{COMFY}/prompt", json.dumps({"prompt": wf}).encode(),
                                 {"Content-Type": "application/json"})
    try:
        rep = json.load(urllib.request.urlopen(req, timeout=30))
    except urllib.error.URLError as err:
        body = err.read().decode() if isinstance(err, urllib.error.HTTPError) else str(err)
        raise RuntimeError(f"ComfyUI is not reachable at {COMFY} or rejected the workflow ({body[:500]}). "
                           "Start ComfyUI or set COMFYUI_URL.") from err
    if rep.get("node_errors"):
        raise RuntimeError(f"Workflow error: {json.dumps(rep['node_errors'])[:800]}")
    pid, start = rep["prompt_id"], time.time()
    while time.time() - start < 900:
        time.sleep(2)
        hist = json.load(urllib.request.urlopen(f"{COMFY}/history/{pid}", timeout=30))
        if pid not in hist:
            continue
        h = hist[pid]
        if h["status"].get("status_str") != "success":
            raise RuntimeError(f"Generation failed: {json.dumps(h['status'])[:800]}")
        im = h["outputs"]["save"]["images"][0]
        q = urllib.parse.urlencode({"filename": im["filename"], "subfolder": im["subfolder"], "type": im["type"]})
        data = urllib.request.urlopen(f"{COMFY}/view?{q}", timeout=60).read()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUTPUT_DIR / f"{out_name}_{time.strftime('%Y%m%d_%H%M%S')}.png"
        out.write_bytes(data)
        return out
    raise TimeoutError("Generation took more than 15 min: is the GPU busy?")


def _preview(path: Path) -> Image:
    im = PILImage.open(path).convert("RGB")
    im.thumbnail((768, 768))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return Image(data=buf.getvalue(), format="jpeg")


def _open(path: str) -> PILImage.Image:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return PILImage.open(p)


@server.tool()
def species() -> str:
    """List the available species, their stages or seasons, trigger words and default model."""
    lines = []
    for key, sp in SPECIES.items():
        kind, _ = _model_file(sp, None)
        lines.append(f"- {key} (trigger {sp['trigger']}, default model {kind}): stages " +
                     ", ".join(f"{s} = \"{d}\"" for s, d in sp["stages"].items()))
    return ("Species (FLUX.1 [dev] + one LoKr / LoRA per species):\n" + "\n".join(lines) +
            "\nClimbing plants - settings seen in training: brick wall, plaster wall, stone wall, pergola, fence, "
            "residential house... Coverage: Sparse / Medium / Dense / Full coverage. "
            "Framing: close-up, medium shot, wide shot.")


@server.tool()
def generate(species: str, stage: str | None = None, setting: str = "stone wall, residential house, white sash windows",
             coverage: str | None = None, framing: str | None = None, seed: int = 42, model: str | None = None,
             weight: float = 1.0, guidance: float = 3.5, width: int | None = None, height: int | None = None) -> list:
    """Generate one image of a plant (txt2img, about 1 min on a 16 GB GPU).

    species: see species(). stage: season or growth stage (default: the first one).
    setting: support and building (climbing plants) or background, in English. The seed sets the composition.
    model: 'lokr' or 'lora' (default per species). Returns the image path and a preview."""
    key, sp, st = _resolve(species, stage)
    prompt = _prompt(sp, st, setting, coverage=coverage, framing=framing)
    kind, lora = _model_file(sp, model)
    size = _setting(sp, "size")
    wf = _base(lora, weight, prompt, guidance)
    wf["9"] = {"class_type": "EmptySD3LatentImage", "inputs": {"width": width or size, "height": height or size,
               "batch_size": 1}}
    wf["16"] = _ksampler(["9", 0], seed, 1.0)
    wf["17"] = {"class_type": "VAEDecode", "inputs": {"samples": ["16", 0], "vae": ["3", 0]}}
    wf["save"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": f"botanical/{key}_{st}", "images": ["17", 0]}}
    out = _run(wf, f"{key}_{st}_s{seed}")
    return [f"Image: {out}\nPrompt: {prompt}\nSeed {seed}, {kind} at {weight}", _preview(out)]


@server.tool()
def plant(photo: str, mask: str, species: str, stage: str | None = None, setting: str = "stone wall",
          seed: int = 42, denoise: float = 1.0, model: str | None = None, weight: float = 1.0,
          guidance: float = 3.5) -> list:
    """Grow a species on a photo, only inside the white area of the mask (inpainting).

    photo, mask: file paths (black and white mask, same aspect ratio as the photo; white = where the plant grows).
    denoise 1.0 = the area is fully redrawn; 0.75-0.85 keeps more of the structure under the plant.
    The rest of the photo is pasted back unchanged."""
    key, sp, st = _resolve(species, stage)
    prompt = _prompt(sp, st, setting)
    kind, lora = _model_file(sp, model)
    p = _upload(_open(photo).convert("RGB"), "botanical_photo.png")
    m = _upload(_open(mask).convert("RGB"), "botanical_mask.png")
    wf = _base(lora, weight, prompt, guidance)
    wf["5"] = {"class_type": "DifferentialDiffusion", "inputs": {"strength": 1.0, "model": ["4", 0]}}
    wf["9"] = {"class_type": "LoadImage", "inputs": {"image": p}}
    wf["11"] = {"class_type": "LoadImageMask", "inputs": {"image": m, "channel": "red"}}
    wf["12"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {"upscale_method": "lanczos", "megapixels": 1.0,
                "resolution_steps": 16, "image": ["9", 0]}}
    wf["13"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["12", 0], "vae": ["3", 0]}}
    wf["14"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["13", 0], "mask": ["11", 0]}}
    wf["16"] = _ksampler(["14", 0], seed, denoise, model=("5", 0))
    wf["17"] = {"class_type": "VAEDecode", "inputs": {"samples": ["16", 0], "vae": ["3", 0]}}
    wf["18"] = {"class_type": "ImageCompositeMasked", "inputs": {"x": 0, "y": 0, "resize_source": False,
                "destination": ["12", 0], "source": ["17", 0], "mask": ["11", 0]}}
    wf["save"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": f"botanical/plant_{key}", "images": ["18", 0]}}
    out = _run(wf, f"plant_{key}_{st}_s{seed}")
    return [f"Image: {out}\nPrompt: {prompt}\nSeed {seed}, denoise {denoise}, {kind} at {weight}", _preview(out)]


@server.tool()
def make_autumn(summer_image: str, seed: int = 43, strength: float = 1.0) -> list:
    """Turn a facade with summer Boston ivy into autumn, without changing the building.

    The foliage mask (green pixels) is computed automatically and only the foliage is repainted.
    strength = denoise: 1.0 = full autumn (red), 0.92 = early autumn (red patches)."""
    im = _open(summer_image).convert("RGB")
    h, s, v = [np.asarray(c, dtype=np.int32) for c in im.convert("HSV").split()]
    green = ((h > 40) & (h < 120) & (s > 60) & (v > 25)).astype(np.uint8) * 255
    mk = (PILImage.fromarray(green).filter(ImageFilter.MedianFilter(5)).filter(ImageFilter.MaxFilter(9))
          .filter(ImageFilter.GaussianBlur(3)))
    share = round((np.asarray(mk) > 127).mean() * 100)
    if share < 3:
        raise ValueError("Almost no green foliage found: is this a summer facade?")
    src, msk = _upload(im, "botanical_summer.png"), _upload(mk.convert("RGB"), "botanical_summer_mask.png")
    _, sp, _ = _resolve("boston_ivy", "autumn")
    prompt = _prompt(sp, "autumn", "the same building", coverage="Dense coverage")
    wf = _base(_model_file(sp, "lokr")[1], 1.0, prompt, 3.5)
    wf["9"] = {"class_type": "LoadImage", "inputs": {"image": src}}
    wf["11"] = {"class_type": "LoadImageMask", "inputs": {"image": msk, "channel": "red"}}
    wf["13"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["9", 0], "vae": ["3", 0]}}
    wf["14"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["13", 0], "mask": ["11", 0]}}
    wf["16"] = _ksampler(["14", 0], seed, strength)
    wf["17"] = {"class_type": "VAEDecode", "inputs": {"samples": ["16", 0], "vae": ["3", 0]}}
    wf["save"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "botanical/autumn", "images": ["17", 0]}}
    out = _run(wf, f"autumn_s{seed}")
    return [f"Image: {out}\nFoliage repainted: {share} % of the image, denoise {strength}, seed {seed}", _preview(out)]


if __name__ == "__main__":
    server.run("stdio")
