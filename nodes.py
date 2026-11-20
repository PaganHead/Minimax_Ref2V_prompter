"""ComfyUI prompt generator using an OpenAI-compatible vision API."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

DEFAULT_SYSTEM = """You write concise, production-ready MiniMax H3 video prompts.
Analyze supplied reference images according to their distinct roles.
Preserve character identity, exact outfit design, location architecture and important props.
Never duplicate objects. Invent natural attractive poses and simple believable actions.
Favor useful still frames, shallow depth of field and non-wide-angle framing.
Respond only with the finished video prompt."""

def encode_image(tensor):
    arr = (tensor[0].detach().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    im = Image.fromarray(arr[:, :, :3], "RGB")
    im.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    im.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

def cache_file_for(unique_id):
    directory = Path(os.environ.get("KIRA_PROMPT_CACHE_DIR", Path(__file__).resolve().parent / ".prompt_cache"))
    directory.mkdir(parents=True, exist_ok=True)
    hashed = hashlib.sha256(str(unique_id).encode()).hexdigest()[:24]
    return directory / (hashed + ".json")

class KiraPromptBuilder:
    CATEGORY = "Kira"
    FUNCTION = "generate"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt_text",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "creative_brief": ("STRING", {"multiline": True, "default": "Kira relaxing in a hammock. Natural playful poses."}),
                "system_prompt": ("STRING", {"multiline": True, "default": DEFAULT_SYSTEM}),
                "shot_mode": (["auto", "selfie", "mirror_selfie", "regular"],),
                "lock_prompt": ("BOOLEAN", {"default": False}),
                "llm_seed": ("INT", {"default": 1, "min": 0, "max": 2147483647}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0, "max": 2, "step": 0.05}),
                "api_base": ("STRING", {"default": "http://127.0.0.1:8000/v1"}),
                "model_name": ("STRING", {"default": "Qwen/Qwen3-VL-8B-Instruct"}),
            },
            "optional": {f"image_{i}": ("IMAGE",) for i in range(1, 5)},
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Queue executions while unlocked regenerate; locked executions can be cached.
        return "locked" if kwargs.get("lock_prompt") else float("nan")

    def generate(self, creative_brief, system_prompt, shot_mode, lock_prompt,
                 llm_seed, temperature, api_base, model_name, unique_id=None, **images):
        path = cache_file_for(unique_id)
        if lock_prompt:
            if not path.exists():
                raise RuntimeError("Generate once with Lock Prompt OFF before locking.")
            return (json.loads(path.read_text(encoding="utf-8"))["prompt"],)

        modes = {
            "auto": "Choose a suitable camera perspective.",
            "selfie": "Camera is Kira's own handheld phone; the phone must not appear.",
            "mirror_selfie": "Mirror selfie with phone potentially visible in reflection.",
            "regular": "Third-person camera, not a selfie.",
        }
        parts = [{"type": "text", "text":
                  f"Write one MiniMax H3 video prompt. Mode: {shot_mode}. {modes[shot_mode]}\n"
                  f"Creative brief: {creative_brief}\n"
                  "Treat each labeled image as a distinct reference, not an extra scene."}]
        roles = ["LOCATION / SCENE", "PERSON / IDENTITY", "OUTFIT", "PROP / STYLE"]
        for i, role in enumerate(roles, 1):
            tensor = images.get(f"image_{i}")
            if tensor is not None:
                parts.append({"type": "text", "text": f"Reference image_{i}: {role}"})
                parts.append({"type": "image_url", "image_url": {"url": encode_image(tensor)}})

        base = api_base.strip().rstrip("/")
        if not (base.startswith("https://") or base.startswith("http://localhost:") or base.startswith("http://127.0.0.1:")):
            raise ValueError("Remote vision APIs must use HTTPS; local HTTP is allowed.")
        key = os.environ.get("KIRA_VLM_API_KEY", "")
        payload = {
            "model": model_name.strip(),
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": parts}],
            "temperature": temperature, "seed": llm_seed, "max_tokens": 1200,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(base + "/chat/completions",
              data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                answer = json.load(response)
        except urllib.error.HTTPError as err:
            detail = err.read(1000).decode("utf-8", "replace")
            raise RuntimeError(f"Vision API returned HTTP {err.code}: {detail}") from err
        except urllib.error.URLError as err:
            raise RuntimeError(f"Cannot reach vision API: {err.reason}") from err

        content = answer["choices"][0]["message"]["content"]
        if isinstance(content, list):
            prompt = "\n".join(item.get("text", "") for item in content if isinstance(item, dict)).strip()
        else:
            prompt = str(content).strip()
        if not prompt:
            raise RuntimeError("Vision API returned no prompt.")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"prompt": prompt}, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)
        return (prompt,)
