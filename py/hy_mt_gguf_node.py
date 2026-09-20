import gc
import re
import urllib.request
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

try:
    import folder_paths
except ImportError:
    class _MockFolderPaths:
        models_dir = Path("models")
        def get_folder_paths(self, name):
            return [str(Path("models") / name)]
    folder_paths = _MockFolderPaths()

try:
    from comfy.utils import ProgressBar
except ImportError:
    class ProgressBar:
        def __init__(self, total):
            self.total = total
        def update_absolute(self, step, total=None, desc=None):
            pass
TARGET_LANGUAGES = [
    "Auto",
    "Chinese",
    "English",
    "French",
    "Portuguese",
    "Spanish",
    "Japanese",
    "Turkish",
    "Russian",
    "Arabic",
    "Korean",
    "Thai",
    "Italian",
    "German",
    "Vietnamese",
    "Malay",
    "Indonesian",
    "Filipino",
    "Hindi",
    "Traditional Chinese",
    "Polish",
    "Czech",
    "Dutch",
    "Khmer",
    "Burmese",
    "Persian",
    "Gujarati",
    "Urdu",
    "Telugu",
    "Marathi",
    "Hebrew",
    "Bengali",
    "Tamil",
    "Ukrainian",
    "Tibetan",
    "Kazakh",
    "Mongolian",
    "Uyghur",
]

TRANSLATE_STYLES = [
    "Default",
    "AI Prompt / Visual Tags",
    "Daily Conversational",
    "Formal Business",
    "Literary & Novel",
    "Anime & ACG",
    "Cinematic & Visual Arts",
    "Scholarly & Academic",
    "Popular Science",
    "Marketing Copy",
    "News & Media",
    "Video Subtitles",
    "Legal & Contract",
    "Ancient / Poetic",
]

STYLE_PROMPTS = {
    "Default": None,
    "AI Prompt / Visual Tags": {
        "en": "concise, comma-separated visual prompt keywords suitable for AI image generation, focusing on subject, lighting, composition, and detail quality",
        "zh": "适合 AI 图像生成的高质量视觉提示词标签（逗号分隔的短语），侧重画面主体、光影、构图与细节描写",
    },
    "Daily Conversational": {
        "en": "natural, relaxed, and everyday colloquial conversation",
        "zh": "自然口语化、亲切生动的日常对话风格",
    },
    "Formal Business": {
        "en": "formal, professional business and corporate tone",
        "zh": "得体严谨的正式商务公文风格",
    },
    "Literary & Novel": {
        "en": "evocative, descriptive, and vivid literary novelistic style",
        "zh": "生动细腻、富有画面感的小说文学叙事风格",
    },
    "Anime & ACG": {
        "en": "lively, expressive anime, manga, and light novel tone fitting ACG conventions",
        "zh": "二次元动漫与轻小说风格，语气生动鲜明，贴合 ACG 语境与角色个性",
    },
    "Cinematic & Visual Arts": {
        "en": "cinematic and visual arts style emphasizing texture, framing, lighting atmosphere, and camera language",
        "zh": "电影镜头与视觉艺术风格，强调质感细节、构图透视、光影氛围与镜头语言",
    },
    "Scholarly & Academic": {
        "en": "rigorous, objective, and precise academic scholarly style",
        "zh": "术语精准、客观严密的学术论文规范风格",
    },
    "Popular Science": {
        "en": "clear, engaging, and accessible popular science tone",
        "zh": "深入浅出、通俗生动的科普语气",
    },
    "Marketing Copy": {
        "en": "catchy, persuasive, and appealing advertising marketing copy",
        "zh": "极具吸引力、感染力与号召力的宣传广告文案风格",
    },
    "News & Media": {
        "en": "objective, clear, concise, and professional journalistic reporting",
        "zh": "客观中立、简练严谨的新闻媒体报道风格",
    },
    "Video Subtitles": {
        "en": "concise and natural subtitle style, punchy phrasing optimized for on-screen pacing",
        "zh": "简练流畅的影视字幕风格，以紧凑短句为主，便于屏幕快速阅读",
    },
    "Legal & Contract": {
        "en": "strict, precise, unambiguous, and formal legal contract terminology",
        "zh": "严密准确、无歧义的法律合同条文规范风格",
    },
    "Ancient / Poetic": {
        "en": "classical, elegant, artistic, and poetic literary style",
        "zh": "辞藻典雅、富有诗意韵律的古风文言质感",
    },
}

OFFICIAL_GGUF_MODELS = {
    "Hy-MT2-1.8B-Q4_K_M.gguf (1.1 GB)": {
        "repo_id": "tencent/Hy-MT2-1.8B-GGUF",
        "ms_repo_id": "Tencent-Hunyuan/Hy-MT2-1.8B-GGUF",
        "filename": "Hy-MT2-1.8B-Q4_K_M.gguf",
    },
    "Hy-MT2-7B-Q4_K_M.gguf (4.5 GB)": {
        "repo_id": "tencent/Hy-MT2-7B-GGUF",
        "ms_repo_id": "Tencent-Hunyuan/Hy-MT2-7B-GGUF",
        "filename": "Hy-MT2-7B-Q4_K_M.gguf",
    },
    "Hy-MT2-1.8B-Q6_K.gguf (1.5 GB)": {
        "repo_id": "tencent/Hy-MT2-1.8B-GGUF",
        "ms_repo_id": "Tencent-Hunyuan/Hy-MT2-1.8B-GGUF",
        "filename": "Hy-MT2-1.8B-Q6_K.gguf",
    },
    "Hy-MT2-7B-Q6_K.gguf (5.9 GB)": {
        "repo_id": "tencent/Hy-MT2-7B-GGUF",
        "ms_repo_id": "Tencent-Hunyuan/Hy-MT2-7B-GGUF",
        "filename": "HY-MT2-7B-Q6_K.gguf",
    },
    "Hy-MT2-1.8B-Q8_0.gguf (1.9 GB)": {
        "repo_id": "tencent/Hy-MT2-1.8B-GGUF",
        "ms_repo_id": "Tencent-Hunyuan/Hy-MT2-1.8B-GGUF",
        "filename": "Hy-MT2-1.8B-Q8_0.gguf",
    },
    "Hy-MT2-7B-Q8_0.gguf (7.5 GB)": {
        "repo_id": "tencent/Hy-MT2-7B-GGUF",
        "ms_repo_id": "Tencent-Hunyuan/Hy-MT2-7B-GGUF",
        "filename": "HY-MT2-7B-Q8_0.gguf",
    },
}


def resolve_target_language(text: str, choice: str) -> str:
    if choice == "Auto":
        has_non_latin = any(
            "\u4e00" <= ch <= "\u9fff"
            or "\u3040" <= ch <= "\u30ff"
            or "\uac00" <= ch <= "\ud7af"
            or "\u0400" <= ch <= "\u04ff"
            or "\u0600" <= ch <= "\u06ff"
            or "\u0e00" <= ch <= "\u0e7f"
            for ch in text
        )
        return "English" if has_non_latin else "Chinese"
    return choice


def clean_model_output(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"<think[^>]*>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|?hy_[^>|]+\|?>", "", text)
    text = re.sub(r"<\|?im_[^>|]+\|?>|<im_[^>]+>|<\|endoftext\|>", "", text)

    cleaned = text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2:
            cleaned = "\n".join(lines[1:-1]).strip()

    return re.sub(r"^(assistant|translation|output|response|result)\s*:\s*", "", cleaned, flags=re.IGNORECASE).strip()


def free_memory():
    try:
        import comfy.model_management
        comfy.model_management.soft_empty_cache()
    except Exception:
        pass

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def get_gguf_search_dirs() -> list[Path]:
    base_dir = Path(folder_paths.models_dir)
    return [
        base_dir / "LLM" / "GGUF",
        base_dir / "llm" / "GGUF",
        base_dir / "LLM",
        base_dir / "hy-mt2",
    ]


def scan_local_gguf_files() -> list[str]:
    found = []
    for sdir in get_gguf_search_dirs():
        if sdir.exists() and sdir.is_dir():
            for f in sdir.glob("**/*.gguf"):
                if f.is_file() and "mmproj" not in f.name.lower():
                    name_l = f.name.lower()
                    path_l = str(f).lower()
                    if ("hy" in name_l and "mt" in name_l) or "hy-mt" in path_l or "hymt" in path_l:
                        if f.name not in found:
                            found.append(f.name)
    return sorted(found)


def split_text_into_chunks(text: str, max_chars: int = 3000) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 2
        if current_len + para_len <= max_chars:
            current_chunk.append(para)
            current_len += para_len
        else:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_len = 0

            if len(para) > max_chars:
                lines = para.split("\n")
                line_chunk = []
                line_len = 0
                for line in lines:
                    if line_len + len(line) + 1 <= max_chars:
                        line_chunk.append(line)
                        line_len += len(line) + 1
                    else:
                        if line_chunk:
                            chunks.append("\n".join(line_chunk))
                            line_chunk = []
                            line_len = 0
                        if len(line) > max_chars:
                            sentences = re.split(r"([。！？.!?]+)", line)
                            sent_chunk = ""
                            for s in sentences:
                                if len(sent_chunk) + len(s) <= max_chars:
                                    sent_chunk += s
                                else:
                                    if sent_chunk:
                                        chunks.append(sent_chunk)
                                    sent_chunk = s
                            if sent_chunk:
                                chunks.append(sent_chunk)
                        else:
                            line_chunk.append(line)
                            line_len += len(line) + 1
                if line_chunk:
                    chunks.append("\n".join(line_chunk))
            else:
                current_chunk.append(para)
                current_len += para_len

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return [c for c in chunks if c.strip()]


PROMPT_PROTECT_PATTERN = re.compile(r"(?is)<d>.*?</d>|<[^>]+>|\[\s*(?:shot|scene|cut)\b[^\]]*\]|__[a-zA-Z0-9_\-\/]+__")


def mask_prompt_tags(text: str) -> tuple[str, dict[str, str]]:
    token_map = {}
    counter = 0

    def replace_tag(match):
        nonlocal counter
        key = f"<<TAG{counter}>>"
        token_map[key] = match.group(0)
        counter += 1
        return key

    masked = PROMPT_PROTECT_PATTERN.sub(replace_tag, text)
    return masked, token_map


def unmask_prompt_tags(text: str, token_map: dict[str, str]) -> str:
    res = text
    for key, orig in token_map.items():
        res = res.replace(key, orig)
    # Robust fallback in case LLM added whitespace like << TAG0 >>
    res = re.sub(r"<<\s*TAG(\d+)\s*>>", lambda m: token_map.get(f"<<TAG{m.group(1)}>>", m.group(0)), res)
    return res


def download_from_modelscope(ms_repo_id: str, filename: str, target_dir: Path) -> str:
    target_path = target_dir / filename
    temp_path = target_dir / f"{filename}.downloading"
    url = f"https://modelscope.cn/api/v1/models/{ms_repo_id}/repo?Revision=master&FilePath={filename}"

    print(f"[HyMT GGUF] Downloading '{filename}' from ModelScope '{ms_repo_id}' ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            total_size = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB
            with open(temp_path, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r[HyMT GGUF] ModelScope Progress: {percent:.1f}% ({downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB)", end="")
        print()
        temp_path.replace(target_path)
        print(f"[HyMT GGUF] ModelScope Download finished: {target_path}")
        return str(target_path)
    except Exception:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        raise


def ensure_gguf_file(model_key: str) -> str:
    direct_p = Path(model_key)
    if direct_p.exists() and direct_p.is_file():
        return str(direct_p)

    target_filename = direct_p.name
    repo_id = None
    ms_repo_id = None
    if model_key in OFFICIAL_GGUF_MODELS:
        target_filename = OFFICIAL_GGUF_MODELS[model_key]["filename"]
        repo_id = OFFICIAL_GGUF_MODELS[model_key]["repo_id"]
        ms_repo_id = OFFICIAL_GGUF_MODELS[model_key].get("ms_repo_id")

    for sdir in get_gguf_search_dirs():
        if sdir.exists():
            matched = list(sdir.glob(f"**/{target_filename}"))
            if matched:
                return str(matched[0])

    if not repo_id:
        raise FileNotFoundError(f"Local GGUF model not found: '{model_key}'")

    target_dir = Path(folder_paths.models_dir) / "LLM" / "GGUF"
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"[HyMT GGUF] Downloading '{target_filename}' from HuggingFace '{repo_id}' to {target_dir} ...")
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=target_filename,
            repo_type="model",
            local_dir=str(target_dir),
        )
        print(f"[HyMT GGUF] HuggingFace Download finished: {downloaded}")
        return str(downloaded)
    except Exception as hf_err:
        print(f"[HyMT GGUF] HuggingFace download failed or unavailable ({hf_err}).")
        if ms_repo_id:
            try:
                try:
                    from modelscope.hub.file_download import model_file_download
                    return str(model_file_download(model_id=ms_repo_id, file_path=target_filename, local_dir=str(target_dir)))
                except ImportError:
                    pass
                return download_from_modelscope(ms_repo_id, target_filename, target_dir)
            except Exception as ms_err:
                raise RuntimeError(
                    f"Failed to download '{target_filename}' from both HuggingFace ({hf_err}) and ModelScope ({ms_err})."
                )
        raise hf_err


class HyMT_Translation_GGUF:
    def __init__(self):
        self.llm = None
        self.current_signature = None

    @classmethod
    def INPUT_TYPES(cls):
        local_files = scan_local_gguf_files()
        preset_names = list(OFFICIAL_GGUF_MODELS.keys())
        known_filenames = {v["filename"].lower() for v in OFFICIAL_GGUF_MODELS.values()}
        all_choices = preset_names + [f for f in local_files if f.lower() not in known_filenames]
        default_choice = preset_names[0] if preset_names else (all_choices[0] if all_choices else "")

        return {
            "required": {
                "model": (all_choices, {"default": default_choice, "tooltip": "Hy-MT2 GGUF model."}),
                "translate_style": (TRANSLATE_STYLES, {"default": "Default", "tooltip": "Target translation style."}),
                "translate_to": (TARGET_LANGUAGES, {"default": "Auto", "tooltip": "Target language. Auto translates non-English to English and English to Chinese."}),
                "text": ("STRING", {"multiline": True, "default": "", "placeholder": "Type or paste text / prompt to translate...", "tooltip": "Text to translate."}),
                "unload_after_run": ("BOOLEAN", {"default": False, "tooltip": "Unload model from VRAM after translation."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "translate"
    CATEGORY = "🧪AILab/🌍Translate"

    @classmethod
    def VALIDATE_INPUTS(cls, model: str, **kwargs):
        return True

    def load_model(self, model: str, gpu_layers: int = -1):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "[HyMT GGUF] llama-cpp-python is not installed. Please install it using: pip install llama-cpp-python"
            )

        signature = (model, int(gpu_layers))
        if self.llm is not None and self.current_signature == signature:
            return

        model_path = ensure_gguf_file(model)
        print(f"[HyMT GGUF] Loading model from '{model_path}' (gpu_layers={gpu_layers}) ...")

        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=int(gpu_layers),
            n_ctx=8192,
            verbose=False,
        )
        self.current_signature = signature
        print(f"[HyMT GGUF] Model loaded successfully.")

    def unload_model(self):
        self.llm = None
        self.current_signature = None
        free_memory()

    def translate(
        self,
        model: str,
        translate_style: str,
        translate_to: str,
        text: str,
        unload_after_run: bool = False,
    ):
        if not text.strip():
            return ("",)

        self.load_model(model, gpu_layers=-1)

        masked_text, token_map = mask_prompt_tags(text.strip())
        target_lang = resolve_target_language(text, translate_to)
        chunks = split_text_into_chunks(masked_text, max_chars=3000)
        pbar = ProgressBar(len(chunks))

        translated_chunks = []
        for i, chunk in enumerate(chunks):
            pbar.update_absolute(i, len(chunks))
            style_info = STYLE_PROMPTS.get(translate_style)
            if style_info:
                if target_lang in ["Chinese", "Traditional Chinese", "Cantonese"]:
                    style_desc = style_info["zh"]
                    prompt = f"请将以下文本翻译为 {target_lang}。注意翻译的风格要严格符合【{style_desc}】，只需要输出翻译后的结果，不要额外解释：\n\n{chunk}"
                else:
                    style_desc = style_info["en"]
                    prompt = f"Please translate the following text into {target_lang}. Note that the translation style must strictly conform to [{style_desc}] and only output the translated result without any additional explanation:\n\n{chunk}"
            else:
                prompt = f"Translate the following text into {target_lang}. Note that you should only output the translated result without any additional explanation:\n\n{chunk}"
            messages = [{"role": "user", "content": prompt}]

            result = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=4096,
                temperature=0.7,
                top_p=0.6,
                top_k=20,
                repeat_penalty=1.05,
                stop=["<|hy_end_of_assistant|>", "<|hy_begin_of_sentence|>", "<|im_end|>", "<|endoftext|>"],
            )

            choices = result.get("choices", [])
            raw_output = choices[0]["message"]["content"] if choices else ""
            clean_out = clean_model_output(raw_output)
            translated_chunks.append(clean_out)

        pbar.update_absolute(len(chunks), len(chunks))
        final_text = "\n\n".join(translated_chunks) if len(chunks) > 1 else (translated_chunks[0] if translated_chunks else "")
        if token_map:
            final_text = unmask_prompt_tags(final_text, token_map)

        if unload_after_run:
            self.unload_model()

        return (final_text,)


NODE_CLASS_MAPPINGS = {
    "HyMT_Translation_GGUF": HyMT_Translation_GGUF,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HyMT_Translation_GGUF": "Hy-MT2 Translate 🌍 (GGUF)",
}
