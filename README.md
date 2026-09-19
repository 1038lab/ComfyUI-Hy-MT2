# ComfyUI-Hy-MT2

[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Tencent%20Hy--MT2-ffc107?color=ffc107&logoColor=white)](https://huggingface.co/collections/tencent/hy-mt2)
[![ModelScope](https://img.shields.io/badge/ModelScope-Tencent%20Hy--MT2-624aff)](https://modelscope.cn/collections/Tencent-Hunyuan/Hy-MT2)
[![GitHub](https://img.shields.io/badge/GitHub-Tencent--Hunyuan%2FHy--MT2-black?logo=github)](https://github.com/Tencent-Hunyuan/Hy-MT2)

A lightweight, fast, and memory-efficient ComfyUI translation node designed for Tencent **Hy-MT2** GGUF models.

---

## Features

- **Lightweight & Low VRAM**: Powered by GGUF (`llama-cpp-python`). The 1.8B Q4 model uses only ~1.1 GB VRAM, avoiding memory contention with diffusion models.
- **Smart Dual-Direction Auto Detection**: Default `Auto (Chinese ⇋ English)` automatically translates Chinese text to English and English to Chinese without manual intervention.
- **Full Official Language Coverage**: All 38 languages and regional dialects officially supported by Tencent Hy-MT2 (Chinese, English, French, Portuguese, Spanish, Japanese, Turkish, Russian, Arabic, Korean, Thai, Italian, German, Vietnamese, Malay, Indonesian, Filipino, Hindi, Traditional Chinese, Polish, Czech, Dutch, Khmer, Burmese, Persian, Gujarati, Urdu, Telugu, Marathi, Hebrew, Bengali, Tamil, Ukrainian, Tibetan, Kazakh, Mongolian, Uyghur, Cantonese).
- **Prompt & Cinematics Syntax Protection**: Automatically shields `<lora:...>`, `<Picture n>`, character dialogues (`<d>[Language] "..."</d>`), shot headers (`[Shot ...]`), and `__wildcards__` from translation distortion while preserving full translation for weighted tags like `(red dress:1.2)` and `[red dress]`.
- **Dual-Channel Auto Download**: Automatically downloads `.gguf` models directly to `models/LLM/GGUF`. Seamlessly falls back to ModelScope CDN if Hugging Face is unreachable.
- **14 Built-in Translation Styles**: Rich stylistic guidance including AI Prompt/Visual Tags, Anime & ACG, Cinematic Arts, Literary, and Subtitles.
- **Output Sanitization**: Strips `<think>` tags, leaked tokens, and code block formatting.

---

## Node Reference

### `Hy-MT2 Translate (GGUF)` / `(GGUF Pro)`
- **Category**: `🧪AILab/🌍Translate`
- **Inputs**:
  - `model`: Official preset or local `.gguf` file.
    - `Hy-MT2-1.8B-Q4_K_M.gguf (1.1 GB)` (Fast & lowest VRAM, default)
    - `Hy-MT2-7B-Q4_K_M.gguf (4.5 GB)` (Balanced 7B quality)
    - `Hy-MT2-1.8B-Q6_K.gguf (1.5 GB)` (Near-lossless 1.8B)
    - `Hy-MT2-7B-Q6_K.gguf (5.9 GB)` (Near-lossless 7B)
    - `Hy-MT2-1.8B-Q8_0.gguf (1.9 GB)` (Maximum precision 1.8B)
    - `Hy-MT2-7B-Q8_0.gguf (7.5 GB)` (Maximum precision 7B)
  - `translate_style`: Stylistic constraint (`Default`, `AI Prompt / Visual Tags`, `Daily Conversational`, `Formal Business`, `Literary & Novel`, `Anime & ACG`, `Cinematic & Visual Arts`, `Scholarly & Academic`, `Popular Science`, `Marketing Copy`, `News & Media`, `Video Subtitles`, `Legal & Contract`, `Ancient / Poetic`). Default is `Default`.
  - `translate_to`: `Auto` (default) or one of the 38 supported languages.
  - `text`: Text or prompt to translate (multiline input with placeholder).
  - `unload_after_run`: Unload model from VRAM after translation (default: `False`).
- **Pro Version Features**:
  - Automatic prompt syntax masking: shields `<lora:...>`, embedding tags, and `(weight:1.2)` values from translation distortion.
  - Automatic comma/spacing formatting normalization for prompt workflows.
  - In-node translation preview & word/character count metrics.
- **Outputs**:
  - `text`: Translated output string.

---

## Model Comparison & Selection Guide

| Model Preset | Params | Quant | File Size | Approx. VRAM | Quality / Fidelity | Recommended Use Case |
|---|---|---|---|---|---|---|
| **Hy-MT2-1.8B-Q4_K_M.gguf** *(Default)* | 1.8B | Q4_K_M | **~1.1 GB** | ~1.4 GB | ⭐⭐⭐⭐ (Fast & Solid) | Everyday prompt translation; minimal VRAM footprint alongside SDXL/Flux |
| **Hy-MT2-1.8B-Q6_K.gguf** | 1.8B | Q6_K | **~1.5 GB** | ~1.8 GB | ⭐⭐⭐⭐⭐ (Near-Lossless 1.8B) | Low-VRAM setups desiring higher translation precision & nuances |
| **Hy-MT2-1.8B-Q8_0.gguf** | 1.8B | Q8_0 | **~1.9 GB** | ~2.2 GB | ⭐⭐⭐⭐⭐ (Full 8-bit 1.8B) | Maximum precision baseline for 1.8B parameter scale |
| **Hy-MT2-7B-Q4_K_M.gguf** | 7B | Q4_K_M | **~4.5 GB** | ~5.0 GB | ⭐⭐⭐⭐⭐ (Strong 7B Quality) | Complex domain, long paragraphs, or challenging literary texts |
| **Hy-MT2-7B-Q6_K.gguf** | 7B | Q6_K | **~5.9 GB** | ~6.4 GB | ⭐⭐⭐⭐⭐+ (Near-Lossless 7B) | Optimal sweet spot for 12GB+ GPUs seeking top-tier translation |
| **Hy-MT2-7B-Q8_0.gguf** | 7B | Q8_0 | **~7.5 GB** | ~8.0 GB | ⭐⭐⭐⭐⭐+ (Pure 8-bit 7B) | Dedicated translation tasks with 16GB+ VRAM or CPU offload |
| *Hy-MT2-1.8B-2bit-GGUF*\* | 1.8B | 2-bit STQ | ~600 MB | ~0.8 GB | ⭐⭐⭐ (Compressed) | Mobile / edge testing *(Requires custom llama.cpp)* |
| *Hy-MT2-1.8B-1.25bit-GGUF*\* | 1.8B | 1.25-bit STQ | ~440 MB | ~0.6 GB | ⭐⭐⭐ (Extreme) | Extreme edge deployment *(Requires custom llama.cpp)* |

*\*Note: 1.25-bit and 2-bit models are not built into the preset dropdown because they depend on Tencent's custom STQ kernel.*

---

## Model Quantization Notes

### Why are 1.25-bit and 2-bit GGUF models not in the default presets?
Tencent released extreme quantization models ([`Hy-MT2-1.8B-1.25bit-GGUF`](https://huggingface.co/tencent/Hy-MT2-1.8B-1.25bit-GGUF) and [`Hy-MT2-1.8B-2bit-GGUF`](https://huggingface.co/tencent/Hy-MT2-1.8B-2bit-GGUF)) based on the **AngelSlim STQ (Sparse Ternary Quantization / Sherry)** algorithm.
- **Custom Kernel Dependency**: These extreme quantizations require a dedicated `STQ_0` / `STQ1_0` computation kernel, introduced via [llama.cpp PR #22836](https://github.com/ggml-org/llama.cpp/pull/22836).
- **Standard Pre-built Binaries**: Standard pre-built wheels of `llama-cpp-python` (used by ComfyUI) do **not** bundle this custom kernel and will throw tensor decoding errors.
- **How to use them**: Advanced users who compile `llama.cpp` / `llama-cpp-python` from source with PR #22836 can manually download the `.gguf` file into `models/LLM/GGUF`. The node will automatically detect and load it from the local file scanner.

---

## Dependencies

```bash
pip install llama-cpp-python huggingface-hub
```

---

If this custom node helps you or you like my work, please give me ⭐ on this repo! It's a great encouragement for my efforts!

## License
GPL-3.0 License
