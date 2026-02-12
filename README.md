# Whisper Dictation (Windows / CUDA)

Speech-to-text dictation for Windows using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with NVIDIA CUDA acceleration and local LLM text polishing.

Forked from [foges/whisper-dictation](https://github.com/foges/whisper-dictation) (macOS-only) and rewritten for Windows.

## Features

- **Batch transcription** — speak freely, text is transcribed when you stop (full context = better quality)
- **ES/EN translation** — speak Spanish, get English text (Shift+Space)
- **LLM text polishing** — local Qwen 1.5B model corrects grammar and punctuation
- **Rewrite button** — rewrite any dictation with a custom prompt (editable in `rewrite_prompt.txt`)
- **Clipboard-based** — text goes to clipboard, paste with Ctrl+V wherever you want (never lost)
- **History panel** — last 10 dictations in a floating panel, click to copy, auto-hides after 30s
- **GPU-accelerated** — Whisper + LLM on secondary GPU, keeps gaming GPU free
- **Flag indicators** — Spain/UK flag shows which mode is active
- **System tray** — status icon with menu (record, translate, show history, quit)
- **Offline** — everything runs locally, no data leaves your machine

## Requirements

- Windows 10/11
- Python 3.10-3.12
- NVIDIA GPU with CUDA support (tested on dual RTX 4090)
- Microphone

## Installation

```bash
git clone https://github.com/josemlopez/whisper-dictation.git
cd whisper-dictation
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### LLM polishing (optional but recommended)

```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
pip install huggingface-hub
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('Qwen/Qwen2.5-1.5B-Instruct-GGUF', 'qwen2.5-1.5b-instruct-q4_k_m.gguf', cache_dir='models')"
```

The first run will also download the Whisper model (~3 GB for `large-v3`).

## Usage

```bash
python whisper-dictation.py -l es --gpu 1
```

Or use `run.bat` / `pythonw.exe` for no terminal window.

### Hotkeys

| Hotkey | Action |
|--------|--------|
| **Ctrl+Space** | Start/stop Spanish dictation |
| **Shift+Space** | Start/stop ES-to-EN translation |
| **Ctrl+V** | Paste the transcribed text |

### Workflow

1. Press **Ctrl+Space** (or **Shift+Space** for English)
2. A flag indicator appears (Spain / UK)
3. Speak as long as you want — no time limit
4. Press the same hotkey to stop
5. Whisper transcribes, LLM polishes, text goes to clipboard
6. A beep signals the text is ready — **Ctrl+V** to paste
7. History panel shows last 10 dictations (auto-hides after 30s)
8. Click any entry to copy it, or click **Rewrite** to rewrite with a custom prompt

### Rewrite prompt

Edit `rewrite_prompt.txt` to customize how the Rewrite button transforms text. Use `{text}` as placeholder. Example:

```
Rewrite the following text in professional English suitable for business emails.
Make it clear, concise, and natural-sounding. Output ONLY the rewritten text:

{text}
```

### Command-line options

| Flag | Description | Default |
|------|-------------|---------|
| `-m`, `--model_name` | Whisper model | `large-v3` |
| `-k`, `--key_combination` | Spanish dictation hotkey | `ctrl_l+space` |
| `-l`, `--language` | Language code(s) | auto-detect |
| `-t`, `--max_time` | Max recording seconds | unlimited |
| `--device` | Inference device (`cuda` / `cpu`) | `cuda` |
| `--compute_type` | Compute type (`float16`, `int8`, `float32`) | `float16` |
| `--gpu` | GPU device index | `1` (secondary) |
| `--no-polish` | Disable LLM text polishing | off |

### Start with Windows

```powershell
New-ItemProperty -Path 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run' -Name 'WhisperDictation' -Value '"C:\path\to\whisper-dictation\.venv\Scripts\pythonw.exe" -u "C:\path\to\whisper-dictation\whisper-dictation.py" -m large-v3 -l es --gpu 1' -PropertyType String -Force
```

To remove from startup:

```powershell
Remove-ItemProperty -Path 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run' -Name 'WhisperDictation'
```

## Troubleshooting

See [FAQ.md](FAQ.md) for common issues and solutions.

## License

MIT License — see [LICENSE](LICENSE).

Copyright (c) 2025 Jose Lopez. Based on [whisper-dictation](https://github.com/foges/whisper-dictation) by Chris Fougner.
