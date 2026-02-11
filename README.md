# Whisper Dictation (Windows / CUDA)

Real-time speech-to-text dictation for Windows using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with NVIDIA CUDA acceleration. Text is typed directly where your cursor has focus — works in any application.

Forked from [foges/whisper-dictation](https://github.com/foges/whisper-dictation) (macOS-only) and rewritten for Windows with real-time streaming transcription.

## Features

- **Real-time streaming** — text appears as you speak, not after you stop
- **GPU-accelerated** — uses faster-whisper with CTranslate2 on CUDA (float16)
- **Works everywhere** — types into whatever window has focus (editor, browser, chat, etc.)
- **System tray** — green/red icon shows ready/recording status
- **Hotkey toggle** — Ctrl+Space to start/stop (configurable)
- **Multilingual** — supports all Whisper languages, with language switching from the tray menu
- **Offline** — everything runs locally, no data leaves your machine

## Requirements

- Windows 10/11
- Python 3.10–3.12
- NVIDIA GPU with CUDA support (tested on RTX 4090)
- Microphone

## Installation

```bash
git clone https://github.com/jmslay/whisper-dictation.git
cd whisper-dictation
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The first run will download the Whisper model (~1.5 GB for `large-v3-turbo`).

## Usage

```bash
python whisper-dictation.py
```

Press **Ctrl+Space** to start dictating. Press again to stop. Text is typed wherever your cursor is.

### Command-line options

| Flag | Description | Default |
|------|-------------|---------|
| `-m`, `--model_name` | Whisper model (`tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`, `large-v3-turbo`, `turbo`) | `large-v3-turbo` |
| `-k`, `--key_combination` | Hotkey to toggle recording | `ctrl_l+space` |
| `-l`, `--language` | Language code(s), comma-separated (e.g. `es` or `es,en`) | auto-detect |
| `-t`, `--max_time` | Max recording duration in seconds | `120` |
| `--chunk` | Streaming chunk duration in seconds | `2.0` |
| `--device` | Inference device (`cuda` or `cpu`) | `cuda` |
| `--compute_type` | Compute type (`float16`, `int8`, `float32`) | `float16` |

### Examples

```bash
# Spanish dictation with large-v3-turbo (recommended)
python whisper-dictation.py -m large-v3-turbo -l es

# English-only, smaller model for less VRAM
python whisper-dictation.py -m small.en -l en

# Multi-language with switching from tray menu
python whisper-dictation.py -l es,en,fr

# CPU-only (slower, no GPU needed)
python whisper-dictation.py --device cpu --compute_type int8
```

### Quick launcher (run.bat)

Edit `run.bat` to set your preferred options, then double-click or add to startup.

## How it works

1. Audio is captured from your microphone at 16kHz
2. Every 2 seconds (configurable), the audio chunk is sent to faster-whisper for transcription
3. New text is incrementally typed into the focused application using pynput
4. A 0.5s overlap between chunks maintains context continuity
5. VAD (Voice Activity Detection) filters out silence

## Troubleshooting

See [FAQ.md](FAQ.md) for common issues and solutions.

## License

MIT License — see [LICENSE](LICENSE).

Copyright (c) 2025 Jose Lopez. Based on [whisper-dictation](https://github.com/foges/whisper-dictation) by Chris Fougner.

If you use this software, please include attribution to **Jose Lopez** as required by the license.
