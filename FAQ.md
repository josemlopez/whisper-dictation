# FAQ

## Installation

### `cublas64_12.dll not found` or similar CUDA DLL errors

The NVIDIA CUDA libraries need to be installed in the venv:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

The script automatically adds the DLL paths at startup. If you still get errors, check that your NVIDIA GPU drivers are up to date.

### `pip install pyaudio` fails

PyAudio requires the PortAudio library. On Windows, install with:

```bash
pip install pyaudio
```

If that fails, download the `.whl` from [Unofficial Windows Binaries](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio) or install via `pipwin`:

```bash
pip install pipwin
pipwin install pyaudio
```

### Which Python version should I use?

Python 3.10, 3.11, or 3.12. Python 3.13 is not yet supported by faster-whisper/CTranslate2.

## Usage

### Whisper hallucinates words like "Gracias", "Thank you", etc.

This is a known Whisper behavior. When there's silence or background noise, the model sometimes generates common phrases. Tips to reduce it:

- Speak continuously without long pauses
- Use a directional microphone close to your mouth
- Set a specific language with `-l` instead of auto-detect
- The VAD filter is already enabled and helps, but doesn't eliminate it completely

### Text is typed too slowly / with lag

- Make sure you're using `cuda` device (default), not `cpu`
- Use `large-v3-turbo` (default) — it's optimized for speed
- Reduce chunk duration with `--chunk 1.5` for faster responses (but less context)
- Use `float16` compute type (default) for best GPU performance

### Text appears in the wrong window

The dictation types into whichever window has focus at the moment. Make sure the target application is in the foreground and has keyboard focus before starting.

### Recording won't start

- Check that your microphone is set as the default recording device in Windows Sound Settings
- Make sure no other application is exclusively using the microphone
- Try running as administrator if hotkey doesn't register

### Hotkey doesn't work

- Default is **Left Ctrl + Space** (`ctrl_l+space`)
- Some applications may intercept Ctrl+Space (e.g., IDE autocomplete). Try a different combo: `-k ctrl_r+space` or `-k alt_l+space`
- The hotkey works globally, even when the terminal is not in focus

## Models

### Which model should I use?

| Model | VRAM | Speed | Accuracy | Best for |
|-------|------|-------|----------|----------|
| `tiny` / `tiny.en` | ~1 GB | Fastest | Low | Quick tests |
| `base` / `base.en` | ~1 GB | Fast | Fair | Low-resource systems |
| `small` / `small.en` | ~2 GB | Good | Good | Balanced |
| `medium` / `medium.en` | ~5 GB | Slower | Very good | Accuracy-focused |
| `large-v3-turbo` | ~6 GB | Fast | Excellent | **Recommended** |
| `large-v3` | ~10 GB | Slow | Best | Maximum accuracy |

The `.en` variants are English-only but slightly better for English. `large-v3-turbo` gives near-`large-v3` accuracy at much higher speed.

### Where are models stored?

Models are downloaded automatically by faster-whisper to `~/.cache/huggingface/hub/`. The `large-v3-turbo` model is approximately 1.5 GB.

## Performance

### How much VRAM does it use?

With `large-v3-turbo` and `float16`: approximately 3-4 GB VRAM during inference. It fits comfortably on any modern NVIDIA GPU (RTX 3060+).

### Can I run it on CPU?

Yes, but it will be significantly slower:

```bash
python whisper-dictation.py --device cpu --compute_type int8
```

With CPU mode, use a smaller model like `small` or `base` for reasonable speed.
