import argparse
import time
import threading
import sys
import os
import glob as globmod
import pyaudio
import numpy as np
from pynput import keyboard
import platform
import winsound
import tkinter as tk

# Add NVIDIA CUDA DLLs to PATH before importing faster_whisper
_venv = os.path.dirname(os.path.dirname(os.path.abspath(sys.executable)))
for _pkg in ("nvidia/cublas/bin", "nvidia/cudnn/bin"):
    _dll_path = os.path.join(_venv, "Lib", "site-packages", _pkg)
    if os.path.isdir(_dll_path):
        os.add_dll_directory(_dll_path)
        os.environ["PATH"] = _dll_path + os.pathsep + os.environ.get("PATH", "")

from faster_whisper import WhisperModel

if platform.system() == "Windows":
    import pystray
    from PIL import Image, ImageDraw, ImageTk


class TextPolisher:
    """Uses a small local LLM to clean up whisper output."""

    def __init__(self, gpu_id=1):
        from llama_cpp import Llama
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        pattern = os.path.join(model_dir, "**", "*.gguf")
        gguf_files = globmod.glob(pattern, recursive=True)
        if not gguf_files:
            raise FileNotFoundError(f"No GGUF model found in {model_dir}")
        model_path = gguf_files[0]
        print(f"Loading LLM from {os.path.basename(model_path)} on GPU {gpu_id}...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=512,
            n_gpu_layers=-1,
            main_gpu=gpu_id,
            verbose=False,
        )
        print("LLM loaded!")

    def polish(self, text, task="transcribe"):
        if not text or len(text.strip()) < 3:
            return text
        if task == "translate":
            prompt = (f"You are a proofreader. Fix any grammar, spelling, and punctuation errors "
                      f"in this English text from speech recognition. Do not add or remove words "
                      f"unless they are clearly wrong. Output ONLY the corrected text:\n\n{text}")
        else:
            prompt = (f"Eres un corrector de textos. Corrige errores de gramatica, ortografia y "
                      f"puntuacion en este texto en espanol que viene de reconocimiento de voz. "
                      f"No anadeas ni quites palabras salvo que sean claramente erroneas. "
                      f"Devuelve SOLO el texto corregido:\n\n{text}")
        try:
            result = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.1,
            )
            polished = result["choices"][0]["message"]["content"].strip()
            if polished and len(polished) > 1:
                return polished
        except:
            pass
        return text


class StreamingTranscriber:
    """Transcribes audio in real-time using faster-whisper, typing text as you speak."""

    def __init__(self, model, chunk_duration=5.0, overlap_duration=0.5,
                 energy_threshold=0.003, polisher=None):
        self.model = model
        self.pykeyboard = keyboard.Controller()
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.energy_threshold = energy_threshold
        self.polisher = polisher
        self.typing = False

    def type_text(self, text):
        self.typing = True
        try:
            for char in text:
                try:
                    self.pykeyboard.type(char)
                    time.sleep(0.002)
                except:
                    pass
        finally:
            self.typing = False

    def has_speech(self, audio_data):
        """Check if audio has enough energy to contain speech."""
        rms = np.sqrt(np.mean(audio_data ** 2))
        return rms > self.energy_threshold

    def warmup(self):
        """Run a tiny silent inference to wake up the GPU/model."""
        silent = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        segments, _ = self.model.transcribe(silent, beam_size=1, best_of=1,
                                            without_timestamps=True)
        for _ in segments:
            pass

    def transcribe_chunk(self, audio_data, language=None, task="transcribe"):
        if not self.has_speech(audio_data):
            return ""

        segments, _ = self.model.transcribe(
            audio_data,
            language=language,
            task=task,
            beam_size=1,
            best_of=1,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            without_timestamps=True,
        )
        text = ""
        for segment in segments:
            text += segment.text
        text = text.strip()

        if text and self.polisher:
            polished = self.polisher.polish(text, task)
            if polished != text:
                print(f"  [RAW] {text}")
                print(f"  [FIX] {polished}")
            text = polished

        return text


class StreamingRecorder:
    """Records audio and transcribes on stop (batch mode)."""

    def __init__(self, transcriber):
        self.recording = False
        self.transcriber = transcriber
        self._thread = None
        self._audio_buffer = None
        self._language = None
        self._task = None

    def start(self, language=None, task="transcribe"):
        self.recording = True
        self._language = language
        self._task = task
        self._audio_buffer = np.array([], dtype=np.float32)
        self._thread = threading.Thread(target=self._record_impl, daemon=True)
        self._thread.start()

    def stop(self):
        self.recording = False
        if self._thread:
            self._thread.join(timeout=5)
        # Transcribe the full recording
        if self._audio_buffer is not None and len(self._audio_buffer) > 0:
            threading.Thread(
                target=self._transcribe_and_type,
                args=(self._audio_buffer, self._language, self._task),
                daemon=True,
            ).start()

    def _record_impl(self):
        sample_rate = 16000
        frames_per_buffer = 1024

        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            frames_per_buffer=frames_per_buffer,
            input=True,
        )

        try:
            while self.recording:
                data = stream.read(frames_per_buffer, exception_on_overflow=False)
                chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                self._audio_buffer = np.concatenate([self._audio_buffer, chunk])
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    def _transcribe_and_type(self, audio, language, task):
        text = self.transcriber.transcribe_chunk(audio, language, task)
        if text and text.strip():
            self.transcriber.type_text(text)


class GlobalKeyListener:
    def __init__(self, app, key_combination, transcriber):
        self.app = app
        self.transcriber = transcriber
        self.action = None  # Set after creation
        self.key1, self.key2 = self.parse_key_combination(key_combination)
        self.key1_pressed = False
        self.key2_pressed = False
        self._toggled = False

    def parse_key_combination(self, key_combination):
        key1_name, key2_name = key_combination.split('+')
        key1 = self._parse_key(key1_name)
        key2 = self._parse_key(key2_name)
        return key1, key2

    def _parse_key(self, key_name):
        if key_name == 'space':
            return keyboard.Key.space
        return getattr(keyboard.Key, key_name, keyboard.KeyCode(char=key_name))

    def on_key_press(self, key):
        if self.transcriber.typing:
            return
        if key == self.key1:
            self.key1_pressed = True
        elif key == self.key2:
            self.key2_pressed = True

        if self.key1_pressed and self.key2_pressed and not self._toggled:
            self._toggled = True
            if self.action:
                self.action()

    def on_key_release(self, key):
        if self.transcriber.typing:
            return
        if key == self.key1:
            self.key1_pressed = False
            self._toggled = False
        elif key == self.key2:
            self.key2_pressed = False
            self._toggled = False


class FloatingIndicator:
    """Small always-on-top flag indicator. Runs on main thread."""

    FLAG_WIDTH = 48
    FLAG_HEIGHT = 32
    MARGIN = 10

    def __init__(self):
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes('-topmost', True)
        self.root.attributes('-transparentcolor', 'magenta')
        self.root.configure(bg='magenta')

        screen_w = self.root.winfo_screenwidth()
        x = screen_w - self.FLAG_WIDTH - self.MARGIN
        y = self.MARGIN
        self.root.geometry(f'{self.FLAG_WIDTH}x{self.FLAG_HEIGHT}+{x}+{y}')

        self.canvas = tk.Canvas(self.root, width=self.FLAG_WIDTH, height=self.FLAG_HEIGHT,
                                bg='magenta', highlightthickness=0)
        self.canvas.pack()

        self._flag_images = {}
        self._create_flags()
        self._current_flag = None
        self.root.withdraw()

    def _create_flags(self):
        """Create flag images using PIL."""
        w, h = self.FLAG_WIDTH, self.FLAG_HEIGHT

        # Spain flag: red-yellow-red (1:2:1)
        es_img = Image.new('RGB', (w, h))
        es_draw = ImageDraw.Draw(es_img)
        stripe = h // 4
        es_draw.rectangle([0, 0, w, stripe], fill='#c60b1e')
        es_draw.rectangle([0, stripe, w, stripe * 3], fill='#ffc400')
        es_draw.rectangle([0, stripe * 3, w, h], fill='#c60b1e')
        self._flag_images['es'] = ImageTk.PhotoImage(es_img)

        # UK flag (simplified): blue bg, white cross, red cross
        en_img = Image.new('RGB', (w, h), '#012169')
        en_draw = ImageDraw.Draw(en_img)
        # White cross
        en_draw.rectangle([w // 2 - 3, 0, w // 2 + 3, h], fill='white')
        en_draw.rectangle([0, h // 2 - 2, w, h // 2 + 2], fill='white')
        # Red cross
        en_draw.rectangle([w // 2 - 1, 0, w // 2 + 1, h], fill='#C8102E')
        en_draw.rectangle([0, h // 2 - 1, w, h // 2 + 1], fill='#C8102E')
        self._flag_images['en'] = ImageTk.PhotoImage(en_img)

    def show_flag(self, lang):
        """Thread-safe: show a flag on the indicator."""
        self.root.after(0, self._do_show_flag, lang)

    def _do_show_flag(self, lang):
        self.canvas.delete('all')
        img = self._flag_images.get(lang)
        if img:
            self._current_flag = img  # Keep reference to prevent GC
            self.canvas.create_image(0, 0, anchor='nw', image=img)
        self.root.deiconify()
        self.root.lift()

    def hide(self):
        """Thread-safe hide."""
        self.root.after(0, self.root.withdraw)

    def run(self):
        """Blocks - runs the tkinter mainloop on the main thread."""
        self.root.mainloop()

    def destroy(self):
        self.root.after(0, self.root.destroy)


def create_tray_icon(color):
    img = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([8, 8, 56, 56], fill=color)
    return img


class WhisperDictationApp:
    def __init__(self, recorder, languages=None, max_time=None):
        self.languages = languages
        self.current_language = languages[0] if languages else None
        self.current_task = None  # None=stopped, "transcribe"=ES, "translate"=ES->EN
        self.started = False
        self.recorder = recorder
        self.max_time = max_time
        self.timer = None
        self.tray = None
        self.indicator = FloatingIndicator()

    def toggle_es(self):
        """Ctrl+Space: toggle Spanish transcription on/off."""
        if self.started:
            self.stop()
        else:
            self.start("transcribe")

    def toggle_en(self):
        """Ctrl+Shift: toggle ES->EN translation on/off."""
        if self.started:
            self.stop()
        else:
            self.start("translate")

    def start(self, task):
        self.current_task = task
        if task == "translate":
            mode_label = "ES->EN"
            flag = "en"
        else:
            mode_label = self.current_language or "auto"
            flag = "es"
        print(f'Recording [{mode_label}]...')
        self.started = True
        self.recorder.start(self.current_language, task)
        self.indicator.show_flag(flag)
        winsound.Beep(800, 150)

        if self.max_time is not None:
            self.timer = threading.Timer(self.max_time, self.stop)
            self.timer.start()

        if self.tray:
            color = '#3b82f6' if task == "translate" else 'red'
            label = f"RECORDING [{mode_label}]"
            self.tray.icon = create_tray_icon(color)
            self.tray.title = f"Whisper - {label}"

    def stop(self):
        if not self.started:
            return

        if self.timer is not None:
            self.timer.cancel()

        self.started = False
        self.recorder.stop()
        self.indicator.hide()
        winsound.Beep(400, 150)
        print('Stopped. Dictation paused.\n')

        if self.tray:
            self.tray.icon = create_tray_icon('green')
            self.tray.title = "Whisper - Paused"

    def quit(self):
        if self.started:
            self.stop()
        if self.tray:
            self.tray.stop()
        self.indicator.destroy()

    def _keepalive_loop(self):
        """Periodically warm up the model so GPU stays ready."""
        while True:
            time.sleep(60)
            if not self.started:
                try:
                    self.recorder.transcriber.warmup()
                except:
                    pass

    def run(self):
        threading.Thread(target=self._run_tray, daemon=True).start()
        threading.Thread(target=self._keepalive_loop, daemon=True).start()
        self.indicator.run()

    def _run_tray(self):
        menu_items = [
            pystray.MenuItem("Spanish (Ctrl+Space)", lambda: self.toggle_es()),
            pystray.MenuItem("English (Shift+Space)", lambda: self.toggle_en()),
            pystray.MenuItem("Quit", lambda: self.quit()),
        ]

        self.tray = pystray.Icon(
            "whisper-dictation",
            create_tray_icon('green'),
            "Whisper - Paused",
            pystray.Menu(*menu_items)
        )
        self.tray.run()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Real-time dictation with faster-whisper. Ctrl+Space to toggle.')
    parser.add_argument('-m', '--model_name', type=str,
                        choices=['tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en',
                                 'medium', 'medium.en', 'large-v2', 'large-v3',
                                 'large-v3-turbo', 'turbo'],
                        default='large-v3',
                        help='Whisper model. Default: large-v3')
    parser.add_argument('-k', '--key_combination', type=str, default='ctrl_l+space',
                        help='Key combo to toggle. Default: ctrl_l+space')
    parser.add_argument('-l', '--language', type=str, default=None,
                        help='Language code, e.g. "es" or "es,en" for multi.')
    parser.add_argument('-t', '--max_time', type=float, default=120,
                        help='Max recording seconds. Default: 120')
    parser.add_argument('--chunk', type=float, default=10.0,
                        help='Chunk duration in seconds for streaming. Default: 10.0')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device for inference. Default: cuda')
    parser.add_argument('--compute_type', type=str, default='float16',
                        choices=['float16', 'int8', 'int8_float16', 'float32'],
                        help='Compute type. float16 for GPU, int8 for CPU. Default: float16')
    parser.add_argument('--gpu', type=int, default=1,
                        help='GPU device index. Default: 1 (secondary GPU)')
    parser.add_argument('--no-polish', action='store_true',
                        help='Disable LLM text polishing')

    args = parser.parse_args()

    if args.language is not None:
        args.language = args.language.split(',')

    return args


if __name__ == "__main__":
    args = parse_args()

    # Pin whisper to the specified GPU
    device = f"{args.device}:{args.gpu}" if args.device == "cuda" else args.device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print(f"Loading faster-whisper model '{args.model_name}' on GPU {args.gpu} ({args.compute_type})...")
    model = WhisperModel(args.model_name, device="cuda", compute_type=args.compute_type)
    print(f"Model loaded!")

    polisher = None
    if not args.no_polish:
        try:
            polisher = TextPolisher(gpu_id=0)  # GPU 0 in visible devices (remapped)
        except Exception as e:
            print(f"LLM polisher not available: {e}")

    transcriber = StreamingTranscriber(model, chunk_duration=args.chunk, polisher=polisher)
    print("Warming up model...")
    transcriber.warmup()
    print("Warm-up done!")
    recorder = StreamingRecorder(transcriber)

    app = WhisperDictationApp(recorder, args.language, args.max_time)

    # Ctrl+Space = Spanish transcription
    es_listener = GlobalKeyListener(app, args.key_combination, transcriber)
    es_listener.action = app.toggle_es

    # Shift+Space = English translation (ES->EN)
    en_listener = GlobalKeyListener(app, 'shift_l+space', transcriber)
    en_listener.action = app.toggle_en

    def on_press(key):
        es_listener.on_key_press(key)
        en_listener.on_key_press(key)

    def on_release(key):
        es_listener.on_key_release(key)
        en_listener.on_key_release(key)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print(f"Running! Press Ctrl+Space for Spanish dictation.")
    print(f"Press Shift+Space for ES->EN translation.")
    print(f"Text will appear where your cursor is, in real-time.")
    app.run()
