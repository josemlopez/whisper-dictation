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
    """Uses a small local LLM to clean up and rewrite whisper output."""

    REWRITE_CONFIG = os.path.join(os.path.dirname(__file__), "rewrite_prompt.txt")

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
            n_ctx=2048,
            n_gpu_layers=-1,
            main_gpu=gpu_id,
            verbose=False,
        )
        print("LLM loaded!")

    def _run_llm(self, prompt, max_tokens=512):
        try:
            result = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.1,
            )
            out = result["choices"][0]["message"]["content"].strip()
            if out and len(out) > 1:
                return out
        except:
            pass
        return None

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
        return self._run_llm(prompt) or text

    def rewrite(self, text):
        """Rewrite text using the custom prompt from rewrite_prompt.txt."""
        if not os.path.exists(self.REWRITE_CONFIG):
            return text
        with open(self.REWRITE_CONFIG, 'r', encoding='utf-8') as f:
            custom_prompt = f.read().strip()
        if not custom_prompt:
            return text
        prompt = custom_prompt.replace("{text}", text)
        return self._run_llm(prompt, max_tokens=1024) or text


class Transcriber:
    """Transcribes audio using faster-whisper with optional LLM polishing."""

    def __init__(self, model, energy_threshold=0.003, polisher=None):
        self.model = model
        self.energy_threshold = energy_threshold
        self.polisher = polisher
        self.typing = False  # Used by key listener to ignore synthetic keys

    def has_speech(self, audio_data):
        rms = np.sqrt(np.mean(audio_data ** 2))
        return rms > self.energy_threshold

    def warmup(self):
        silent = np.zeros(16000, dtype=np.float32)
        segments, _ = self.model.transcribe(silent, beam_size=1, best_of=1,
                                            without_timestamps=True)
        for _ in segments:
            pass

    def transcribe(self, audio_data, language=None, task="transcribe"):
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


class Recorder:
    """Records audio and transcribes on stop. Sends text via callback."""

    def __init__(self, transcriber, on_text_ready=None):
        self.recording = False
        self.transcriber = transcriber
        self.on_text_ready = on_text_ready
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
        if self._audio_buffer is not None and len(self._audio_buffer) > 0:
            threading.Thread(
                target=self._transcribe_and_deliver,
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

    def _transcribe_and_deliver(self, audio, language, task):
        text = self.transcriber.transcribe(audio, language, task)
        if text and text.strip() and self.on_text_ready:
            self.on_text_ready(text)


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


class FloatingUI:
    """Flag indicator + history panel. Runs on main thread."""

    FLAG_WIDTH = 48
    FLAG_HEIGHT = 32
    PANEL_WIDTH = 420
    ENTRY_HEIGHT = 60
    MAX_HISTORY = 10
    MARGIN = 10

    def __init__(self, polisher=None):
        self.root = tk.Tk()
        self.root.withdraw()
        self._history = []
        self._hide_timer = None
        self._polisher = polisher

        # --- Flag window ---
        self._flag_win = tk.Toplevel(self.root)
        self._flag_win.overrideredirect(True)
        self._flag_win.attributes('-topmost', True)
        self._flag_win.attributes('-transparentcolor', 'magenta')
        self._flag_win.configure(bg='magenta')
        screen_w = self._flag_win.winfo_screenwidth()
        x = screen_w - self.FLAG_WIDTH - self.MARGIN
        self._flag_win.geometry(f'{self.FLAG_WIDTH}x{self.FLAG_HEIGHT}+{x}+{self.MARGIN}')
        self._flag_canvas = tk.Canvas(self._flag_win, width=self.FLAG_WIDTH,
                                       height=self.FLAG_HEIGHT, bg='magenta',
                                       highlightthickness=0)
        self._flag_canvas.pack()
        self._flag_images = {}
        self._create_flags()
        self._current_flag = None
        self._flag_win.withdraw()

        # --- History panel ---
        self._panel = tk.Toplevel(self.root)
        self._panel.overrideredirect(True)
        self._panel.attributes('-topmost', True)
        self._panel.configure(bg='#1e1e1e')
        screen_h = self._panel.winfo_screenheight()
        panel_h = self.ENTRY_HEIGHT * self.MAX_HISTORY + 30
        px = screen_w - self.PANEL_WIDTH - self.MARGIN
        py = screen_h - panel_h - 60
        self._panel.geometry(f'{self.PANEL_WIDTH}x{panel_h}+{px}+{py}')

        header = tk.Frame(self._panel, bg='#1e1e1e')
        header.pack(fill='x', padx=4, pady=(4, 0))
        title = tk.Label(header, text="Whisper History (click to copy)",
                         bg='#1e1e1e', fg='#888888', font=('Segoe UI', 9))
        title.pack(side='left', padx=4)
        close_btn = tk.Label(header, text="\u2715", bg='#1e1e1e', fg='#666666',
                             font=('Segoe UI', 11), cursor='hand2')
        close_btn.pack(side='right', padx=4)
        close_btn.bind('<Button-1>', lambda e: self._panel.withdraw())
        prompt_btn = tk.Label(header, text="Edit Prompt", bg='#3a3a5c', fg='#cccccc',
                              font=('Segoe UI', 8), padx=6, pady=2, cursor='hand2')
        prompt_btn.pack(side='right', padx=4)
        prompt_btn.bind('<Button-1>', lambda e: self._open_prompt_editor())

        self._scrollable = tk.Frame(self._panel, bg='#1e1e1e')
        self._scrollable.pack(fill='both', expand=True, padx=4, pady=4)
        self._entry_rows = []
        for i in range(self.MAX_HISTORY):
            row = tk.Frame(self._scrollable, bg='#2d2d2d')
            row.pack(fill='x', pady=2)
            lbl = tk.Label(row, text="", bg='#2d2d2d', fg='#cccccc',
                           font=('Segoe UI', 10), anchor='nw', justify='left',
                           wraplength=self.PANEL_WIDTH - 90, padx=8, pady=6,
                           cursor='hand2')
            lbl.pack(side='left', fill='x', expand=True)
            lbl.bind('<Button-1>', lambda e, idx=i: self._copy_entry(idx))
            rw_btn = tk.Label(row, text="Rewrite", bg='#3a5a3a', fg='#cccccc',
                              font=('Segoe UI', 8), padx=6, pady=4, cursor='hand2')
            rw_btn.pack(side='right', padx=4, pady=4)
            rw_btn.bind('<Button-1>', lambda e, idx=i: self._rewrite_entry(idx))
            self._entry_rows.append({'frame': row, 'label': lbl, 'rewrite': rw_btn})
        self._panel.withdraw()

    def _create_flags(self):
        w, h = self.FLAG_WIDTH, self.FLAG_HEIGHT
        es_img = Image.new('RGB', (w, h))
        es_draw = ImageDraw.Draw(es_img)
        stripe = h // 4
        es_draw.rectangle([0, 0, w, stripe], fill='#c60b1e')
        es_draw.rectangle([0, stripe, w, stripe * 3], fill='#ffc400')
        es_draw.rectangle([0, stripe * 3, w, h], fill='#c60b1e')
        self._flag_images['es'] = ImageTk.PhotoImage(es_img)

        en_img = Image.new('RGB', (w, h), '#012169')
        en_draw = ImageDraw.Draw(en_img)
        en_draw.rectangle([w // 2 - 3, 0, w // 2 + 3, h], fill='white')
        en_draw.rectangle([0, h // 2 - 2, w, h // 2 + 2], fill='white')
        en_draw.rectangle([w // 2 - 1, 0, w // 2 + 1, h], fill='#C8102E')
        en_draw.rectangle([0, h // 2 - 1, w, h // 2 + 1], fill='#C8102E')
        self._flag_images['en'] = ImageTk.PhotoImage(en_img)

    def show_flag(self, lang):
        self.root.after(0, self._do_show_flag, lang)

    def _do_show_flag(self, lang):
        self._flag_canvas.delete('all')
        img = self._flag_images.get(lang)
        if img:
            self._current_flag = img
            self._flag_canvas.create_image(0, 0, anchor='nw', image=img)
        self._flag_win.deiconify()
        self._flag_win.lift()

    def hide_flag(self):
        self.root.after(0, self._flag_win.withdraw)

    def add_text(self, text):
        """Thread-safe: add text to history, copy to clipboard, show panel."""
        self.root.after(0, self._do_add_text, text)

    def _do_add_text(self, text):
        self._history.insert(0, text)
        self._history = self._history[:self.MAX_HISTORY]
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._refresh_entries()
        self._panel.deiconify()
        self._panel.lift()
        if self._hide_timer is not None:
            self.root.after_cancel(self._hide_timer)
        self._hide_timer = self.root.after(30000, self._panel.withdraw)
        winsound.Beep(600, 100)

    def _refresh_entries(self):
        for i, row in enumerate(self._entry_rows):
            if i < len(self._history):
                display = self._history[i]
                if len(display) > 150:
                    display = display[:147] + "..."
                bg = '#3a3a5c' if i == 0 else '#2d2d2d'
                row['label'].configure(text=display, bg=bg)
                row['frame'].configure(bg=bg)
            else:
                row['label'].configure(text="", bg='#2d2d2d')
                row['frame'].configure(bg='#2d2d2d')

    def _copy_entry(self, idx):
        if idx < len(self._history):
            self.root.clipboard_clear()
            self.root.clipboard_append(self._history[idx])
            row = self._entry_rows[idx]
            row['label'].configure(bg='#4a4a7c')
            self.root.after(300, lambda: row['label'].configure(
                bg='#3a3a5c' if idx == 0 else '#2d2d2d'))
            winsound.Beep(600, 80)

    def _rewrite_entry(self, idx):
        if idx >= len(self._history) or not self._polisher:
            return
        row = self._entry_rows[idx]
        row['rewrite'].configure(text="...", bg='#5a5a3a')
        threading.Thread(
            target=self._do_rewrite, args=(idx,), daemon=True
        ).start()

    def _do_rewrite(self, idx):
        text = self._history[idx]
        rewritten = self._polisher.rewrite(text)
        print(f"  [REWRITE] {rewritten[:80]}...")
        self.root.after(0, self._apply_rewrite, idx, rewritten)

    def _apply_rewrite(self, idx, rewritten):
        self._history[idx] = rewritten
        self.root.clipboard_clear()
        self.root.clipboard_append(rewritten)
        self._refresh_entries()
        row = self._entry_rows[idx]
        row['rewrite'].configure(text="Rewrite", bg='#3a5a3a')
        row['label'].configure(bg='#3a5c3a')
        self.root.after(500, lambda: row['label'].configure(
            bg='#3a3a5c' if idx == 0 else '#2d2d2d'))
        winsound.Beep(700, 80)

    def _open_prompt_editor(self):
        config_path = TextPolisher.REWRITE_CONFIG
        current = ""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                current = f.read()

        editor = tk.Toplevel(self.root)
        editor.title("Rewrite Prompt")
        editor.attributes('-topmost', True)
        editor.configure(bg='#1e1e1e')
        editor.geometry('500x350')

        info = tk.Label(editor, text="Use {text} where your dictation should go.",
                        bg='#1e1e1e', fg='#888888', font=('Segoe UI', 9))
        info.pack(anchor='w', padx=10, pady=(8, 4))

        text_widget = tk.Text(editor, bg='#2d2d2d', fg='#cccccc', insertbackground='white',
                              font=('Consolas', 11), wrap='word', padx=8, pady=8)
        text_widget.pack(fill='both', expand=True, padx=10, pady=(0, 4))
        text_widget.insert('1.0', current)

        btn_frame = tk.Frame(editor, bg='#1e1e1e')
        btn_frame.pack(fill='x', padx=10, pady=(0, 8))

        def save():
            content = text_widget.get('1.0', 'end-1c')
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(content)
            winsound.Beep(600, 80)
            editor.destroy()

        save_btn = tk.Button(btn_frame, text="Save", command=save,
                             bg='#3a5a3a', fg='white', font=('Segoe UI', 10),
                             padx=16, pady=4, cursor='hand2')
        save_btn.pack(side='right')
        cancel_btn = tk.Button(btn_frame, text="Cancel", command=editor.destroy,
                               bg='#5a3a3a', fg='white', font=('Segoe UI', 10),
                               padx=16, pady=4, cursor='hand2')
        cancel_btn.pack(side='right', padx=(0, 8))

    def show_panel(self):
        """Thread-safe: show the history panel."""
        self.root.after(0, self._do_show_panel)

    def _do_show_panel(self):
        self._panel.deiconify()
        self._panel.lift()
        if self._hide_timer is not None:
            self.root.after_cancel(self._hide_timer)
        self._hide_timer = self.root.after(30000, self._panel.withdraw)

    def hide_panel(self):
        self.root.after(0, self._panel.withdraw)

    def run(self):
        self.root.mainloop()

    def destroy(self):
        self.root.after(0, self.root.destroy)


def create_tray_icon(color):
    img = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([8, 8, 56, 56], fill=color)
    return img


class WhisperDictationApp:
    def __init__(self, recorder, languages=None, max_time=None, polisher=None):
        self.languages = languages
        self.current_language = languages[0] if languages else None
        self.current_task = None
        self.started = False
        self.recorder = recorder
        self.recorder.on_text_ready = self._on_text_ready
        self.max_time = max_time
        self.timer = None
        self.tray = None
        self.ui = FloatingUI(polisher=polisher)

    def _on_text_ready(self, text):
        """Called from background thread when transcription is done."""
        print(f"  [CLIPBOARD] {text[:80]}...")
        self.ui.add_text(text)

    def toggle_es(self):
        if self.started:
            self.stop()
        else:
            self.start("transcribe")

    def toggle_en(self):
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
        self.ui.show_flag(flag)
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
        self.ui.hide_flag()
        winsound.Beep(400, 150)
        print('Stopped. Processing...\n')

        if self.tray:
            self.tray.icon = create_tray_icon('green')
            self.tray.title = "Whisper - Paused"

    def quit(self):
        if self.started:
            self.stop()
        if self.tray:
            self.tray.stop()
        self.ui.destroy()

    def _keepalive_loop(self):
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
        self.ui.run()

    def _run_tray(self):
        menu_items = [
            pystray.MenuItem("Spanish (Ctrl+Space)", lambda: self.toggle_es()),
            pystray.MenuItem("English (Shift+Space)", lambda: self.toggle_en()),
            pystray.MenuItem("Show History", lambda: self.ui.show_panel()),
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
    parser.add_argument('-t', '--max_time', type=float, default=None,
                        help='Max recording seconds. Default: unlimited')
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

    transcriber = Transcriber(model, polisher=polisher)
    print("Warming up model...")
    transcriber.warmup()
    print("Warm-up done!")
    recorder = Recorder(transcriber)

    app = WhisperDictationApp(recorder, args.language, args.max_time, polisher=polisher)

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
    print(f"Text goes to clipboard - paste with Ctrl+V.")
    app.run()
