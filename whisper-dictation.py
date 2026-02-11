import argparse
import time
import threading
import sys
import os
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
    from PIL import Image, ImageDraw


class StreamingTranscriber:
    """Transcribes audio in real-time using faster-whisper, typing text as you speak."""

    def __init__(self, model, chunk_duration=5.0, overlap_duration=0.5,
                 energy_threshold=0.003):
        self.model = model
        self.pykeyboard = keyboard.Controller()
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.energy_threshold = energy_threshold
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

    def transcribe_chunk(self, audio_data, language=None):
        if not self.has_speech(audio_data):
            return ""

        segments, _ = self.model.transcribe(
            audio_data,
            language=language,
            beam_size=1,
            best_of=1,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            without_timestamps=True,
        )
        text = ""
        for segment in segments:
            text += segment.text
        return text.strip()


class StreamingRecorder:
    """Records audio and streams chunks to the transcriber in real-time."""

    def __init__(self, transcriber):
        self.recording = False
        self.transcriber = transcriber
        self._thread = None

    def start(self, language=None):
        self.recording = True
        self._thread = threading.Thread(target=self._stream_impl, args=(language,), daemon=True)
        self._thread.start()

    def stop(self):
        self.recording = False
        if self._thread:
            self._thread.join(timeout=5)

    def _stream_impl(self, language):
        sample_rate = 16000
        frames_per_buffer = 1024
        chunk_samples = int(self.transcriber.chunk_duration * sample_rate)
        overlap_samples = int(self.transcriber.overlap_duration * sample_rate)

        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            frames_per_buffer=frames_per_buffer,
            input=True,
        )

        audio_buffer = np.array([], dtype=np.float32)
        prev_text = ""

        try:
            while self.recording:
                data = stream.read(frames_per_buffer, exception_on_overflow=False)
                chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                audio_buffer = np.concatenate([audio_buffer, chunk])

                if len(audio_buffer) >= chunk_samples:
                    text = self.transcriber.transcribe_chunk(audio_buffer, language)

                    if text and text != prev_text:
                        if prev_text and text.startswith(prev_text):
                            new_text = text[len(prev_text):]
                        elif prev_text:
                            new_text = " " + text
                        else:
                            new_text = text

                        if new_text.strip():
                            self.transcriber.type_text(new_text)

                        prev_text = text

                    # Keep overlap for context continuity
                    audio_buffer = audio_buffer[-overlap_samples:]

            # Process remaining audio
            if len(audio_buffer) > sample_rate * 0.3:  # At least 0.3s of audio
                text = self.transcriber.transcribe_chunk(audio_buffer, language)
                if text and text != prev_text:
                    if prev_text and text.startswith(prev_text):
                        new_text = text[len(prev_text):]
                    else:
                        new_text = " " + text
                    if new_text.strip():
                        self.transcriber.type_text(new_text)

        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()


class GlobalKeyListener:
    def __init__(self, app, key_combination, transcriber):
        self.app = app
        self.transcriber = transcriber
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
            self.app.toggle()

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
    """Small always-on-top dot in the corner of the screen. Runs on main thread."""

    def __init__(self, size=24, margin=10):
        self.size = size
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes('-topmost', True)
        self.root.attributes('-transparentcolor', 'black')
        self.root.configure(bg='black')

        screen_w = self.root.winfo_screenwidth()
        x = screen_w - size - margin
        y = margin
        self.root.geometry(f'{size}x{size}+{x}+{y}')

        self.canvas = tk.Canvas(self.root, width=size, height=size,
                                bg='black', highlightthickness=0)
        self.canvas.pack()
        self.dot = self.canvas.create_oval(2, 2, size - 2, size - 2, fill='#ef4444',
                                           outline='')
        self.root.withdraw()  # Start hidden

    def set_recording(self, recording):
        """Thread-safe: schedules the update on the tkinter main thread."""
        self.root.after(0, self._do_set_recording, recording)

    def _do_set_recording(self, recording):
        if recording:
            self.canvas.itemconfig(self.dot, fill='#ef4444')
            self.root.deiconify()
            self.root.lift()
        else:
            self.root.withdraw()

    def run(self):
        """Blocks — runs the tkinter mainloop on the main thread."""
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
        self.started = False
        self.recorder = recorder
        self.max_time = max_time
        self.timer = None
        self.tray = None
        self.indicator = FloatingIndicator()

    def toggle(self):
        if self.started:
            self.stop()
        else:
            self.start()

    def start(self):
        print('Recording + streaming transcription...')
        self.started = True
        self.recorder.start(self.current_language)
        self.indicator.set_recording(True)
        winsound.Beep(800, 150)  # High beep = recording

        if self.max_time is not None:
            self.timer = threading.Timer(self.max_time, self.stop)
            self.timer.start()

        if self.tray:
            self.tray.icon = create_tray_icon('red')
            self.tray.title = "Whisper - RECORDING"

    def stop(self):
        if not self.started:
            return

        if self.timer is not None:
            self.timer.cancel()

        self.started = False
        self.recorder.stop()
        self.indicator.set_recording(False)
        winsound.Beep(400, 150)  # Low beep = paused
        print('Stopped. Dictation paused.\n')

        if self.tray:
            self.tray.icon = create_tray_icon('green')
            self.tray.title = "Whisper - Paused (Ctrl+Space)"

    def quit(self):
        if self.started:
            self.stop()
        if self.tray:
            self.tray.stop()
        self.indicator.destroy()

    def run(self):
        # pystray runs in a background thread
        threading.Thread(target=self._run_tray, daemon=True).start()
        # tkinter runs on the main thread (required by Windows)
        self.indicator.run()

    def _run_tray(self):
        menu_items = [
            pystray.MenuItem("Record / Pause (Ctrl+Space)", lambda: self.toggle()),
        ]

        if self.languages and len(self.languages) > 1:
            lang_items = []
            for lang in self.languages:
                lang_items.append(
                    pystray.MenuItem(lang, lambda _, l=lang: self._set_language(l),
                                     checked=lambda item, l=lang: self.current_language == l)
                )
            menu_items.append(pystray.MenuItem("Language", pystray.Menu(*lang_items)))

        menu_items.append(pystray.MenuItem("Quit", lambda: self.quit()))

        self.tray = pystray.Icon(
            "whisper-dictation",
            create_tray_icon('green'),
            "Whisper - Paused (Ctrl+Space)",
            pystray.Menu(*menu_items)
        )
        self.tray.run()

    def _set_language(self, lang):
        self.current_language = lang
        print(f"Language: {lang}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Real-time dictation with faster-whisper. Ctrl+Space to toggle.')
    parser.add_argument('-m', '--model_name', type=str,
                        choices=['tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en',
                                 'medium', 'medium.en', 'large-v2', 'large-v3',
                                 'large-v3-turbo', 'turbo'],
                        default='large-v3-turbo',
                        help='Whisper model. Default: large-v3-turbo')
    parser.add_argument('-k', '--key_combination', type=str, default='ctrl_l+space',
                        help='Key combo to toggle. Default: ctrl_l+space')
    parser.add_argument('-l', '--language', type=str, default=None,
                        help='Language code, e.g. "es" or "es,en" for multi.')
    parser.add_argument('-t', '--max_time', type=float, default=120,
                        help='Max recording seconds. Default: 120')
    parser.add_argument('--chunk', type=float, default=5.0,
                        help='Chunk duration in seconds for streaming. Default: 5.0')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device for inference. Default: cuda')
    parser.add_argument('--compute_type', type=str, default='float16',
                        choices=['float16', 'int8', 'int8_float16', 'float32'],
                        help='Compute type. float16 for GPU, int8 for CPU. Default: float16')

    args = parser.parse_args()

    if args.language is not None:
        args.language = args.language.split(',')

    return args


if __name__ == "__main__":
    args = parse_args()

    print(f"Loading faster-whisper model '{args.model_name}' on {args.device} ({args.compute_type})...")
    model = WhisperModel(args.model_name, device=args.device, compute_type=args.compute_type)
    print(f"Model loaded!")

    transcriber = StreamingTranscriber(model, chunk_duration=args.chunk)
    recorder = StreamingRecorder(transcriber)

    app = WhisperDictationApp(recorder, args.language, args.max_time)
    key_listener = GlobalKeyListener(app, args.key_combination, transcriber)
    listener = keyboard.Listener(on_press=key_listener.on_key_press, on_release=key_listener.on_key_release)
    listener.start()

    print(f"Running! Press Ctrl+Space to start/stop dictation.")
    print(f"Text will appear where your cursor is, in real-time.")
    app.run()
