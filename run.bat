@echo off
"C:\Users\jmsla\whisper-dictation\.venv\Scripts\python.exe" -u "C:\Users\jmsla\whisper-dictation\whisper-dictation.py" -m large-v3-turbo -l es 2>&1 | tee C:\Users\jmsla\whisper-dictation\whisper.log
pause
