@echo off
REM Models from https://huggingface.co/...

set LLAMA_SERVER_EXE=C:\llama\llama-server.exe

set MODEL_PATH=C:\llama\models\LightOnOCR-3-1B-Q8_0.gguf

set MMPROJ_PATH=C:\llama\models\mmproj-LightOnOCR-3-1B-F16.gguf

set MODEL_ARGS=-m "%MODEL_PATH%" --mmproj "%MMPROJ_PATH%"


"%LLAMA_SERVER_EXE%" %MODEL_ARGS% --host 127.0.0.1 --port 8081 -c 8192 --temp 0.0 --timeout 1200 --parallel 1 --repeat-penalty 1.15 --threads 4
