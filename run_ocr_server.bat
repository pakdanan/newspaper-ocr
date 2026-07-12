@echo off
REM Models from https://huggingface.co/wangjazz/LightOnOCR-2-1B-gguf

set LLAMA_SERVER_EXE=C:\llama\llama-server.exe

set MODEL_PATH=C:\llama\models\LightOnOCR-2-1B-f16.gguf

set MMPROJ_PATH=C:\llama\models\LightOnOCR-2-1B-mmproj-f16.gguf

set MODEL_ARGS=-m "%MODEL_PATH%" --mmproj "%MMPROJ_PATH%"


"%LLAMA_SERVER_EXE%" %MODEL_ARGS% --host 127.0.0.1 --port 8081 -c 8192 --temp 0.2 --repeat-penalty 1.15 --threads 6
