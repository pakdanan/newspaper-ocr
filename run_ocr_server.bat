@echo off
REM Models from https://huggingface.co/prithivMLmods/LightOnOCR-3-0.8B-GGUF

set LLAMA_SERVER_EXE=C:\llama\llama-server.exe

set MODEL_PATH=C:\llama\models\prithiv\LightOnOCR-3-0.8B.Q6_K.gguf

set MMPROJ_PATH=C:\llama\models\prithiv\LightOnOCR-3-0.8B.mmproj-bf16.gguf

set MODEL_ARGS=-m "%MODEL_PATH%" --mmproj "%MMPROJ_PATH%"


"%LLAMA_SERVER_EXE%" %MODEL_ARGS% --host 127.0.0.1 --port 8081 -c 8192 -n 4096 --temp 0.0 --timeout 1500 --parallel 1 --repeat-penalty 1.15 --threads 4
