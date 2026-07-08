@echo off
setlocal
cd /d "%~dp0"
set NO_PROXY=127.0.0.1,localhost
set no_proxy=127.0.0.1,localhost
set GRADIO_ANALYTICS_ENABLED=False
set TMP=%~dp0tmp
set TEMP=%~dp0tmp
set TMPDIR=%~dp0tmp
set GRADIO_TEMP_DIR=%~dp0gradio_tmp
if "%IMAGEMAGICK_BINARY%"=="" set IMAGEMAGICK_BINARY=magick
if "%ASR_MODEL%"=="" set ASR_MODEL=fun-asr-nano
funclip-env\Scripts\python.exe funclip\launch.py -m %ASR_MODEL% -p 7860 --listen
