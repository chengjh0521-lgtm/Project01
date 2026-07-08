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
funclip-env\Scripts\python.exe funclip\launch.py -m paraformer -p 7860 --listen
