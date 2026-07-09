# Project01 FunClip Deployment

This repository is a FunClip-based local video highlight clipping service.

## Quick Start

```bat
python -m venv funclip-env
funclip-env\Scripts\python.exe -m pip install -r requirements.txt
start_funclip.bat
```

Open:

```text
http://127.0.0.1:7860
```

## DeepSeek

The LLM clipping panel defaults to `deepseek-v4-flash`.

You can either paste the DeepSeek API key in the UI, or set:

```bat
set DEEPSEEK_API_KEY=your_api_key
```

The UI has a `保存提示词/API | Save Settings` button. It saves the current system prompt, user prompt, selected model, and API key to:

```text
user_settings.json
```

This file is ignored by Git. On Linux, the app writes it with `600` permissions.

## Output And Temp Files

Runtime files are intentionally ignored by Git:

- `tmp/`
- `gradio_tmp/`
- `output/`
- `local_videos/`
- `funclip-env/`
- `*.log`

For generated clips, set `File Output Dir` in the UI to:

```text
D:\pandas_project\pandagent\FunClip\output
```

On a server, change this path to a writable directory with enough disk space.

## Server Local Videos

To avoid browser uploads for large videos, place source videos in:

```text
local_videos/
```

Example:

```bash
scp ./your-video.mp4 root@your-server:/opt/funclip-project01/local_videos/
```

Then open the web UI, click `Refresh Local Videos`, choose the video from `Server Local Video`, and run `ASR`.

## Server Mode

Windows:

```bat
start_funclip_server.bat
```

Linux:

```bash
python3 -m venv funclip-env
./funclip-env/bin/python -m pip install -r requirements.txt
chmod +x start_funclip_server.sh
PORT=7860 ./start_funclip_server.sh
```

The server script binds Gradio to `0.0.0.0`, so place it behind your reverse proxy or firewall as needed.

## Background ASR Tasks

`ASR` and `ASR+SD` start background jobs and return an `ASR Job ID` immediately. The web page polls the job every 10 seconds, and you can also click `Query ASR Task` manually.

This avoids browser or reverse-proxy timeouts during long speech recognition. Keep the backend process running until the task reports `Done`; then the transcript, SRT, and download files are loaded into the page. If the page reloads, paste the job id back into `ASR Job ID` and click `Query ASR Task`.

## Subtitle Preview

Use `Subtitle Preview Text`, `Subtitle Font Size`, `Subtitle Color`, `Subtitle X`, and `Subtitle Y` to preview burned-in subtitles before clipping.

Click `Preview Subtitle` after selecting a server local video or uploading a video. The same subtitle size, color, and position are used by `Clip+Subtitles` and `AI Clip+Subtitles`.
