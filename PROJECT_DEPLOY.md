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

The same button also saves subtitle settings used after refresh:

- subtitle font size and color
- subtitle X/Y position
- subtitle highlight color
- subtitle highlight prompt
- expected subtitle highlight count
- sound effect word bindings

## Output And Temp Files

Runtime files are intentionally ignored by Git:

- `tmp/`
- `gradio_tmp/`
- `output/`
- `local_videos/`
- `local_sfx/`
- `funclip-env/`
- `*.log`

For generated clips, set `File Output Dir` in the UI to:

```text
/opt/funclip-project01/output
```

On Windows local development, use your local repository path instead, for example:

```text
D:\pandas_project\pandagent\FunClip\output
```

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

## Server Sound Effects

On the server, place reusable sound effects in either of these project-root folders:

```text
/opt/funclip-project01/local_sfx/
/opt/funclip-project01/music/
```

Example:

```bash
scp ./ding.mp3 root@your-server:/opt/funclip-project01/local_sfx/
scp -r ./music root@your-server:/opt/funclip-project01/music
```

The UI scans both folders recursively. Open the web UI and click `Refresh Sound Effects` to see available files. The read-only `Sound Effect Folders` box shows the absolute folders the running service is reading.

If your `music/` folder is somewhere else, set it explicitly before starting the service:

```bash
cd /opt/funclip-project01
export FUNCLIP_MUSIC_DIR=/opt/funclip-project01/music
export FUNCLIP_LOCAL_SFX_DIR=/opt/funclip-project01/local_sfx
PORT=7860 ./start_funclip_server.sh
```

Quick server check:

```bash
ls -lah /opt/funclip-project01/music
find /opt/funclip-project01/music -type f | head
```

Choose one sound effect from `Server Sound Effects`, then enter trigger words for that sound effect in `Selected Sound Effect Trigger Words`. Use commas or new lines:

```text
糖尿病,戒烟
心梗脑梗
```

When you switch to another sound effect, the trigger word box automatically switches to that sound effect's saved words. Internally the app stores bindings in this format:

```text
sound_file | trigger words | volume | cooldown seconds
```

`Save Settings` saves these bindings to `user_settings.json`. The bindings are applied to video clipping outputs, including `Clip`, `Clip+Subtitles`, `AI Clip`, and `AI Clip+Subtitles`.

Sound effects are timed to the trigger word position inside the subtitle line. For example, if a trigger word appears near the end of a 6-second subtitle, the sound effect is placed near the end of that subtitle instead of always at the subtitle start.

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

## Subtitle Highlights

Use `Subtitle Highlight Color` to choose the emphasis color. The default is `yellow`, and CSS color names or hex colors such as `#ffcc00` are accepted.

After ASR finishes, run `LLM Inference` first so `LLM Clipper Result` contains timestamp ranges. Then edit `Subtitle Highlight Prompt` and `Expected Highlight Count` if needed, and click `LLM Pick Subtitle Highlights`.

The subtitle highlighter only sends SRT lines that overlap the LLM-selected highlight timestamp ranges to the selected LLM. It uses the same `LLM Model Name` and API key saved by `Save Settings`.

Timestamp ranges in `LLM Clipper Result` can use standard SRT arrows or bracket ranges such as `[00:06:04,460-00:06:15,370]`.

You can also edit `Subtitle Highlight Terms` manually. Put one term per line, or separate terms with commas.

The highlight terms and color are applied in `Preview Subtitle`, `Clip+Subtitles`, and `AI Clip+Subtitles`.
