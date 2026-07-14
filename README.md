# FunClip Module Gateway

简化网页入口，只有三个模块：

* `subtitle_generation`：ASR 和 SRT 生成。
* `subtitle_processing`：通过 DeepSeek 从 SRT 选择高光时间段。
* `video_generation`：按照高光时间戳剪辑并烧录字幕。

服务器可在项目根目录创建未被 Git 跟踪的 `.env` 文件，保存 `DEEPSEEK_API_KEY` 和
`FUNCLIP_LLM_MODEL`。启动脚本会自动加载它，网页中的 API Key 可留空。

把服务器上的待处理视频放入 `subtitle_generation/pending_videos/`。刷新网页后，可从“服务器待处理视频”下拉框直接选择；该选择优先于本地上传。

仓库中的 `upstream_funclip` 是官方 `modelscope/FunClip` Git 子模块。首次部署后执行：

```bash
git submodule update --init --recursive
python3 -m venv .venv
.venv/bin/pip install -r requirements-web.txt
python prepare_funclip_runtime.py
```

该脚本将上游的 `funclip` 和 `font` 复制到被忽略的 `runtime/` 目录，保留一份
`launch.py.before-importable`，再把复制后的 `runtime/funclip/launch.py` 改成可导入形式。
导入该副本时只会初始化处理方法，不会构建官方 Gradio 页面。
官方子模块不会被修改；上游更新后，停止服务并运行：

```bash
git submodule update --remote upstream_funclip
python prepare_funclip_runtime.py --replace
```

Example:

```python
from subtitle_generation import generate_subtitles
from subtitle_processing import process_subtitles
from video_generation import render_highlight_video

text, srt, video_state, _, _, _ = generate_subtitles("input.mp4")
corrected_srt, highlight_display, keywords, llm_result, video_state = process_subtitles(
    srt, "api-key", 8, video_state
)
video, _, message, clip_srt = render_highlight_video(llm_result, video_state)
```

## Minimal Web UI

`web_app.py` provides three buttons and three outputs: subtitles, LLM highlight
results, and the generated video. It uses the same modules as the Python API.

```powershell
$env:FUNCLIP_LAUNCH_DIR = "D:\path\to\gateway\runtime\funclip"
$env:DEEPSEEK_API_KEY = "sk-..."
python .\web_app.py
```
