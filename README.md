# FunClip Module Gateway

简化网页入口，只有三个模块：

* `字幕生成`：ASR 和 SRT 生成。
* `字幕处理`：通过 DeepSeek 从 SRT 选择高光时间段。
* `视频生成`：按照高光时间戳剪辑并烧录字幕。

仓库中的 `upstream_funclip` 是官方 `modelscope/FunClip` Git 子模块。首次部署后执行：

```bash
git submodule update --init --recursive
python3 -m venv .venv
.venv/bin/pip install -r requirements-web.txt
python prepare_funclip_runtime.py
```

该脚本将上游的 `funclip` 和 `font` 复制到被忽略的 `runtime/` 目录，保留一份
`launch.py.before-importable`，再把复制后的 `runtime/funclip/launch.py` 改成可导入形式。
官方子模块不会被修改；上游更新后，停止服务并运行：

```bash
git submodule update --remote upstream_funclip
python prepare_funclip_runtime.py --replace
```

Example:

```python
from 字幕生成 import generate_subtitles
from 字幕处理 import choose_highlights
from 视频生成 import render_highlight_video

text, srt, video_state, _, _, _ = generate_subtitles("input.mp4")
llm_result = choose_highlights(srt, "api-key", "system prompt", "user prompt")
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
