# Importable Launch

`make_launch_importable.py` changes only `funclip/launch.py`.

It exposes the existing module-level callbacks to another Python program while
keeping `funclip_service.launch()` restricted to direct script execution.

Run once after cloning:

```powershell
python .\make_launch_importable.py
```

After an upstream update replaces `funclip/launch.py`, run:

```powershell
git restore .\funclip\launch.py
git pull --ff-only
python .\make_launch_importable.py
```

The original file is backed up as `funclip/launch.py.before-importable` the
first time the patch is applied. Use `--check` to test whether a new upstream
file needs the patch.

Once patched, importing `funclip.launch` initializes its models and builds the
Gradio object but does not start the web server. Existing callbacks such as
`mix_recog`, `llm_inference`, `AI_clip`, and `AI_clip_subti` are then available
as module attributes.
