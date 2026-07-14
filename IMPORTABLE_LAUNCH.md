# Importable Launch

`make_launch_importable.py` changes only `funclip/launch.py`.

It exposes the existing module-level callbacks to another Python program while
keeping `funclip_service.launch()` restricted to direct script execution.

Run once after cloning:

```powershell
python .\make_launch_importable.py
```

After an upstream update places a fresh `funclip/launch.py` in this directory,
run the patch again:

```powershell
python .\make_launch_importable.py
```

The unmodified file is backed up as `funclip/launch.py.before-importable` each
time the patch is applied. Use `--check` to test whether a new upstream file
needs the patch. If an upstream release changes the expected `launch.py`
startup structure, the script stops with a clear error instead of changing the
wrong code.

Once patched, importing `funclip.launch` initializes its models and builds the
Gradio object but does not start the web server. Existing callbacks such as
`mix_recog`, `llm_inference`, `AI_clip`, and `AI_clip_subti` are then available
as module attributes.
