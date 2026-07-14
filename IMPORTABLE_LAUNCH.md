# Importable Upstream Launch

This directory starts from the upstream `modelscope/FunClip` repository, not
from the customized Project01 repository.

Run the patch after cloning or after placing an updated upstream `launch.py` in
`funclip/`:

```powershell
python .\make_launch_importable.py
```

The script changes only `funclip/launch.py`. It makes the processing callbacks
available after `import launch`, while keeping `funclip_service.launch()` for
direct script execution only. It backs up the unmodified file as
`funclip/launch.py.before-importable`.

Use `--check` before patching. If an upstream release changes the startup
structure, the script fails explicitly instead of applying an unsafe patch.
