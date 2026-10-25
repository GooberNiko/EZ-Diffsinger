"""Assemble src/*.py into DiffSinger_Colab_Improved.ipynb.

Run:  python src/build.py
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "DiffSinger_Colab_Improved.ipynb"

MAGIC = re.compile(r"^\s*[!%]")


def load_cell(name):
    return (HERE / name).read_text(encoding="utf-8").rstrip("\n")


def check_syntax(name, source):
    """Compile the cell with IPython magics stripped so typos never ship."""
    stripped = "\n".join("" if MAGIC.match(line) else line for line in source.split("\n"))
    try:
        compile(stripped, name, "exec")
    except SyntaxError as exc:
        print(f"SYNTAX ERROR in {name}:{exc.lineno}: {exc.msg}")
        print(f"    {(exc.text or '').rstrip()}")
        return False
    return True


def as_source(text):
    """
    Split into the list-of-lines an .ipynb wants.

    Every line except the last MUST keep its trailing newline: readers join the
    list with no separator, so dropping them collapses the whole cell onto one
    unreadable line.

    Done by hand rather than with splitlines(keepends=True), which also splits
    on form feeds and other Unicode line boundaries, and would mangle a cell
    containing one.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n").split("\n")
    return [line + "\n" for line in lines[:-1]] + lines[-1:]


def markdown(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": as_source(text.strip("\n"))}


def code(name):
    source = load_cell(name)
    if not check_syntax(name, source):
        raise SystemExit(1)
    metadata = {"cellView": "form"} if source.startswith("#@title") else {}
    return {"cell_type": "code", "execution_count": None, "metadata": metadata,
            "outputs": [], "source": as_source(source)}


INTRO = """
<div align="center" style="padding:28px 16px;border-radius:16px;background:linear-gradient(135deg,#6366f1,#8b5cf6 55%,#db2777);color:#fff">
<div style="font-size:34px;font-weight:800;letter-spacing:.5px">&#127908; DiffSinger Colab</div>
<div style="font-size:15px;opacity:.92;margin-top:6px">Singing voice synthesis via shallow diffusion &mdash; rebuilt, faster, and it tells you what went wrong</div>
<div style="margin-top:14px;font-size:13px">
<a style="color:#fff" href="https://github.com/openvpi/DiffSinger">DiffSinger</a> &nbsp;&middot;&nbsp;
<a style="color:#fff" href="https://github.com/xunmengshe/OpenUtau/wiki/Voicebank-Development">OpenUtau voicebank docs</a>
</div>
</div>

### &#129517; How to use it

| I want to&hellip; | Run |
|---|---|
| &#127926; Train an acoustic model | **0 &rarr; 1 &rarr; 2 &rarr; 3 &rarr; 4 &rarr; 5** |
| &#127932; Train a variance model | **0 &rarr; 1 &rarr; 2 &rarr; 3 &rarr; 4 &rarr; 5** &nbsp;<font size="-1">(pick `variance` in step 3)</font> |
| &#128260; Continue yesterday's run | **0 &rarr; 1 &rarr; 5** &nbsp;<font size="-1">(fill in `resume_from_model_dir`)</font> |
| &#128230; Only export ONNX / build a voicebank | **0 &rarr; 1** <font size="-1">(`profile = onnx export only`)</font> **&rarr; 6 &rarr; 7** |

### &#9889; Pick a training mode

Chosen in **step 3** (and can be changed again in **step 5**).

| | &#128994; Original | &#9889; Woah's Epic Faster Training <sup>experimental</sup> |
|---|---|---|
| Config | stock DiffSinger, exactly as before | stock DiffSinger keys, tuned to the runtime |
| Binarize | single process | one worker per CPU core (falls back to single-process if it fails) |
| Batch size | your sliders in step 5 | sized to the GPU's VRAM (T4 / L4 / A100 tiers) |
| Precision | what you picked in step 3 | **fp16 tensor cores** on a T4, **bf16** on L4 / A100 |
| Data loading | 4 workers, prefetch 2 | one core left free for the training loop, prefetch 4, expandable CUDA allocator |
| Progress bar | redrawn every step | every 10 steps (less CPU stolen from the GPU on 2-core runtimes) |
| Compiler | eager PyTorch | `torch.compile` on the diffusion backbone (auto-falls back to eager) |
| Optimizer | stock | fused AdamW kernel (Muon branch on any GPU, `main` under bf16) |
| Validation / logging | 10 rendered samples + sanity pass, log every 100 | 3 samples, no sanity pass, log every 500 |
| Binary data | local or Drive | always local disk |

Fast mode changes **how quickly** steps run, never the model you get: architecture, schedule,
learning rate and checkpoints are identical. It is tuned for the free **T4** first: that
card is limited by memory bandwidth and Colab gives it only 2 CPU cores, so the wins there come
from `torch.compile` fusing the backbone's many small kernels and from keeping the CPU free
to feed the GPU. On an L4 / A100, bf16 and bigger batches add on top. If
training dies with *CUDA out of memory*, switch step 5 back to **Original** &mdash; the cell
tells you when that is the cause.

### &#128161; Worth knowing

- Every step says what it did and how long it took. When something fails you get the problem,
  the relevant log lines, and what to do about it.
- State lives in `/content/ds_state.json`, not in Python variables, so a runtime restart costs
  you nothing.
- Nothing in the DiffSinger source tree is patched, so re-running any cell is safe.
- DiffSinger gets **its own Python 3.11 environment**: its `numpy<2` / `librosa<0.10` pins have
  no wheels on the Python 3.13 Colab now ships, and compiling them is what made the old setup
  take an hour.

<font size="-1">Terminology: what OpenUtau calls **parameters** (tension, voicing, breathiness,
energy) is what DiffSinger's configs call **variance**.</font>
"""

SETUP_MD = """
---
## &#128295; Setup

Run cell **0** once, then cell **1**. Both are safe to re-run at any point.
"""

DATA_MD = """
---
## &#128193; Dataset and config

Step 2 unpacks and segments your data. Step 3 turns your choices into a config, picks your
**training mode**, and checks it will actually work. Step 4 binarizes.

Your archive should contain **one folder per singer**:

```
data.zip
 +- my_singer/
     +- song1.wav
     +- song1.lab
     +- ...
```

For multiple languages, add a `lang_config.yaml` at the top level and suffix each
folder with its language code (`my_singer.ja`, `other_singer.en`).
"""

TRAIN_MD = """
---
## &#128640; Train

Training runs until `max_updates` or until you interrupt the cell. Checkpoints are
already on disk either way &mdash; interrupting loses nothing since the last save.

Colab will disconnect eventually. To pick up where you left off in a new session:
run cell 0, cell 1, then cell 5 with `resume_from_model_dir` pointing at your model
folder. Steps 2&ndash;4 are not needed.

> &#9889; **Fast mode on a T4:** keep `torch_compile` ticked. On a T4 it is the biggest single
> win, because the card spends most of its time moving data between small kernels and the
> compiler fuses them. The first few minutes are slower while it compiles. Fast mode also
> forces fp16 on a T4, so a `bf16-mixed` or `32-true` choice in step 3 cannot quietly run
> at a fraction of the speed.
"""

EXPORT_MD = """
---
## &#128230; Export and package

Step 6 converts checkpoints to ONNX; step 7 assembles them into an OpenUtau voicebank.

A complete voicebank needs **both** an acoustic and a variance model. Acoustic alone
renders audio but cannot predict duration, pitch or parameters on its own.
"""

UTIL_MD = """
---
## &#129520; Utilities

Optional. Neither is part of the normal train &rarr; export path.
"""

CHANGES_MD = """
---
## &#128221; What is different from the original notebook

**Two training modes**

Original keeps the stock DiffSinger behaviour. Woah's Epic Faster Training (experimental)
parallelises binarize, sizes batches to the GPU, uses bf16 where the hardware has it, keeps
the GPU fed with more dataloader workers and trims validation &mdash; all through stock config
keys, so switching back is one dropdown.

**Setup went from ~1 hour to a few minutes**

| | Original | Here |
|---|---|---|
| Python | installed into Colab's interpreter, now 3.13, where `numpy<2`, `onnx~=1.16` and `onnxsim` have **no wheels** &mdash; pip silently falls back to compiling them from source on two cores | a Python 3.11 environment built by `uv`, where every pin has a wheel; the kernel is left alone |
| PyTorch | reinstalled from PyPI over Colab's CUDA build, often landing a CPU-only wheel | pinned `torch==2.5.1` (before `torch.load` flipped to `weights_only=True` and broke Lightning 2.3 checkpoint loading) |
| SOME deps | installed `SOME/requirements.txt`: `fairseq==0.12.2`, `gradio`, and `onnx==1.14.0` fighting DiffSinger's `onnx~=1.16.0` | skipped; SOME's script only imports `click`, `librosa`, `tqdm`, `yaml`, all already required |
| pip runs | 7 separate invocations, 7 dependency resolves | 1 resolve, all wheels |
| Downloads | ~1.5 GB serially over single-connection `wget` | `aria2c`, 4 files at a time, 16 connections each |
| ONNX export | installed Miniconda over `/usr/local`, replacing the interpreter and forcing a restart that wiped every variable | a CPU-only Python 3.10 venv built on demand in about a minute; your runtime is untouched |

Why two environments: training wants a modern torch, but `requirements-onnx.txt` pins
`torch==1.13.1`, which only exists for Python 3.10 or older. Rather than destroy the
runtime to satisfy both, each gets its own interpreter and the notebook calls into
them. Nothing is ever installed into the kernel except `uv` itself.

**Bugs that made things fail for real**

- `hnsep` was unpacked to `checkpoints/hnsep_240512/` while every config asks for
  `checkpoints/vr/model.pt`, so tension / voicing / breathiness extraction could
  never find its model. Now installed where the config points.
- The SOME checkpoint was looked up as `0119_continuous256_5spk` while the download
  is `0119_continuous128_5spk`. Now located by search, not by a hard-coded name.
- Label normalisation (`SP`&rarr;`pau`, `br`&rarr;`AP`) ran on the csv/ds branch instead of
  the lab branch &mdash; a comment in the original admits it &mdash; so it never touched a
  `.lab` file. It also used a raw string replace that corrupted any phoneme
  containing those letters. Now it rewrites the phoneme column only.
- Choosing `parselmouth` vs `harvest` for MIDI estimation did nothing: the choice
  was stored and never passed to the segmenter. Now passed as `-p`.
- Fine-tuning from your own base model raised `NameError: finetune_ckpt_path` &mdash; it
  was only assigned on the download branch.
- `--exp_name ${save_dir}` is bash expansion of a variable that does not exist, so
  the experiment name was always empty. The original compensated by rewriting
  `utils/hparams.py` and `utils/training_utils.py` with regexes on every run, which
  broke resuming and checkpoint deletion. DiffSinger resolves its work dir as
  `checkpoints/<exp_name>`, so a symlink does the whole job &mdash; no source patching.
- The ONNX exporter writes `<exp>.phonemes.json`, but the voicebank builder copied
  `acoustic_phonemes.json`. The failure was swallowed by a bare `except` and you got
  a voicebank that silently would not load. Filenames are now normalised explicitly,
  and nothing is caught and ignored.
- A bank with `pitch.onnx` but no `variance.onnx` hit `NameError: predict_dur`.
- Only the first row of each `transcriptions.csv` got its leading `SP` turned into
  `AP`; every other segment kept the wrong token. Now all of them.
- `!cmd {python_var}` broke on any path containing a space and threw away exit
  codes. Every command now runs through `subprocess` with real arguments.

**Things that are simply nicer**

- One voicebank builder instead of four (three of which the original labelled
  "legacy", "obsolete" and "even more obsolete").
- Pre-flight validation before binarizing: missing checkpoints, dictionaries that do
  not match your dataset's languages, pitch training with no MIDI in the data.
- A dataset summary after step 2 &mdash; speakers, segment counts, hours &mdash; so you find
  out about the singer whose files did not convert now, not 40 minutes into training.
- Binarized data goes to local disk by default; reading it over Drive's FUSE mount
  measurably slows training.
- Dictionaries are copied into your model folder, so the model folder is
  self-contained and still works in a fresh runtime.
- Interrupting the training cell also stops the training process, instead of leaving
  it holding the GPU so the next run dies with an out-of-memory error.

Credit where it is due: the original is by **usamireko**, maintained before that by
**MLo7Ghinsan**, and started life as **Kei's** notebook. The voicebank builder is
adapted from **agentasteriski's** DiffTrainer. This is a rewrite of their work, not a
replacement for it.
"""


def main():
    cells = [
        markdown(INTRO),
        markdown(SETUP_MD),
        markdown("### 0 &middot; Helper library\n"
                 "Writes `/content/dsnb.py`. Everything else imports it, which is why the "
                 "notebook survives a runtime restart. Run it once."),
        code("00_helpers.py"),
        code("01_setup.py"),
        markdown(DATA_MD),
        code("02_dataset.py"),
        code("03_config.py"),
        code("04_binarize.py"),
        markdown(TRAIN_MD),
        code("05_train.py"),
        markdown(EXPORT_MD),
        code("06_export_onnx.py"),
        code("07_build_voicebank.py"),
        markdown(UTIL_MD),
        code("08_util_drop_speakers.py"),
        code("09_util_package_dataset.py"),
        markdown(CHANGES_MD),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = f"cell{index:02d}"

    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    # Read it back and confirm every cell still has its line breaks. Joining the
    # source list must reproduce the text exactly - if the trailing newlines go
    # missing the notebook renders as one unreadable run-on line.
    written = json.loads(OUT.read_text(encoding="utf-8"))
    for index, cell in enumerate(written["cells"]):
        joined = "".join(cell["source"])
        if "\n" not in joined and len(joined) > 200:
            raise SystemExit(f"cell {index} lost its line breaks")
        for line in cell["source"][:-1]:
            if not line.endswith("\n"):
                raise SystemExit(f"cell {index} has a source line without a newline: {line[:60]!r}")
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB, {len(cells)} cells, "
          f"{sum(len(c['source']) for c in written['cells'])} source lines verified)")


if __name__ == "__main__":
    sys.exit(main())
