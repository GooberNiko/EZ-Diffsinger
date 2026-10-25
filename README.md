# DiffSinger Colab notebook — rebuilt

`DiffSinger_Colab_Improved.ipynb` is a rewrite of the community DiffSinger notebook
(usamireko's, maintained before that by MLo7Ghinsan, originally Kei's). It targets the
two things people actually complain about: setup takes an hour, and when something
fails there is no way to tell what or why.

Upload the `.ipynb` to Colab, or open it from Drive.

## Layout

| File | What it is |
|---|---|
| `DiffSinger_Colab_Improved.ipynb` | The notebook. This is the deliverable. |
| `src/*.py` | One file per notebook cell, in run order. Edit these, not the `.ipynb`. |
| `src/build.py` | Assembles `src/` into the notebook. Syntax-checks every cell first. |
| `DiffSinger_colab_notebook OGRIGINAL.ipynb` | The original, kept for reference. |

Rebuild after editing a cell:

```bash
python src/build.py
```

`build.py` compiles each cell with IPython magics stripped and refuses to write the
notebook if anything fails to parse.

## Cell map

| Cell | Purpose |
|---|---|
| 0 | Writes `/content/dsnb.py`, the helper library every other cell imports |
| 1 | Setup: repos, packages, checkpoints |
| 2 | Unpack, segment, build phoneme dictionaries |
| 3 | Build and validate the training config |
| 4 | Binarize |
| 5 | Train (also resumes a previous run) |
| 6 | Export ONNX |
| 7 | Build an OpenUtau voicebank |
| — | Utilities: drop speakers, package the prepared dataset |

## Training modes

Picked in step 3 (`training_mode`), overridable in step 5.

- **Original**: stock DiffSinger behaviour, unchanged.
- **Woah's Epic Faster Training (experimental)**: changes only stock hparams.
  `binarization_args.num_workers` is set to cores - 1 (at least 2), and binary data always
  goes on local disk. At train time `max_batch_size` / `max_batch_frames` come from VRAM tiers.
  Precision becomes `bf16-mixed` on compute capability 8.0+ and is forced to `16-mixed`
  below that (a T4 emulates bf16). `ds_workers` is cores - 1 so the training loop keeps a core,
  `dataloader_prefetch_factor` is 4, `num_valid_plots` is 3, `num_sanity_val_steps` is 0 and
  `log_interval` is 500. Fused AdamW is turned on with `adamw_args.fused` for Muon_AdamW at
  any precision, since its ChainedOptimizer leaves unscaling to Lightning so clipping
  still works under fp16. Plain AdamW gets `optimizer_args.fused` only under bf16. The
  process also gets `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1` and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
  Training is started through `/content/nb_fast_train.py` (`FAST_LAUNCHER`). Before it
  runs the stock `scripts/train.py`, it turns on TF32, sets the progress bar to redraw every
  `NB_PROGRESS_EVERY` (10) steps, and wraps each backbone class's `forward` in
  `torch.compile(dynamic=True)`. State-dict keys stay the same, and dynamo's
  `suppress_errors` falls back to eager. A run that crashes on a compiler error (not OOM)
  restarts once with compile off. If parallel binarize fails it retries
  single-process. Switching back to Original restores the stock values and your step-3
  precision. The logic lives in `fast_train_overrides()` in `00_helpers.py`.

## Why DiffSinger gets its own Python

Colab now ships **Python 3.13**. DiffSinger pins `numpy<2.0.0`, `librosa<0.10.0`,
`onnx~=1.16.0` and `onnxsim~=0.4.36` — none of which publish cp313 wheels. `pip
install -r requirements.txt` therefore falls back to *building numpy, onnx and
onnxsim from source* on two vCPUs, which takes tens of minutes and frequently fails
outright. That is the real reason the original setup never finished.

Relaxing the pins would mean running DiffSinger against numpy 2 and librosa 0.10,
whose APIs its code predates. So instead:

| Environment | Python | For | Built |
|---|---|---|---|
| `/content/ds_env` | 3.11 | binarize, train, segmentation, SOME, drop_spk | step 1 |
| `/content/onnx_env` | 3.10 | `scripts/export.py` (pins `torch==1.13.1`) | step 6, on demand |

Both are created with [`uv`](https://github.com/astral-sh/uv), which downloads a
standalone CPython in seconds. The notebook kernel gets nothing installed into it
except `uv` itself, so Colab's own packages can never conflict with DiffSinger's —
and a future Colab Python bump cannot break setup again.

Three things a venv needs that a Colab-wide install gets for free, all handled in
`build_env()` and `SUBPROCESS_ENV`:

- **`pkg_resources`.** `uv venv` ships no seed packages by default, and both
  `lightning` and `librosa` import `pkg_resources` on sight. Fixed with
  `uv venv --seed` plus an explicit `setuptools<81` pin (81 drops `pkg_resources`).
- **`onnxruntime`.** On first import `onnxsim` shells out to `pip install
  onnxruntime` if it is absent, which fails in a venv and takes the import with it.
  Installed up front instead.
- **`MPLBACKEND`.** Colab exports `module://matplotlib_inline.backend_inline`, which
  only exists in the kernel. Anything importing matplotlib in the venv —
  `torchmetrics` does — dies on "not a valid value for backend". Every subprocess
  gets `MPLBACKEND=Agg` and an emptied `PYTHONPATH` so the 3.11 interpreter cannot
  reach Colab's 3.13 site-packages.

## Design notes

**No cross-cell globals.** The original passed state between cells in Python
variables (`all_shits_not_wav_n_lab`, `dictionary_files`, `export_mode`, …), so a
runtime restart or running cells out of order gave a `NameError` several cells later.
Here everything goes through `/content/ds_state.json`, and each cell says which
earlier step to run if something it needs is missing.

**No source patching.** The original rewrote `utils/hparams.py` and
`utils/training_utils.py` with regexes on every run to redirect the work directory.
DiffSinger already resolves its work dir as `checkpoints/<exp_name>`, so a symlink
does the job — and unlike the regex, it works for resuming and for the exporter.

**Failures are explained.** `dsnb.die()` prints the problem, the tail of the relevant
log, and a concrete fix. Nothing is wrapped in a bare `except` that prints and
continues, which is how the original produced voicebanks that silently would not load.

**Every command is a real subprocess.** `!cmd {python_var}` broke on paths with
spaces and discarded exit codes. `dsnb.run()` passes an argument list, streams output,
logs it, and raises with context on a non-zero exit.

The last markdown cell in the notebook lists the specific bugs this fixes.

## What was not tested

The notebook has not been executed on Colab from here — no GPU runtime was available
in this environment. Every cell is syntax-checked and pyflakes-clean, and the facts it
relies on (requirements files, config keys, `hparams.set_hparams`, exporter output
filenames, `db_converter` flags) were read from the upstream repositories rather than
recalled.

Watch these on a first run:

- **The 3.11 environment build.** Every pin should resolve to a wheel. If you see
  `Downloading <something>.tar.gz` followed by "Getting requirements to build wheel",
  something is still compiling and the pin needs adjusting.
- **The module/version table at the end of step 1.** It is the contract: every row
  must show a version, none may say `MISSING`. Anything that fails there will fail
  again during binarize, just later and less legibly.
- **`torch==2.5.1` with Lightning 2.3.** Chosen because torch 2.6 flipped
  `torch.load` to `weights_only=True`, which breaks checkpoint loading in Lightning
  2.3. If training starts but resuming fails on a pickle error, that pairing is why.
- **The 3.10 export environment in cell 6**, which is only built when you use it.
