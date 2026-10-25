%%writefile /content/dsnb.py
"""
dsnb.py - shared helper library for the DiffSinger Colab notebook.

Written to /content so every other cell can `import dsnb` even after a runtime
restart.  This is why the notebook does not depend on variables left over in
memory from earlier cells - the single biggest source of "NameError" in the
original notebook.
"""

import contextlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import wave
import zipfile
from pathlib import Path

CONTENT = Path("/content")
REPO = CONTENT / "DiffSinger"
STATE_PATH = CONTENT / "ds_state.json"
LOG_DIR = CONTENT / "nb_logs"

# DiffSinger pins numpy<2 and librosa<0.10, which means it needs Python 3.12 or
# older.  Colab now ships 3.13, where those pins have no wheels at all and pip
# falls back to compiling numpy, onnx and onnxsim from source on two cores - the
# single biggest reason setup "takes an hour and then fails".  So the training
# code gets its own interpreter, and the notebook kernel is left alone.
DS_ENV = CONTENT / "ds_env"          # python 3.11 + CUDA torch: everything but export
DS_ENV_PYTHON_VERSION = "3.11"
ONNX_ENV = CONTENT / "onnx_env"      # python 3.10 + CPU torch 1.13: ONNX export only
ONNX_ENV_PYTHON_VERSION = "3.10"

# Environment fixes applied to every subprocess we launch.
#   MPLBACKEND       Colab exports module://matplotlib_inline.backend_inline, which
#                    only exists in the kernel. Anything importing matplotlib in the
#                    venv (torchmetrics does) dies on "not a valid value for backend".
#   PYTHONPATH       Colab points this at /env/python, i.e. 3.13 site-packages. A
#                    3.11 interpreter must not see those.
#   PYTHONNOUSERSITE Same reasoning for ~/.local.
SUBPROCESS_ENV = {
    "MPLBACKEND": "Agg",
    "PYTHONPATH": "",
    "PYTHONNOUSERSITE": "1",
}

# --------------------------------------------------------------------------- #
#  Errors + pretty output
# --------------------------------------------------------------------------- #


class NotebookError(RuntimeError):
    """A problem we understand and can explain. Never a bare traceback."""


# Colab renders ANSI colour in cell output, which keeps the [ok]/[warn] lines readable
# at a glance. Log files get the same text; the escapes are harmless there.
_C = {"green": "\033[92m", "cyan": "\033[96m", "yellow": "\033[93m", "red": "\033[91m",
      "magenta": "\033[95m", "bold": "\033[1m", "dim": "\033[2m", "end": "\033[0m"}


def _html(markup):
    """Show rich HTML when running inside a notebook. Returns False elsewhere."""
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display
        if get_ipython() is None:
            return False
        display(HTML(markup))
        return True
    except Exception:
        return False


def _esc(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_PANEL = ("font-family:Inter,Segoe UI,Roboto,sans-serif;border-radius:12px;"
          "padding:14px 18px;margin:12px 0 6px;color:#fff;")


def die(problem, fix=None, detail=None):
    """Print a readable failure panel and stop the cell."""
    width = 78
    print("\n" + "=" * width)
    print(f"  {_C['red']}{_C['bold']}STOPPED:{_C['end']} " + problem)
    if detail:
        print("-" * width)
        for line in str(detail).rstrip().splitlines()[-30:]:
            print("  " + line)
    if fix:
        print("-" * width)
        print("  How to fix:")
        for line in str(fix).strip().splitlines():
            print("    " + line)
    print("=" * width + "\n")
    _html(f'<div style="{_PANEL}background:linear-gradient(135deg,#dc2626,#7f1d1d);">'
          f'<div style="font-size:16px;font-weight:700">&#9940; Stopped: {_esc(problem)}</div>'
          + (f'<div style="margin-top:8px;font-size:13px"><b>How to fix</b><br>'
             f'<span style="white-space:pre-wrap;font-family:monospace">'
             f'{_esc(str(fix).strip())}</span></div>' if fix else "")
          + "</div>")
    # SystemExit rather than an ordinary exception: the panel above already says
    # everything useful, and IPython prints SystemExit without a traceback.
    raise SystemExit(problem)


def ok(msg):
    print(f"  {_C['green']}✔{_C['end']}  " + str(msg))


def info(msg):
    print(f"  {_C['cyan']}ℹ{_C['end']}  " + str(msg))


def warn(msg):
    print(f"  {_C['yellow']}⚠  " + str(msg) + _C["end"])


DEFAULT_ACCENT = ("#6366f1", "#8b5cf6")
FAST_ACCENT = ("#f97316", "#db2777")
DONE_ACCENT = ("#059669", "#0d9488")


def header(title, subtitle=None, accent=DEFAULT_ACCENT):
    """A gradient banner in Colab; a ruled text header anywhere else."""
    shown = _html(
        f'<div style="{_PANEL}background:linear-gradient(135deg,{accent[0]},{accent[1]});'
        f'box-shadow:0 4px 14px rgba(0,0,0,.18)">'
        f'<div style="font-size:17px;font-weight:700;letter-spacing:.2px">{_esc(title)}</div>'
        + (f'<div style="font-size:12.5px;opacity:.9;margin-top:3px">{_esc(subtitle)}</div>'
           if subtitle else "")
        + "</div>")
    if not shown:
        print("\n" + "=" * 78)
        print("  " + title + (f"  -  {subtitle}" if subtitle else ""))
        print("=" * 78)


def human_time(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m {seconds % 60:02d}s"


def human_size(num_bytes):
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


@contextlib.contextmanager
def step(name):
    """Time a stage and always say whether it worked."""
    print(f"\n{_C['bold']}{_C['magenta']}▶ {name}{_C['end']}")
    started = time.time()
    try:
        yield
    except NotebookError:
        raise
    except Exception as exc:
        print(f"{_C['red']}✖ {name} FAILED after {human_time(time.time() - started)}{_C['end']}")
        die(f"{name} failed", detail=f"{type(exc).__name__}: {exc}")
    else:
        print(f"{_C['dim']}  └ done in {human_time(time.time() - started)}{_C['end']}")


# --------------------------------------------------------------------------- #
#  Running commands
# --------------------------------------------------------------------------- #


def run(cmd, cwd=None, check=True, quiet=False, log_file=None, env=None):
    """
    Run a command from a *list* of arguments.

    Everything the original notebook did with `!cmd {python_var}` broke as soon
    as a path contained a space, and swallowed the exit code.  This does not:
    arguments are passed through without a shell, and a non-zero exit raises
    with the tail of the output attached.
    """
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    cmd = [str(part) for part in cmd]

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handle = open(log_file, "a", encoding="utf-8", errors="replace") if log_file else None
    tail = []
    merged_env = {**os.environ, **SUBPROCESS_ENV, **(env or {})}

    process = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        errors="replace",
        env=merged_env,
    )
    try:
        for line in process.stdout:
            if handle:
                handle.write(line)
            tail.append(line)
            if len(tail) > 200:
                del tail[0]
            if not quiet:
                sys.stdout.write(line)
        process.wait()
    except KeyboardInterrupt:
        # Stop the child too. Without this, an interrupted training cell leaves a
        # python process holding the GPU and the next run dies with "out of memory".
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        print("\n  [info] interrupted - child process stopped")
        raise
    finally:
        if handle:
            handle.close()

    if check and process.returncode != 0:
        die(
            f"command failed (exit {process.returncode}): {' '.join(cmd[:4])} ...",
            fix=f"Full log: {log_file}" if log_file else None,
            detail="".join(tail),
        )
    return process.returncode


def run_out(cmd):
    """Run and return stdout as text; empty string if the command fails."""
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    try:
        return subprocess.run(
            [str(c) for c in cmd], capture_output=True, text=True, errors="replace"
        ).stdout.strip()
    except Exception:
        return ""


def have(binary):
    return shutil.which(binary) is not None


# --------------------------------------------------------------------------- #
#  Interpreters
# --------------------------------------------------------------------------- #


def uv_path(log_file=None, quiet=True):
    """Path to `uv`, installing it into the kernel first if need be.

    uv ships as a self-contained binary, so this is a few seconds and cannot
    disturb the kernel's own packages.
    """
    found = shutil.which("uv")
    if found:
        return found
    run([sys.executable, "-m", "pip", "install", "-q", "uv"],
        log_file=log_file, quiet=quiet)
    found = shutil.which("uv")
    if found:
        return found
    candidate = Path(sys.executable).parent / "uv"
    if candidate.exists():
        return str(candidate)
    die("could not install uv",
        fix="Run `!pip install uv` in a new cell, then run setup again.")


def env_python(env_dir):
    candidate = Path(env_dir) / "bin" / "python"
    return str(candidate) if candidate.exists() else None


def training_python():
    """The interpreter that actually has DiffSinger's dependencies."""
    found = env_python(DS_ENV)
    if found is None:
        die("the DiffSinger environment is missing",
            fix="Run step 1 (Setup). It builds a Python "
                f"{DS_ENV_PYTHON_VERSION} environment at {DS_ENV}.")
    return found


def req_name(requirement_line):
    """'librosa<0.10.0' -> 'librosa'."""
    # maxsplit by keyword: passing it positionally is deprecated as of python 3.13.
    return re.split(r"[<>=!~\[; ]", str(requirement_line).strip(), maxsplit=1)[0].lower()


def venv_extras(requirements):
    """
    Packages every DiffSinger venv needs but no requirements file lists.

    A venv is not a Colab-wide install: things that are simply present system-wide
    have to be asked for.
    """
    present = {req_name(entry) for entry in requirements}
    extras = []
    if "setuptools" not in present:
        # lightning and librosa both `import pkg_resources`, which lives in
        # setuptools and is gone from 81 onwards.
        extras.append("setuptools<81")
    if "onnxsim" in present and "onnxruntime" not in present:
        # onnxsim shells out to `pip install onnxruntime` on first import if it is
        # missing, which fails inside a venv and takes the import down with it.
        extras.append("onnxruntime")
    return extras


def build_env(env_dir, python_version, requirements, log_file=None, quiet=True):
    """
    Create a throwaway interpreter and install `requirements` into it.

    Everything goes in ONE resolve, torch included.  Installing torch first and
    the rest afterwards does not work: `requirements-onnx.txt` leaves torchmetrics
    unpinned, modern torchmetrics wants torch>=2, and the second pass happily
    upgrades the torch you just pinned - which is how you end up with torch 2.14
    in an environment that asked for 1.13.1.  A single resolve makes the pin
    binding and forces the resolver to pick companions that fit it.
    """
    env_dir = Path(env_dir)
    requirements = list(requirements) + venv_extras(requirements)
    uv = uv_path(log_file=log_file, quiet=quiet)
    run([uv, "python", "install", python_version], log_file=log_file, quiet=quiet)
    # --seed puts pip and setuptools in the venv. Without it there is no
    # pkg_resources, and both lightning and librosa import it on sight.
    run([uv, "venv", "--seed", "--python", python_version, str(env_dir)],
        log_file=log_file, quiet=quiet)
    python = env_dir / "bin" / "python"

    spec = env_dir.with_suffix(".requirements.txt")
    spec.write_text("\n".join(requirements) + "\n", encoding="utf-8")
    run([uv, "pip", "install", "--python", str(python), "-r", str(spec)],
        log_file=log_file, quiet=quiet)
    return str(python)


def assert_pinned(python, module, expected, log_file=None):
    """
    Confirm a package really is the version we asked for.

    A resolver that moves torch is the single most destructive thing that can
    happen to one of these environments, and it fails much later and much less
    legibly than it should.
    """
    actual = run_out([str(python), "-c",
                      f"import {module};print({module}.__version__)"])
    if not actual:
        warn(f"could not read {module}.__version__ from {python}")
        return None
    if not actual.startswith(expected):
        die(f"{module} is {actual}, but this environment pinned {expected}",
            fix="Something in the requirements pulled a different version in.\n"
                "Re-run this cell with the force/rebuild option ticked. If it happens\n"
                "again, the log shows which package demanded it.\n"
                + (f"Log: {log_file}" if log_file else ""))
    return actual


def module_versions(python, checks):
    """[(module, pip_name)] -> rows of {module, version} using that interpreter.

    Results are collected first and printed last behind a sentinel, because some
    packages print during import - onnxsim shells out to pip - and anything they
    write to stdout would otherwise be parsed as if it were a result row.
    """
    probe = (
        "import importlib.metadata as md\n"
        "rows = []\n"
        "for module, dist in %r:\n"
        "    try:\n"
        "        __import__(module)\n"
        "    except Exception as exc:\n"
        "        reason = type(exc).__name__ + ': ' + str(exc).replace(chr(10), ' ')[:70]\n"
        "        rows.append((module, 'MISSING', reason))\n"
        "        continue\n"
        "    try:\n"
        "        rows.append((module, md.version(dist), ''))\n"
        "    except Exception:\n"
        "        rows.append((module, 'ok', ''))\n"
        "for row in rows:\n"
        "    print('@@ROW@@' + chr(9).join(row))\n"
    ) % (list(checks),)
    result = subprocess.run([str(python), "-c", probe], capture_output=True, text=True,
                            env={**os.environ, **SUBPROCESS_ENV})
    rows, broken = [], []
    for line in result.stdout.splitlines():
        if not line.startswith("@@ROW@@"):
            continue
        parts = line[len("@@ROW@@"):].split("\t")
        while len(parts) < 3:
            parts.append("")
        rows.append({"module": parts[0], "version": parts[1]})
        if parts[1] == "MISSING":
            broken.append((parts[0], parts[2] or "import failed"))
    if not rows:
        die("could not query the environment",
            detail=(result.stderr or result.stdout)[-1500:],
            fix="Re-run setup with force_reinstall ticked.")
    return rows, broken


# --------------------------------------------------------------------------- #
#  Downloading
# --------------------------------------------------------------------------- #


def download_many(items, log_file=None, quiet=True):
    """
    items: list of (url, destination_path).

    Uses aria2c with 16 connections per file and 4 files at a time, which is
    where most of the original setup's wall-clock went - it fetched ~1.5 GB of
    checkpoints one at a time with single-connection wget.
    """
    items = [(url, Path(dest)) for url, dest in items]
    todo = []
    for url, dest in items:
        if dest.exists() and dest.stat().st_size > 0:
            info(f"already have {dest.name} ({human_size(dest.stat().st_size)})")
        else:
            todo.append((url, dest))
    if not todo:
        return

    if have("aria2c"):
        spec = CONTENT / ".aria2_input"
        lines = []
        for url, dest in todo:
            dest.parent.mkdir(parents=True, exist_ok=True)
            lines.append(url)
            lines.append(f"  dir={dest.parent}")
            lines.append(f"  out={dest.name}")
        spec.write_text("\n".join(lines) + "\n", encoding="utf-8")
        run(
            [
                "aria2c", "--input-file", str(spec),
                "--max-concurrent-downloads", "4",
                "--max-connection-per-server", "16",
                "--split", "16",
                "--min-split-size", "1M",
                "--continue", "true",
                "--auto-file-renaming", "false",
                "--allow-overwrite", "true",
                "--console-log-level", "warn",
                "--summary-interval", "0",
            ],
            log_file=log_file,
            quiet=quiet,
        )
        spec.unlink(missing_ok=True)
    else:
        for url, dest in todo:
            dest.parent.mkdir(parents=True, exist_ok=True)
            run(["curl", "-L", "--fail", "--silent", "--show-error", "-o", str(dest), url],
                log_file=log_file, quiet=quiet)

    missing = [dest.name for _, dest in todo if not dest.exists() or dest.stat().st_size == 0]
    if missing:
        die("some downloads did not arrive: " + ", ".join(missing),
            fix="Re-run this cell. If it keeps failing the release URL may have moved.")
    for _, dest in todo:
        ok(f"{dest.name} ({human_size(dest.stat().st_size)})")


def unzip(archive, dest):
    """Extract a zip with the stdlib - no dependency on p7zip being present."""
    archive, dest = Path(archive), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)


def extract_archive(archive, dest):
    """
    Extract .zip/.7z/.rar/.tar.* for user datasets.

    Prefers 7z when available (handles rar/7z), falls back to Python for zip and
    tar so a missing p7zip is not a hard failure like it was before.
    """
    archive, dest = Path(archive), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    suffix = archive.suffix.lower()

    if have("7z") and suffix in (".7z", ".rar", ".zip"):
        run(["7z", "x", "-y", str(archive), f"-o{dest}"], quiet=True)
        return
    if suffix == ".zip":
        unzip(archive, dest)
        return
    if suffix in (".gz", ".bz2", ".xz", ".tar", ".tgz"):
        import tarfile

        with tarfile.open(archive) as tf:
            tf.extractall(dest)
        return
    if have("7z"):
        run(["7z", "x", "-y", str(archive), f"-o{dest}"], quiet=True)
        return
    die(f"do not know how to extract {archive.name}",
        fix="Re-pack your dataset as a .zip file and try again.")


# --------------------------------------------------------------------------- #
#  Cross-cell state
# --------------------------------------------------------------------------- #


def load_state(required=None):
    if not STATE_PATH.exists():
        die("no saved notebook state found",
            fix="Run the Setup cell (step 1) first. It creates /content/ds_state.json.")
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    for key in required or []:
        if key not in state:
            die(f"state is missing '{key}'",
                fix=_state_hint(key))
    return state


def save_state(**values):
    state = {}
    if STATE_PATH.exists():
        try:
            state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            state = {}
    state.update(values)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def _state_hint(key):
    hints = {
        "setup_done": "Run step 1 (Setup).",
        "data_dir": "Run step 2 (Prepare dataset).",
        "dictionary_files": "Run step 2 (Prepare dataset).",
        "config_path": "Run step 3 (Build config).",
        "model_dir": "Run step 3 (Build config).",
    }
    return hints.get(key, "Re-run the earlier cells in order.")


# --------------------------------------------------------------------------- #
#  Environment reporting
# --------------------------------------------------------------------------- #


def gpu_report():
    out = run_out(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
         "--format=csv,noheader"]
    )
    if not out:
        return None
    name, mem, driver = [part.strip() for part in out.splitlines()[0].split(",")]
    return {"name": name, "memory": mem, "driver": driver}


def gpu_specs():
    """Name, VRAM in GB and CUDA compute capability of the first GPU, or None."""
    out = run_out(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                   "--format=csv,noheader,nounits"])
    if not out:
        return None
    try:
        name, mem, cap = [part.strip() for part in out.splitlines()[0].split(",")]
        return {"name": name, "vram_gb": float(mem) / 1024, "compute_cap": float(cap)}
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
#  Training modes
# --------------------------------------------------------------------------- #

ORIGINAL_MODE = "Original"
FAST_MODE = "Woah's Epic Faster Training (experimental)"


def is_fast(mode):
    return str(mode).startswith("Woah")


def fast_binarize_workers():
    """
    Parallel binarize workers. Each one loads its own copy of the pitch and
    harmonic/noise models, so leave a core for the kernel on big runtimes. On a
    2-core T4 two workers still win: while one waits on the GPU (RMVPE, VR) the
    other is decoding audio.
    """
    return min(8, max(2, cpu_count() - 1))


def fast_train_overrides(model_type, gpu=None, optimizer_cls=""):
    """
    What Woah's fast mode layers on top of the config: (dotted.key, value, reason).

    Every key is a stock DiffSinger hparam - nothing is patched, and Original mode
    stays exactly the stock behaviour. Where the speed comes from:
      - the right precision for the card: bf16 on Ampere+ (L4, A100), fp16 tensor
        cores on a T4 - which also rescues a T4 from bf16 / fp32 picked in step 3,
        both of which run at a fraction of fp16 speed there
      - fused AdamW: one CUDA kernel per step instead of one per tensor
      - batches sized to the GPU's VRAM instead of one T4-safe number
      - more dataloader workers and deeper prefetch so the GPU never waits on CPU
      - cheaper validation and logging
    torch.compile is handled separately by the launcher (FAST_LAUNCHER below).
    """
    gpu = gpu if gpu is not None else gpu_specs()
    vram = gpu["vram_gb"] if gpu else 15.0
    cap = gpu["compute_cap"] if gpu else 7.5

    # (min VRAM GB, max_batch_size, max_batch_frames). Variance models cost far
    # less per frame, which is why upstream already defaults them higher.
    if model_type == "acoustic":
        tiers = [(70, 64, 160000), (36, 48, 100000), (22, 32, 64000), (0, 24, 48000)]
    else:
        tiers = [(70, 96, 240000), (36, 64, 160000), (22, 48, 120000), (0, 32, 80000)]
    batch, frames = next((b, f) for limit, b, f in tiers if vram >= limit)

    # The training loop itself needs a core. Colab's T4 runtime has two, and two
    # loader workers there starve the process that launches the GPU kernels;
    # DiffSinger's loading is light (pre-binarized arrays), so one keeps up.
    workers = max(1, min(8, cpu_count() - 1))
    rows = [
        ("max_batch_size", batch, f"sized for {vram:.0f} GB of VRAM"),
        ("max_batch_frames", frames, "fills the GPU instead of the T4-safe default"),
        ("ds_workers", workers, f"{cpu_count()} CPU cores: one left for the training loop"),
        ("dataloader_prefetch_factor", 4, "keeps batches queued ahead of the GPU"),
        ("num_valid_plots", 3, "validation renders 3 samples instead of 10"),
        ("num_sanity_val_steps", 0, "skips the warm-up validation pass"),
        ("log_interval", 500, "TensorBoard scalars every 500 steps, not 100"),
    ]
    name = gpu["name"] if gpu else "this GPU"
    if cap >= 8.0:
        rows.insert(0, ("pl_trainer_precision", "bf16-mixed",
                        f"{name} supports bfloat16 natively"))
    else:
        rows.insert(0, ("pl_trainer_precision", "16-mixed",
                        f"fp16 tensor cores - bf16 is emulated on {name}"))
    # Under fp16, Lightning refuses gradient clipping (DiffSinger always clips) on
    # an optimizer that unscales gradients itself, which fused AdamW does. Muon_AdamW
    # wraps its AdamW in a ChainedOptimizer that leaves unscaling to Lightning, so
    # fused is safe there at any precision; plain AdamW only gets it under bf16.
    if optimizer_cls.endswith("Muon_AdamW"):
        rows.append(("optimizer_args.adamw_args.fused", True, "fused AdamW kernel"))
    elif cap >= 8.0:
        rows.append(("optimizer_args.fused", True, "fused AdamW kernel (bf16)"))
    return rows


# Keys fast mode may touch. Original mode puts each back to its stock value, so
# flipping the dropdown back after a fast run really is the stock setup again.
FAST_KEYS = ("pl_trainer_precision", "max_batch_size", "max_batch_frames", "ds_workers",
             "dataloader_prefetch_factor", "num_valid_plots", "num_sanity_val_steps",
             "log_interval", "optimizer_args.fused", "optimizer_args.adamw_args.fused")


def _dotted(cfg, key, value=None, delete=False):
    *parents, leaf = key.split(".")
    node = cfg
    for part in parents:
        if not isinstance(node.get(part), dict):
            if delete:
                return
            node[part] = {}
        node = node[part]
    if delete:
        node.pop(leaf, None)
    else:
        node[leaf] = value


def apply_training_mode(cfg, rows=None, stock=None):
    """Fast mode: write `rows`. Original mode: restore FAST_KEYS from `stock`."""
    if rows is not None:
        for key, value, _ in rows:
            _dotted(cfg, key, value)
        return
    stock = stock or {}
    for key in FAST_KEYS:
        value = stock
        for part in key.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        if value is None:
            _dotted(cfg, key, delete=True)
        else:
            _dotted(cfg, key, value)


FAST_ENV = {
    # TF32 for whatever still runs in fp32 on Ampere+. No effect on a T4.
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1",
    # Fewer allocator stalls and less fragmentation with variable-length batches.
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    # Read by FAST_LAUNCHER: redraw the progress bar every 10 steps, not every step.
    "NB_PROGRESS_EVERY": "10",
}

# Fast-mode launcher. It applies runtime tweaks that have no config key, then hands
# over to the unmodified scripts/train.py. The DiffSinger tree is never edited.
#   - TF32 matmuls / convolutions for anything still in fp32 (Ampere+)
#   - progress bar redrawn every NB_PROGRESS_EVERY steps instead of every step.
#     Each redraw formats every metric and pushes a line through the pipe to the
#     notebook; on a 2-core T4 that is CPU time stolen from launching kernels.
#   - torch.compile on the diffusion backbone, which is where nearly all of the
#     FLOPs are. The class `forward` is wrapped, not the module, so state_dict
#     keys - and therefore checkpoints and ONNX export - are unchanged.
#     dynamic=True because DiffSinger batches vary in length every step.
FAST_LAUNCHER = r'''
import os
import runpy
import sys

repo = sys.argv[1]
script = os.path.join(repo, "scripts", "train.py")
sys.argv = [script] + sys.argv[2:]
sys.path.insert(0, repo)

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

every = int(os.environ.get("NB_PROGRESS_EVERY", "1"))
if every > 1:
    try:
        import utils.training_utils as training_utils
        bar_init = training_utils.DsTQDMProgressBar.__init__

        def init_with_refresh(self, refresh_rate=1, *args, **kwargs):
            bar_init(self, every, *args, **kwargs)

        training_utils.DsTQDMProgressBar.__init__ = init_with_refresh
    except Exception as exc:
        print(f"| nb: progress bar left as-is ({type(exc).__name__}: {exc})", flush=True)

if os.environ.get("NB_TORCH_COMPILE") == "1":
    try:
        import torch._dynamo
        # A graph dynamo cannot handle runs eagerly instead of killing training.
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.cache_size_limit = 64
        import modules.backbones as backbones
        for name, cls in backbones.BACKBONES.items():
            cls.forward = torch.compile(cls.forward, dynamic=True)
        print("| nb: torch.compile enabled for backbones: " + ", ".join(backbones.BACKBONES),
              flush=True)
    except Exception as exc:
        print(f"| nb: torch.compile unavailable ({type(exc).__name__}: {exc}) - running eager",
              flush=True)

runpy.run_path(script, run_name="__main__")
'''

# Log lines that mean the compiler, not the model, killed a run.
COMPILE_FAILURE_MARKERS = ("torch._dynamo", "torch._inductor", "BackendCompilerFailed",
                           "InductorError", "triton.compiler", "CompilationError")


def write_fast_launcher():
    path = CONTENT / "nb_fast_train.py"
    path.write_text(FAST_LAUNCHER.lstrip("\n"), encoding="utf-8")
    return path


def disk_free(path="/content"):
    usage = shutil.disk_usage(path)
    return usage.free


def ram_total():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


def cpu_count():
    return os.cpu_count() or 2


# --------------------------------------------------------------------------- #
#  Checkpoints / experiment plumbing
# --------------------------------------------------------------------------- #


SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_exp_name(name):
    cleaned = SAFE_NAME.sub("_", str(name).strip()).strip("._-")
    return cleaned or "model"


def link_experiment(exp_name, target_dir):
    """
    Make `DiffSinger/checkpoints/<exp_name>` point at the user's model folder.

    DiffSinger computes its work dir as `checkpoints/<exp_name>`, so a symlink
    is all that is needed to train straight into Google Drive.  The original
    notebook instead rewrote utils/hparams.py and utils/training_utils.py with
    regexes on every run, which broke resuming and broke checkpoint deletion.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    link = REPO / "checkpoints" / exp_name
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        link.unlink()
    elif link.is_dir():
        # A real directory of the same name - never delete it silently.
        if link.resolve() != target_dir.resolve():
            die(f"checkpoints/{exp_name} already exists as a real folder",
                fix=f"Rename your model folder, or remove {link}")
        return link
    elif link.exists():
        link.unlink()
    link.symlink_to(target_dir, target_is_directory=True)
    return link


CKPT_STEPS = re.compile(r"steps_(\d+)")


def find_latest_ckpt(folder):
    folder = Path(folder)
    best, best_step = None, -1
    for path in folder.glob("*.ckpt"):
        match = CKPT_STEPS.search(path.name)
        step_num = int(match.group(1)) if match else 0
        if step_num > best_step:
            best, best_step = path, step_num
    return best


# --------------------------------------------------------------------------- #
#  Config loading / validation
# --------------------------------------------------------------------------- #


def load_yaml(path):
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def dump_yaml(data, path):
    import yaml

    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, sort_keys=False, allow_unicode=True)


def resolve_merged_config(config_path):
    """
    Reproduce DiffSinger's base_config chaining so we can validate the *effective*
    config, not just the leaf file.
    """
    config_path = Path(config_path)
    seen = set()

    def load_chain(path):
        path = Path(path).resolve()
        if path in seen:
            return {}
        seen.add(path)
        raw = load_yaml(path)
        merged = {}
        bases = raw.get("base_config") or []
        if isinstance(bases, str):
            bases = [bases]
        for base in bases:
            base_path = base
            if str(base).startswith("."):
                base_path = os.path.normpath(os.path.join(path.parent, base))
            elif not os.path.isabs(base):
                base_path = os.path.normpath(os.path.join(REPO, base))
            merged.update(load_chain(base_path))
        merged.update({k: v for k, v in raw.items() if k != "base_config"})
        return merged

    return load_chain(config_path)


def preflight(config_path, model_type):
    """
    Check everything binarize/train will reach for *before* spending 20 minutes
    finding out the hard way.  Returns nothing; raises with a specific fix.
    """
    cfg = resolve_merged_config(config_path)
    problems = []

    def check_file(key, label):
        value = cfg.get(key)
        if not value:
            return
        path = Path(value)
        if not path.is_absolute():
            path = REPO / path
        if not path.exists():
            problems.append(f"{label}: config '{key}' points at {value} which does not exist")

    if cfg.get("pe") == "rmvpe":
        check_file("pe_ckpt", "pitch extractor")
    wants_hnsep = any(
        cfg.get(k) for k in (
            "use_tension_embed", "use_voicing_embed", "use_breathiness_embed",
            "predict_tension", "predict_voicing", "predict_breathiness",
        )
    )
    if wants_hnsep and cfg.get("hnsep") == "vr":
        check_file("hnsep_ckpt", "harmonic/noise separator")
    if model_type == "acoustic":
        check_file("vocoder_ckpt", "vocoder")

    dictionaries = cfg.get("dictionaries") or {}
    for lang, dict_path in dictionaries.items():
        path = Path(dict_path)
        if not path.is_absolute():
            path = REPO / path
        if not path.exists():
            problems.append(f"dictionary '{lang}' missing: {dict_path}")

    languages_used = {ds.get("language") for ds in cfg.get("datasets") or []}
    unknown = {lang for lang in languages_used if lang and lang not in dictionaries}
    if unknown:
        problems.append(
            "dataset language(s) "
            + ", ".join(sorted(unknown))
            + " have no matching dictionary (have: "
            + ", ".join(sorted(dictionaries)) + ")"
        )

    for dataset in cfg.get("datasets") or []:
        raw_dir = Path(dataset.get("raw_data_dir", ""))
        if not raw_dir.exists():
            problems.append(f"raw_data_dir missing: {raw_dir}")
            continue
        if not (raw_dir / "transcriptions.csv").exists() and not list(raw_dir.glob("*.ds")):
            problems.append(f"{raw_dir.name}: no transcriptions.csv and no .ds files")

    binary_dir = cfg.get("binary_data_dir")
    if binary_dir:
        parent = Path(binary_dir).parent
        if not parent.exists():
            problems.append(f"binary_data_dir parent does not exist: {parent}")

    if problems:
        die("the config will not work as-is",
            fix="Fix the items above, then re-run the config cell.",
            detail="\n".join("- " + p for p in problems))
    ok("pre-flight checks passed")


# --------------------------------------------------------------------------- #
#  Dataset inspection
# --------------------------------------------------------------------------- #


def wav_seconds(path):
    """Duration from the WAV header only - no numpy, no librosa, instant."""
    try:
        with contextlib.closing(wave.open(str(path), "rb")) as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:
        return 0.0


def dataset_summary(data_dir):
    """Return per-speaker stats for a prepared DiffSinger dataset folder."""
    data_dir = Path(data_dir)
    rows = []
    for speaker_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        wavs = list((speaker_dir / "wavs").glob("*.wav"))
        ds_files = list(speaker_dir.glob("*.ds"))
        seconds = sum(wav_seconds(w) for w in wavs)
        rows.append({
            "speaker": speaker_dir.name,
            "segments": len(wavs) or len(ds_files),
            "hours": seconds / 3600.0,
            "has_csv": (speaker_dir / "transcriptions.csv").exists(),
        })
    return rows


def print_table(rows, columns):
    """Summary table: styled HTML in Colab, fixed-width text anywhere else."""
    if not rows:
        return
    cell = "padding:5px 14px;border-bottom:1px solid rgba(127,127,127,.25);text-align:left"
    markup = ('<table style="border-collapse:collapse;font-family:Inter,Segoe UI,Roboto,'
              'sans-serif;font-size:13px;margin:6px 0 10px">'
              + "<tr>" + "".join(f'<th style="{cell};font-weight:700;opacity:.75">{_esc(t)}</th>'
                                 for _, t in columns) + "</tr>"
              + "".join("<tr>" + "".join(f'<td style="{cell}">{_esc(row.get(k, ""))}</td>'
                                         for k, _ in columns) + "</tr>" for row in rows)
              + "</table>")
    if _html(markup):
        return
    widths = {}
    for key, title in columns:
        widths[key] = max(len(title), *(len(str(row.get(key, ""))) for row in rows))
    line = "  " + "  ".join(title.ljust(widths[key]) for key, title in columns)
    print(line)
    print("  " + "  ".join("-" * widths[key] for key, _ in columns))
    for row in rows:
        print("  " + "  ".join(str(row.get(key, "")).ljust(widths[key]) for key, _ in columns))
