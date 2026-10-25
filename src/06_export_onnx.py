#@title 6. Export to ONNX
#@markdown Converts trained checkpoints into the ONNX files OpenUtau loads.
#@markdown Fill in whichever models you have - blanks are skipped.

acoustic_model_dir = ""  #@param {type:"string"}
variance_model_dir = ""  #@param {type:"string"}
output_dir = ""  #@param {type:"string"}
#@markdown <font size="-1">An `onnx/acoustic` and `onnx/variance` folder are created here.</font>

export_backend = "isolated python 3.10 (recommended)"  #@param ["isolated python 3.10 (recommended)", "current runtime"]
#@markdown <font size="-1">DiffSinger pins `torch==1.13.1` for ONNX export, which needs Python 3.10.
#@markdown The isolated option builds a small CPU-only 3.10 environment in about a minute.
#@markdown <br>The original notebook installed Miniconda over `/usr/local` instead, which replaced
#@markdown the whole interpreter, forced a runtime restart, and wiped every variable the later cells needed.</font>

show_full_log = False  #@param {type:"boolean"}

# --------------------------------------------------------------------------- #

import hashlib
import importlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/content")
import dsnb
importlib.reload(dsnb)
from dsnb import (LOG_DIR, REPO, die, find_latest_ckpt, header, human_size, info,
                  link_experiment, load_state, ok, print_table, run, sanitize_exp_name,
                  save_state, step, warn)

state = load_state(required=["setup_done"])
LOG = LOG_DIR / "export.log"
QUIET = not show_full_log
ENV_DIR = dsnb.ONNX_ENV
ENV_PY = ENV_DIR / "bin" / "python"

header("Export ONNX")


def clean_path(value):
    return value.strip().strip('"').strip("'")


jobs = []
for kind, raw in (("acoustic", acoustic_model_dir), ("variance", variance_model_dir)):
    if not clean_path(raw):
        continue
    folder = Path(clean_path(raw))
    if not folder.exists():
        die(f"{kind} model folder not found: {folder}",
            fix="Point at the folder that holds model_ckpt_steps_*.ckpt and config.yaml.")
    ckpt = find_latest_ckpt(folder)
    if ckpt is None:
        die(f"no *.ckpt in {folder}",
            fix="Give the folder containing the checkpoints, not the .ckpt file itself.")
    if not (folder / "config.yaml").exists():
        die(f"no config.yaml in {folder}",
            fix="Training writes config.yaml next to the checkpoints. Without it the\n"
                "exporter cannot know how the model was built.")
    jobs.append({"kind": kind, "dir": folder, "ckpt": ckpt,
                 "exp": sanitize_exp_name(folder.name)})

if not jobs:
    die("nothing to export",
        fix="Fill in acoustic_model_dir and/or variance_model_dir.")
if not clean_path(output_dir):
    die("output_dir is empty", fix="e.g. /content/drive/MyDrive/DiffSinger/my_singer")
OUT_ROOT = Path(clean_path(output_dir))

for job in jobs:
    info(f"{job['kind']}: {job['ckpt'].name} from {job['dir'].name}")


# --------------------------------------------------------------------------- #
#  Export environment
# --------------------------------------------------------------------------- #
# Plain PyPI, not the +cpu index. The CPU wheels are a smaller download, but they
# live on a separate index, and mixing indexes needs --index-strategy juggling that
# is far more likely to go wrong than a one-off 900 MB fetch is to be slow.
EXPORT_TORCH = ["torch==1.13.1", "torchvision==0.14.1", "torchaudio==0.13.1"]
EXPORT_IMPORTS = [
    ("torch", "torch"),
    ("lightning", "lightning"),
    ("numpy", "numpy"),
    ("librosa", "librosa"),
    ("torchmetrics", "torchmetrics"),
    ("onnx", "onnx"),
    ("onnxruntime", "onnxruntime"),
    ("onnxsim", "onnxsim"),
    ("yaml", "PyYAML"),
    ("pkg_resources", "setuptools"),
]

if export_backend.startswith("isolated"):
    with step(f"Export environment (python {dsnb.ONNX_ENV_PYTHON_VERSION})"):
        requirements = []
        for raw in (REPO / "requirements-onnx.txt").read_text(encoding="utf-8").splitlines():
            entry = raw.split("#", 1)[0].strip()
            if entry and not entry.lower().startswith(("torch", "torchvision", "torchaudio")):
                requirements.append(entry)

        # Keyed to the requirement list, not just "does the folder exist" - an env
        # built by an older version of this notebook has to be rebuilt, not reused.
        fingerprint = hashlib.sha256(
            "|".join([dsnb.ONNX_ENV_PYTHON_VERSION] + EXPORT_TORCH + requirements).encode()
        ).hexdigest()[:16]
        marker = ENV_DIR.with_name(".onnx_env_complete")
        usable = (ENV_PY.exists()
                  and marker.exists()
                  and marker.read_text(encoding="utf-8").strip() == fingerprint
                  and not dsnb.module_versions(str(ENV_PY), EXPORT_IMPORTS)[1])

        if usable:
            ok("already built and everything imports - reusing it")
        else:
            if ENV_PY.exists():
                info("rebuilding: the existing environment is stale or incomplete")
                shutil.rmtree(ENV_DIR)
            marker.unlink(missing_ok=True)
            print(f"  torch 1.13.1 + {len(requirements)} packages, one resolve")
            info("torchmetrics is unpinned upstream and its recent versions demand")
            info("torch>=2, so torch has to be pinned inside the same resolve or it")
            info("gets silently upgraded and export.py dies on a missing symbol.")
            dsnb.build_env(
                ENV_DIR,
                dsnb.ONNX_ENV_PYTHON_VERSION,
                EXPORT_TORCH + requirements,
                log_file=LOG,
                quiet=QUIET,
            )
            dsnb.assert_pinned(ENV_PY, "torch", "1.13.1", log_file=LOG)
            rows, broken = dsnb.module_versions(str(ENV_PY), EXPORT_IMPORTS)
            if broken:
                print_table(rows, [("module", "module"), ("version", "version")])
                for module, reason in broken:
                    warn(f"{module}: {reason}")
                die(f"{len(broken)} module(s) missing from the export environment",
                    fix=f"Install them and re-run:\n"
                        f"  !{dsnb.uv_path()} pip install --python {ENV_PY} "
                        + " ".join(dict(EXPORT_IMPORTS)[m] for m, _ in broken))
            marker.write_text(fingerprint, encoding="utf-8")
            ok(f"built and verified at {ENV_DIR}")
    PYTHON = str(ENV_PY)
else:
    PYTHON = dsnb.training_python()
    warn("using the training environment - export.py pins torch 1.13, so if it")
    warn("complains, switch back to 'isolated python 3.10' and re-run.")


# --------------------------------------------------------------------------- #
#  Export
# --------------------------------------------------------------------------- #
# What the exporter writes is `{exp_name}.<thing>`; OpenUtau wants fixed names.
# Because we know exp_name exactly, this is a rename table rather than the
# substring guessing the original notebook did (which left `*.phonemes.json`
# untouched and quietly broke the voicebank builder further down).
RENAMES = {
    "acoustic": {
        "onnx": "acoustic.onnx",
        "phonemes.json": "acoustic_phonemes.json",
        "languages.json": "acoustic_languages.json",
    },
    "variance": {
        "linguistic.onnx": "linguistic.onnx",
        "dur.onnx": "dur.onnx",
        "pitch.onnx": "pitch.onnx",
        "variance.onnx": "variance.onnx",
        "phonemes.json": "variance_phonemes.json",
        "languages.json": "variance_languages.json",
    },
}

results = []
for job in jobs:
    kind, folder, exp = job["kind"], job["dir"], job["exp"]
    out = OUT_ROOT / "onnx" / kind
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    with step(f"Exporting {kind}"):
        link_experiment(exp, folder)
        run([PYTHON, "scripts/export.py", kind, "--exp", exp, "--out", str(out)],
            cwd=REPO, log_file=LOG, quiet=QUIET)

    with step(f"Normalising {kind} filenames"):
        prefix = exp + "."
        renamed = []
        for path in sorted(out.iterdir()):
            if not path.name.startswith(prefix):
                continue
            tail = path.name[len(prefix):]
            if tail.endswith(".emb"):
                new_name = tail                      # {speaker}.emb
            else:
                new_name = RENAMES[kind].get(tail)
            if not new_name:
                continue
            path.rename(out / new_name)
            renamed.append(f"{path.name} -> {new_name}")
        for line in renamed:
            ok(line)
        if not renamed:
            warn("no files matched the expected naming - check the export log")

        # The voicebank builder reads the training config for extra phonemes and
        # audio settings, so keep a copy right next to the ONNX files.
        shutil.copy(folder / "config.yaml", out / "config.yaml")
        for dictionary in (folder / "dictionaries").glob("*.txt"):
            shutil.copy(dictionary, out / dictionary.name)
        ok("config.yaml and dictionaries copied alongside the ONNX files")

    files = sorted(p for p in out.iterdir() if p.is_file())
    results += [{"model": kind, "file": p.name, "size": human_size(p.stat().st_size)}
                for p in files]
    job["out"] = str(out)


header("Export complete", accent=dsnb.DONE_ACCENT)
print_table(results, [("model", "model"), ("file", "file"), ("size", "size")])

expected = {"acoustic": ["acoustic.onnx", "acoustic_phonemes.json", "acoustic_languages.json",
                         "dsconfig.yaml"],
            "variance": ["linguistic.onnx", "dsconfig.yaml"]}
for job in jobs:
    missing = [name for name in expected[job["kind"]]
               if not (Path(job["out"]) / name).exists()]
    if missing:
        warn(f"{job['kind']} is missing: {', '.join(missing)}")

save_state(
    onnx_acoustic=next((j["out"] for j in jobs if j["kind"] == "acoustic"), None),
    onnx_variance=next((j["out"] for j in jobs if j["kind"] == "variance"), None),
)
print("\n  Next: step 7 - build the OpenUtau voicebank.")
