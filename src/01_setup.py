#@title 1. Setup  -  installs everything (safe to re-run)
#@markdown Installs everything DiffSinger needs. Safe to re-run: finished steps are skipped.
#@markdown <br><font size="-1">Typical cold run: **4-7 minutes**. Re-runs: a few seconds.</font>

#@markdown ---
branch = "main"  #@param ["main", "muon_lynxnet2"]
#@markdown <font size="-1">`muon_lynxnet2` is experimental - things can and do break there.</font>

profile = "training"  #@param ["training", "onnx export only", "everything"]
#@markdown <font size="-1">`onnx export only` skips the ~1 GB of training checkpoints.</font>

vocoder_choice = "pc_nsf_hifigan (recommended)"  #@param ["pc_nsf_hifigan (recommended)", "nsf_hifigan", "both"]

mount_drive = True  #@param {type:"boolean"}
install_some = True  #@param {type:"boolean"}
#@markdown <font size="-1">SOME is only needed if you want it to estimate MIDI for variance/pitch training.</font>

show_full_log = False  #@param {type:"boolean"}
force_reinstall = False  #@param {type:"boolean"}

# --------------------------------------------------------------------------- #

import hashlib
import importlib
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, "/content")
if not Path("/content/dsnb.py").exists():
    raise SystemExit("Run the cell above (step 0) first - it writes the helper library.")
import dsnb
importlib.reload(dsnb)
from dsnb import (CONTENT, DS_ENV_PYTHON_VERSION, LOG_DIR, REPO, die, download_many, have,
                  header, info, human_size, human_time, ok, print_table, run, run_out,
                  save_state, step, unzip, warn)

QUIET = not show_full_log
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG = LOG_DIR / "setup.log"
STAMP = CONTENT / ".setup_complete"
STARTED = time.time()

WANT_TRAINING = profile in ("training", "everything")
WANT_EXPORT = profile in ("onnx export only", "everything")

REPOS = {
    "DiffSinger": ("https://github.com/usamireko/DiffSinger.git", branch),
    "nnsvs-db-converter": ("https://github.com/UtaUtaUtau/nnsvs-db-converter", None),
    "MakeDiffSinger": ("https://github.com/openvpi/MakeDiffSinger", None),
    "ghin_shenanigans": ("https://github.com/MLo7Ghinsan/ghin_shenanigans", None),
    "SOME": ("https://github.com/openvpi/SOME", None),
}

ASSETS = {
    "rmvpe": (
        "https://github.com/openvpi/DiffSinger/releases/download/v2.1.0/rmvpe.zip",
        CONTENT / "dl" / "rmvpe.zip",
    ),
    "hnsep": (
        "https://github.com/yxlllc/vocal-remover/releases/download/hnsep_240512/hnsep_240512.zip",
        CONTENT / "dl" / "hnsep_240512.zip",
    ),
    "pc_nsf_hifigan": (
        "https://github.com/openvpi/vocoders/releases/download/"
        "pc-nsf-hifigan-44.1k-hop512-128bin-2025.02/"
        "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.zip",
        CONTENT / "dl" / "pc_nsf_hifigan.zip",
    ),
    "nsf_hifigan": (
        "https://github.com/openvpi/vocoders/releases/download/"
        "nsf-hifigan-44.1k-hop512-128bin-2024.02/"
        "nsf_hifigan_44.1k_hop512_128bin_2024.02.zip",
        CONTENT / "dl" / "nsf_hifigan.zip",
    ),
    "some": (
        "https://github.com/openvpi/SOME/releases/download/v1.0.0-baseline/"
        "0119_continuous128_5spk.zip",
        CONTENT / "dl" / "some_baseline.zip",
    ),
}

TARGET_PYTHON = tuple(int(part) for part in DS_ENV_PYTHON_VERSION.split("."))

# torch is pinned here rather than taken from requirements.txt (which does not
# list it at all - it says "install PyTorch manually").
#   2.5.1 because lightning~=2.3 predates torch 2.6, where torch.load flipped to
#   weights_only=True by default and checkpoint loading breaks.
TORCH_PIN = "torch==2.5.1"
TORCHAUDIO_PIN = "torchaudio==2.5.1"

# Packages we deliberately do NOT install, and why.
#   torch / torchvision / torchaudio  -> pinned above instead, so the resolver
#                                        cannot wander onto an incompatible one.
#   SOME/requirements.txt             -> pins fairseq==0.12.2 + gradio==3.47.1 and
#                                        onnx==1.14.0, which fight DiffSinger's
#                                        onnx~=1.16.0 and take ~30 min to resolve.
#                                        SOME's batch_infer only imports click,
#                                        librosa, tqdm and yaml, all of which
#                                        DiffSinger already requires.
SKIP_PACKAGES = {"torch", "torchvision", "torchaudio"}

# Not in requirements.txt but used by the MakeDiffSinger scripts. (setuptools and
# onnxruntime are added by dsnb.venv_extras, which every venv gets.)
EXTRA_PACKAGES = ["mido"]

# Pins with no wheel for a given Python or newer. Empty in practice at 3.11 -
# which is the whole point of not using the kernel's 3.13.
PIN_OVERRIDES = {
    "praat-parselmouth": ("praat-parselmouth>=0.4.5", (3, 12)),
}

IMPORT_CHECKS = [
    ("torch", "torch"),
    ("lightning", "lightning"),
    ("numpy", "numpy"),
    ("librosa", "librosa"),
    ("parselmouth", "praat-parselmouth"),
    ("pyworld", "pyworld-prebuilt"),
    ("onnx", "onnx"),
    ("onnxruntime", "onnxruntime"),
    ("onnxsim", "onnxsim"),
    ("yaml", "PyYAML"),
    ("einops", "einops"),
    ("tensorboard", "tensorboard"),
    ("torchmetrics", "torchmetrics"),
    ("matplotlib", "matplotlib"),
    ("h5py", "h5py"),
    ("resampy", "resampy"),
    ("mido", "mido"),
    ("pkg_resources", "setuptools"),
]


# --------------------------------------------------------------------------- #
header("DiffSinger setup")

if STAMP.exists() and not force_reinstall:
    info("setup already completed in this runtime - verifying instead of reinstalling")
    info("tick 'force_reinstall' if you want it done again from scratch")

# --- environment ----------------------------------------------------------- #
gpu = dsnb.gpu_report()
free = dsnb.disk_free("/content")
print(f"  python      {sys.version.split()[0]}")
print(f"  gpu         {gpu['name'] + ' (' + gpu['memory'] + ')' if gpu else 'NONE - training will not work'}")
print(f"  cpu / ram   {dsnb.cpu_count()} cores / {human_size(dsnb.ram_total())}")
print(f"  disk free   {human_size(free)}")

if not gpu and WANT_TRAINING:
    warn("no GPU detected. Runtime > Change runtime type > T4 GPU, then re-run.")
if free < 12 * 1024 ** 3 and WANT_TRAINING:
    warn(f"only {human_size(free)} free - binarized data plus checkpoints usually needs 15 GB+")

# --- drive ----------------------------------------------------------------- #
if mount_drive:
    with step("Mounting Google Drive"):
        if Path("/content/drive/MyDrive").exists():
            ok("already mounted")
        else:
            from google.colab import drive

            drive.mount("/content/drive")
            ok("mounted at /content/drive")

# --- system packages ------------------------------------------------------- #
with step("System packages"):
    needed = [pkg for pkg, binary in (("aria2", "aria2c"), ("p7zip-full", "7z"),
                                      ("unzip", "unzip")) if not have(binary)]
    if needed:
        if run(["apt-get", "install", "-y", "-qq"] + needed,
               log_file=LOG, quiet=QUIET, check=False) != 0:
            # Stale package lists are the usual cause and the usual cure.
            info("retrying after apt-get update")
            run(["apt-get", "update", "-qq"], log_file=LOG, quiet=QUIET)
            run(["apt-get", "install", "-y", "-qq"] + needed, log_file=LOG, quiet=QUIET)
        ok("installed " + ", ".join(needed))
    else:
        ok("aria2 / 7z / unzip already present")

# --- repositories ---------------------------------------------------------- #
with step("Repositories"):
    for name, (url, want_branch) in REPOS.items():
        if name == "SOME" and not install_some:
            continue
        target = CONTENT / name
        if (target / ".git").exists():
            current = run_out(["git", "-C", str(target), "rev-parse", "--abbrev-ref", "HEAD"])
            if want_branch and current != want_branch:
                info(f"{name}: switching {current} -> {want_branch}")
                run(["git", "-C", str(target), "remote", "set-branches", "origin", want_branch],
                    log_file=LOG, quiet=QUIET)
                run(["git", "-C", str(target), "fetch", "--depth", "1", "origin", want_branch],
                    log_file=LOG, quiet=QUIET)
                run(["git", "-C", str(target), "checkout", "-B", want_branch,
                     f"origin/{want_branch}"], log_file=LOG, quiet=QUIET)
            ok(f"{name} present ({want_branch or current})")
            continue
        cmd = ["git", "clone", "--depth", "1", "--quiet"]
        if want_branch:
            cmd += ["-b", want_branch]
        cmd += [url, str(target)]
        run(cmd, log_file=LOG, quiet=QUIET)
        ok(f"{name} cloned" + (f" ({want_branch})" if want_branch else ""))

if not (REPO / "requirements.txt").exists():
    die("DiffSinger did not clone correctly",
        fix="Re-run this cell. If it still fails, Runtime > Disconnect and delete runtime.")

# --- python environment ---------------------------------------------------- #
if not WANT_TRAINING:
    DS_PYTHON = None
    info("export-only profile: step 6 builds its own python 3.10 environment")
else:
    with step(f"DiffSinger environment (python {DS_ENV_PYTHON_VERSION})"):
        requirements, dropped, retargeted = [], [], []
        for raw in (REPO / "requirements.txt").read_text(encoding="utf-8").splitlines():
            entry = raw.split("#", 1)[0].strip()
            if not entry:
                continue
            name = dsnb.req_name(entry)
            if name in SKIP_PACKAGES:
                dropped.append(entry)
                continue
            override, min_python = PIN_OVERRIDES.get(name, (None, None))
            if override and TARGET_PYTHON >= min_python:
                retargeted.append(f"{entry} -> {override}")
                entry = override
            requirements.append(entry)
        requirements += EXTRA_PACKAGES

        fingerprint = hashlib.sha256(
            ("|".join([DS_ENV_PYTHON_VERSION, TORCH_PIN] + requirements)).encode()
        ).hexdigest()[:16]
        marker = CONTENT / ".ds_env_complete"
        marker_matches = (marker.exists()
                          and marker.read_text(encoding="utf-8").strip() == fingerprint)

        print(f"  kernel python   {sys.version.split()[0]}  (left untouched)")
        print(f"  target python   {DS_ENV_PYTHON_VERSION} at {dsnb.DS_ENV}")
        print(f"  torch           {TORCH_PIN}")
        for line in retargeted:
            info(line)
        if dropped:
            info("torch is pinned separately, not taken from requirements.txt: "
                 + ", ".join(dropped))
        print(f"  packages        {len(requirements)}: " + ", ".join(requirements))

        existing = dsnb.env_python(dsnb.DS_ENV)
        broken = dsnb.module_versions(existing, IMPORT_CHECKS)[1] if existing else []
        reusable = bool(existing) and marker_matches and not broken and not force_reinstall

        if reusable:
            ok("already built and everything imports - nothing to do")
            DS_PYTHON = existing
        else:
            if existing and marker_matches and broken:
                warn("the environment exists but these do not import: "
                     + ", ".join(module for module, _ in broken))
                warn("rebuilding")
            marker.unlink(missing_ok=True)
            if dsnb.DS_ENV.exists():
                shutil.rmtree(dsnb.DS_ENV)

            print("  ---- install output ------------------------------------------------")
            # Deliberately NOT quiet. This is the longest step, so watching it is the
            # difference between "working" and "hung".
            # torch goes in the same resolve as everything else - see build_env.
            DS_PYTHON = dsnb.build_env(
                dsnb.DS_ENV,
                DS_ENV_PYTHON_VERSION,
                [TORCH_PIN, TORCHAUDIO_PIN] + requirements,
                log_file=LOG,
                quiet=False,
            )
            dsnb.assert_pinned(DS_PYTHON, "torch",
                               TORCH_PIN.split("==")[1], log_file=LOG)
            print("  ---- end install output --------------------------------------------")
            marker.write_text(fingerprint, encoding="utf-8")
            ok(f"built at {dsnb.DS_ENV}")

# --- checkpoints ----------------------------------------------------------- #
checkpoints = REPO / "checkpoints"
checkpoints.mkdir(parents=True, exist_ok=True)

if WANT_TRAINING:
    wanted = ["rmvpe", "hnsep"]
    if vocoder_choice != "nsf_hifigan":
        wanted.append("pc_nsf_hifigan")
    if vocoder_choice in ("nsf_hifigan", "both"):
        wanted.append("nsf_hifigan")
    if install_some:
        wanted.append("some")

    with step(f"Downloading {len(wanted)} checkpoint archives in parallel"):
        download_many([ASSETS[key] for key in wanted], log_file=LOG, quiet=QUIET)

    with step("Installing checkpoints"):
        # rmvpe -> checkpoints/rmvpe/model.pt  (and a copy MakeDiffSinger expects)
        if not (checkpoints / "rmvpe" / "model.pt").exists():
            unzip(ASSETS["rmvpe"][1], checkpoints)
        assets_dir = CONTENT / "MakeDiffSinger" / "variance-temp-solution" / "assets"
        if assets_dir.exists() and not (assets_dir / "rmvpe" / "model.pt").exists():
            unzip(ASSETS["rmvpe"][1], assets_dir)
        ok("rmvpe -> checkpoints/rmvpe/model.pt")

        # hnsep: every config ships `hnsep_ckpt: checkpoints/vr/model.pt`, but the
        # release unpacks to hnsep_240512/.  The original notebook left it there,
        # so tension / voicing / breathiness extraction could never find it.
        vr_dir = checkpoints / "vr"
        if not (vr_dir / "model.pt").exists():
            staging = CONTENT / ".hnsep_tmp"
            if staging.exists():
                shutil.rmtree(staging)
            unzip(ASSETS["hnsep"][1], staging)
            found = next(staging.rglob("model.pt"), None)
            if found is None:
                die("hnsep archive did not contain model.pt",
                    fix="Delete /content/dl/hnsep_240512.zip and re-run this cell.")
            vr_dir.mkdir(parents=True, exist_ok=True)
            for item in found.parent.iterdir():
                shutil.move(str(item), vr_dir / item.name)
            shutil.rmtree(staging, ignore_errors=True)
        ok("hnsep -> checkpoints/vr/model.pt  (matches hnsep_ckpt in the configs)")

        for key in ("pc_nsf_hifigan", "nsf_hifigan"):
            if key not in wanted:
                continue
            prefix = "pc_nsf_hifigan_44.1k" if key.startswith("pc_") else "nsf_hifigan_44.1k"
            found = [p for p in checkpoints.glob(prefix + "*")
                     if p.is_dir() and (p / "model.ckpt").exists()]
            if not found:
                unzip(ASSETS[key][1], checkpoints)
                found = [p for p in checkpoints.glob(prefix + "*") if p.is_dir()]
            if found:
                ok(f"{key} -> checkpoints/{found[0].name}")
            else:
                warn(f"{key} did not unpack as expected")

        some_ckpt = None
        if install_some:
            some_root = checkpoints / "SOME"
            if not list(some_root.rglob("*.ckpt")):
                unzip(ASSETS["some"][1], some_root)
            # Do not hard-code the folder name.  The original notebook looked for
            # `0119_continuous256_5spk` while downloading `...128_5spk`, so SOME
            # MIDI estimation could never start.
            candidates = sorted(some_root.rglob("*.ckpt"))
            simplified = [c for c in candidates if "simplified" in c.name]
            some_ckpt = str((simplified or candidates)[0]) if candidates else None
            if some_ckpt:
                ok(f"SOME -> {Path(some_ckpt).relative_to(CONTENT)}")
            else:
                warn("SOME checkpoint not found; MIDI estimation via SOME will be unavailable")
else:
    some_ckpt = None
    info("export-only profile: skipping training checkpoints")

# --- verify ---------------------------------------------------------------- #
if not DS_PYTHON:
    info("export-only profile: nothing to verify here")
else:
    with step("Verifying the environment"):
        # A version for every module. "Did it actually install?" should never be
        # something you have to guess at.
        rows, broken = dsnb.module_versions(DS_PYTHON, IMPORT_CHECKS)
        print_table(rows, [("module", "module"), ("version", "version")])

        if broken:
            print()
            for module, reason in broken:
                warn(f"{module}: {reason}")
            by_module = dict(IMPORT_CHECKS)
            die(f"{len(broken)} module(s) failed to import",
                fix="Install them into the DiffSinger environment:\n"
                    f"  !{dsnb.uv_path()} pip install --python {DS_PYTHON} "
                    + " ".join(by_module[m] for m, _ in broken)
                    + f"\nThen re-run this cell. Full log: {LOG}")
        ok(f"all {len(rows)} required modules import cleanly")

        cuda = run_out([DS_PYTHON, "-c",
                        "import torch;print(torch.__version__, torch.cuda.is_available())"])
        if cuda:
            version, available = cuda.split()
            if available == "True":
                ok(f"torch {version} sees the GPU")
            elif WANT_TRAINING:
                warn(f"torch {version} but CUDA is NOT available - training will be unusably slow")
            warn("could not query torch in the DiffSinger environment")

# --------------------------------------------------------------------------- #
save_state(
    setup_done=True,
    branch=branch,
    profile=profile,
    repo=str(REPO),
    some_ckpt=some_ckpt,
    vocoder_choice=vocoder_choice,
    ds_python=DS_PYTHON,
)
STAMP.write_text("ok", encoding="utf-8")

header(f"Setup finished in {human_time(time.time() - STARTED)}", accent=dsnb.DONE_ACCENT)
if DS_PYTHON:
    print(f"  DiffSinger runs on {DS_PYTHON}")
    print(f"  This notebook's kernel stays on python {sys.version.split()[0]} and is untouched.")
print("  Next: step 2 - prepare your dataset.")
print(f"  Log:  {LOG}")
