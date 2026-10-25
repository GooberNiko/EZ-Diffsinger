#@title 5. Train  -  or continue an earlier run
#@markdown Starts (or continues) training and opens TensorBoard above the log.

#@markdown ---
#@markdown ### Training mode
training_mode = "same as step 3"  #@param ["same as step 3", "Original", "Woah's Epic Faster Training (experimental)"]
#@markdown <font size="-1">&#9889; **Woah's Epic Faster Training** replaces `batch_size` and
#@markdown `max_batch_frames` below with values sized to this GPU's VRAM, picks the fastest precision
#@markdown for the card (fp16 on a T4, bf16 on L4/A100), fuses AdamW, keeps a CPU core free for the
#@markdown training loop and trims validation, logging and progress-bar redraws. Your schedule
#@markdown (`save_every`, `max_updates`, checkpoints kept) is never touched. **Original** uses the
#@markdown sliders as-is.</font>
torch_compile = True  #@param {type:"boolean"}
#@markdown <font size="-1">Fast mode only. Compiles the diffusion backbone with `torch.compile` - the
#@markdown single biggest speed-up on a T4, after a few minutes of compiling while the first steps run.
#@markdown If the compiler fails, training restarts without it automatically.</font>

#@markdown ---
#@markdown ### Schedule
batch_size = 16  #@param {type:"slider", min:1, max:64, step:1}
#@markdown <font size="-1">Too high gives CUDA out-of-memory; too low wastes the GPU. 16 suits a T4 at default sizes.</font>
max_batch_frames = 32000  #@param {type:"slider", min:4000, max:80000, step:1000}
save_every = 2000  #@param {type:"slider", min:200, max:10000, step:100}
max_updates = 160000  #@param {type:"slider", min:1000, max:1000000, step:1000}
keep_last_n_checkpoints = 3  #@param {type:"slider", min:1, max:10, step:1}
open_tensorboard = True  #@param {type:"boolean"}

#@markdown ---
#@markdown ### Continue an earlier run &nbsp;<font size="-1">(new runtime? just fill this in)</font>
resume_from_model_dir = ""  #@param {type:"string"}
#@markdown <font size="-1">Point at the model folder from a previous session. Steps 2 and 3 are not needed -
#@markdown it reads `notebook_config.yaml` that step 3 saved there.</font>
copy_binary_back_to_local = True  #@param {type:"boolean"}

# --------------------------------------------------------------------------- #

import importlib
import shlex
import shutil
import sys
from pathlib import Path

from IPython import get_ipython

sys.path.insert(0, "/content")
import dsnb
importlib.reload(dsnb)
from dsnb import (DEFAULT_ACCENT, DONE_ACCENT, FAST_ACCENT, FAST_ENV, LOG_DIR, ORIGINAL_MODE,
                  REPO, die, dump_yaml, find_latest_ckpt, header, human_size, info, is_fast,
                  link_experiment, load_state, load_yaml, ok, print_table,
                  sanitize_exp_name, save_state, step, warn)

state = load_state(required=["setup_done"])
PY = dsnb.training_python()

# --------------------------------------------------------------------------- #
#  Where are we training?
# --------------------------------------------------------------------------- #
if resume_from_model_dir.strip():
    MODEL_DIR = Path(resume_from_model_dir.strip().strip('"').strip("'"))
    if not MODEL_DIR.exists():
        die(f"model folder not found: {MODEL_DIR}",
            fix="Check the path, or leave this blank to train the model from step 3.")
    CONFIG = MODEL_DIR / "notebook_config.yaml"
    if not CONFIG.exists():
        candidate = MODEL_DIR / "config.yaml"
        if not candidate.exists():
            die(f"no config found in {MODEL_DIR}",
                fix="Expected notebook_config.yaml (written by step 3) or config.yaml\n"
                    "(written by a previous training run).")
        CONFIG = candidate
        warn("using config.yaml from the previous run")
    EXP_NAME = sanitize_exp_name(MODEL_DIR.name)
    MODEL_TYPE = "variance" if "predict_dur" in dsnb.resolve_merged_config(CONFIG) else "acoustic"
    info(f"resuming a {MODEL_TYPE} model from {MODEL_DIR}")
else:
    state = load_state(required=["config_path", "model_dir", "exp_name"])
    MODEL_DIR = Path(state["model_dir"])
    CONFIG = Path(state["config_path"])
    EXP_NAME = state["exp_name"]
    MODEL_TYPE = state["model_type"]

MODE = (state.get("training_mode", ORIGINAL_MODE) if training_mode == "same as step 3"
        else training_mode)
FAST = is_fast(MODE)
header(f"Train {EXP_NAME}",
       subtitle=("⚡ Woah's Epic Faster Training mode (experimental)" if FAST
                 else "Original training mode"),
       accent=FAST_ACCENT if FAST else DEFAULT_ACCENT)

with step("Preparing work directory"):
    link_experiment(EXP_NAME, MODEL_DIR)
    ok(f"checkpoints/{EXP_NAME} -> {MODEL_DIR}")


# --------------------------------------------------------------------------- #
#  Binary data - after a runtime restart the local copy is gone
# --------------------------------------------------------------------------- #
cfg = load_yaml(CONFIG)
binary_dir = Path(cfg.get("binary_data_dir", ""))

with step("Locating binarized data"):
    def has_data(path):
        return path.exists() and any(path.glob("*.data"))

    if has_data(binary_dir):
        ok(f"{binary_dir}")
    else:
        backup = MODEL_DIR / "binary"
        if has_data(backup):
            if copy_binary_back_to_local and str(binary_dir).startswith("/content/binary"):
                info("restoring the Drive backup to local disk (much faster to read from)")
                if binary_dir.exists():
                    shutil.rmtree(binary_dir)
                shutil.copytree(backup, binary_dir)
                ok(f"restored to {binary_dir}")
            else:
                cfg["binary_data_dir"] = str(backup)
                binary_dir = backup
                warn("training straight off Drive - expect slower steps")
        else:
            die("no binarized data found",
                fix=f"Looked in:\n  {binary_dir}\n  {backup}\n"
                    "Run steps 2-4 to rebuild it, or tick 'backup_binary_to_drive' in step 4\n"
                    "next time so a lost runtime does not cost you the binarize pass.")


# --------------------------------------------------------------------------- #
#  Apply the schedule
# --------------------------------------------------------------------------- #
with step("Applying training schedule"):
    cfg["val_check_interval"] = save_every
    cfg["max_updates"] = max_updates
    cfg["num_ckpt_keep"] = keep_last_n_checkpoints
    # The precision you chose in step 3 is remembered, so switching back to
    # Original after a fast run really is the original setup again.
    cfg.setdefault("nb_user_precision", cfg.get("pl_trainer_precision", "16-mixed"))
    stock = dsnb.resolve_merged_config(REPO / "configs" / f"{MODEL_TYPE}.yaml")
    if FAST:
        gpu = dsnb.gpu_specs()
        optimizer_cls = (dsnb.resolve_merged_config(CONFIG).get("optimizer_args")
                         or {}).get("optimizer_cls", "")
        tuned = dsnb.fast_train_overrides(MODEL_TYPE, gpu, optimizer_cls)
        dsnb.apply_training_mode(cfg, rows=tuned)
        if not gpu:
            warn("could not read the GPU - fast mode assumed a 16 GB T4")
    else:
        dsnb.apply_training_mode(cfg, stock=stock)
        cfg["pl_trainer_precision"] = cfg["nb_user_precision"]
        cfg["max_batch_size"] = batch_size
        cfg["max_batch_frames"] = max_batch_frames
    dump_yaml(cfg, CONFIG)
    ok(f"batch {cfg['max_batch_size']} / {cfg['max_batch_frames']} frames / "
       f"save every {save_every} / stop at {max_updates}")

if FAST:
    header("⚡ Fast-mode tuning", subtitle="stock DiffSinger settings, sized to this runtime",
           accent=FAST_ACCENT)
    print_table([{"k": k, "v": v, "why": why} for k, v, why in tuned]
                + [{"k": k, "v": v, "why": "environment"} for k, v in FAST_ENV.items()]
                + [{"k": "torch.compile", "v": "on" if torch_compile else "off",
                    "why": "backbone compiled, falls back to eager on failure"
                    if torch_compile else "ticked off in this cell"},
                   {"k": "TF32", "v": "on", "why": "fp32 matmuls / convs on Ampere+"}],
                [("k", "setting"), ("v", "value"), ("why", "why")])
    if torch_compile:
        info("The first steps are slow while torch.compile builds kernels - that is expected.")
    info("If the first steps fail with CUDA out-of-memory, re-run with training_mode = Original.")

existing = find_latest_ckpt(MODEL_DIR)
if existing:
    info(f"found {existing.name} - training continues from there")
else:
    info("no checkpoint yet - starting from step 0")

free = dsnb.disk_free("/content")
if free < 5 * 1024 ** 3:
    warn(f"only {human_size(free)} free on /content; checkpoints may fail to write")


# --------------------------------------------------------------------------- #
#  TensorBoard
# --------------------------------------------------------------------------- #
if open_tensorboard:
    tb_dir = MODEL_DIR / "lightning_logs"
    tb_dir.mkdir(parents=True, exist_ok=True)
    ipython = get_ipython()
    ipython.run_line_magic("reload_ext", "tensorboard")
    ipython.run_line_magic("tensorboard", f"--logdir {shlex.quote(str(tb_dir))}")


# --------------------------------------------------------------------------- #
header("Training", subtitle="interrupt this cell to stop - your checkpoints are already saved",
       accent=FAST_ACCENT if FAST else DEFAULT_ACCENT)
save_state(model_dir=str(MODEL_DIR), config_path=str(CONFIG), exp_name=EXP_NAME,
           model_type=MODEL_TYPE, training_mode=MODE)

TRAIN_LOG = LOG_DIR / "train.log"
TRAIN_ARGS = ["--config", str(CONFIG), "--exp_name", EXP_NAME, "--reset"]


def train(compile_backbone):
    """Run one training process; returns (exit code, this run's log text)."""
    start = TRAIN_LOG.stat().st_size if TRAIN_LOG.exists() else 0
    if FAST:
        # The launcher applies the runtime tweaks, then runs the stock train.py.
        cmd = [PY, dsnb.write_fast_launcher(), REPO] + TRAIN_ARGS
        env = {**FAST_ENV, "NB_TORCH_COMPILE": "1" if compile_backbone else "0"}
    else:
        cmd, env = [PY, "scripts/train.py"] + TRAIN_ARGS, None
    code = dsnb.run(cmd, cwd=REPO, log_file=TRAIN_LOG, check=False, env=env)
    with TRAIN_LOG.open("rb") as fh:
        fh.seek(start)
        return code, fh.read().decode("utf-8", errors="replace")


def ran_out_of_memory(log_text):
    return "CUDA out of memory" in log_text or "OutOfMemoryError" in log_text


exit_code, this_run = train(FAST and torch_compile)
# Compile errors dynamo recovers from still leave warnings in the log, so only a
# crash that was not an out-of-memory counts as the compiler's fault.
if (FAST and torch_compile and exit_code != 0 and not ran_out_of_memory(this_run)
        and any(marker in this_run for marker in dsnb.COMPILE_FAILURE_MARKERS)):
    # Checkpoints are already on disk, so the retry resumes rather than restarts.
    warn("torch.compile failed on this runtime - restarting training without it")
    exit_code, this_run = train(False)

# --exp_name is what DiffSinger turns into checkpoints/<exp_name>. The original
# notebook passed `${save_dir}`, which bash expands to an empty string, and then
# worked around the resulting mess by rewriting utils/hparams.py on every run.

latest = find_latest_ckpt(MODEL_DIR)
header("Training stopped", accent=DONE_ACCENT if latest else DEFAULT_ACCENT)
if exit_code != 0 and ran_out_of_memory(this_run):
    warn("the GPU ran out of memory.")
    if FAST:
        warn("fast mode's batch size is too big for this dataset - set training_mode to")
        warn("Original (or keep fast mode on a bigger GPU) and run this cell again.")
    else:
        warn("lower batch_size / max_batch_frames above and run this cell again.")
if latest:
    ok(f"latest checkpoint: {latest.name} ({human_size(latest.stat().st_size)})")
    print(f"\n  Model folder: {MODEL_DIR}")
    print("  Next: step 6 - export to ONNX.")
else:
    warn("no checkpoint was written - scroll up for the reason")
