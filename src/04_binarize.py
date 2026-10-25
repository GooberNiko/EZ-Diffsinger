#@title 4. Binarize  -  pack the dataset for training
#@markdown Turns your dataset into the packed format training reads. Run once per dataset.

backup_binary_to_drive = False  #@param {type:"boolean"}
#@markdown <font size="-1">Only relevant when step 3 used local disk. Copies the result next to your
#@markdown model so a dead runtime does not cost you the binarize pass. Costs upload time and Drive space.</font>

show_full_log = True  #@param {type:"boolean"}

# --------------------------------------------------------------------------- #

import importlib
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, "/content")
import dsnb
importlib.reload(dsnb)
from dsnb import (DEFAULT_ACCENT, DONE_ACCENT, FAST_ACCENT, LOG_DIR, REPO, die, header,
                  human_size, human_time, info, is_fast, load_state, ok, preflight,
                  print_table, step, warn)

state = load_state(required=["setup_done", "config_path", "binary_dir"])
PY = dsnb.training_python()
CONFIG = Path(state["config_path"])
BINARY_DIR = Path(state["binary_dir"])
MODEL_DIR = Path(state["model_dir"])
LOG = LOG_DIR / "binarize.log"

FAST = is_fast(state.get("training_mode", ""))
workers = (dsnb.resolve_merged_config(CONFIG).get("binarization_args") or {}).get("num_workers", 0)     if CONFIG.exists() else 0
header("Binarize",
       subtitle=(f"⚡ fast mode - {workers} parallel workers" if FAST
                 else "Original mode - single process"),
       accent=FAST_ACCENT if FAST else DEFAULT_ACCENT)

if not CONFIG.exists():
    die(f"config not found: {CONFIG}", fix="Re-run step 3.")

with step("Re-checking the config"):
    preflight(CONFIG, state["model_type"])

if BINARY_DIR.exists() and any(BINARY_DIR.iterdir()):
    warn(f"{BINARY_DIR} already has data - it will be overwritten")

started = time.time()
BINARIZE = [PY, "scripts/binarize.py", "--config", str(CONFIG), "--reset"]
with step("Running binarize.py"):
    code = dsnb.run(BINARIZE, cwd=REPO, log_file=LOG, quiet=not show_full_log, check=not FAST)
    if FAST and code != 0:
        # Parallel binarize is the experimental part. Rather than leave you stuck,
        # fall back to the stock single-process pass and say so.
        warn(f"parallel binarize failed (exit {code}) - retrying single-process")
        cfg = dsnb.load_yaml(CONFIG)
        cfg.setdefault("binarization_args", {})["num_workers"] = 0
        dsnb.dump_yaml(cfg, CONFIG)
        dsnb.run(BINARIZE, cwd=REPO, log_file=LOG, quiet=not show_full_log)
        ok("single-process binarize worked; the config now uses it")

# --------------------------------------------------------------------------- #
produced = sorted(p for p in BINARY_DIR.glob("*") if p.is_file())
if not produced:
    die("binarize finished but produced no files",
        fix=f"Check the log: {LOG}")

total = sum(p.stat().st_size for p in produced)
rows = [{"file": p.name, "size": human_size(p.stat().st_size)} for p in produced]
header(f"Binarized in {human_time(time.time() - started)}", accent=DONE_ACCENT)
print_table(rows, [("file", "file"), ("size", "size")])
print()
info(f"{BINARY_DIR}  ({human_size(total)})")

if backup_binary_to_drive and not state.get("binary_on_drive"):
    with step("Copying binary data to Drive"):
        target = MODEL_DIR / "binary"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(BINARY_DIR, target)
        ok(f"backup at {target}")
        info("Step 5 will pick this up automatically if the local copy is gone.")

print("\n  Next: step 5 - train.")
