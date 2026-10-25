#@title Utility: drop speakers from a checkpoint
#@markdown Trims a multi-speaker checkpoint down to the speakers you want to publish.
#@markdown Run it on the **acoustic and the variance** checkpoint, then export those.

checkpoint_path = ""  #@param {type:"string"}
#@markdown <font size="-1">Path to a single `model_ckpt_steps_*.ckpt`, or to the folder holding it.</font>
keep_speaker_ids = ""  #@param {type:"string"}
#@markdown <font size="-1">IDs to **keep**, comma separated, e.g. `0,1`. Open `spk_map.json` in the
#@markdown model folder to see them: `{"natural": 0, "power": 1, "silly": 2}`.</font>
fill_dropped_embeds = "zeros"  #@param ["zeros", "random", "mean", "cyclic"]

# --------------------------------------------------------------------------- #

import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "/content")
import dsnb
importlib.reload(dsnb)
from dsnb import REPO, die, find_latest_ckpt, header, human_size, info, ok, run, step

PY = dsnb.training_python()

header("Drop speakers")

raw = checkpoint_path.strip().strip('"').strip("'")
if not raw:
    die("checkpoint_path is empty")
source = Path(raw)
if source.is_dir():
    found = find_latest_ckpt(source)
    if found is None:
        die(f"no *.ckpt inside {source}")
    source = found
    info(f"using {source.name}")
if not source.exists():
    die(f"checkpoint not found: {source}")

keep = [part.strip() for part in keep_speaker_ids.split(",") if part.strip()]
if not keep:
    die("keep_speaker_ids is empty",
        fix="List the IDs you want to keep, e.g. 0,1")
if not all(part.isdigit() for part in keep):
    die(f"keep_speaker_ids must be numbers, got: {keep_speaker_ids}",
        fix='Use the numeric IDs from spk_map.json, e.g. 0,2 - not the names.')

spk_map = source.parent / "spk_map.json"
if spk_map.exists():
    mapping = json.loads(spk_map.read_text(encoding="utf-8"))
    inverted = {str(value): name for name, value in mapping.items()}
    unknown = [i for i in keep if i not in inverted]
    if unknown:
        die("no such speaker id: " + ", ".join(unknown),
            fix="spk_map.json says: " + json.dumps(mapping))
    info("keeping: " + ", ".join(f"{i} ({inverted[i]})" for i in keep))

destination = source.with_name(source.stem + "_spk-dropped.ckpt")
with step("Rewriting checkpoint"):
    run([PY, "scripts/drop_spk.py", str(source), str(destination),
         "--retain", ",".join(keep), "--fill", fill_dropped_embeds],
        cwd=REPO, quiet=True)

ok(f"{destination}  ({human_size(destination.stat().st_size)})")
info("Move this into a folder with the model's config.yaml before exporting it.")
