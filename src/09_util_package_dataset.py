#@title Utility: package the prepared dataset
#@markdown Zips whatever step 2 produced so you can keep it and skip segmentation next time.
#@markdown Feed the zip back in as `csv + wav (DiffSinger)`.

save_to = ""  #@param {type:"string"}
#@markdown <font size="-1">Folder to write `<name>.zip` into, e.g. `/content/drive/MyDrive/DiffSinger`</font>
zip_name = "prepared_dataset"  #@param {type:"string"}

# --------------------------------------------------------------------------- #

import importlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/content")
import dsnb
importlib.reload(dsnb)
from dsnb import (dataset_summary, die, header, human_size, info, load_state, ok,
                  print_table, step)

state = load_state(required=["data_dir"])
DATA_DIR = Path(state["data_dir"])

header("Package prepared dataset")

if not DATA_DIR.exists():
    die(f"{DATA_DIR} is gone", fix="Run step 2 again to rebuild it.")
if not save_to.strip():
    die("save_to is empty", fix="e.g. /content/drive/MyDrive/DiffSinger")

DEST = Path(save_to.strip().strip('"').strip("'"))
DEST.mkdir(parents=True, exist_ok=True)

rows = dataset_summary(DATA_DIR)
for row in rows:
    row["hours"] = f"{row['hours']:.2f}"
print_table(rows, [("speaker", "speaker"), ("segments", "segments"), ("hours", "hours")])

with step("Zipping"):
    # root_dir=DATA_DIR keeps the speaker folders at the top of the archive, which
    # is exactly the shape step 2 expects to read back in. The original zipped
    # absolute /content paths, so the result could not be re-imported.
    archive = shutil.make_archive(str(DEST / zip_name.strip()), "zip", root_dir=str(DATA_DIR))

ok(f"{archive}  ({human_size(Path(archive).stat().st_size)})")
info("Re-import it in step 2 with data_type = 'csv + wav (DiffSinger)'.")
