#@title 2. Prepare dataset  -  unpack, segment, write dictionaries
#@markdown Unpacks your data into `/content/raw_data`, segments it, and writes the phoneme
#@markdown dictionaries. Prints a summary so you can see what actually made it through.

#@markdown ---
#@markdown ### Input
data_type = "lab + wav (NNSVS)"  #@param ["lab + wav (NNSVS)", "csv + wav (DiffSinger)", "ds (DiffSinger)"]
data_zip_path = ""  #@param {type:"string"}
#@markdown <font size="-1">Path to your archive on Drive - `.zip`, `.7z` and `.rar` all work, spaces included.</font>

#@markdown ---
#@markdown ### Segmentation &nbsp;<font size="-1">(lab + wav only)</font>
segment_length = 10  #@param {type:"slider", min:5, max:35, step:1}
max_silences = 2  #@param {type:"slider", min:0, max:20, step:1}
detect_breaths = True  #@param {type:"boolean"}
#@markdown <font size="-1">`detect_breaths` finds AP inside pauses automatically - recommended.</font>

normalize_silence_labels = True  #@param {type:"boolean"}
#@markdown <font size="-1">Rewrites `SP`/`sil` to `pau` and `br` to `AP` in your labels so the segmenter understands them.</font>

write_ds_and_labels = False  #@param {type:"boolean"}
#@markdown <font size="-1">Also emit `.ds` files and HTK labels beside the segments - handy for
#@markdown checking segmentation in SlurCutter, and for the "package dataset" utility at the bottom.</font>

#@markdown ---
#@markdown ### MIDI estimation &nbsp;<font size="-1">(needed for variance *pitch* training)</font>
estimate_midi = "off"  #@param ["off", "parselmouth", "harvest", "SOME (most accurate)"]

# --------------------------------------------------------------------------- #

import csv
import importlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/content")
import dsnb
importlib.reload(dsnb)
from dsnb import (CONTENT, REPO, dataset_summary, die, extract_archive, header, info,
                  load_state, ok, print_table, run, save_state, step, warn)

state = load_state(required=["setup_done"])
PY = dsnb.training_python()   # DiffSinger's python 3.11 env, not the notebook kernel

RAW_ROOT = CONTENT / "raw_data"
DATA_DIR = RAW_ROOT / "diffsinger_db"
DICT_DIR = REPO / "dictionaries"
LANG_JSON = CONTENT / "nnsvs-db-converter" / "lang.sample.json"

SILENCE_TOKENS = {"pau", "AP", "SP", "sil", "br", "silB", "silE", ""}
LABEL_FIXES = {"SP": "pau", "sil": "pau", "silB": "pau", "silE": "pau", "br": "AP"}
VOWEL_STARTS = {"a", "i", "u", "e", "o", "A", "I", "U", "E", "O", "N", "M"}
LIQUID_STARTS = {"y", "w", "l", "r"}

header("Prepare dataset")

if not data_zip_path.strip():
    die("no data_zip_path given",
        fix="Paste the full path to your dataset archive, e.g.\n"
            "  /content/drive/MyDrive/my singer/data.zip")
archive = Path(data_zip_path.strip().strip('"').strip("'"))
if not archive.exists():
    die(f"archive not found: {archive}",
        fix="Check the path. Use the file browser on the left, right-click > Copy path.")


# --------------------------------------------------------------------------- #
#  1. Extract
# --------------------------------------------------------------------------- #
with step(f"Extracting {archive.name}"):
    if RAW_ROOT.exists():
        shutil.rmtree(RAW_ROOT)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    extract_archive(archive, DATA_DIR)

    # Tolerate an archive that wraps everything in one extra folder.
    entries = [p for p in DATA_DIR.iterdir() if not p.name.startswith(".")]
    dirs = [p for p in entries if p.is_dir()]
    if len(dirs) == 1 and len(entries) == 1:
        inner = dirs[0]
        looks_like_speaker = any(inner.glob("*.lab")) or any(inner.glob("*.wav")) \
            or (inner / "wavs").exists() or any(inner.glob("*.ds"))
        if not looks_like_speaker:
            info(f"unwrapping extra top-level folder '{inner.name}'")
            for item in list(inner.iterdir()):
                shutil.move(str(item), DATA_DIR / item.name)
            inner.rmdir()

    speaker_dirs = sorted(p for p in DATA_DIR.iterdir() if p.is_dir())
    if not speaker_dirs:
        die("no speaker folders found inside the archive",
            fix="Your archive must contain one folder per singer, e.g.\n"
                "  data.zip\n"
                "   +- my_singer/\n"
                "       +- *.wav and *.lab")
    ok(f"{len(speaker_dirs)} speaker folder(s): " + ", ".join(p.name for p in speaker_dirs))


# --------------------------------------------------------------------------- #
#  2. Normalise label files
#
#  The original notebook ran this on the csv/ds branch instead of the lab branch
#  ("using 'if not' bc i edited the wrong section"), so it never touched a single
#  .lab file.  It also did a raw string replace, which corrupted any phoneme that
#  merely contained "br" or "SP".  This rewrites the phoneme column only.
# --------------------------------------------------------------------------- #
if data_type == "lab + wav (NNSVS)" and normalize_silence_labels:
    with step("Normalising label files"):
        changes, touched = {}, 0
        for lab in DATA_DIR.rglob("*.lab"):
            lines, dirty = [], False
            for line in lab.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.split()
                if len(parts) >= 3 and parts[2] in LABEL_FIXES:
                    changes[parts[2]] = changes.get(parts[2], 0) + 1
                    parts[2] = LABEL_FIXES[parts[2]]
                    dirty = True
                    lines.append(" ".join(parts))
                else:
                    lines.append(line)
            if dirty:
                lab.write_text("\n".join(lines) + "\n", encoding="utf-8")
                touched += 1
        if changes:
            ok(f"{touched} file(s) updated: "
               + ", ".join(f"{k}->{LABEL_FIXES[k]} x{v}" for k, v in sorted(changes.items())))
        else:
            ok("labels already use pau / AP - nothing to change")


# --------------------------------------------------------------------------- #
#  3. Collect phonemes and write dictionaries
# --------------------------------------------------------------------------- #
def phonemes_in(path):
    """Yield phonemes from a .lab, .csv or .ds file, minus silence tokens."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".lab":
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.split()
                if len(parts) >= 3:
                    yield parts[2]
        elif suffix == ".csv":
            with path.open("r", newline="", encoding="utf-8", errors="replace") as fh:
                for row in csv.DictReader(fh):
                    for token in (row.get("ph_seq") or "").split():
                        yield token
        elif suffix == ".ds":
            payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            for entry in payload if isinstance(payload, list) else [payload]:
                for token in (entry.get("ph_seq") or "").split():
                    yield token
    except Exception as exc:
        warn(f"could not read {path.name}: {exc}")


def strip_lang(token):
    return token.split("/", 1)[1] if "/" in token else token


lang_config_path = DATA_DIR / "lang_config.yaml"
extra_phonemes, merged_phoneme_groups = [], []
speakers = []          # [{folder, speaker, language}]
lang_phonemes = {}     # language -> set(phonemes)

with step("Building phoneme dictionaries"):
    if lang_config_path.exists():
        lang_config = dsnb.load_yaml(lang_config_path)
        languages = list(lang_config.get("languages") or [])
        extra_phonemes = list(lang_config.get("extra_phonemes") or [])
        merged_phoneme_groups = list(lang_config.get("merged_phoneme_groups") or [])
        if not languages:
            die("lang_config.yaml has no 'languages' list",
                fix="Add e.g.\n  languages:\n    - ja\n    - en")
        info("multi-language mode: " + ", ".join(languages))
        lang_phonemes = {lang: set() for lang in languages}

        unlabelled = []
        for folder in speaker_dirs:
            if "." not in folder.name:
                unlabelled.append(folder.name)
                continue
            base, _, suffix = folder.name.rpartition(".")
            if suffix not in languages:
                unlabelled.append(folder.name)
                continue
            speakers.append({"folder": str(folder), "speaker": base, "language": suffix})
            for path in folder.rglob("*"):
                if path.suffix.lower() not in (".lab", ".csv", ".ds"):
                    continue
                for token in phonemes_in(path):
                    hint, _, tail = token.partition("/")
                    if tail and hint in languages:
                        if tail not in SILENCE_TOKENS:
                            lang_phonemes[hint].add(tail)
                    elif token not in SILENCE_TOKENS:
                        lang_phonemes[suffix].add(token)

        if unlabelled:
            die("these folders have no language suffix: " + ", ".join(unlabelled),
                fix="With lang_config.yaml present, every speaker folder must end in a\n"
                    "language code from the list, e.g. 'my_singer.ja'.")
    else:
        info("single-language mode (no lang_config.yaml) - dictionary will be 'custom'")
        collected = set()
        for folder in speaker_dirs:
            speakers.append({"folder": str(folder), "speaker": folder.name, "language": "custom"})
            for path in folder.rglob("*"):
                if path.suffix.lower() not in (".lab", ".csv", ".ds"):
                    continue
                for token in phonemes_in(path):
                    token = strip_lang(token)
                    if token not in SILENCE_TOKENS:
                        collected.add(token)
        lang_phonemes = {"custom": collected}

    if not any(lang_phonemes.values()):
        die("no phonemes found anywhere in the dataset",
            fix={"lab + wav (NNSVS)": "Expected *.lab files next to your *.wav files.",
                 "csv + wav (DiffSinger)": "Expected transcriptions.csv with a ph_seq column.",
                 "ds (DiffSinger)": "Expected *.ds files with a ph_seq field."}[data_type])

    DICT_DIR.mkdir(parents=True, exist_ok=True)
    dictionaries, dictionary_files = {}, []
    for lang, phoneme_set in lang_phonemes.items():
        path = DICT_DIR / f"dictionary-{lang}.txt"
        path.write_text(
            "".join(f"{ph}\t{ph}\n" for ph in sorted(phoneme_set)), encoding="utf-8"
        )
        dictionaries[lang] = str(path)
        dictionary_files.append(str(path))
        ok(f"{path.name}: {len(phoneme_set)} phonemes")

    # Language definition for the segmenter's MIDI estimator.
    all_phonemes = sorted({ph for group in lang_phonemes.values() for ph in group})
    vowels = [p for p in all_phonemes if p[0] in VOWEL_STARTS]
    liquids = {p: True for p in all_phonemes if p[0] in LIQUID_STARTS}
    LANG_JSON.parent.mkdir(parents=True, exist_ok=True)
    LANG_JSON.write_text(json.dumps({"vowels": vowels, "liquids": liquids}, indent=2),
                         encoding="utf-8")
    info(f"{len(vowels)} vowels / {len(liquids)} liquids written for the segmenter")


# --------------------------------------------------------------------------- #
#  4. Convert to DiffSinger layout
# --------------------------------------------------------------------------- #
if data_type == "lab + wav (NNSVS)":
    converter = CONTENT / "nnsvs-db-converter" / "db_converter.py"
    use_some = estimate_midi.startswith("SOME")
    inline_midi = estimate_midi in ("parselmouth", "harvest")

    with step("Segmenting audio"):
        for entry in speakers:
            folder = Path(entry["folder"])
            cmd = [PY, str(converter),
                   "-s", str(max_silences),
                   "-l", str(segment_length),
                   "-T", "0",                      # use every core; the original used one
                   "-L", str(LANG_JSON)]
            if detect_breaths:
                cmd.append("-B")
            if write_ds_and_labels:
                cmd += ["-D", "-w", "htk"]
            if inline_midi:
                # The original hard-coded parselmouth: it read the menu choice into
                # `midi_pitch_ext` and then never passed it to anything.
                cmd += ["-m", "-c", "-p", estimate_midi]
            cmd.append(str(folder))
            info(f"{folder.name}: segmenting (this is the slow part)")
            run(cmd, quiet=True)

            produced = folder / "diffsinger_db"
            if not produced.exists():
                die(f"the segmenter produced nothing for '{folder.name}'",
                    fix="Check that every .wav has a matching .lab with the same name.")
            for stale in list(folder.glob("*.wav")) + list(folder.glob("*.lab")):
                stale.unlink()
            for item in produced.iterdir():
                destination = folder / item.name
                if destination.exists():
                    shutil.rmtree(destination) if destination.is_dir() else destination.unlink()
                shutil.move(str(item), destination)
            produced.rmdir()
            segments = len(list((folder / "wavs").glob("*.wav")))
            ok(f"{folder.name}: {segments} segments")

    if use_some:
        some_ckpt = state.get("some_ckpt")
        if not some_ckpt or not Path(some_ckpt).exists():
            die("SOME checkpoint is not available",
                fix="Re-run step 1 with 'install_some' ticked.")
        with step("Estimating MIDI with SOME"):
            for entry in speakers:
                info(f"{Path(entry['folder']).name}: running SOME")
                run([PY, "batch_infer.py",
                     "--model", some_ckpt,
                     "--dataset", entry["folder"],
                     "--overwrite"],
                    cwd=CONTENT / "SOME", quiet=True)
                ok(f"{Path(entry['folder']).name}: note_seq written")

elif data_type == "ds (DiffSinger)":
    segmentor = CONTENT / "ghin_shenanigans" / "scripts" / "ds_segmentor.py"
    ds2csv = CONTENT / "MakeDiffSinger" / "variance-temp-solution" / "convert_ds.py"
    with step("Segmenting .ds and converting to transcriptions.csv"):
        for entry in speakers:
            folder = Path(entry["folder"])
            ds_out = folder / "ds"
            run([PY, str(segmentor), str(folder), "--export_path", str(ds_out)],
                quiet=True)
            for stale in folder.glob("*.ds"):
                stale.unlink()
            run([PY, str(ds2csv), "ds2csv", str(ds_out),
                 str(folder / "transcriptions.csv")], quiet=True)
            ok(f"{folder.name}: {len(list(ds_out.glob('*.ds')))} .ds files")
else:
    info("csv + wav data used as-is")


# --------------------------------------------------------------------------- #
#  5. Leading SP -> AP
#
#  A segment that opens on silence should open on AP, not SP.  The original only
#  ever patched row 1 of each file, leaving every other segment wrong.
# --------------------------------------------------------------------------- #
with step("Fixing leading silence tokens"):
    fixed_rows = 0
    for csv_path in DATA_DIR.rglob("transcriptions.csv"):
        with csv_path.open("r", newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        if not rows or "ph_seq" not in rows[0]:
            continue
        column = rows[0].index("ph_seq")
        changed = False
        for row in rows[1:]:
            if len(row) <= column:
                continue
            tokens = row[column].split()
            if tokens and tokens[0] == "SP":
                tokens[0] = "AP"
                row[column] = " ".join(tokens)
                fixed_rows += 1
                changed = True
        if changed:
            with csv_path.open("w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerows(rows)
    ok(f"{fixed_rows} segment(s) now start on AP")


# --------------------------------------------------------------------------- #
#  6. Summary
# --------------------------------------------------------------------------- #
rows = dataset_summary(DATA_DIR)
for row in rows:
    row["hours"] = f"{row['hours']:.2f}"
    row["has_csv"] = "yes" if row["has_csv"] else "NO"

header("Dataset ready", accent=dsnb.DONE_ACCENT)
print_table(rows, [("speaker", "speaker"), ("segments", "segments"),
                   ("hours", "hours"), ("has_csv", "transcriptions")])

missing_csv = [r["speaker"] for r in rows if r["has_csv"] == "NO"]
if missing_csv:
    warn("no transcriptions.csv for: " + ", ".join(missing_csv))

total_hours = sum(float(r["hours"]) for r in rows)
print()
info(f"{len(rows)} speaker(s), {sum(int(r['segments']) for r in rows)} segments, "
     f"{total_hours:.2f} hours total")
if total_hours and total_hours < 0.5:
    warn("under 30 minutes of audio - expect a rough model")

save_state(
    data_dir=str(DATA_DIR),
    data_type=data_type,
    speakers=speakers,
    dictionaries=dictionaries,
    dictionary_files=dictionary_files,
    extra_phonemes=extra_phonemes,
    merged_phoneme_groups=merged_phoneme_groups,
    languages=sorted(lang_phonemes),
    has_midi=estimate_midi != "off",
    total_hours=total_hours,
)
print("\n  Next: step 3 - build the training config.")
