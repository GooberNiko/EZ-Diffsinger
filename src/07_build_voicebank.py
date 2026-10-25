#@title 7. Build OpenUtau voicebank
#@markdown Assembles the exported ONNX files into an installable OpenUtau voicebank and zips it.
#@markdown <br><font size="-1">Leave the paths blank to reuse whatever step 6 just exported.</font>

voicebank_name = ""  #@param {type:"string"}
acoustic_onnx_folder = ""  #@param {type:"string"}
variance_onnx_folder = ""  #@param {type:"string"}
save_to = ""  #@param {type:"string"}
custom_vocoder_onnx = ""  #@param {type:"string"}
#@markdown <font size="-1">Path to a fine-tuned vocoder `.onnx` (its `vocoder.yaml` must sit beside it). Leave blank for the standard one.</font>
make_zip = True  #@param {type:"boolean"}

# --------------------------------------------------------------------------- #

import importlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/content")
import dsnb
importlib.reload(dsnb)
from dsnb import (die, dump_yaml, header, human_size, info, load_state, load_yaml, ok,
                  print_table, sanitize_exp_name, step, warn)

state = load_state()

# OpenUtau classifies every symbol; a wrong guess only affects automatic
# phoneme timing hints, never whether the bank loads.
PHONEME_TYPES = {
    "a": "vowel", "i": "vowel", "u": "vowel", "e": "vowel", "o": "vowel",
    "A": "vowel", "I": "vowel", "U": "vowel", "E": "vowel", "O": "vowel",
    "N": "vowel", "M": "vowel",
    "b": "stop", "d": "stop", "g": "stop", "k": "stop", "p": "stop",
    "q": "stop", "t": "stop",
    "c": "affricate", "j": "affricate",
    "f": "fricative", "h": "fricative", "s": "fricative", "v": "fricative",
    "z": "fricative",
    "l": "liquid", "r": "liquid",
    "m": "nasal", "n": "nasal",
    "w": "semivowel", "y": "semivowel",
}
# OpenUtau's built-in phonemizers already understand these language codes, so a
# dsdict for them does not need a replacements table.
OU_CANON_LANGUAGES = {"zh", "en", "de", "it", "ja", "yue", "ko", "pt", "ru", "es"}

header("Build OpenUtau voicebank")


def clean(value):
    return value.strip().strip('"').strip("'")


ACOUSTIC = Path(clean(acoustic_onnx_folder) or state.get("onnx_acoustic") or "")
VARIANCE = Path(clean(variance_onnx_folder) or state.get("onnx_variance") or "")
if not clean(voicebank_name):
    die("voicebank_name is empty", fix="This is the name OpenUtau shows in the singer list.")
NAME = clean(voicebank_name)
FOLDER_NAME = sanitize_exp_name(NAME.replace(" ", ""))
if not clean(save_to):
    die("save_to is empty", fix="e.g. /content/drive/MyDrive/DiffSinger")
DEST = Path(clean(save_to))

if not ACOUSTIC or not ACOUSTIC.exists():
    die(f"acoustic ONNX folder not found: {ACOUSTIC}",
        fix="Run step 6 first, or point at the folder containing acoustic.onnx.")
if not (ACOUSTIC / "acoustic.onnx").exists():
    die(f"no acoustic.onnx in {ACOUSTIC}",
        fix="Step 6 renames the export to acoustic.onnx. If you exported by hand,\n"
            "rename `<exp>.onnx` to `acoustic.onnx` and its `.phonemes.json` /\n"
            "`.languages.json` to `acoustic_phonemes.json` / `acoustic_languages.json`.")

has_variance_folder = bool(VARIANCE) and VARIANCE.exists()
if not has_variance_folder:
    warn("no variance model - the bank will render audio but will not predict")
    warn("duration, pitch or parameters on its own.")

parts = {
    "linguistic": has_variance_folder and (VARIANCE / "linguistic.onnx").exists(),
    "dur": has_variance_folder and (VARIANCE / "dur.onnx").exists(),
    "pitch": has_variance_folder and (VARIANCE / "pitch.onnx").exists(),
    "variance": has_variance_folder and (VARIANCE / "variance.onnx").exists(),
}
info("components found: " + (", ".join(k for k, v in parts.items() if v) or "acoustic only"))

BUILD = DEST / FOLDER_NAME
if BUILD.exists():
    shutil.rmtree(BUILD)

DSMAIN = BUILD / "dsmain"
SUBFOLDERS = {"dur": BUILD / "dsdur", "pitch": BUILD / "dspitch",
              "variance": BUILD / "dsvariance"}


# --------------------------------------------------------------------------- #
#  1. Copy files
# --------------------------------------------------------------------------- #
with step("Laying out the bank"):
    (DSMAIN / "embeds").mkdir(parents=True, exist_ok=True)
    for filename in ("acoustic.onnx", "acoustic_phonemes.json", "acoustic_languages.json"):
        source = ACOUSTIC / filename
        if not source.exists():
            die(f"{filename} is missing from {ACOUSTIC}",
                fix="Re-run step 6; the export did not finish.")
        shutil.copy(source, DSMAIN / filename)
    shutil.copy(ACOUSTIC / "dsconfig.yaml", BUILD / "dsconfig.yaml")

    acoustic_embeds = []
    for emb in sorted(ACOUSTIC.glob("*.emb")):
        shutil.copy(emb, DSMAIN / "embeds" / emb.name)
        acoustic_embeds.append(emb.stem)

    if parts["linguistic"]:
        shutil.copy(VARIANCE / "linguistic.onnx", DSMAIN / "linguistic.onnx")

    variance_embeds = []
    for key, folder in SUBFOLDERS.items():
        if not parts[key]:
            continue
        (folder / "embeds").mkdir(parents=True, exist_ok=True)
        shutil.copy(VARIANCE / f"{key}.onnx", folder / f"{key}.onnx")
        for emb in sorted(VARIANCE.glob("*.emb")):
            shutil.copy(emb, folder / "embeds" / emb.name)
        variance_embeds = [emb.stem for emb in sorted(VARIANCE.glob("*.emb"))]
    ok(f"{len(acoustic_embeds)} acoustic embed(s), {len(variance_embeds)} variance embed(s)")


# --------------------------------------------------------------------------- #
#  2. Configs
# --------------------------------------------------------------------------- #
with step("Writing configs"):
    acoustic_train_cfg = load_yaml(ACOUSTIC / "config.yaml") if (
        ACOUSTIC / "config.yaml").exists() else {}
    variance_train_cfg = load_yaml(VARIANCE / "config.yaml") if (
        has_variance_folder and (VARIANCE / "config.yaml").exists()) else {}
    variance_dsconfig = load_yaml(VARIANCE / "dsconfig.yaml") if (
        has_variance_folder and (VARIANCE / "dsconfig.yaml").exists()) else {}

    # Name the vocoder the model was actually trained against rather than assuming.
    trained_vocoder = Path(acoustic_train_cfg.get("vocoder_ckpt", "")).parent.name
    main_config = load_yaml(BUILD / "dsconfig.yaml")
    main_config.update({
        "singer_type": "diffsinger",
        "acoustic": "dsmain/acoustic.onnx",
        "phonemes": "dsmain/acoustic_phonemes.json",
        "languages": "dsmain/acoustic_languages.json",
        "vocoder": trained_vocoder or "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02",
    })
    if acoustic_embeds:
        main_config["speakers"] = [f"embeds/{name}" for name in acoustic_embeds]
    dump_yaml(main_config, BUILD / "dsconfig.yaml")

    # Shared values for the sub-model configs. Everything is read from the two
    # training configs, and every lookup has a default - the original crashed
    # with a NameError whenever a bank had pitch.onnx but no variance.onnx.
    shared = {
        "phonemes": "../dsmain/acoustic_phonemes.json",
        "languages": "../dsmain/acoustic_languages.json",
        "linguistic": "../dsmain/linguistic.onnx",
        "sample_rate": acoustic_train_cfg.get("audio_sample_rate", 44100),
        "hop_size": acoustic_train_cfg.get("hop_size", 512),
        "hidden_size": variance_dsconfig.get(
            "hidden_size", variance_train_cfg.get("hidden_size", 256)),
        "use_lang_id": bool(acoustic_train_cfg.get("use_lang_id", False)),
        "use_continuous_acceleration": variance_dsconfig.get(
            "use_continuous_acceleration", True),
    }
    if variance_embeds:
        shared["speakers"] = [f"embeds/{name}" for name in variance_embeds]

    if parts["dur"]:
        dump_yaml({**shared, "dur": "dur.onnx", "predict_dur": True},
                  SUBFOLDERS["dur"] / "dsconfig.yaml")
    if parts["variance"]:
        dump_yaml({
            **shared,
            "variance": "variance.onnx",
            "predict_dur": bool(variance_train_cfg.get("predict_dur", True)),
            "predict_tension": bool(variance_train_cfg.get("predict_tension", False)),
            "predict_voicing": bool(variance_train_cfg.get("predict_voicing", False)),
            "predict_energy": bool(variance_train_cfg.get("predict_energy", False)),
            "predict_breathiness": bool(variance_train_cfg.get("predict_breathiness", False)),
        }, SUBFOLDERS["variance"] / "dsconfig.yaml")
    if parts["pitch"]:
        dump_yaml({
            **shared,
            "pitch": "pitch.onnx",
            "use_expr": True,
            "predict_dur": bool(variance_train_cfg.get("predict_dur", True)),
            "use_note_rest": bool(
                variance_dsconfig.get("use_note_rest",
                                      variance_train_cfg.get("use_melody_encoder", False))),
        }, SUBFOLDERS["pitch"] / "dsconfig.yaml")
    ok("dsconfig.yaml written for " + ", ".join(
        ["dsmain"] + [f"ds{k}" for k, v in parts.items() if v and k != "linguistic"]))

    # character.txt / character.yaml
    (BUILD / "character.txt").write_text(
        f"name={NAME}\nauthor=\nweb=\nversion=\n", encoding="utf-8")
    character = {
        "name": NAME,
        "singer_type": "diffsinger",
        "text_file_encoding": "utf-8",
        "default_phonemizer": "OpenUtau.Core.DiffSinger.DiffSingerPhonemizer",
        "image": None,
        "portrait": None,
        "portrait_opacity": 0.45,
    }
    if acoustic_embeds:
        character["subbanks"] = [
            {"color": f"{index:02d}: {name}", "suffix": f"embeds/{name}"}
            for index, name in enumerate(acoustic_embeds, start=1)
        ]
    dump_yaml(character, BUILD / "character.yaml")
    ok("character.txt / character.yaml written")


# --------------------------------------------------------------------------- #
#  3. Dictionaries
# --------------------------------------------------------------------------- #
with step("Generating dsdict files"):
    dictionary_files = sorted(ACOUSTIC.glob("dictionary-*.txt"))
    if not dictionary_files:
        dictionary_files = sorted(ACOUSTIC.glob("*.txt"))
    if not dictionary_files:
        die("no phoneme dictionary found next to the acoustic ONNX",
            fix="Step 6 copies dictionary-*.txt from your model folder. Copy the\n"
                "dictionary-<lang>.txt files from your model folder into\n"
                f"{ACOUSTIC} and run this cell again.")

    extra_phonemes = acoustic_train_cfg.get("extra_phonemes") or []
    general_extra, per_language_extra = [], {}
    for phoneme in extra_phonemes:
        if "/" in str(phoneme):
            language, base = str(phoneme).split("/", 1)
            per_language_extra.setdefault(language, []).append(base)
        elif phoneme not in ("AP", "SP") and "trash" not in str(phoneme):
            general_extra.append(phoneme)

    multi_language = len(dictionary_files) > 1

    def language_of(path):
        stem = path.stem
        return stem.split("-", 1)[1] if "-" in stem else "custom"

    def symbol_type(phoneme):
        return PHONEME_TYPES.get(phoneme[0], "stop")

    def write_dsdict(path, symbols, entries, replacements=None):
        payload = {
            "symbols": [{"symbol": "SP", "type": "vowel"},
                        {"symbol": "AP", "type": "vowel"}] + symbols,
        }
        if replacements:
            payload["replacements"] = replacements
        payload["entries"] = entries + [{"grapheme": "SP", "phonemes": ["SP"]},
                                        {"grapheme": "AP", "phonemes": ["AP"]}]
        dump_yaml(payload, path)

    all_symbols, all_entries = [], []
    for phoneme in general_extra:
        all_symbols.append({"symbol": phoneme, "type": symbol_type(phoneme)})
        all_entries.append({"grapheme": phoneme, "phonemes": [phoneme]})

    per_language = {}
    for path in dictionary_files:
        language = language_of(path)
        phonemes = []
        for line in path.read_text(encoding="utf-8").splitlines():
            parts_of_line = line.split()
            if parts_of_line:
                phonemes.append(parts_of_line[0])
        phonemes += per_language_extra.get(language, [])
        per_language[language] = phonemes
        for phoneme in phonemes:
            symbol = f"{language}/{phoneme}" if multi_language else phoneme
            all_symbols.append({"symbol": symbol, "type": symbol_type(phoneme)})
            all_entries.append({"grapheme": phoneme, "phonemes": [symbol]})

    written = []
    if multi_language:
        for language, phonemes in per_language.items():
            symbols = [{"symbol": f"{language}/{p}", "type": symbol_type(p)} for p in phonemes]
            entries = [{"grapheme": p, "phonemes": [f"{language}/{p}"]} for p in phonemes]
            replacements = None
            if language not in OU_CANON_LANGUAGES:
                replacements = [{"from": p, "to": f"{language}/{p}"} for p in phonemes]
            # Every per-language dict carries the full symbol table so a phoneme
            # typed by hand from another language still resolves.
            write_dsdict(BUILD / f"dsdict-{language}.yaml", all_symbols, entries, replacements)
            written.append(f"dsdict-{language}.yaml ({len(phonemes)} phonemes)")
    write_dsdict(BUILD / "dsdict.yaml", all_symbols, all_entries)
    written.append(f"dsdict.yaml ({len(all_symbols)} symbols)")

    # Copy every dsdict into the sub-model folders as well. OpenUtau versions
    # differ on where they look; duplicating a few KB removes the guesswork.
    for source in sorted(BUILD.glob("dsdict*.yaml")):
        for key, folder in SUBFOLDERS.items():
            if parts[key]:
                shutil.copy(source, folder / source.name)
    for line in written:
        ok(line)


# --------------------------------------------------------------------------- #
#  4. Optional custom vocoder
# --------------------------------------------------------------------------- #
if clean(custom_vocoder_onnx):
    with step("Adding custom vocoder"):
        vocoder = Path(clean(custom_vocoder_onnx))
        if not vocoder.exists():
            die(f"vocoder not found: {vocoder}")
        vocoder_yaml = vocoder.parent / "vocoder.yaml"
        if not vocoder_yaml.exists():
            die(f"vocoder.yaml not found beside {vocoder.name}",
                fix="A custom vocoder needs both the .onnx and its vocoder.yaml.")
        target = BUILD / "dsvocoder"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy(vocoder, target / vocoder.name)
        shutil.copy(vocoder_yaml, target / "vocoder.yaml")
        main_config = load_yaml(BUILD / "dsconfig.yaml")
        main_config["vocoder"] = vocoder.stem
        dump_yaml(main_config, BUILD / "dsconfig.yaml")
        ok(f"dsvocoder/{vocoder.name}")


# --------------------------------------------------------------------------- #
#  5. Package
# --------------------------------------------------------------------------- #
tree = []
for path in sorted(BUILD.rglob("*")):
    if path.is_file():
        tree.append({"file": str(path.relative_to(BUILD)),
                     "size": human_size(path.stat().st_size)})

header(f"Voicebank built: {BUILD}", accent=dsnb.DONE_ACCENT)
print_table(tree, [("file", "file"), ("size", "size")])

if make_zip:
    with step("Zipping"):
        archive = shutil.make_archive(str(DEST / FOLDER_NAME), "zip",
                                      root_dir=str(DEST), base_dir=FOLDER_NAME)
        ok(f"{archive} ({human_size(Path(archive).stat().st_size)})")

print("\n  Install it by dropping the zip onto OpenUtau, or unzip it into")
print("  OpenUtau/Singers/. Enable the DiffSinger phonemizer if it is not already the default.")
