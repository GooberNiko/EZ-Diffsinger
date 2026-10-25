#@title 3. Build training config  -  and check it will actually work
#@markdown Writes `configs/<your model>.yaml` inside the DiffSinger repo. Your stock
#@markdown configs are never modified, so this cell is safe to re-run as often as you like.

#@markdown ---
#@markdown ### Training mode
training_mode = "Original"  #@param ["Original", "Woah's Epic Faster Training (experimental)"]
#@markdown <font size="-1">**Original** is the stock DiffSinger setup, exactly as before.<br>
#@markdown **Woah's Epic Faster Training** &#9889; binarizes on every CPU core, forces local-disk
#@markdown binary data, and in step 5 compiles the backbone, picks the fastest precision for your
#@markdown GPU (fp16 on a T4, bf16 on L4/A100), sizes batches to it and cuts CPU and validation
#@markdown overhead. Tuned for the free T4 first. Only stock
#@markdown DiffSinger settings are changed. *Experimental:* if you hit CUDA out-of-memory, switch
#@markdown back to Original.</font>

#@markdown ---
#@markdown ### Model
model_type = "acoustic"  #@param ["acoustic", "variance"]
model_dir = "/content/drive/MyDrive/DiffSinger/my_singer_acoustic"  #@param {type:"string"}
#@markdown <font size="-1">Folder for this model's checkpoints - **edit the example above**. It is
#@markdown created for you. Keep it under `/content/drive/MyDrive/` or you lose it when Colab
#@markdown disconnects. Use a **separate folder per model**, since a full voicebank needs an
#@markdown acoustic *and* a variance model. This is also what you paste into step 5 to resume
#@markdown and into step 6 to export.</font>

diffusion_type = "reflow"  #@param ["reflow", "ddpm"]
backbone = "lynxnet"  #@param ["lynxnet", "lynxnet2", "wavenet"]
#@markdown <font size="-1">`lynxnet2` requires the `muon_lynxnet2` branch in step 1.</font>
sampling_algorithm = "euler"  #@param ["euler", "rk2", "rk4", "rk5"]
diff_accelerator = "unipc"  #@param ["unipc", "ddim", "pndm", "dpm-solver"]
loss_type = "l2"  #@param ["l2", "l1"]
precision = "16-mixed"  #@param ["16-mixed", "bf16-mixed", "32-true"]

#@markdown ---
#@markdown ### Features to train
#@markdown <font size="-1">Acoustic *embeds* and variance *predictions* must match, or OpenUtau
#@markdown will ask for a parameter your model cannot produce. This cell checks that for you.</font>
tension = True  #@param {type:"boolean"}
voicing = True  #@param {type:"boolean"}
breathiness = False  #@param {type:"boolean"}
energy = False  #@param {type:"boolean"}
#@markdown <font size="-1">Energy is deprecated upstream; leave it off unless you know why you want it.</font>

#@markdown <font size="-1">Variance only:</font>
duration_training = True  #@param {type:"boolean"}
pitch_training = "off"  #@param ["off", "standard", "melody encoder"]

#@markdown ---
#@markdown ### Extraction
f0_extractor = "rmvpe"  #@param ["rmvpe", "parselmouth", "harvest"]
parameter_extraction = "vr"  #@param ["vr", "world"]
shallow_diffusion = "on (gt_val)"  #@param ["on (gt_val)", "on (aux_val)", "off"]

#@markdown ---
#@markdown ### Fine-tuning
finetune = False  #@param {type:"boolean"}
base_model_path = ""  #@param {type:"string"}
#@markdown <font size="-1">Leave blank to download haru0l's public base model for the selected model type.</font>

#@markdown ---
#@markdown ### Storage
binary_location = "local disk (fastest)"  #@param ["local disk (fastest)", "next to the model (Drive)"]
#@markdown <font size="-1">Reading binarized data over Drive's FUSE mount slows training down noticeably.
#@markdown Local disk is rebuilt by re-running step 4 if the runtime dies.</font>

#@markdown ---
#@markdown ### Network size &nbsp;<font size="-1">- leave these alone unless you know what they do</font>
acoustic_hidden_size = 256  #@param {type:"slider", min:64, max:1024, step:32}
acoustic_num_layers = 6  #@param {type:"slider", min:2, max:24, step:1}
acoustic_num_channels = 1024  #@param {type:"slider", min:64, max:2048, step:64}
variance_hidden_size = 256  #@param {type:"slider", min:64, max:1024, step:32}
duration_hidden_size = 512  #@param {type:"slider", min:64, max:1024, step:32}
melody_encoder_hidden_size = 128  #@param {type:"slider", min:32, max:512, step:32}
pitch_num_layers = 6  #@param {type:"slider", min:2, max:24, step:1}
pitch_num_channels = 512  #@param {type:"slider", min:64, max:1024, step:64}
variance_num_layers = 6  #@param {type:"slider", min:2, max:24, step:1}
variance_num_channels = 384  #@param {type:"slider", min:64, max:1024, step:64}

# --------------------------------------------------------------------------- #

import csv
import importlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/content")
import dsnb
importlib.reload(dsnb)
from dsnb import (CONTENT, DEFAULT_ACCENT, FAST_ACCENT, REPO, die, download_many, dump_yaml,
                  header, info, is_fast, link_experiment, load_state, ok, preflight,
                  print_table, sanitize_exp_name, save_state, step, warn)

state = load_state(required=["setup_done", "data_dir", "dictionaries"])
DATA_DIR = Path(state["data_dir"])
speakers = state["speakers"]

FAST = is_fast(training_mode)
header(f"Build {model_type} config",
       subtitle=("⚡ Woah's Epic Faster Training mode (experimental)" if FAST
                 else "Original training mode"),
       accent=FAST_ACCENT if FAST else DEFAULT_ACCENT)

if not model_dir.strip():
    die("model_dir is empty",
        fix="Pick a folder for your checkpoints, e.g.\n"
            "  /content/drive/MyDrive/DiffSinger/my_singer_acoustic")
MODEL_DIR = Path(model_dir.strip().strip('"').strip("'"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)
EXP_NAME = sanitize_exp_name(MODEL_DIR.name)

if not str(MODEL_DIR).startswith("/content/drive/"):
    warn(f"{MODEL_DIR} is not on Google Drive.")
    warn("Everything there is deleted when this Colab runtime ends - including")
    warn("your checkpoints. Use a path under /content/drive/MyDrive/ instead.")
elif not Path("/content/drive/MyDrive").exists():
    die("that path is on Drive, but Drive is not mounted",
        fix="Re-run step 1 with 'mount_drive' ticked.")

existing_ckpt = dsnb.find_latest_ckpt(MODEL_DIR)
if existing_ckpt:
    warn(f"{MODEL_DIR.name} already contains {existing_ckpt.name}.")
    warn("Training will continue from it rather than start over. For a fresh model,")
    warn("point model_dir at a new folder.")

if backbone == "lynxnet2" and state.get("branch") != "muon_lynxnet2":
    die("lynxnet2 needs the muon_lynxnet2 branch",
        fix="Re-run step 1 with branch = muon_lynxnet2, or pick a different backbone.")


with step("Copying dictionaries next to the model"):
    # Keeping the dictionaries inside the model folder makes it self-contained:
    # the config still resolves after a fresh runtime, and the OpenUtau builder
    # in step 7 can find them without you hunting for /content paths.
    dict_out = MODEL_DIR / "dictionaries"
    dict_out.mkdir(parents=True, exist_ok=True)
    dictionaries = {}
    for lang, source in state["dictionaries"].items():
        target = dict_out / Path(source).name
        shutil.copy(source, target)
        dictionaries[lang] = str(target)
        ok(f"{lang} -> {target.name}")


# --------------------------------------------------------------------------- #
#  Datasets block
# --------------------------------------------------------------------------- #
with step("Describing datasets"):
    datasets = []
    per_speaker_tests = max(1, 4 // max(1, len(speakers)))

    for spk_id, entry in enumerate(speakers):
        folder = Path(entry["folder"])
        if state["data_type"] == "ds (DiffSinger)":
            items = sorted(p.stem for p in folder.glob("*.ds"))
        else:
            items = sorted(p.stem for p in (folder / "wavs").glob("*.wav"))
        if not items:
            die(f"'{folder.name}' has no audio segments",
                fix="Re-run step 2; the segmenter did not produce anything for this speaker.")
        # Whole item names, not prefixes: a short prefix can silently match every
        # item in the set and leave the training split empty.
        chosen = items[:per_speaker_tests]
        datasets.append({
            "raw_data_dir": str(folder),
            "speaker": entry["speaker"],
            "spk_id": spk_id,
            "language": entry["language"],
            "test_prefixes": chosen,
        })
        ok(f"spk_id {spk_id}: {entry['speaker']} [{entry['language']}] - {len(items)} items")

    num_spk = len(datasets)
    languages = sorted({d["language"] for d in datasets})
    # use_lang_id depends on how many languages are in play. The original tied it
    # to merged_phoneme_groups, which is a different setting entirely.
    use_lang_id = len(dictionaries) > 1
    if len(dictionaries) > 1 and len(languages) == 1:
        info("multiple dictionaries but one language in use - keeping use_lang_id on")


# --------------------------------------------------------------------------- #
#  Fine-tuning
# --------------------------------------------------------------------------- #
PRETRAINED = {
    "acoustic": "https://github.com/haru0l/diffsinger_models/releases/download/"
                "acoustic/model_ckpt_steps_49000.ckpt",
    "variance": "https://github.com/haru0l/diffsinger_models/releases/download/"
                "variance/model_ckpt_steps_51000.ckpt",
}

finetune_ckpt = None
if finetune:
    with step("Preparing base model"):
        if base_model_path.strip():
            finetune_ckpt = Path(base_model_path.strip().strip('"').strip("'"))
            if not finetune_ckpt.exists():
                die(f"base model not found: {finetune_ckpt}",
                    fix="Check the path, or clear it to use the public base model.")
        else:
            finetune_ckpt = CONTENT / "pretrain_models" / f"{model_type}_pretrain.ckpt"
            download_many([(PRETRAINED[model_type], finetune_ckpt)], quiet=True)
        # The original set finetune_ckpt_path only on the *download* branch, so a
        # custom base_model_path raised NameError a few lines later.
        ok(f"fine-tuning from {finetune_ckpt}")


# --------------------------------------------------------------------------- #
#  Storage
# --------------------------------------------------------------------------- #
if FAST and not binary_location.startswith("local"):
    info("fast mode keeps binarized data on local disk - Drive's FUSE mount is the slowest")
    info("part of a training step. Tick 'backup_binary_to_drive' in step 4 for a safety copy.")
    binary_location = "local disk (fastest)"

if binary_location.startswith("local"):
    BINARY_DIR = CONTENT / "binary" / EXP_NAME
else:
    BINARY_DIR = MODEL_DIR / "binary"
BINARY_DIR.parent.mkdir(parents=True, exist_ok=True)

with step("Linking experiment folder"):
    link = link_experiment(EXP_NAME, MODEL_DIR)
    ok(f"checkpoints/{EXP_NAME} -> {MODEL_DIR}")
    info("DiffSinger resolves its work dir as checkpoints/<exp_name>, so this symlink is")
    info("all it takes to train straight into Drive - no source files are patched.")


# --------------------------------------------------------------------------- #
#  Compose the override config
# --------------------------------------------------------------------------- #
shallow_on = shallow_diffusion.startswith("on")
gt_start = shallow_diffusion == "on (gt_val)"
predict_pitch = pitch_training != "off"

cfg = {
    "base_config": [str(REPO / "configs" / f"{model_type}.yaml")],
    "datasets": datasets,
    "num_spk": num_spk,
    "use_spk_id": num_spk > 1,
    "dictionaries": dictionaries,
    "num_lang": len(dictionaries),
    "use_lang_id": use_lang_id,
    "extra_phonemes": state.get("extra_phonemes") or [],
    "merged_phoneme_groups": state.get("merged_phoneme_groups") or [],
    "binary_data_dir": str(BINARY_DIR),
    "pl_trainer_precision": precision,
    "diffusion_type": diffusion_type,
    "diff_accelerator": diff_accelerator,
    "main_loss_type": loss_type,
    "pe": f0_extractor,
    "pe_ckpt": "checkpoints/rmvpe/model.pt" if f0_extractor == "rmvpe" else None,
    "hnsep": parameter_extraction,
    "hnsep_ckpt": "checkpoints/vr/model.pt",
    "tension_smooth_width": 0.06,
    "finetune_enabled": bool(finetune),
    "finetune_ckpt_path": str(finetune_ckpt) if finetune_ckpt else None,
    "finetune_strict_shapes": not finetune,
    "pretrain": bool(finetune),
}

if model_type == "acoustic":
    # Point at whichever vocoder step 1 actually downloaded, so choosing the plain
    # nsf_hifigan there does not leave the config asking for the pc_ one.
    vocoder_dir = ("nsf_hifigan_44.1k_hop512_128bin_2024.02"
                   if state.get("vocoder_choice") == "nsf_hifigan"
                   else "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02")
    cfg.update({
        "vocoder_ckpt": f"checkpoints/{vocoder_dir}/model.ckpt",
        "use_tension_embed": tension,
        "use_voicing_embed": voicing,
        "use_breathiness_embed": breathiness,
        "use_energy_embed": energy,
        "use_shallow_diffusion": shallow_on,
        "shallow_diffusion_args": {"val_gt_start": gt_start},
        "sampling_algorithm": sampling_algorithm,
        "hidden_size": acoustic_hidden_size,
        "backbone_type": backbone,
        "backbone_args": {"num_layers": acoustic_num_layers,
                          "num_channels": acoustic_num_channels},
        # Augmentation is on by default and stays on - it is close to free quality.
        "use_key_shift_embed": True,
        "use_speed_embed": True,
        "augmentation_args": {
            "random_pitch_shifting": {"enabled": True},
            "random_time_stretching": {"enabled": True},
        },
    })
else:
    cfg.update({
        "predict_tension": tension,
        "predict_voicing": voicing,
        "predict_breathiness": breathiness,
        "predict_energy": energy,
        "predict_dur": duration_training,
        "predict_pitch": predict_pitch,
        "use_melody_encoder": pitch_training == "melody encoder",
        "use_glide_embed": False,
        "hidden_size": variance_hidden_size,
        "dur_prediction_args": {"hidden_size": duration_hidden_size},
        "melody_encoder_args": {"hidden_size": melody_encoder_hidden_size},
        "variances_prediction_args": {
            "backbone_type": backbone,
            "backbone_args": {"num_layers": variance_num_layers,
                              "num_channels": variance_num_channels},
        },
        "pitch_prediction_args": {
            "backbone_type": backbone,
            "backbone_args": {"num_layers": pitch_num_layers,
                              "num_channels": pitch_num_channels},
        },
        "binarization_args": {"prefer_ds": state["data_type"] == "ds (DiffSinger)"},
    })

if FAST:
    # Parallel binarize. Stock is 0 (single process), which on a large dataset is
    # most of the time spent in step 4.
    cfg.setdefault("binarization_args", {})["num_workers"] = dsnb.fast_binarize_workers()

cfg = {key: value for key, value in cfg.items() if value is not None}


# --------------------------------------------------------------------------- #
#  Deep-merge our overrides onto the stock config so nested dicts keep their
#  untouched keys (setting backbone_args wholesale would drop kernel_size etc.)
# --------------------------------------------------------------------------- #
def deep_merge(base, override):
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


with step("Writing config"):
    stock = dsnb.resolve_merged_config(REPO / "configs" / f"{model_type}.yaml")
    unknown = [k for k in cfg if k not in stock and k != "base_config"]
    resolved = {"base_config": cfg["base_config"]}
    for key, value in cfg.items():
        if key == "base_config":
            continue
        resolved[key] = deep_merge(stock[key], value) if (
            isinstance(value, dict) and isinstance(stock.get(key), dict)) else value

    CONFIG_PATH = REPO / "configs" / f"nb_{EXP_NAME}_{model_type}.yaml"
    dump_yaml(resolved, CONFIG_PATH)
    dump_yaml(resolved, MODEL_DIR / "notebook_config.yaml")   # copy for resuming later
    ok(f"{CONFIG_PATH}")
    if unknown:
        # These still take effect - hparams merges our file over the stock one - but
        # a key DiffSinger never reads is inert, so it is worth naming them.
        info("settings added on top of the stock config: " + ", ".join(sorted(unknown)))


# --------------------------------------------------------------------------- #
#  Sanity checks that save you an hour
# --------------------------------------------------------------------------- #
with step("Checking the setup makes sense"):
    if model_type == "variance" and predict_pitch:
        sample = next(DATA_DIR.rglob("transcriptions.csv"), None)
        if sample is None:
            die("pitch training needs transcriptions.csv", fix="Re-run step 2.")
        with sample.open("r", newline="", encoding="utf-8") as fh:
            columns = next(csv.reader(fh), [])
        if "note_seq" not in columns:
            die("pitch training is on but your data has no note_seq (MIDI)",
                fix="Re-run step 2 with 'estimate_midi' set to parselmouth, harvest or SOME.")
        ok("note_seq present - pitch training can proceed")

    if model_type == "variance" and not (duration_training or predict_pitch
                                         or tension or voicing or breathiness or energy):
        die("a variance model that predicts nothing has nothing to train",
            fix="Turn on duration, pitch, or at least one parameter.")

    preflight(CONFIG_PATH, model_type)

    if model_type == "variance":
        info("Reminder: your acoustic model must embed exactly the parameters this")
        info("variance model predicts (tension/voicing/breathiness/energy).")


# --------------------------------------------------------------------------- #
header("Config summary", accent=FAST_ACCENT if FAST else DEFAULT_ACCENT)
summary = [
    {"k": "training mode", "v": ("⚡ " + training_mode) if FAST else training_mode},
    {"k": "model", "v": f"{model_type} ({'multi' if num_spk > 1 else 'single'}-speaker)"},
    {"k": "speakers", "v": ", ".join(d["speaker"] for d in datasets)},
    {"k": "languages", "v": ", ".join(sorted(dictionaries))},
    {"k": "backbone", "v": f"{backbone} / {diffusion_type} / {diff_accelerator}"},
    {"k": "precision", "v": precision},
    {"k": "f0 / params", "v": f"{f0_extractor} / {parameter_extraction}"},
    {"k": "fine-tune", "v": str(finetune_ckpt) if finetune_ckpt else "no (training from scratch)"},
    {"k": "binary data", "v": str(BINARY_DIR)},
    {"k": "binarize workers", "v": str(dsnb.fast_binarize_workers()) if FAST else "0 (stock)"},
    {"k": "checkpoints", "v": str(MODEL_DIR)},
    {"k": "exp_name", "v": EXP_NAME},
    {"k": "config", "v": str(CONFIG_PATH)},
]
if model_type == "acoustic":
    summary.insert(3, {"k": "embeds", "v": ", ".join(
        [n for n, on in (("tension", tension), ("voicing", voicing),
                         ("breathiness", breathiness), ("energy", energy)) if on]) or "none"})
else:
    summary.insert(3, {"k": "predicts", "v": ", ".join(
        [n for n, on in (("duration", duration_training), ("pitch", predict_pitch),
                         ("tension", tension), ("voicing", voicing),
                         ("breathiness", breathiness), ("energy", energy)) if on]) or "none"})
print_table(summary, [("k", "setting"), ("v", "value")])

save_state(
    model_type=model_type,
    model_dir=str(MODEL_DIR),
    exp_name=EXP_NAME,
    config_path=str(CONFIG_PATH),
    binary_dir=str(BINARY_DIR),
    binary_on_drive=not binary_location.startswith("local"),
    training_mode=training_mode,
)
print("\n  Next: step 4 - binarize.")
