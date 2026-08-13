#!/usr/bin/env python3
"""qwen_layers.py -- decompose a raster image into editable layers, via Kaggle GPUs.

WHAT THIS IS FOR
  AI image generators produce a single flat raster with defects baked in: a wrong
  hand, a mangled word, an object in the wrong place. You cannot fix those without
  repainting around them. Qwen-Image-Layered (Alibaba, Apache-2.0) decomposes one
  raster into several semantically separated RGBA layers with real alpha, so each
  element becomes independently editable -- and then GIMP/Inkscape (or their MCP
  servers) can fix ONE layer without touching the rest.

  This script is the whole round trip: local image in, layered GIMP/Inkscape
  artifacts out. It runs the model on Kaggle's free dual-T4 notebooks, so no local
  GPU is needed -- which is the whole point of the controller pattern.

WHY KAGGLE AND NOT COLAB
  The model instance is 115 GB uncompressed (measured via `kaggle models get`, NOT
  the ~55 GB widely quoted). On Kaggle it ATTACHES as a mounted read-only Model, so
  it never touches your working disk or download budget. Colab's free tier has to
  pull weights down and dies doing it. And since Kaggle CLI 2.x you can finally
  DEMAND the accelerator -- `machine_shape: NvidiaTeslaT4` gives 2x T4 = 30 GB
  VRAM deterministically, instead of being handed a 16 GB P100 by lottery.

THE OUTPUT, AND WHY THERE IS AN SVG
  Upstream exports PNG / PSD / PPTX -- there is no SVG exporter, which is usually
  reported as "Inkscape is not supported". That is a gap in the exporter, not in
  Inkscape: a layered SVG that wraps each RGBA layer in an
  `inkscape:groupmode="layer"` group opens natively with the layer stack intact.
  This script generates that locally. The layers stay raster (this model emits
  pixels, not paths) -- so you get separable, individually editable objects, not
  vector art. Trace Bitmap is still your problem.

VERIFICATION -- AN IDENTITY, NOT A VIBE
  A decomposition can fail by returning layers that look plausible and do not
  reconstruct the input. So every run alpha-composites the returned layers back
  together and compares against the original. That identity is the gate; the file
  count is not. `--max-recompose-mae` sets the threshold, and the check reports the
  measured error either way so a "pass" is never silent.

EXIT CODES
  0 ok · 1 real observed negative · 2 usage · 3 could-not-tell
  4 not authenticated · 5 timeout · 6 refused (would spend GPU quota)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

EXIT_OK, EXIT_NO, EXIT_USAGE, EXIT_UNKNOWN, EXIT_AUTH, EXIT_TIMEOUT, EXIT_REFUSED = 0, 1, 2, 3, 4, 5, 6

MODEL_SOURCE = "qwen-lm/qwen-image-layered/transformers/qwen-image-layered/1"
MODEL_MOUNT_HINT = "qwen-image-layered"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
PSD_MAGIC = b"8BPS"
_JSON = False


def emit(ok: bool, human: str, data: Any = None, remediation: str = "", code: int = EXIT_OK) -> int:
    if _JSON:
        print(json.dumps({"ok": ok, "human": human, "remediation": remediation,
                          "data": data}, indent=2, default=str))
    else:
        stream = sys.stdout if ok else sys.stderr
        print(human, file=stream)
        if remediation:
            print(f"  try: {remediation}", file=stream)
    return code


def die(human: str, code: int = EXIT_UNKNOWN, remediation: str = "") -> None:
    sys.exit(emit(False, f"qwen_layers: {human}", None, remediation, code))


def log(msg: str) -> None:
    if not _JSON:
        print(f"[qwen_layers] {msg}", flush=True)


# ------------------------------------------------------------------ kaggle CLI

def kaggle_bin() -> str:
    # shutil.which first: a bare name is not guaranteed to be executable by
    # subprocess on every platform, and the resulting error is indistinguishable
    # from "not installed".
    exe = shutil.which("kaggle")
    if not exe:
        die("kaggle CLI not found on PATH", EXIT_UNKNOWN,
            "uv tool install --python 3.12 kaggle")
    return exe


def kag(*args: str, timeout: int = 900, check: bool = True) -> str:
    proc = subprocess.run([kaggle_bin(), *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout,
                          stdin=subprocess.DEVNULL)
    out = (proc.stdout or "") + (proc.stderr or "")
    if check and proc.returncode != 0:
        low = out.lower()
        if "401" in out or "unauthor" in low or "credentials" in low or "access_token" in low:
            die(f"Kaggle rejected the credentials: {out.strip()[:300]}", EXIT_AUTH,
                "set KAGGLE_USERNAME and KAGGLE_KEY, or place ~/.kaggle/kaggle.json")
        die(f"`kaggle {' '.join(args)}` failed: {out.strip()[:400]}", EXIT_UNKNOWN)
    return out


def kaggle_username() -> str:
    for var in ("KAGGLE_USERNAME",):
        if os.environ.get(var):
            return os.environ[var]
    out = kag("config", "view", check=False)
    for line in out.splitlines():
        if "username" in line.lower():
            parts = line.replace(":", " ").split()
            if parts:
                return parts[-1].strip()
    die("could not determine your Kaggle username", EXIT_AUTH,
        "export KAGGLE_USERNAME=<your-handle>")
    return ""


# ------------------------------------------------------------------- the kernel

KERNEL_SCRIPT = r'''
"""Generated by qwen_layers.py -- runs on a Kaggle dual-T4 notebook."""
import glob, json, os, sys, traceback

# Must be set before the CUDA allocator initialises, i.e. before torch is imported.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

REPORT = {"stage": "start", "ok": False}

def bail(stage, err):
    REPORT.update(stage=stage, ok=False, error=str(err)[:2000])
    with open("/kaggle/working/report.json", "w") as fh:
        json.dump(REPORT, fh, indent=2)
    traceback.print_exc()
    sys.exit(1)

try:
    import torch
    # Prove which accelerator we ACTUALLY got before spending 20 minutes loading a
    # 115 GB model. Requesting NvidiaTeslaT4 is a request, not a guarantee.
    n = torch.cuda.device_count()
    caps = [torch.cuda.get_device_capability(i) for i in range(n)]
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    vram = [round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1) for i in range(n)]
    REPORT.update(gpus=n, gpu_names=names, capabilities=[f"{a}.{b}" for a, b in caps], vram_gb=vram)
    print("GPUs:", names, caps, vram, flush=True)
    if n == 0:
        bail("no_gpu", "no CUDA device -- enable_gpu did not take effect")
    if caps[0][0] < 7:
        print("WARNING: pre-Turing GPU; not the T4 that was requested", flush=True)
except Exception as e:
    bail("torch", e)

try:
    REPORT["stage"] = "install"
    # PIN TRANSFORMERS BELOW 5. Diagnosed 2026-08-13 with a CPU-only kernel:
    # Kaggle's image ships transformers 5.0.0, but this checkpoint was saved
    # under 4.5x. The 5.x refactor renamed Qwen2.5-VL's vision tower from
    # `visual.*` to `model.visual.*`, so 390 of the text encoder's 729 weights
    # match nothing and are silently left on the META device. Nothing raises at
    # load; it detonates ~20 min later as "Cannot copy out of meta tensor".
    # The model card's "transformers>=4.51.3" predates 5.0 and is now a trap:
    # it is satisfied by the very version that breaks it.
    # Do NOT use os.system(... -q ... | tail -3): it discards the output AND the
    # return code, so a failed install looks identical to a successful one. That
    # cost a full run -- the report said transformers 5.0.0 / diffusers 0.37.1,
    # i.e. NOTHING was installed, and nothing had complained.
    import subprocess as _sp

    def pip(*pkgs):
        p = _sp.run([sys.executable, "-m", "pip", "install", "--no-input", *pkgs],
                    capture_output=True, text=True, errors="replace", timeout=1800)
        tail = ((p.stdout or "") + (p.stderr or "")).strip().splitlines()[-6:]
        REPORT.setdefault("pip", []).append(
            {"pkgs": list(pkgs), "rc": p.returncode, "tail": tail})
        print("pip", pkgs, "rc", p.returncode, flush=True)
        for line in tail:
            print("   ", line, flush=True)
        return p.returncode

    # ORDER MATTERS, and getting it wrong cost a run. Installing transformers
    # first worked (5.0.0 -> 4.57.6), and then diffusers-from-git upgraded
    # huggingface-hub to 1.27.0, which transformers 4.x refuses (<1.0). Both pips
    # returned rc=0; the breakage was only in the resolver's warning text.
    # So: heavy/loose packages first, the constrained pair LAST and together, so
    # nothing installed afterwards can bump hub again.
    # DO NOT install diffusers from git. That was the root of the whole mess:
    # git main needs huggingface-hub 1.x (`get_cached_repo_tree`), hub 1.x breaks
    # transformers 4.x, and transformers 4.x is required because the checkpoint
    # predates the 5.x `visual.*` -> `model.visual.*` rename. A three-way deadlock
    # entirely of my own making -- because the RELEASED diffusers 0.37.1 already
    # ships QwenImageLayeredPipeline (verified: the file exists in tag v0.37.1)
    # and it accepts huggingface-hub<2.0. It is also what the Kaggle image
    # already has, so this is mostly a pin, not an install.
    pip("python-pptx", "bitsandbytes", "accelerate")
    pip("diffusers==0.37.1", "transformers>=4.51.3,<5", "huggingface-hub>=0.34.0,<1.0")

    # importlib.metadata, not the module attribute: a downgrade during this same
    # process can leave an already-imported module reporting its OLD version.
    import importlib.metadata as _md
    REPORT["diffusers"] = _md.version("diffusers")
    REPORT["transformers"] = _md.version("transformers")
    REPORT["huggingface_hub"] = _md.version("huggingface_hub")

    # VERIFY BY IMPORTING, in a FRESH interpreter. Correct-looking version
    # strings are not a working install: the last run had transformers 4.57.6
    # recorded and still could not import it.
    # Test the CLASS WE ACTUALLY NEED, not just the packages. `import diffusers`
    # succeeded on the previous run while `from diffusers import
    # QwenImageLayeredPipeline` was the thing that failed -- diffusers imports
    # its pipelines lazily, so the module-level check was one layer too shallow.
    for _mod, _stmt in (("transformers", "import transformers"),
                        ("diffusers", "import diffusers"),
                        ("QwenImageLayeredPipeline",
                         "from diffusers import QwenImageLayeredPipeline")):
        _p = _sp.run([sys.executable, "-c", _stmt],
                     capture_output=True, text=True, errors="replace", timeout=900)
        REPORT.setdefault("import_check", {})[_mod] = _p.returncode
        if _p.returncode != 0:
            bail("install", f"{_mod} unusable: "
                            f"{((_p.stderr or '') + (_p.stdout or '')).strip()[-600:]}")
    print("versions:", REPORT["diffusers"], REPORT["transformers"], flush=True)
    if int(REPORT["transformers"].split(".")[0]) >= 5:
        bail("install", f"transformers {REPORT['transformers']} still >=5 after the "
                        "pin; see REPORT['pip'] for what pip actually said")
except Exception as e:
    bail("install", e)

try:
    REPORT["stage"] = "locate_model"
    # The model is ATTACHED, not downloaded. Find its mount rather than assuming a
    # path: Kaggle's input layout for model instances has changed before.
    cands = [p for p in glob.glob("/kaggle/input/**/", recursive=True)
             if "qwen-image-layered" in p.lower()]
    cands.sort(key=len)
    model_path = None
    for c in cands:
        if any(os.path.exists(os.path.join(c, f)) for f in ("model_index.json",)):
            model_path = c
            break
    if model_path is None and cands:
        model_path = cands[0]
    if model_path is None:
        bail("locate_model", f"no mounted qwen-image-layered under /kaggle/input; saw {glob.glob('/kaggle/input/*')}")
    REPORT["model_path"] = model_path
    print("model at", model_path, flush=True)
except Exception as e:
    bail("locate_model", e)

try:
    REPORT["stage"] = "load"
    from diffusers import QwenImageLayeredPipeline
    from diffusers.quantizers import PipelineQuantizationConfig
    def make_qc(components):
        return PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                          "bnb_4bit_compute_dtype": torch.bfloat16},
            components_to_quantize=components,
        )

    # MEASURED, do not re-derive:
    #  * device_map accepts only {balanced, cuda, cpu} here -- "auto" is
    #    transformers' vocabulary and raises. An earlier rung wasted a run on it,
    #    which also meant max_memory was never actually tried with a valid map.
    #  * Plain "balanced" left Qwen2.5-VL's VISION TOWER on meta
    #    (text_encoder.model.visual.*) while placing everything else.
    #  * Both single-device offload paths OOM: model_offload tried to allocate
    #    9.51 GiB with 1.25 GiB free. ~23 GB wanted on one 14.6 GB card, so both
    #    cards are required and only "balanced" spans them.
    MAXMEM = {0: "13GiB", 1: "13GiB", "cpu": "20GiB"}
    qc_both = make_qc(["transformer", "text_encoder"])
    qc_tx = make_qc(["transformer"])
    # WHY A LADDER, NOT ONE CALL: the first attempt used
    # enable_model_cpu_offload(), which shuttles modules between CPU and a SINGLE
    # GPU. On a dual-T4 box that used 14.44 of 14.56 GiB on card 0 and left card 1
    # completely idle -- OOM with half the VRAM unused. That was a configuration
    # bug, not a capacity limit. device_map="balanced" shards across both cards.
    # Each attempt here costs ~15 min of quota, so fall back rather than fail.
    def meta_check(p):
        """A load that did not raise is NOT a load that works.

        device_map='balanced' returned a pipeline whose parameters were still on
        the `meta` device -- placeholders with no data -- and the failure only
        surfaced at inference as "Cannot copy out of meta tensor; no data!", 20
        minutes later. So the rung's success criterion is materialisation, not
        the absence of an exception.
        """
        offenders, devices = [], {}
        for comp_name in ("transformer", "text_encoder", "text_encoder_2", "vae"):
            comp = getattr(p, comp_name, None)
            if comp is None or not hasattr(comp, "named_parameters"):
                continue
            seen = set()
            for pname, param in comp.named_parameters():
                seen.add(param.device.type)
                if param.device.type == "meta":
                    offenders.append(f"{comp_name}.{pname}")
                    if len(offenders) >= 3:
                        break
            devices[comp_name] = sorted(seen)
        return offenders, devices

    # --strategy forces ONE rung. The ladder runs every rung in the SAME process,
    # so a failed device_map attempt leaves allocations behind and the next rung
    # OOMs on its scraps -- sequential_offload died asking for 34 MiB with 2.81
    # MiB free, which says nothing about sequential_offload. Any rung being
    # judged on its merits has to run first, alone.
    _only = "__STRATEGY__"
    pipe = None
    attempts = []
    for name, qcfg, kwargs, post in (
        # Give accelerate an explicit budget: without max_memory it declined to
        # place the vision tower at all and left it on meta.
        ("balanced_maxmem", qc_both, {"device_map": "balanced", "max_memory": MAXMEM}, None),
        # If the quantizer is what skips the vision tower, leave the text encoder
        # unquantised and let balanced split it across the two cards instead.
        ("balanced_tx_only", qc_tx, {"device_map": "balanced", "max_memory": MAXMEM}, None),
        ("balanced_plain", qc_both, {"device_map": "balanced"}, None),
        ("sequential_offload", qc_both, {}, "enable_sequential_cpu_offload"),
        ("model_offload", qc_both, {}, "enable_model_cpu_offload"),
    ):
        if _only not in ("", "all", name):
            continue
        try:
            pipe = QwenImageLayeredPipeline.from_pretrained(
                model_path, quantization_config=qcfg, torch_dtype=torch.bfloat16, **kwargs)
            if post:
                getattr(pipe, post)()
            offenders, devices = meta_check(pipe)
            if offenders:
                raise RuntimeError(
                    f"loaded but NOT materialised -- params still on meta: {offenders}")
            attempts.append({"strategy": name, "ok": True, "devices": devices})
            REPORT["load_strategy"] = name
            REPORT["component_devices"] = devices
            print("loaded via", name, devices, flush=True)
            break
        except Exception as le:
            attempts.append({"strategy": name, "ok": False, "error": str(le)[:400]})
            print("load strategy", name, "failed:", str(le)[:200], flush=True)
            pipe = None
            try:
                import gc
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass
    REPORT["load_attempts"] = attempts
    if pipe is None:
        bail("load", "every load strategy failed; see load_attempts")
    pipe.set_progress_bar_config(disable=None)
except Exception as e:
    bail("load", e)

try:
    REPORT["stage"] = "infer"
    from PIL import Image
    # DISCOVER the input; do not assume /kaggle/input/<slug>/. Kaggle 2.x
    # namespaces some inputs -- the model mounts under /kaggle/input/models/<owner>/...
    # -- and a wrong path assumption here surfaces as "your dataset is empty",
    # which sends you auditing the upload instead of the path. Depth-bounded on
    # purpose: a recursive walk would enumerate the 115 GB model tree.
    # Measured layout: datasets mount at /kaggle/input/datasets/<owner>/<slug>/<file>
    # and models at /kaggle/input/models/<owner>/<model>/<fw>/<inst>/<ver>/.
    # Two earlier runs died here because the depth was guessed -- first
    # /kaggle/input/<slug>/, then a 3-deep glob that stopped one level short. So
    # WALK and prune instead of guessing: pruning `models` keeps this off the
    # 115 GB tree, and everything else is small.
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    root = "/kaggle/input"
    src, seen_dirs = [], []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d.lower() != "models" and "qwen-image-layered" not in d.lower()]
        seen_dirs.append(dirpath)
        for f in filenames:
            if f.lower().endswith(exts):
                src.append(os.path.join(dirpath, f))
    src.sort(key=len)
    if not src:
        # Name the evidence: an error that does not say what it saw is a dead end.
        bail("infer", "no input image under /kaggle/input; walked: "
                      + json.dumps(sorted(seen_dirs)[:40]))
    image = Image.open(src[0]).convert("RGBA")
    REPORT["input_file"] = os.path.basename(src[0])
    REPORT["input_size"] = list(image.size)

    inputs = {
        "image": image,
        "generator": torch.Generator(device="cuda").manual_seed(__SEED__),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": __STEPS__,
        "num_images_per_prompt": 1,
        "layers": __LAYERS__,
        "resolution": __RES__,
        "cfg_normalize": True,
        "use_en_prompt": True,
    }
    with torch.inference_mode():
        out = pipe(**inputs)
    layers = out.images[0]
    REPORT["layers_returned"] = len(layers)
except Exception as e:
    bail("infer", e)

try:
    REPORT["stage"] = "save"
    os.makedirs("/kaggle/working/layers", exist_ok=True)
    saved = []
    for i, im in enumerate(layers):
        if im.mode != "RGBA":
            im = im.convert("RGBA")
        p = f"/kaggle/working/layers/layer_{i:02d}.png"
        im.save(p)
        saved.append(os.path.basename(p))
    REPORT["saved"] = saved

    try:
        from psd_tools import PSDImage
        psd = PSDImage.frompil(layers[0])
        psd.save("/kaggle/working/output.psd")
        REPORT["psd"] = True
    except Exception as pe:
        # Not fatal: PNG layers + the generated SVG already cover GIMP/Inkscape.
        REPORT["psd"] = False
        REPORT["psd_error"] = str(pe)[:300]

    REPORT.update(stage="done", ok=True)
    with open("/kaggle/working/report.json", "w") as fh:
        json.dump(REPORT, fh, indent=2)
    print("DONE", json.dumps(REPORT)[:500], flush=True)
except Exception as e:
    bail("save", e)
'''


def build_kernel(workdir: Path, slug: str, input_ds: str, layers: int,
                 steps: int, res: int, seed: int, strategy: str = "all") -> None:
    script = (KERNEL_SCRIPT
              .replace("__INPUT_DS__", input_ds)
              .replace("__LAYERS__", str(layers))
              .replace("__STEPS__", str(steps))
              .replace("__RES__", str(res))
              .replace("__STRATEGY__", strategy)
              .replace("__SEED__", str(seed)))
    (workdir / "main.py").write_text(script, encoding="utf-8")
    meta = {
        "id": slug,
        "title": slug.split("/")[-1],
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        # The whole point: demand dual T4 rather than accept a P100.
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "model_sources": [MODEL_SOURCE],
        "kernel_sources": [],
        "competition_sources": [],
    }
    (workdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


# -------------------------------------------------------------------- artifacts

def svg_for(layer_files: list[str], size: tuple[int, int], embed: bool,
            outdir: Path) -> str:
    import base64
    w, h = size
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
        f'xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd" '
        f'width="{w}" height="{h}" viewBox="0 0 {w} {h}" version="1.1">',
        '  <sodipodi:namedview inkscape:document-units="px" />',
        '  <title>Qwen-Image-Layered decomposition</title>',
    ]
    # Back-to-front: layer 0 is background, so document order == stacking order.
    for i, name in enumerate(layer_files):
        if embed:
            b64 = base64.b64encode((outdir / "layers" / name).read_bytes()).decode()
            href = f"data:image/png;base64,{b64}"
        else:
            href = f"layers/{name}"
        parts += [
            f'  <g inkscape:groupmode="layer" id="layer{i}" '
            f'inkscape:label="{i:02d} {Path(name).stem}" style="display:inline">',
            f'    <image xlink:href="{href}" x="0" y="0" width="{w}" height="{h}" '
            f'preserveAspectRatio="none" image-rendering="optimizeQuality" />',
            '  </g>',
        ]
    parts.append('</svg>')
    return "\n".join(parts)


def verify(outdir: Path, original: Path, max_mae: float) -> dict:
    """Gate on an identity: recompositing the layers must reproduce the input."""
    from PIL import Image, ImageChops  # noqa: PLC0415

    layer_dir = outdir / "layers"
    files = sorted(p.name for p in layer_dir.glob("layer_*.png"))
    if not files:
        die("no layer PNGs came back", EXIT_NO,
            "read report.json in the output for the stage that failed")

    bad = [f for f in files if (layer_dir / f).read_bytes()[:8] != PNG_MAGIC]
    if bad:
        die(f"not valid PNGs: {bad}", EXIT_UNKNOWN)

    imgs = [Image.open(layer_dir / f).convert("RGBA") for f in files]
    no_alpha = [f for f, im in zip(files, imgs) if im.getchannel("A").getextrema() == (255, 255)]

    canvas = Image.new("RGBA", imgs[0].size, (0, 0, 0, 0))
    for im in imgs:
        canvas = Image.alpha_composite(canvas, im)

    src = Image.open(original).convert("RGBA").resize(canvas.size, Image.LANCZOS)
    diff = ImageChops.difference(canvas.convert("RGB"), src.convert("RGB"))
    hist = diff.convert("L").histogram()
    total = sum(hist) or 1
    mae = sum(i * n for i, n in enumerate(hist)) / total

    return {
        "layers": len(files),
        "files": files,
        "size": list(canvas.size),
        "layers_without_alpha": no_alpha,
        "recompose_mae": round(mae, 3),
        "recompose_ok": mae <= max_mae,
        "max_recompose_mae": max_mae,
    }


# ------------------------------------------------------------------------ main

def run(args) -> int:
    src = Path(args.image).expanduser().resolve()
    if not src.is_file():
        die(f"no such image: {src}", EXIT_USAGE)
    head = src.read_bytes()[:12]
    if not (head.startswith(PNG_MAGIC) or head[:3] == b"\xff\xd8\xff"
            or head[:4] == b"RIFF" or head[:2] == b"BM"):
        die(f"{src.name} is not a PNG/JPEG/WebP/BMP (magic {head[:4]!r})", EXIT_USAGE)

    from PIL import Image  # noqa: PLC0415
    with Image.open(src) as im:
        in_size = im.size
    log(f"input {src.name} {in_size[0]}x{in_size[1]}")

    user = kaggle_username()
    stamp = args.tag or time.strftime("%Y%m%d-%H%M%S")
    ds_slug = f"{user}/qwl-in-{stamp}"
    k_slug = args.slug or f"{user}/qwl-{stamp}"
    outdir = Path(args.out).expanduser().resolve() if args.out else Path.cwd() / f"qwl-{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    if args.fetch:
        k_slug = args.fetch
    else:
        if args.dry_run:
            return emit(True, f"dry run: would push {k_slug} with model {MODEL_SOURCE} "
                              f"(dual T4, {args.layers} layers, {args.res}px)",
                        {"kernel": k_slug, "dataset": ds_slug, "out": str(outdir)})

        with tempfile.TemporaryDirectory() as td:
            dsdir = Path(td) / "ds"
            dsdir.mkdir()
            shutil.copy2(src, dsdir / src.name)
            (dsdir / "dataset-metadata.json").write_text(json.dumps({
                "title": f"qwl-in-{stamp}", "id": ds_slug,
                "licenses": [{"name": "CC0-1.0"}]}, indent=2), encoding="utf-8")
            log(f"uploading input as private dataset {ds_slug}")
            kag("datasets", "create", "-p", str(dsdir), "-q", "--dir-mode", "zip")
            # Kaggle needs a moment before a fresh dataset can be attached.
            time.sleep(args.settle)

            kdir = Path(td) / "k"
            kdir.mkdir()
            build_kernel(kdir, k_slug, ds_slug.split("/")[-1], args.layers,
                         args.steps, args.res, args.seed, args.strategy)
            meta = json.loads((kdir / "kernel-metadata.json").read_text())
            meta["dataset_sources"] = [ds_slug]
            (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
            log(f"pushing {k_slug} (machine_shape=NvidiaTeslaT4)")
            kag("kernels", "push", "-p", str(kdir))

        if args.no_wait:
            return emit(True, f"pushed {k_slug}; collect later",
                        {"kernel": k_slug},
                        f"qwen_layers.py {src} --fetch {k_slug} --out {outdir}")

    deadline = time.time() + args.timeout
    last = ""
    while time.time() < deadline:
        out = kag("kernels", "status", k_slug, check=False, timeout=180)
        if "COMPLETE" in out:
            log("kernel COMPLETE")
            break
        if "ERROR" in out or "CANCEL" in out:
            die(f"kernel failed: {out.strip()[:300]}", EXIT_NO,
                f"kaggle kernels output {k_slug} -p . && cat report.json")
        if out.strip() != last:
            last = out.strip()
            log(last[:120])
        time.sleep(args.poll)
    else:
        die(f"kernel still running after {args.timeout}s", EXIT_TIMEOUT,
            f"qwen_layers.py {src} --fetch {k_slug} --out {outdir}")

    log("downloading output")
    kag("kernels", "output", k_slug, "-p", str(outdir), "--quiet")

    report_path = outdir / "report.json"
    kreport = json.loads(report_path.read_text()) if report_path.exists() else {}
    if kreport and not kreport.get("ok"):
        die(f"kernel reported failure at stage '{kreport.get('stage')}': "
            f"{kreport.get('error', '')[:300]}", EXIT_NO)

    v = verify(outdir, src, args.max_recompose_mae)
    svg = svg_for(v["files"], tuple(v["size"]), args.embed, outdir)
    (outdir / "layers.svg").write_text(svg, encoding="utf-8")

    psd = outdir / "output.psd"
    v["psd"] = psd.exists() and psd.read_bytes()[:4] == PSD_MAGIC

    data = {"kernel": k_slug, "out": str(outdir), "svg": str(outdir / "layers.svg"),
            "gpu": kreport.get("gpu_names"), "capability": kreport.get("capabilities"),
            **v}

    if not v["recompose_ok"]:
        return emit(False,
                    f"layers returned ({v['layers']}) but they do NOT reconstruct the "
                    f"input: mean abs error {v['recompose_mae']} > {args.max_recompose_mae}. "
                    "Treat this decomposition as unreliable.",
                    data, "re-run with --layers 3 or --res 1024", EXIT_NO)

    return emit(True,
                f"{v['layers']} layers -> {outdir}\n"
                f"  inkscape: {outdir/'layers.svg'}\n"
                f"  gimp:     {'output.psd' if v['psd'] else 'open layers.svg or the PNGs'}\n"
                f"  recompose MAE {v['recompose_mae']} (<= {args.max_recompose_mae})",
                data)


def main(argv: list[str] | None = None) -> int:
    global _JSON
    p = argparse.ArgumentParser(
        prog="qwen_layers.py",
        description="Decompose a raster image into editable RGBA layers on Kaggle's "
                    "free dual-T4 GPUs, and emit GIMP/Inkscape-ready artifacts.")
    p.add_argument("image", help="path to the raster image (relative or absolute)")
    p.add_argument("-o", "--out", help="output directory (default ./qwl-<stamp>)")
    p.add_argument("-n", "--layers", type=int, default=4, help="layer count (3-10)")
    p.add_argument("--res", type=int, default=640, choices=[640, 1024],
                   help="model resolution bucket; 640 is upstream's recommendation")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--embed", action="store_true",
                   help="inline PNGs as data URIs so layers.svg is self-contained")
    p.add_argument("--max-recompose-mae", type=float, default=12.0,
                   help="fail if recompositing the layers differs from the input by more")
    p.add_argument("--slug", help="kernel slug (default <user>/qwl-<stamp>)")
    p.add_argument("--tag", help="stamp to use instead of the timestamp")
    p.add_argument("--fetch", help="skip the push; collect an existing kernel by slug")
    p.add_argument("--no-wait", action="store_true", help="push and exit")
    p.add_argument("--timeout", type=int, default=5400)
    p.add_argument("--poll", type=int, default=20)
    p.add_argument("--settle", type=int, default=25,
                   help="seconds to let a fresh dataset become attachable")
    p.add_argument("--strategy", default="all",
                   help="force ONE load strategy (balanced_maxmem, balanced_tx_only, "
                        "balanced_plain, sequential_offload, model_offload). The ladder\n"
                        "shares one process, so later rungs inherit earlier rungs' leaked VRAM.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    _JSON = args.json

    if not 2 <= args.layers <= 10:
        die("--layers must be between 2 and 10", EXIT_USAGE)
    return run(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(EXIT_UNKNOWN)
    except subprocess.TimeoutExpired as exc:
        sys.exit(emit(False, f"qwen_layers: subprocess timed out: {exc}", code=EXIT_TIMEOUT))
