# Field notes: running a large model on Kaggle's free dual T4

Everything here was **measured**, on a free Kaggle account, in August 2026. Where
something is unresolved it says so — a guess written as a fact is worse than a
gap, because the next person stops looking.

The worked example is [Qwen-Image-Layered](https://github.com/QwenLM/Qwen-Image-Layered)
(~29B params, Apache-2.0), driven by [`qwen_layers.py`](../qwen_layers.py).

---

## 1. You can now choose the accelerator (and it works)

For a long time the Kaggle API could not express *which* GPU you wanted;
`enable_gpu` gave you whatever was free, usually a P100. Kaggle staff confirmed
it wasn't supported
([discussion 664303](https://www.kaggle.com/discussions/product-feedback/664303)).

**Kaggle CLI 2.x supports it.** In `kernel-metadata.json`:

```json
"machine_shape": "NvidiaTeslaT4"
```

or on the command line, which overrides the metadata:

```sh
kaggle kernels push --accelerator NvidiaTeslaT4
```

| Value | Hardware |
|---|---|
| `NvidiaTeslaT4` | 2× Tesla T4, 14.6 GB each (compute capability **7.5**) |
| `NvidiaTeslaP100` | 1× Tesla P100, 16 GB (capability 6.0) |
| `Tpu1VmV38` | TPU v3-8 |

Confirmed from inside a kernel: `2 x Tesla T4, cap 7.5, 14.6 GB each`.

> **The discussion thread was never updated**, so it still reads as unresolved.
> If you search for this, you will find the "not supported" answer first.

**Check what you actually got before doing anything expensive.** Requesting a
shape is a request, not a guarantee, and discovering you're on a P100 twenty
minutes into a model load is avoidable:

```python
import torch
print(torch.cuda.get_device_capability(0))   # (7, 5) = Turing/T4
```

Quota: `kaggle quota` (2.x only) — 30 h GPU/week, 20 h TPU, resets weekly.

## 2. Kaggle CLI 2.x needs Python ≥ 3.11

Ubuntu 22.04 ships 3.10, so `pip install --upgrade kaggle` **silently keeps
1.7.x and reports success** — pip is correctly installing the newest *compatible*
version, and nothing tells you a newer generation exists.

```sh
pip3 install --user uv
uv tool install --python 3.12 kaggle
kaggle --version          # Kaggle CLI 2.2.4 or newer
```

2.x authenticates with the same `KAGGLE_USERNAME` / `KAGGLE_KEY` as 1.x, and
`kaggle kernels status` still prints `KernelWorkerStatus.COMPLETE`, so scripts
that grep for it keep working. Both verified before upgrading.

## 3. Attach big models — never download them

The Qwen-Image-Layered instance is **115 GB uncompressed**
(`kaggle models get qwen-lm/qwen-image-layered` → `totalUncompressedBytes`).
Widely-quoted figures of ~55 GB are wrong by roughly half.

Attached models mount read-only and never touch your working disk or download
budget. This is the single biggest reason to prefer Kaggle over Colab's free
tier for large weights.

`model_sources` requires the **five-part** form — a two-part `owner/model` ref is
rejected at push:

```json
"model_sources": ["qwen-lm/qwen-image-layered/transformers/qwen-image-layered/1"]
```

`{owner}/{model-slug}/{framework}/{instance-slug}/{version}`. Get the real values
from `kaggle models get <owner>/<model>` — the `instances[]` array has
`framework`, `slug` and `versionNumber`.

## 4. Mount layout — discover it, don't assume it

Two runs died guessing this. **Measured:**

```
/kaggle/input/datasets/<owner>/<dataset-slug>/<your files>
/kaggle/input/models/<owner>/<model>/<framework>/<instance>/<version>/
```

Note datasets are **four levels deep**, not `/kaggle/input/<slug>/`. Walk and
prune rather than hardcoding a depth — and prune `models`, or you will enumerate
a 115 GB tree:

```python
import os
exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
found = []
for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
    dirnames[:] = [d for d in dirnames if d.lower() != "models"]
    found += [os.path.join(dirpath, f) for f in filenames if f.lower().endswith(exts)]
```

**Make the failure name what it saw.** `"no input image found"` sent us auditing
a perfectly good upload; `"walked: [...]"` printed the real path and ended the
question in one run.

## 5. The dependency matrix (August 2026 image)

Kaggle's image ships **transformers 5.0.0** and **diffusers 0.37.1**. For a
checkpoint published in December 2025 that combination does not work, and the
compatible set is narrow:

| package | version | why |
|---|---|---|
| `diffusers` | **0.37.1** | already on the image, and already contains `QwenImageLayeredPipeline` |
| `transformers` | **≥4.51.3, <5** | the model card's `>=4.51.3` predates 5.0 and is *satisfied by* the version that breaks it |
| `huggingface-hub` | **≥0.34.0, <1.0** | transformers 4.x refuses hub 1.x |

**Do not install diffusers from git.** It requires `huggingface-hub` 1.x
(`get_cached_repo_tree`), hub 1.x breaks transformers 4.x, and you end up in a
three-way deadlock — one that dissolves entirely once you check whether the
*released* version already has what you need. It did; the file exists in tag
`v0.37.1`. Two `curl`s would have saved four runs.

Install the loose packages first and the constrained set **last and together**,
so nothing installed afterwards can bump the hub back:

```python
pip("python-pptx", "bitsandbytes", "accelerate")
pip("diffusers==0.37.1", "transformers>=4.51.3,<5", "huggingface-hub>=0.34.0,<1.0")
```

**Both failing `pip` calls returned rc=0.** The breakage lived only in the
resolver's warning text, so capture pip's output — checking the exit code is not
enough. And never `os.system("pip -q install ... | tail -3")`: that discards the
output *and* the return code, and a silently-failing install is indistinguishable
from a working one.

**Verify by importing the class you need, in a fresh interpreter.** Diffusers
imports pipelines lazily, so `import diffusers` succeeding tells you nothing:

```sh
python -c "from diffusers import QwenImageLayeredPipeline"
```

## 6. Memory: what fits on 2× T4, and what doesn't — UNRESOLVED

Honest status: **we have not got this model to run on free dual T4.**

Measured:

* At 4-bit NF4 the model fills **14.56 GB on a single card** during load, so
  every single-device path (`enable_model_cpu_offload`,
  `enable_sequential_cpu_offload`) OOMs. Confirmed in a *clean* process, not as
  a side effect of earlier attempts — the error was byte-identical either way.
* Both cards are only reachable via `device_map="balanced"`. In diffusers the
  accepted values are **`balanced`, `cuda`, `cpu`** — `"auto"` is transformers'
  vocabulary and raises.
* With any `device_map`, Qwen2.5-VL's **vision tower stays on the `meta`
  device** (`text_encoder.model.visual.*`), while everything else places
  correctly. Nothing raises at load; it fails ~20 minutes later at inference as
  `Cannot copy out of meta tensor; no data!`.

The vision-tower weights **are** present in the mirror (390 of the text
encoder's 729 keys). The failure is identical across memory budgets,
quantisation targets, and transformers 4.x *and* 5.x — so it is not a memory or
version problem, and we stopped attributing it to one.

> **A failure that is identical across independent configurations is a fact
> about your instrument, not about the configurations.** Noticing that is what
> stopped four more runs of tuning `max_memory`.

**Guard against it cheaply** — a load that doesn't raise is not a load that
works:

```python
bad = [n for n, p in pipe.text_encoder.named_parameters() if p.device.type == "meta"]
if bad:
    raise RuntimeError(f"loaded but not materialised: {bad[:3]}")
```

If you need this model working today, use a single card with ≥24 GB (A10G,
L4, A100). ~29B params at NF4 is simply a poor fit for 14.6 GB slices.

## 7. Verify the output with an identity, not a vibe

A decomposition can return the right *number* of plausible layers that don't
reconstruct the input. Alpha-composite them back and compare:

```python
canvas = Image.new("RGBA", size, (0, 0, 0, 0))
for layer in layers:
    canvas = Image.alpha_composite(canvas, layer)
# mean abs error vs the original; large => the decomposition is unreliable
```

Test the verifier in **both** directions before trusting it. Ours was checked
against a correct decomposition (MAE 0.000) *and* against layers with the right
file count and wrong content (MAE 94.5 → correctly rejected). A check that has
never failed is not known to work.

## 8. Getting layers into Inkscape

Upstream exports PNG / PSD / PPTX and there is no SVG exporter, which gets
reported as "Inkscape isn't supported". That's a gap in the exporter, not in
Inkscape — wrap each RGBA layer in an `inkscape:groupmode="layer"` group and it
opens with the stack intact:

```xml
<g inkscape:groupmode="layer" inkscape:label="00 background">
  <image xlink:href="layers/layer_00.png" x="0" y="0" width="W" height="H"/>
</g>
```

Document order is stacking order, and the model emits layers back-to-front, so
write them in order. The layers stay **raster** — this model produces pixels,
not paths. For vectors you still need Trace Bitmap, but tracing a single flat
layer works far better than tracing a flattened composite.

## 9. Running long jobs from the controller

The controller is a 1 GB micro-VM; the GPU work happens on Kaggle. Two things
will bite you:

* **A foreground job over SSH dies with the connection** — exit 255 is *ssh's*
  error, not the job's. Detach it:

  ```sh
  nohup setsid sh -c 'python3 qwen_layers.py img.png > run.log 2>&1; echo $? > run.done' \
      < /dev/null > /dev/null 2>&1 &
  ```

  then poll for `run.done`.

* **Do not poll with `pgrep -f <script>` over SSH.** The probe's own command
  line contains the pattern, so it matches *itself* and reports a finished job
  as still running. Use the done-file.

Kernels run on Kaggle's servers, so once `kernels push` succeeds the job
survives losing the controller entirely. Give each run a fixed `--tag` and you
can reattach with `--fetch <owner>/<slug>` instead of hunting a timestamp.
