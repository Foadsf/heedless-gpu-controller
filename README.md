# The Heedless GPU Command Center

> **Run heavy AI training jobs on high-end Cloud GPUs (Tesla T4/P100) for free, controlled from a persistent free-tier micro-VM.**

This repository documents the architecture and setup for a "Heedless" Command Center using **Oracle Cloud Infrastructure (OCI)** and **Kaggle**.

## The Concept

We want a persistent, always-on cloud environment to prototype AI models, but:
1.  **Always-Free Cloud VMs** usually have no GPUs or weak CPUs.
2.  **Free GPU Notebooks** (Colab/Kaggle) are ephemeral (sessions die, data is wiped).

**The Solution:**
We build a "Command Center" on a free OCI Micro-VM. It acts as a permanent remote control that dispatches heavy training jobs to Kaggle's powerful GPUs via CLI.

## Prerequisites

1.  **Oracle Cloud Account:** [Sign up for Free Tier](https://www.oracle.com/cloud/free/).
2.  **Kaggle Account:** [Sign up here](https://www.kaggle.com).
3.  **Mobile Phone:** Required for verifying your Kaggle account (crucial for unlocking GPU access).

---

## Phase 1: Creating the OCI "Command Center" VM

The goal is to snag an "Always Free" VM. The Ampere A1 (ARM) shape is far better on
paper — but see the capacity note below before you plan around it. We use the
**AMD Micro** instance as the reliable fallback.

### 0. Check capacity *before* you build anything

The single most frustrating part of this setup is provisioning a shape that turns
out to be unavailable. OCI can answer that question directly instead of failing at
the end of the launch wizard:

```
Compute -> Instances -> Create instance -> pick shape
```

…or, without the console, via the Compute Capacity Report API
(`POST /20160918/computeCapacityReports`), which returns one of `AVAILABLE` /
`OUT_OF_HOST_CAPACITY` / `HARDWARE_NOT_SUPPORTED` per shape.

**Measured reality (August 2026, `eu-amsterdam-1`):**

| Shape | Free tier | Status |
|---|---|---|
| `VM.Standard.A1.Flex` (Ampere ARM) | yes | `OUT_OF_HOST_CAPACITY` at **every** size tried — 4/24, 2/12, 1/6, even the 1 OCPU / 1 GB minimum |
| `VM.Standard.E2.1.Micro` (AMD) | yes | `AVAILABLE` |

So A1 being refused is **not** a sizing problem you can shrink your way around —
when a region is out of Ampere, it is out at every size. Note also that Oracle
**reduced** the Always Free A1 allowance to **2 OCPUs / 12 GB** (it was 4/24), and
that some regions have only one availability domain, so there is no second AD to
retry in.

### 1. Networking (The "Public Subnet" Fix)
*The OCI instance creation wizard often glitches and fails to assign a Public IP. We fix this by creating the network first.*

1.  Log in to OCI Console.
2.  Go to **Networking** -> **Virtual Cloud Networks**.
3.  Click **"Start VCN Wizard"**.
4.  Select **"Create VCN with Internet Connectivity"**.
5.  Name it `kaggle-network` and click **Create**.

### 2. Launching the Instance
1.  Go to **Compute** -> **Instances** -> **Create Instance**.
2.  **Name:** `kaggle-controller`
3.  **Image:** Click "Change Image" -> **Canonical Ubuntu**.
    * *Recommendation:* Choose **Canonical Ubuntu 22.04 Minimal**.
    * *Why:* The standard version uses ~500MB RAM. The Minimal version uses ~150MB, leaving more room for your Python scripts on the 1GB RAM Micro instance.
    * *Caveat added later:* Minimal ships **Python 3.10** and **no `rsyslog`**. Both
      matter — see Phase 3 and Troubleshooting.
4.  **Shape:** Click "Change Shape" -> **Specialty and Legacy**.
    * Select **VM.Standard.E2.1.Micro** (Always Free-eligible).
5.  **Networking:**
    * Select "Select existing virtual cloud network".
    * VCN: `kaggle-network`.
    * Subnet: `public subnet-kaggle-network`.
    * **CRITICAL:** Ensure "Assign a public IPv4 address" says **Yes**.
6.  **SSH Keys:**
    * Generate a key on your local machine: `ssh-keygen -t ed25519 -f ~/.ssh/oracle_key`
    * Select "Paste public keys" in OCI and paste the content of your `.pub` file.
7.  Click **Create**.

> **Generate one key per machine.** Do not copy a private key between your laptops.
> Adding a second public key later is easy (Troubleshooting), copying private keys
> around is a habit that ends badly.

---

## Phase 2: Configuration & Future-Proofing

The "Minimal" image saves RAM but lacks critical tools. We must install them manually.

```bash
# 1. Update and install Python/Pip/Git/Snap
sudo apt update
sudo apt install python3-pip unzip git snapd -y

# 2. Install Oracle Cloud Agent
sudo snap install oracle-cloud-agent --classic
sudo snap start oracle-cloud-agent

# 3. Install uv (needed for modern Python tooling -- see Phase 3)
pip3 install --user uv
```

### THE TRAP THAT WILL BITE YOU: `.bashrc` and non-interactive shells

Every guide (including earlier versions of this one) tells you to append your
environment to `~/.bashrc`:

```bash
# DON'T DO THIS -- it only works when you are typing at a prompt
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
echo 'export KAGGLE_USERNAME="..."' >> ~/.bashrc
```

Ubuntu's stock `~/.bashrc` begins with:

```bash
case $- in
    *i*) ;;
      *) return;;      # <-- non-interactive shells STOP HERE
esac
```

Anything appended **after** that guard is invisible to `ssh host ./run_gpu.sh`, to
`cron`, and to `systemd`. The failure is nasty because it is *misleading*: the
Kaggle CLI reports

```
OSError: Could not find kaggle.json. Make sure it's located in /home/ubuntu/.config/kaggle
```

…which sends you hunting for a missing credentials file that was never the problem.
The variables exist; your shell just never read them. Equally, `which kaggle`
returns nothing and you conclude it isn't installed, when it is only *not on PATH
in that kind of shell*.

**The fix** — keep secrets in their own mode-600 file and source it **above** the
guard:

```bash
# ~/.kaggle_env  (chmod 600 -- .bashrc is world-readable at 644!)
export KAGGLE_USERNAME="your_username_here"
export KAGGLE_KEY="your_key_here"
```

```bash
# insert at the TOP of ~/.bashrc, before the `case $- in` guard
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) export PATH="$HOME/.local/bin:$PATH";; esac
[ -f "$HOME/.kaggle_env" ] && . "$HOME/.kaggle_env"
```

Verify with a genuinely non-interactive invocation — this is the test that matters,
because an interactive `ssh` login will pass even while `cron` fails:

```bash
ssh youruser@YOUR_INSTANCE_IP 'kaggle competitions list | head -3'
```

---

## Phase 3: Kaggle CLI & Authentication

Install the `kaggle` package with pip:

```bash
pip3 install --user kaggle
```

### Ubuntu 22.04 needs `uv` for the current CLI

**Kaggle CLI 2.x requires Python >= 3.11**, and Ubuntu 22.04 ships **3.10**. `pip`
handles this by silently installing the newest *compatible* version (1.7.x) — so
`pip install --upgrade kaggle` reports success and leaves you on the old CLI. That
silence is the whole problem: nothing tells you a newer generation exists.

Install it with its own managed Python instead, without touching the system one:

```bash
pip3 install --user uv
uv tool install --python 3.12 kaggle
kaggle --version        # Kaggle CLI 2.2.4 or newer
```

Version 2.x authenticates with the **same** `KAGGLE_USERNAME` / `KAGGLE_KEY`
variables as 1.x, and `kaggle kernels status` still prints
`KernelWorkerStatus.COMPLETE`, so `run_gpu.sh` keeps working unchanged. (Both
verified before upgrading — worth re-checking yourself rather than trusting it.)

### Credentials

1. **Obtain your Credentials:**
   * Go to Kaggle.com -> **Settings** -> **API**.
   * Click **"Create Legacy API Key"**.
   * Open the downloaded `kaggle.json`: `{"username":"your_user","key":"your_hex_key"}`.
2. **Configure** as shown in Phase 2 (`~/.kaggle_env`, mode 600).
3. **Test:** `kaggle competitions list`

> Newer `KGAT…`-prefixed tokens are for Kaggle's **MCP server** (`Bearer` auth),
> not for the `KAGGLE_KEY` variable. They are different credentials for different
> things; do not put one where the other belongs.

---

## Phase 4: The GPU Workflow

### 1. Create a Script

Write your PyTorch/TensorFlow code in a standard `.py` file. See `examples/000_hello_gpu/main.py`.

### 2. Initialize Metadata

Run `kaggle kernels init`. Edit `kernel-metadata.json` to include:

```json
{
  "id": "YOUR_KAGGLE_USERNAME/project-name",
  "title": "GPU Test",
  "code_file": "main.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_internet": "true",
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": [],
  "kernel_sources": [],
  "competition_sources": []
}
```

### Choosing your accelerator (T4 vs P100 vs TPU)

For a long time the API could not express this: `enable_gpu` gave you whatever
Kaggle picked (usually a P100), and setting `machine_shape` was ignored. Kaggle
staff confirmed at the time that *"the Kaggle API does not currently support
accelerators"*
([discussion](https://www.kaggle.com/discussions/product-feedback/664303)).

**As of Kaggle CLI 2.x this works.** The client reads `machine_shape` straight from
`kernel-metadata.json`, with a CLI flag that overrides it:

```bash
kaggle kernels push --accelerator NvidiaTeslaT4
```

Accepted values:

| Value | Hardware |
|---|---|
| `NvidiaTeslaT4` | Tesla T4 (Turing) — needed for anything requiring SM 7.5+ |
| `NvidiaTeslaP100` | Tesla P100 (Pascal) |
| `Tpu1VmV38` | TPU v3-8 |

2.x also adds a weekly accelerator quota command, so you can check your remaining
GPU/TPU hours from the controller.

> **Verify, don't assume.** The historical complaint was that `machine_shape` was
> *accepted and then silently reset* to P100 server-side. Confirm what you actually
> got by printing it from inside the kernel — `torch.cuda.get_device_name(0)` — and
> trust that over the request you sent.

### 3. Execution (One-Command)

```bash
chmod +x run_gpu.sh
./run_gpu.sh YOUR_USERNAME/project-name
```

This will automatically upload code, wait for the GPU, and stream logs back to your terminal.

---

## Optional: Google Colab as a second GPU backend

Colab now has an official CLI, which suits this architecture well — a headless
controller dispatching work to someone else's GPU:

```bash
uv tool install --python 3.12 google-colab-cli   # provides `colab`
colab new / exec / install / ls / download / log
```

It authenticates via OAuth, so the first run needs a browser once. There is also an
official [`colab-mcp`](https://github.com/googlecolab/colab-mcp) server for agent
use — but note it exposes a single tool, `open_colab_browser_connection`, and
bridges to a Colab session **in a local browser**. It is therefore not useful on a
headless controller; run it on your workstation, not the VM.

---

## Troubleshooting

### "Locked Out / Lost SSH Key"

**Do not rely on Run Command — it may not exist on your instance.** Earlier versions
of this guide recommended OCI **Run Command** for key recovery. On an Ubuntu
**Minimal** image with the snap-installed Oracle Cloud Agent, the
`Compute Instance Run Command` plugin is **not in the plugin list at all** — the
console still lets you create a command, which then sits at `Accepted` until it
`Expired`. That is a bad thing to discover on the day you are locked out.

Check first: **Instance -> Management tab -> Oracle Cloud Agent**. If
`Compute Instance Run Command` is absent, use **Bastion** instead:

1. In the same Oracle Cloud Agent list, enable the **Bastion** plugin and wait for
   it to report `Running` (a few minutes).
2. **Identity & Security -> Bastion -> Create bastion**, targeting your VCN and the
   instance's subnet. Restrict the CIDR allowlist to your own IP.
   *(Bastion names are alphanumeric only — no hyphens.)*
3. **Create session -> Managed SSH**, username `ubuntu`, pick the instance, paste
   your **new** public key.
4. Use the **View SSH command** dialog; it gives you an `ssh -o ProxyCommand=…` line.
5. Once inside, append your new key to `~/.ssh/authorized_keys` — the Bastion
   session is time-limited (3 h max), so make the access permanent while you are in.

### "FAILURE: No GPU detected"

1. Go to Kaggle Settings -> **Phone Verification**.
2. Verify your number.
3. Manually switch the Accelerator to "GPU T4" in a web notebook once to "unlock" the feature.

### Reading logs on a Minimal image

Ubuntu Minimal ships **no `rsyslog`**, so `/var/log/auth.log` **does not exist**.
Grepping it for failed logins returns nothing, which reads as "no attacks" when in
fact nothing was measured. Use the journal:

```bash
sudo journalctl -u ssh -g "Accepted|Failed password|Invalid user" --no-pager
```

On a box with port 22 open to the internet, expect a constant stream of
`Invalid user` attempts — that is normal background noise. What matters is whether
any line says `Accepted` from an address that is not yours.
