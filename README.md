# The Heedless GPU Command Center

> **Run heavy AI training jobs on high-end Cloud GPUs (Tesla P100/T4) for free, controlled from a persistent free-tier micro-VM.**

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

The goal is to snag an "Always Free" VM. While the Ampere A1 (ARM) instances are best, they are often out of stock. We use the **AMD Micro** instance as a reliable fallback.

### 1. Networking (The "Public Subnet" Fix)
*The OCI instance creation wizard often glitches and fails to assign a Public IP. We fix this by creating the network first.*

1.  Log in to OCI Console.
2.  Go to **Networking** -> **Virtual Cloud Networks**.
3.  Click **"Start VCN Wizard"**.
4.  Select **"Create VCN with Internet Connectivity"**.
5.  Name it `kaggle-network` and click **Create**.
    * *This ensures you have a Public Subnet and an Internet Gateway ready.*

### 2. Launching the Instance
1.  Go to **Compute** -> **Instances** -> **Create Instance**.
2.  **Name:** `kaggle-controller`
3.  **Image:** Click "Change Image" -> **Canonical Ubuntu**.
    * *Recommendation:* Choose **Canonical Ubuntu 22.04 Minimal**.
    * *Why:* The standard version uses ~500MB RAM. The Minimal version uses ~150MB, leaving more room for your Python scripts on the 1GB RAM Micro instance.
4.  **Shape:** Click "Change Shape" -> **Specialty and Legacy**.
    * Select **VM.Standard.E2.1.Micro** (Always Free-eligible).
    * *Specs:* 1 OCPU, 1 GB Memory.
5.  **Networking:**
    * Select "Select existing virtual cloud network".
    * VCN: `kaggle-network`.
    * Subnet: `public subnet-kaggle-network`.
    * **CRITICAL:** Ensure "Assign a public IPv4 address" says **Yes**.
6.  **SSH Keys:**
    * Generate a key on your local machine (PowerShell): `ssh-keygen -t rsa -b 4096`
    * Select "Paste public keys" in OCI and paste the content of your `.pub` file.
7.  Click **Create**.

### 3. Connection
Once the instance status is **Green (Running)**, grab the Public IP and connect:
```bash
ssh -i /path/to/private/key ubuntu@YOUR_PUBLIC_IP

```

---

## Phase 2: Configuration & Future-Proofing

The "Minimal" image saves RAM but lacks critical tools. We must install them manually.

**CRITICAL:** We also install the **Oracle Cloud Agent** and enable permissions. This safeguards you against losing your SSH key by allowing you to inject new keys via the OCI Web Console ("Run Command" feature).

```bash
# 1. Update and install Python/Pip/Git/Snap
sudo apt update
sudo apt install python3-pip unzip git snapd -y

# 2. Install Oracle Cloud Agent (Crucial for recovery)
sudo snap install oracle-cloud-agent --classic
sudo snap start oracle-cloud-agent

# 3. Allow Agent to run sudo (Required for OCI "Run Command")
echo "ocarun ALL=(ALL) NOPASSWD:ALL" | sudo tee /etc/sudoers.d/101-oracle-cloud-agent-run-command
sudo chmod 440 /etc/sudoers.d/101-oracle-cloud-agent-run-command

# 4. Install Kaggle CLI
pip3 install kaggle

# 5. Add local bin to PATH (so you can type 'kaggle' instead of the full path)
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

```

---

## Phase 3: Kaggle Authentication

To control the GPUs, we authenticate using environment variables. We use the **Legacy API Key** method as it is most reliable for the CLI.

1. **Obtain your Credentials:**
* Go to Kaggle.com -> **Settings** -> **API**.
* Click **"Create Legacy API Key"**.
* Open the downloaded `kaggle.json` file. It looks like: `{"username":"your_user","key":"your_hex_key"}`.


2. **Configure Environment:**
* Run these commands to save your credentials permanently to your shell configuration (replace values with the text from your file):


```bash
echo 'export KAGGLE_USERNAME="your_username_here"' >> ~/.bashrc
echo 'export KAGGLE_KEY="your_key_here"' >> ~/.bashrc
source ~/.bashrc

```


3. **Test:**
```bash
kaggle competitions list

```


*If you see a list of competitions, you are connected.*

---

## Phase 4: The GPU Workflow

This is how you run code on the cloud.

### 1. Create a Script

Write your PyTorch/TensorFlow code in a standard `.py` file. See `examples/000_hello_gpu/main.py`.

### 2. Initialize Metadata

Run `kaggle kernels init` to generate `kernel-metadata.json`. You **must** edit this file to enable the GPU.

**Crucial Configuration:**

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
  "dataset_sources": [],
  "kernel_sources": [],
  "competition_sources": []
}

```

### 3. Execution (The "One-Command" Method)

Instead of manually pushing and checking status repeatedly, use the included automation script:

1. **Make it executable:**
```bash
chmod +x run_gpu.sh

```


2. **Run it:**
```bash
./run_gpu.sh YOUR_USERNAME/project-name

```


*This will automatically upload your code, wait for the remote GPU to finish, and stream the logs back to your terminal.*

---

## Troubleshooting

### "Locked Out / Lost SSH Key"

If you lose your private key, do **not** delete the VM immediately. Because we set up the Oracle Cloud Agent in Phase 2:

1. Go to OCI Console -> Instance -> **Resources** -> **Run Command**.
2. Create a command script to append your *new* public key to `~/.ssh/authorized_keys`.
3. The agent will execute this as root/sudo, restoring your access.

### "FAILURE: No GPU detected"

If the logs say CUDA is not available, it is usually because your Kaggle account is not phone verified.

1. Go to Kaggle Settings -> **Phone Verification**.
2. Verify your number.
3. Go to any notebook on the web interface and manually switch the Accelerator to "GPU T4" once to "unlock" the feature.

### "Command not found"

Run `export PATH=$HOME/.local/bin:$PATH` or add it to your `.bashrc`.

