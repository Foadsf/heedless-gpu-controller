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
````

-----

## Phase 2: Configuration

Since we used the "Minimal" image, we need to install the basics.

```bash
# 1. Update and install Python/Pip
sudo apt update
sudo apt install python3-pip unzip -y

# 2. Install Kaggle CLI
pip3 install kaggle

# 3. Add local bin to PATH (so you can type 'kaggle' instead of the full path)
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

-----

## Phase 3: Kaggle Authentication

To control the GPUs, we authenticate using an environment variable.

1.  **Obtain your API Token:**
    * Go to Kaggle.com -> **Settings** -> **API** -> **Create New Token**.
    * Copy the token string provided (e.g., `KGAT_...`). *Note: If a file downloads, you can ignore it; we only need the token string.*

2.  **Configure Environment:**
    * Run this command to save your token permanently to your shell configuration (replace `YOUR_TOKEN_STRING` with your actual token):

    ```bash
    echo 'export KAGGLE_API_TOKEN="YOUR_TOKEN_STRING"' >> ~/.bashrc
    source ~/.bashrc
    ```

3.  **Test:**

    ```bash
    kaggle competitions list
    ```

    *If you see a list of competitions, you are connected.*

-----

## Phase 4: The GPU Workflow

This is how you run code on the cloud.

### 1\. Create a Script

Write your PyTorch/TensorFlow code in a standard `.py` file. See `examples/000_hello_gpu/main.py`.

### 2\. Initialize Metadata

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

### 3\. The "Push" (Execute)

```bash
kaggle kernels push
```

### 4\. Check Status & Logs

```bash
# Check status
kaggle kernels status YOUR_USERNAME/project-name

# Download logs (only works after status is COMPLETE)
kaggle kernels output YOUR_USERNAME/project-name
cat project-name.log
```

-----

## Troubleshooting

### "FAILURE: No GPU detected"

If the logs say CUDA is not available, it is usually because your Kaggle account is not phone verified.

1.  Go to Kaggle Settings -\> **Phone Verification**.
2.  Verify your number.
3.  Go to any notebook on the web interface and manually switch the Accelerator to "GPU T4" once to "unlock" the feature.


### "Command not found"

Run `export PATH=$HOME/.local/bin:$PATH` or add it to your `.bashrc`.
