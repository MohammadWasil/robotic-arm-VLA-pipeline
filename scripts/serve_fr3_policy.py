from pathlib import Path
import subprocess
import sys
import tyro
from huggingface_hub import snapshot_download


def main(
    repo_id: str = "wasil11/pi0-franka-fr3-pick-and-place-lora",
    step: int = 9999,
    config_name: str = "pi0_fr3_unity_low_mem_finetune",
    local_checkpoint_root: Path = Path("checkpoints/pi0_fr3_unity_low_mem_finetune/fr3_v1"),
    port: int = 8000,
) -> None:
    """
    Downloads the fine-tuned Pi-0 checkpoint from Hugging Face and launches openpi's WebSocket policy server.

    This script is just a thn wrapper around openpi's `serve_policy.py` script, which is the actual policy server. 
    It downloads the checkpoint from Hugging Face if it doesn't exist locally, and Launches openpi's WebSocket 
    inference server on port 8000, loading the finetuned model and config. This way, any WebSocket client can send 
    images and state and receive 50-step action chunks.

    Args:
        repo_id: The Hugging Face repository ID where the fine-tuned Pi-0 model is stored.
        step: The specific training step of the checkpoint to download and serve.
        config_name: The configuration name for the policy server that we defined in openpi's config file.
        local_checkpoint_root: The local directory where the checkpoint will be cached.
        port: The port on which to run the policy server.
    """
    step_dir = local_checkpoint_root / str(step)
    if not (step_dir / "params").exists():
        print(f"Download step {step} from https://huggingface.co{repo_id} ...")
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=[f"{step}/*"],
            local_dir=str(local_checkpoint_root),
        )
    else:
        print(f"Found cached checkpoint at {step_dir}")

    cmd = [
        sys.executable,
        "third_party/openpi/scripts/serve_policy.py",
        "--port",
        str(port),
        "policy:checkpoint",
        f"--policy.config={config_name}",
        f"--policy.dir={step_dir}",
    ]
    print(f"Starting pi-0 policy server on port {port}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    tyro.cli(main)
