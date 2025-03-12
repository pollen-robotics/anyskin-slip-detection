"""Converts a *.pt to a *.safetensors."""

import torch
from safetensors.torch import save_file


def convert_pt_to_safetensors(pt_path: str, safetensors_path: str) -> None:
    """Convert a PyTorch model checkpoint from *.pt format to *.safetensors format.

    Args:
        pt_path (str): Path to the input PyTorch model checkpoint file.
        safetensors_path (str): Path to save the converted safetensors model file.

    Returns:
        None
    """
    model_state_dict = torch.load(pt_path)

    save_file(model_state_dict, safetensors_path)
    model_state_dict = torch.load(pt_path)

    save_file(model_state_dict, safetensors_path)


if __name__ == "__main__":
    pt_path = "checkpoints/best_model.pt"
    safetensors_path = "checkpoints/best_model.safetensors"

    convert_pt_to_safetensors(pt_path, safetensors_path)
    print(f"Model saved to {safetensors_path}")
