import torch


def determine_device(gpu_flag_enabled):
    """Determines device given gpu flag and the availability of cuda"""
    return torch.device(
        "cuda" if torch.cuda.is_available() and gpu_flag_enabled else "cpu"
    )
