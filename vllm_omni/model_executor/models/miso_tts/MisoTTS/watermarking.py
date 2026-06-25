import argparse

import torch
import torchaudio

# Warning: When using MisoTTS in another application, you must set this key
# and keep the watermark key secret.
MISO_TTS_WATERMARK = [0, 0, 0, 0, 0]


def cli_check_audio() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, required=True)
    args = parser.parse_args()

    check_audio_from_file(args.audio_path)


def load_watermarker(device: str = "cuda"):
    """Stub function - watermarking disabled."""
    return None


@torch.inference_mode()
def watermark(
    watermarker,
    audio_array: torch.Tensor,
    sample_rate: int,
    watermark_key: list[int],
) -> tuple[torch.Tensor, int]:
    """Stub function - returns audio unchanged without watermarking."""
    return audio_array, sample_rate


@torch.inference_mode()
def verify(
    watermarker,
    watermarked_audio: torch.Tensor,
    sample_rate: int,
    watermark_key: list[int],
) -> bool:
    """Stub function - always returns False."""
    return False


def check_audio_from_file(audio_path: str) -> None:
    audio_array, sample_rate = load_audio(audio_path)
    print(f"Watermarking disabled: {audio_path}")


def load_audio(audio_path: str) -> tuple[torch.Tensor, int]:
    audio_array, sample_rate = torchaudio.load(audio_path)
    audio_array = audio_array.mean(dim=0)
    return audio_array, int(sample_rate)


if __name__ == "__main__":
    cli_check_audio()
