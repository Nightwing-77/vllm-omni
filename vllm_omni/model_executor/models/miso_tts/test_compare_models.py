"""
Test to compare vLLM-Omni Miso TTS implementation with the official Miso TTS.

This test loads both models and compares their outputs stage by stage:
1. Load official Miso TTS from the cloned repo
2. Load vLLM-Omni Miso TTS implementation
3. Run inference on the same test case
4. Compare logits, tokens, and final audio outputs
5. Report similarity metrics

Run with:
    python -m vllm_omni.model_executor.models.miso_tts.test_compare_models
"""
import os
import sys
from typing import Any

import torch
import torch.nn.functional as F

# Add official Miso TTS repo to path
OFFICIAL_REPO_PATH = os.path.join(os.path.dirname(__file__), "MisoTTS")
if OFFICIAL_REPO_PATH not in sys.path:
    sys.path.insert(0, OFFICIAL_REPO_PATH)

# Import official Miso TTS
from models import MISO_TTS_8B_CONFIG, Model as OfficialModel
from generator import Generator, Segment, load_miso_8b

# Import vLLM-Omni Miso TTS
from modeling_miso_tts import MISO_TTS_8B_CONFIG as VLLM_CONFIG, MisoTTSModel as VLLMModel, load_miso_model_weights


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two tensors."""
    if a.shape != b.shape:
        return -1.0
    a_flat = a.flatten()
    b_flat = b.flatten()
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def max_absolute_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute max absolute difference between two tensors."""
    if a.shape != b.shape:
        return float('inf')
    return torch.max(torch.abs(a - b)).item()


def compare_tensors(name: str, a: torch.Tensor, b: torch.Tensor, threshold: float = 0.99) -> None:
    """Compare two tensors and print similarity metrics."""
    print(f"\n{'='*60}")
    print(f"Comparing: {name}")
    print(f"{'='*60}")
    print(f"Shape A: {a.shape}, Shape B: {b.shape}")
    
    if a.shape != b.shape:
        print(f"❌ SHAPE MISMATCH!")
        return
    
    cos_sim = cosine_similarity(a, b)
    max_diff = max_absolute_diff(a, b)
    
    print(f"Cosine Similarity: {cos_sim:.6f}")
    print(f"Max Absolute Diff: {max_diff:.6f}")
    
    if cos_sim >= threshold:
        print(f"✅ PASS (similarity >= {threshold})")
    else:
        print(f"❌ FAIL (similarity < {threshold})")


def test_model_architecture() -> None:
    """Test that both models have the same architecture."""
    print("\n" + "="*60)
    print("TESTING MODEL ARCHITECTURE")
    print("="*60)
    
    # Create models without loading weights
    official_model = OfficialModel(MISO_TTS_8B_CONFIG)
    vllm_model = VLLMModel(VLLM_CONFIG)
    
    # Compare parameter counts
    official_params = sum(p.numel() for p in official_model.parameters())
    vllm_params = sum(p.numel() for p in vllm_model.parameters())
    
    print(f"Official model parameters: {official_params:,}")
    print(f"vLLM model parameters: {vllm_params:,}")
    
    if official_params == vllm_params:
        print("✅ Parameter counts match")
    else:
        print(f"❌ Parameter count mismatch: {abs(official_params - vllm_params):,}")
    
    # Compare layer names
    official_keys = set(official_model.state_dict().keys())
    vllm_keys = set(vllm_model.state_dict().keys())
    
    print(f"\nOfficial model layers: {len(official_keys)}")
    print(f"vLLM model layers: {len(vllm_keys)}")
    
    missing_in_vllm = official_keys - vllm_keys
    extra_in_vllm = vllm_keys - official_keys
    
    if missing_in_vllm:
        print(f"\n❌ Missing in vLLM ({len(missing_in_vllm)}):")
        for k in sorted(missing_in_vllm)[:10]:  # Show first 10
            print(f"  - {k}")
    
    if extra_in_vllm:
        print(f"\n❌ Extra in vLLM ({len(extra_in_vllm)}):")
        for k in sorted(extra_in_vllm)[:10]:  # Show first 10
            print(f"  - {k}")
    
    if not missing_in_vllm and not extra_in_vllm:
        print("✅ Layer names match")


def test_generate_frame() -> None:
    """Test generate_frame output between both models."""
    print("\n" + "="*60)
    print("TESTING GENERATE_FRAME")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    # Load official model
    print("Loading official Miso TTS...")
    official_gen = load_miso_8b(device=device, dtype=dtype)
    official_model = official_gen._model
    
    # Load vLLM model
    print("Loading vLLM-Omni Miso TTS...")
    vllm_model = load_miso_model_weights("MisoLabs/MisoTTS", device, dtype)
    
    # Setup caches
    official_model.setup_caches(1)
    vllm_model.setup_caches(1, dtype)
    
    # Create test input
    batch_size = 1
    seq_len = 10
    num_codebooks = 32
    
    tokens = torch.randint(0, 2051, (batch_size, seq_len, num_codebooks + 1)).to(device)
    tokens_mask = torch.ones(batch_size, seq_len, num_codebooks + 1, dtype=torch.bool).to(device)
    input_pos = torch.arange(seq_len).unsqueeze(0).to(device)
    
    temperature = 0.9
    topk = 50
    
    # Generate frame with official model
    print("Generating frame with official model...")
    with torch.inference_mode():
        official_frame = official_model.generate_frame(tokens, tokens_mask, input_pos, temperature, topk)
    
    # Generate frame with vLLM model
    print("Generating frame with vLLM model...")
    with torch.inference_mode():
        vllm_frame = vllm_model.generate_frame(tokens, tokens_mask, input_pos, temperature, topk)
    
    # Compare outputs
    compare_tensors("Generated Frame", official_frame, vllm_frame, threshold=0.95)
    
    # Test with multiple steps
    print("\n" + "="*60)
    print("TESTING MULTI-STEP GENERATION")
    print("="*60)
    
    official_model.reset_caches()
    vllm_model.reset_caches()
    
    official_frames = []
    vllm_frames = []
    
    curr_tokens = tokens.clone()
    curr_tokens_mask = tokens_mask.clone()
    curr_pos = input_pos.clone()
    
    for step in range(5):
        print(f"Step {step + 1}/5")
        
        official_frame = official_model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
        vllm_frame = vllm_model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
        
        official_frames.append(official_frame.clone())
        vllm_frames.append(vllm_frame.clone())
        
        compare_tensors(f"Frame {step + 1}", official_frame, vllm_frame, threshold=0.95)
        
        # Update state for next step (if not zero frame)
        if not (official_frame == 0).all():
            curr_tokens = torch.cat([official_frame, torch.zeros(1, 1).long().to(device)], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat([torch.ones_like(official_frame).bool(), torch.zeros(1, 1).bool().to(device)], dim=1).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1


def test_full_generation() -> None:
    """Test full generation pipeline with both models."""
    print("\n" + "="*60)
    print("TESTING FULL GENERATION PIPELINE")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    # Load official generator
    print("Loading official Miso TTS Generator...")
    official_gen = load_miso_8b(device=device, dtype=dtype)
    
    # Load vLLM model components
    print("Loading vLLM-Omni Miso TTS components...")
    from miso_tts_talker import _llama3_text_tokenizer, load_mimi_codec
    vllm_model = load_miso_model_weights("MisoLabs/MisoTTS", device, dtype)
    vllm_model.setup_caches(1, dtype)
    vllm_text_tok = _llama3_text_tokenizer()
    vllm_mimi = load_mimi_codec(device, vllm_model.config.audio_num_codebooks)
    
    # Test case
    text = "Hello, this is a test."
    speaker = 0
    max_audio_length_ms = 2000  # Short test
    
    # Generate with official
    print(f"\nGenerating with official model: '{text}'")
    official_audio = official_gen.generate(
        text=text,
        speaker=speaker,
        context=[],
        max_audio_length_ms=max_audio_length_ms,
        temperature=0.9,
        topk=50,
    )
    
    # Generate with vLLM (manual pipeline)
    print(f"Generating with vLLM model: '{text}'")
    vllm_model.reset_caches()
    
    # Build prompt
    fs = vllm_model.config.audio_num_codebooks + 1
    ids = vllm_text_tok.encode(f"[{speaker}] {text.lstrip()}")
    prompt = torch.zeros(len(ids), fs).long().to(device)
    prompt_mask = torch.zeros(len(ids), fs).bool().to(device)
    prompt[:, -1] = torch.tensor(ids)
    prompt_mask[:, -1] = True
    
    curr_tokens = prompt.unsqueeze(0)
    curr_tokens_mask = prompt_mask.unsqueeze(0)
    curr_pos = torch.arange(prompt.size(0)).unsqueeze(0).long().to(device)
    
    max_generation_len = int(max_audio_length_ms / 80)
    samples = []
    
    for i in range(max_generation_len):
        frame = vllm_model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, 0.9, 50)
        if (frame == 0).all():
            print(f"  Stopped at frame {i+1} (zero frame)")
            break
        samples.append(frame)
        
        curr_tokens = torch.cat([frame, torch.zeros(1, 1).long().to(device)], dim=1).unsqueeze(1)
        curr_tokens_mask = torch.cat([torch.ones_like(frame).bool(), torch.zeros(1, 1).bool().to(device)], dim=1).unsqueeze(1)
        curr_pos = curr_pos[:, -1:] + 1
    
    if samples:
        vllm_audio = vllm_mimi.decode(torch.stack(samples).permute(1, 2, 0)).squeeze(0).squeeze(0)
    else:
        vllm_audio = torch.zeros(24000, device=device)
    
    # Compare audio outputs
    print(f"\nOfficial audio length: {len(official_audio)} samples")
    print(f"vLLM audio length: {len(vllm_audio)} samples")
    
    # Resample to same length if needed
    min_len = min(len(official_audio), len(vllm_audio))
    official_audio_trimmed = official_audio[:min_len]
    vllm_audio_trimmed = vllm_audio[:min_len]
    
    compare_tensors("Audio Output", official_audio_trimmed, vllm_audio_trimmed, threshold=0.90)
    
    # Compare frame counts
    print(f"\nOfficial frames generated: {len(samples) if samples else 0}")
    print(f"vLLM frames generated: {len(samples)}")


def main() -> None:
    """Run all comparison tests."""
    print("="*60)
    print("MISO TTS MODEL COMPARISON TEST")
    print("="*60)
    print(f"Official repo path: {OFFICIAL_REPO_PATH}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    try:
        test_model_architecture()
        test_generate_frame()
        test_full_generation()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
