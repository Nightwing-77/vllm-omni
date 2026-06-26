# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processor for Miso TTS: Talker → Mimi decoder."""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct, to_dict
from vllm_omni.model_executor.stage_input_processors.chunk_size_utils import (
    compute_dynamic_initial_chunk_size,
    max_ic_for_chunk_size,
)

logger = init_logger(__name__)

_MISO_NUM_CODEBOOKS = 32


def _extract_last_frame(pooling_output: dict) -> torch.Tensor | None:
    audio_codes = pooling_output.get("codes", {}).get("audio")
    if isinstance(audio_codes, list):
        tensors = [x for x in audio_codes if isinstance(x, torch.Tensor) and x.numel() > 0]
        if not tensors:
            return None
        frame = tensors[-1]
    elif isinstance(audio_codes, torch.Tensor):
        if audio_codes.numel() == 0:
            return None
        if audio_codes.ndim == 2:
            frame = audio_codes[-1]
        elif audio_codes.ndim == 1:
            frame = audio_codes
        else:
            return None
    else:
        return None
    if frame.numel() == 0 or not bool(frame.any().item()):
        return None
    return frame.to(torch.long).reshape(-1)


def talker_preprocess_input(
    request: Any,
    model_intermediate_buffer: dict[str, dict] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Extract additional_information (text, speaker) into runtime info for talker."""
    logger.info(f"[MisoTalkerPreprocess] Called, request type={type(request)}, dir={dir(request)[:5]}")
    
    if model_intermediate_buffer is None:
        logger.warning("[MisoTalkerPreprocess] model_intermediate_buffer is None")
        return {}
    
    # Try different possible req_id attribute names
    req_id = getattr(request, "req_id", None)
    if req_id is None:
        req_id = getattr(request, "request_id", None)
    if req_id is None:
        req_id = getattr(request, "external_req_id", None)
    
    logger.info(f"[MisoTalkerPreprocess] req_id={req_id}, buffer keys={list(model_intermediate_buffer.keys()) if model_intermediate_buffer else 'None'}")
    
    if req_id is None:
        logger.warning("[MisoTalkerPreprocess] No req_id found on request")
        return {}
    
    info = model_intermediate_buffer.get(req_id, {})
    logger.info(f"[MisoTalkerPreprocess] Extracted info for req_id={req_id}: keys={list(info.keys())}")
    
    # Return the info dict which will be merged into runtime_additional_information
    return info


def _audio_codes_as_frames(audio_codes: object) -> torch.Tensor | None:
    """Normalize talker ``codes.audio`` to ``[num_frames, Q]``."""
    if isinstance(audio_codes, list):
        rows = [x.reshape(-1).to(torch.long) for x in audio_codes if isinstance(x, torch.Tensor) and x.numel() > 0]
        if not rows:
            return None
        if all(r.numel() == _MISO_NUM_CODEBOOKS for r in rows):
            return torch.stack(rows, dim=0)
        if len(rows) == 1 and rows[0].numel() % _MISO_NUM_CODEBOOKS == 0:
            return rows[0].reshape(-1, _MISO_NUM_CODEBOOKS)
        return None
    if not isinstance(audio_codes, torch.Tensor) or audio_codes.numel() == 0:
        return None
    audio_codes = audio_codes.to(torch.long)
    if audio_codes.ndim == 1 and audio_codes.numel() % _MISO_NUM_CODEBOOKS == 0:
        return audio_codes.reshape(-1, _MISO_NUM_CODEBOOKS)
    if audio_codes.ndim == 2:
        return audio_codes
    return None


def talker2mimi(
    source_outputs: list[Any],
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
) -> list[Any]:
    """Non-async: pass full codec sequence to Mimi after talker finishes."""
    from vllm_omni.inputs.data import OmniTokensPrompt

    logger.info(f"[MisoFullPayloadProcess] Called with {len(source_outputs)} source_outputs")
    code2wav_inputs: list[OmniTokensPrompt] = []
    for idx, talker_output in enumerate(source_outputs):
        logger.info(f"[MisoFullPayloadProcess] Processing output {idx}, finished={talker_output.finished}")
        if not talker_output.finished:
            logger.info(f"[MisoFullPayloadProcess] Skipping output {idx} (not finished)")
            continue
        output = talker_output.outputs[0]
        mm = output.multimodal_output
        logger.info(f"[MisoFullPayloadProcess] multimodal_output keys={list(mm.keys()) if mm else 'None'}")
        mm_codes = mm.get("codes", {})
        audio_codes = mm_codes.get("audio")
        logger.info(f"[MisoFullPayloadProcess] audio_codes type={type(audio_codes)}, len={len(audio_codes) if isinstance(audio_codes, list) else 'N/A'}")
        audio_codes = _audio_codes_as_frames(audio_codes)
        if audio_codes is None:
            logger.warning(f"[MisoFullPayloadProcess] audio_codes is None after _audio_codes_as_frames")
            continue
        logger.info(f"[MisoFullPayloadProcess] audio_codes as frames shape={audio_codes.shape}")
        valid_mask = (audio_codes >= 0).all(dim=1) & audio_codes.any(dim=1)
        audio_codes = audio_codes[valid_mask]
        logger.info(f"[MisoFullPayloadProcess] After valid filter shape={audio_codes.shape}")
        flat = audio_codes.transpose(0, 1).reshape(-1).tolist()
        logger.info(f"[MisoFullPayloadProcess] flat codes length={len(flat)}")
        if not flat:
            logger.warning(f"[MisoFullPayloadProcess] flat codes is empty")
            continue
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=flat,
                multi_modal_data=None,
                mm_processor_kwargs=None,
                additional_information=None,
            )
        )
    logger.info(f"[MisoFullPayloadProcess] Returning {len(code2wav_inputs)} inputs")
    return code2wav_inputs


def talker2mimi_async_chunk(
    transfer_manager: Any,
    multimodal_output: dict | None,
    request: Any,
    is_finished: bool = False,
    **_: Any,
) -> OmniPayloadStruct | None:
    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())

    # Check if talker signaled done via multimodal_output
    if isinstance(multimodal_output, dict):
        frame = _extract_last_frame(multimodal_output)
        if frame is not None:
            transfer_manager.code_prompt_token_ids[request_id].append(frame.cpu().tolist())
        # Check done flag from talker
        done_flags = multimodal_output.get("done")
        logger.info(f"[MisoAsyncChunk] done_flags={done_flags}, type={type(done_flags)}")
        if isinstance(done_flags, (list, tuple)) and len(done_flags) > 0:
            logger.info(f"[MisoAsyncChunk] done_flags[0]={done_flags[0]}, bool={bool(done_flags[0])}")
            finished = finished or bool(done_flags[0])
            logger.info(f"[MisoAsyncChunk] finished after check={finished}")
        elif hasattr(done_flags, 'item'):  # torch.Tensor
            finished = finished or bool(done_flags.item())
            logger.info(f"[MisoAsyncChunk] finished after tensor check={finished}")

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", 25))
    left_context_size_config = int(cfg.get("codec_left_context_frames", 25))
    initial_chunk = int(cfg.get("initial_codec_chunk_frames") or 0)

    length = len(transfer_manager.code_prompt_token_ids[request_id])
    if length <= 0:
        if finished:
            return OmniPayloadStruct(
                codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
                meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
            )
        return None

    if initial_chunk > 0 and length < initial_chunk and not finished:
        return None

    if length < chunk_size and not finished:
        return None

    # Sliding window (chunk + left-context) just like Qwen3-TTS / Moss.
    context_length = min(length, chunk_size)
    end_index = min(length, left_context_size_config + context_length)
    left_context_size = max(0, end_index - context_length)
    window = transfer_manager.code_prompt_token_ids[request_id][-end_index:]
    if not window:
        return None

    num_frames = len(window)
    
    # Debug logging
    logger.info(f"[MisoAsyncChunk] window length={num_frames}, chunk_size={chunk_size}, left_context={left_context_size}")
    logger.info(f"[MisoAsyncChunk] first frame sample: {window[0][:5] if window else 'N/A'}")
    logger.info(f"[MisoAsyncChunk] last frame sample: {window[-1][:5] if window else 'N/A'}")
    
    code_tensor = torch.tensor(
        [window[f][q] for q in range(_MISO_NUM_CODEBOOKS) for f in range(num_frames)],
        dtype=torch.long,
    )
    meta = MetaStruct(
        left_context_size=left_context_size,
        codec_chunk_frames=chunk_size,
        codec_left_context_frames=left_context_size_config,
        code_flat_numel=int(code_tensor.numel()),
        finished=torch.tensor(finished, dtype=torch.bool),
    )
    if finished:
        transfer_manager.code_prompt_token_ids[request_id].clear()

    return OmniPayloadStruct(codes=CodesStruct(audio=code_tensor), meta=meta)


def talker2mimi_full_payload(
    transfer_manager: Any,
    pooling_output: dict | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    logger.info(f"[MisoFullPayloadConnector] Called, is_finished={is_finished}, request_finished={request.is_finished()}")
    if not is_finished and not request.is_finished():
        frame = _extract_last_frame(pooling_output) if isinstance(pooling_output, dict) else None
        logger.info(f"[MisoFullPayloadConnector] Extracted frame: {frame.shape if frame is not None else 'None'}")
        if frame is not None:
            rid = request.external_req_id
            transfer_manager.code_prompt_token_ids[rid].append(frame.cpu().tolist())
            logger.info(f"[MisoFullPayloadConnector] Buffered frame, total={len(transfer_manager.code_prompt_token_ids[rid])}")
        return None

    rid = request.external_req_id
    frames = transfer_manager.code_prompt_token_ids.get(rid, [])
    logger.info(f"[MisoFullPayloadConnector] Finished, buffered frames={len(frames)}")
    if not frames:
        logger.warning(f"[MisoFullPayloadConnector] No buffered frames, returning empty")
        return OmniPayloadStruct(
            codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
            meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
        )
    flat: list[int] = []
    for frame in frames:
        flat.extend(frame)
    logger.info(f"[MisoFullPayloadConnector] Flattened {len(flat)} codes")
    transfer_manager.code_prompt_token_ids[rid].clear()
    return OmniPayloadStruct(
        codes=CodesStruct(audio=torch.tensor(flat, dtype=torch.long)),
        meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
    )


def talker2mimi_token_only(
    source_outputs: list[Any],
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
) -> list[Any]:
    """Sync mode placeholder tokens; real codec payload arrives via connector."""
    from vllm_omni.inputs.data import OmniTokensPrompt

    return [
        OmniTokensPrompt(
            prompt_token_ids=[0],
            multi_modal_data=None,
            mm_processor_kwargs=None,
            additional_information=to_dict(OmniPayloadStruct()),
        )
        for out in source_outputs
        if getattr(out, "finished", False)
    ]


# Re-export dynamic IC helper for tests / parity with Qwen3-TTS processors.
__all__ = [
    "talker2mimi",
    "talker2mimi_async_chunk",
    "talker2mimi_full_payload",
    "talker2mimi_token_only",
    "compute_dynamic_initial_chunk_size",
    "max_ic_for_chunk_size",
]
