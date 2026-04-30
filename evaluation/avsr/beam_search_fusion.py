"""
Beam search with shallow fusion between Whisper (audio) and VSRo (video).

Fusion modes:
  - "hibrid_logp": shallow fusion at log-prob level (Whisper + VSR averaged
                   in log-probability space). This is the main reported method.
  - "whisper":     audio-only baseline (Whisper alone).
  - "multivsr":    video-only baseline (VSR alone).

Decoder structure follows MultiVSR's beam search (search.py), with the
modification that the next-token log-probabilities are computed by combining
both modalities at each step.
"""

import numpy as np
import torch
import torch.nn.functional as F


def tile(x, count, dim=0):
    """Replicate the tensor `count` times along dimension `dim`."""
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = (
        x.view(batch, -1)
        .transpose(0, 1)
        .repeat(count, 1)
        .transpose(0, 1)
        .contiguous()
        .view(*out_size)
    )
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def subsequent_mask(size: int) -> torch.Tensor:
    """Upper-triangular causal mask for the decoder."""
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(mask) == 0


def beam_search_fusion(
    whisper_model,        # WhisperForConditionalGeneration.model (encoder + decoder)
    whisper_proj,         # WhisperForConditionalGeneration.proj_out
    audio_embeds,         # (1, T_a, D_a) — Whisper encoder output
    multivsr_model,       # the encoder-decoder VSR model
    video_memory,         # (1, T_v, D_v) — VSR encoder output
    src_mask,             # (1, 1, T_v) — VSR encoder mask
    bos_indices,          # list of BOS token IDs (start-of-transcript prompt)
    eos_index,            # EOS token ID
    pad_index,            # PAD token ID (typically 0)
    max_output_length,    # maximum tokens to generate
    size=5,               # beam size
    alpha=1.0,            # length penalty exponent
    n_best=1,             # number of hypotheses to return
    mode="hibrid_logp",   # "hibrid_logp" | "whisper" | "multivsr"
):
    """
    Beam search decoding with shallow fusion. Supports three modes:

    - "hibrid_logp": at each step, log_probs = (log_softmax(whisper_logits)
                     + log_softmax(vsr_logits)) / 2
    - "whisper":     log_probs = log_softmax(whisper_logits)
    - "multivsr":    log_probs = log_softmax(vsr_logits)

    Currently supports batch_size=1 only.
    """
    batch_size = src_mask.size(0)
    assert batch_size == 1, "Currently only batch_size=1 is supported"

    device = src_mask.device

    # Replicate encoder outputs `size` times for parallel beam search
    audio_embeds = tile(audio_embeds.contiguous(), size, dim=0)
    video_memory = tile(video_memory.contiguous(), size, dim=0)
    src_mask = tile(src_mask, size, dim=0)

    batch_offset = torch.arange(batch_size, dtype=torch.long, device=device)
    beam_offset = torch.arange(0, batch_size * size, step=size,
                               dtype=torch.long, device=device)

    # Initialize sequences with BOS prompt
    alive_seq = torch.zeros([batch_size * size, len(bos_indices)],
                            dtype=torch.long, device=device)
    for i, tok in enumerate(bos_indices):
        alive_seq[:, i] = tok

    # First beam gets log-prob 0, the rest -inf so they don't compete on step 1
    topk_log_probs = torch.zeros(batch_size, size, device=device)
    topk_log_probs[:, 1:] = float("-inf")

    hypotheses = [[] for _ in range(batch_size)]
    results = {
        "predictions": [[] for _ in range(batch_size)],
        "scores": [[] for _ in range(batch_size)],
    }

    for step in range(max_output_length):
        decoder_input = alive_seq

        # ── Whisper logits ────────────────────────────────────────────
        if mode != "multivsr":
            out_audio = whisper_model.decoder(
                input_ids=decoder_input,
                encoder_hidden_states=audio_embeds,
            )
            logits_audio = whisper_proj(out_audio.last_hidden_state[:, -1])
            logp_audio = F.log_softmax(logits_audio, dim=-1)
        else:
            logp_audio = None

        # ── VSR logits ───────────────────────────────────────────────
        if mode != "whisper":
            tgt_mask = subsequent_mask(decoder_input.size(1)).to(device).long()
            logits_video = multivsr_model.decode(
                video_memory, src_mask, decoder_input, tgt_mask
            )
            logits_video = logits_video.reshape(
                (decoder_input.size(0), -1, logits_video.size(-1))
            )
            logits_video = logits_video[:, -1]
            logp_video = F.log_softmax(logits_video, dim=-1)
        else:
            logp_video = None

        # ── Fusion ───────────────────────────────────────────────────
        if mode == "whisper":
            log_probs = logp_audio
        elif mode == "multivsr":
            log_probs = logp_video
        elif mode == "hibrid_logp":
            # Align vocab sizes if Whisper and VSR differ slightly (rare)
            min_vocab = min(logp_audio.size(-1), logp_video.size(-1))
            log_probs = (logp_audio[:, :min_vocab] + logp_video[:, :min_vocab]) / 2.0
        else:
            raise ValueError(f"Unknown fusion mode: {mode}")

        # ── Standard beam search expansion ───────────────────────────
        log_probs += topk_log_probs.view(-1).unsqueeze(1)
        curr_scores = log_probs.clone()

        output_size = log_probs.size(-1)

        if alpha > -1:
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha
            curr_scores /= length_penalty

        curr_scores = curr_scores.reshape(-1, size * output_size)
        topk_scores, topk_ids = curr_scores.topk(size, dim=-1)

        if alpha > -1:
            topk_log_probs = topk_scores * length_penalty
        else:
            topk_log_probs = topk_scores.clone()

        try:
            topk_beam_index = topk_ids.div(output_size, rounding_mode="floor")
        except TypeError:
            topk_beam_index = topk_ids.div(output_size)
        topk_ids = topk_ids.fmod(output_size)

        batch_index = (
            topk_beam_index + beam_offset[:topk_beam_index.size(0)].unsqueeze(1)
        )
        select_indices = batch_index.view(-1)

        alive_seq = torch.cat(
            [alive_seq.index_select(0, select_indices), topk_ids.view(-1, 1)], -1
        )

        # ── Termination ──────────────────────────────────────────────
        is_finished = topk_ids.eq(eos_index)
        if step + 1 == max_output_length:
            is_finished.fill_(True)
        end_condition = is_finished[:, 0].eq(True)

        if is_finished.any():
            predictions = alive_seq.view(-1, size, alive_seq.size(-1))
            for i in range(is_finished.size(0)):
                b = batch_offset[i]
                if end_condition[i]:
                    is_finished[i].fill_(True)
                finished_hyp = is_finished[i].nonzero(as_tuple=False).view(-1)
                for j in finished_hyp:
                    if (predictions[i, j, 1:] == eos_index).nonzero(
                        as_tuple=False
                    ).numel() < 2:
                        hypotheses[b].append(
                            (topk_scores[i, j], predictions[i, j, 1:])
                        )
                if end_condition[i]:
                    best_hyp = sorted(hypotheses[b], key=lambda x: x[0], reverse=True)
                    for n, (score, pred) in enumerate(best_hyp):
                        if n >= n_best:
                            break
                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)

            non_finished = end_condition.eq(False).nonzero(as_tuple=False).view(-1)
            if len(non_finished) == 0:
                break
            topk_log_probs = topk_log_probs.index_select(0, non_finished)
            batch_index = batch_index.index_select(0, non_finished)
            batch_offset = batch_offset.index_select(0, non_finished)
            alive_seq = predictions.index_select(0, non_finished).view(
                -1, alive_seq.size(-1)
            )

        select_indices = batch_index.view(-1)
        audio_embeds = audio_embeds.index_select(0, select_indices)
        video_memory = video_memory.index_select(0, select_indices)
        src_mask = src_mask.index_select(0, select_indices)

    return results["predictions"], results["scores"]
