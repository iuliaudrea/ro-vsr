"""
Beam search with n-gram blocking and repetition penalty.

Identical to the beam search in MultiVSR (`search.py`), with two additions:
  - repetition_penalty over the last `repetition_window` tokens
  - blocking of repeated n-grams (parameter `no_repeat_ngram_size`)

Used to reduce the hallucinations and "și și și ..." style loops
that we observed during autoregressive decoding.
"""

import torch
import torch.nn.functional as F


def tile(x, count, dim=0):
    """Repeat the tensor `count` times along dimension `dim`."""
    if isinstance(x, tuple):
        h, c = x
        return tile(h, count, dim=dim), tile(c, count, dim=dim)
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


def apply_repetition_penalty(log_probs, alive_seq, penalty, window, special_token_ids):
    """Penalize tokens that have appeared in the last `window` steps."""
    if penalty <= 1.0 or window <= 0:
        return log_probs

    seq_len = alive_seq.size(1)
    start = max(0, seq_len - window)
    recent_tokens = alive_seq[:, start:]

    log_penalty = torch.log(torch.tensor(penalty, device=log_probs.device))

    mask = torch.zeros_like(log_probs, dtype=torch.bool)
    mask.scatter_(1, recent_tokens, True)

    if special_token_ids:
        for special_id in special_token_ids:
            if 0 <= special_id < log_probs.size(1):
                mask[:, special_id] = False

    return torch.where(mask, log_probs - log_penalty, log_probs)


def apply_no_repeat_ngram(log_probs, alive_seq, ngram_size):
    """Block tokens that would form an n-gram already present in the sequence."""
    if ngram_size <= 0:
        return log_probs

    batch_beam_size, vocab_size = log_probs.shape
    seq_len = alive_seq.size(1)

    if seq_len < ngram_size - 1:
        return log_probs

    log_probs = log_probs.clone()
    alive_seq_list = alive_seq.tolist()

    for i in range(batch_beam_size):
        seq = alive_seq_list[i]
        if len(seq) < ngram_size - 1:
            continue
        prefix = tuple(seq[-(ngram_size - 1):]) if ngram_size > 1 else ()

        for k in range(len(seq) - ngram_size + 1):
            ngram = tuple(seq[k:k + ngram_size])
            if ngram[:-1] == prefix:
                blocked_token = ngram[-1]
                if 0 <= blocked_token < vocab_size:
                    log_probs[i, blocked_token] = float("-inf")

    return log_probs


def beam_search_with_rep_penalty(
    model,
    size,
    bos_index,
    eos_index,
    pad_index,
    encoder_output,
    src_mask,
    max_output_length,
    alpha=1.0,
    n_best=1,
    repetition_penalty=1.0,
    repetition_window=0,
    no_repeat_ngram_size=0,
    special_token_ids=None,
):
    """
    Beam search identical to MultiVSR's (`search.py`), plus repetition
    penalty and n-gram blocking.
    """
    from vsr_inference.dataloader_utils import subsequent_mask

    if special_token_ids is None:
        special_token_ids = set()

    assert size > 0
    assert n_best <= size

    batch_size = src_mask.size(0)
    encoder_output = tile(encoder_output.contiguous(), size, dim=0)
    src_mask = tile(src_mask, size, dim=0)

    batch_offset = torch.arange(
        batch_size, dtype=torch.long, device=encoder_output.device
    )
    beam_offset = torch.arange(
        0, batch_size * size, step=size,
        dtype=torch.long, device=encoder_output.device,
    )

    bos_len = len(bos_index)
    alive_seq = torch.zeros(
        [batch_size * size, bos_len],
        dtype=torch.long, device=encoder_output.device,
    )
    for i in range(bos_len):
        alive_seq[:, i] = bos_index[i]

    topk_log_probs = torch.zeros(batch_size, size, device=encoder_output.device)
    topk_log_probs[:, 1:] = float("-inf")

    hypotheses = [[] for _ in range(batch_size)]
    results = {
        "predictions": [[] for _ in range(batch_size)],
        "scores": [[] for _ in range(batch_size)],
        "attentions": [[] for _ in range(batch_size)],
    }

    for step in range(max_output_length):
        decoder_input = alive_seq

        logits = model.decode(
            encoder_output, src_mask, decoder_input,
            subsequent_mask(decoder_input.size(1)).to(encoder_output.device).long(),
        )
        logits = logits.reshape((decoder_input.size(0), -1, logits.size(-1)))
        logits = logits[:, -1]

        log_probs = F.log_softmax(logits, dim=-1).squeeze(1)

        log_probs = apply_repetition_penalty(
            log_probs, alive_seq,
            penalty=repetition_penalty,
            window=repetition_window,
            special_token_ids=special_token_ids,
        )
        log_probs = apply_no_repeat_ngram(
            log_probs, alive_seq,
            ngram_size=no_repeat_ngram_size,
        )

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
            [alive_seq.index_select(0, select_indices), topk_ids.view(-1, 1)],
            -1,
        )

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
                    best_hyp = sorted(
                        hypotheses[b], key=lambda x: x[0], reverse=True
                    )
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
        encoder_output = encoder_output.index_select(0, select_indices)
        src_mask = src_mask.index_select(0, select_indices)

    return results["predictions"], results["scores"]
