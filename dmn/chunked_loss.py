"""Full-context, target-only Gemma loss for the reviewed NF4 v2 recipe.

Recompute small vocabulary projections; never split or truncate an example.
"""


def loss_for(model, row, *, chunk_tokens=64):
    import torch
    from torch.utils.checkpoint import checkpoint
    from transformers import Gemma4ForCausalLM, Gemma4ForConditionalGeneration

    if type(chunk_tokens) is not int or not 1 <= chunk_tokens <= 256:
        raise ValueError('output chunk size must be 1..256 tokens')
    base = model.get_base_model()
    if type(base) not in (Gemma4ForCausalLM, Gemma4ForConditionalGeneration):
        raise ValueError('chunked loss supports only explicit Gemma4 classes')
    if any(p.requires_grad for p in base.lm_head.parameters()):
        raise ValueError('chunked loss requires a frozen vocabulary projection')
    if len(row['tokens']) != len(row['labels']) or len(row['tokens']) < 2:
        raise ValueError('unaligned example or loss mask')
    targets = row['labels'][1:]
    count = sum(label != -100 for label in targets)
    if not count:
        raise ValueError('example has no supervised shifted targets')
    device = base.get_input_embeddings().weight.device
    tokens = torch.tensor([row['tokens']], device=device, dtype=torch.long)
    labels = torch.tensor(targets, device=device, dtype=torch.long)
    # LoRA layers are already installed in the decoder. This path is restricted
    # to ordinary single-adapter LoRA, not prompt learning or mixed-adapter batches.
    if (len(model.peft_config) != 1 or
            next(iter(model.peft_config.values())).peft_type != 'LORA'):
        raise ValueError('chunked loss requires one ordinary LoRA adapter')
    hidden = base.model(input_ids=tokens, use_cache=False, return_dict=True).last_hidden_state
    cap = base.config.get_text_config().final_logit_softcapping

    def chunk_loss(states, selected_labels):
        logits = base.lm_head(states)
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap
        return torch.nn.functional.cross_entropy(
            logits[0].float(), selected_labels, ignore_index=-100, reduction='sum')

    terms = []
    for start in range(0, len(targets), chunk_tokens):
        end = min(start + chunk_tokens, len(targets))
        if all(label == -100 for label in targets[start:end]):
            continue
        states, selected = hidden[:, start:end], labels[start:end]
        # A scalar is retained for each chunk. Backward recomputes its projection
        # and soft-cap, avoiding a sequence-length x vocabulary activation graph.
        if torch.is_grad_enabled():
            terms.append(checkpoint(chunk_loss, states, selected, use_reentrant=False))
        else:
            terms.append(chunk_loss(states, selected))
    return torch.stack(terms).sum() / count
