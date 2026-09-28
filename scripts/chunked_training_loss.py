"""Research-only Gemma loss: keep the full context, recompute small output chunks.

Not connected to production recipes. Decoder attention and target masks are
unchanged; different matrix/reduction shapes may introduce rounding differences.
"""


def loss_for(model, row, *, chunk_tokens=64):
    import torch
    from torch.utils.checkpoint import checkpoint
    from transformers import Gemma4ForCausalLM, Gemma4ForConditionalGeneration

    if type(chunk_tokens) is not int or not 1 <= chunk_tokens <= 256:
        raise ValueError('output chunk size must be 1..256 tokens')
    base = model.get_base_model()
    if type(base) not in (Gemma4ForCausalLM, Gemma4ForConditionalGeneration):
        raise ValueError('research loss supports only explicit Gemma4 classes')
    if any(p.requires_grad for p in base.lm_head.parameters()):
        raise ValueError('research loss requires a frozen vocabulary projection')
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
        raise ValueError('research loss requires one ordinary LoRA adapter')
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


def compare_gradients(model, row, reference_loss):
    """Disposable numerical check after training, without another optimizer step."""
    import torch
    model.train()
    model.zero_grad(set_to_none=True)
    full = reference_loss(model, row)
    full.backward()
    expected = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    reference_value = float(full.detach())
    del full
    model.zero_grad(set_to_none=True)
    chunked = loss_for(model, row)
    chunked.backward()
    torch.testing.assert_close(chunked.detach().cpu(), torch.tensor(reference_value), atol=2e-6, rtol=2e-5)
    maximum = 0.
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            actual = parameter.grad.detach().cpu()
            torch.testing.assert_close(actual, expected[name], atol=2e-6, rtol=2e-4)
            maximum = max(maximum, float((actual - expected[name]).abs().max()))
    difference = abs(float(chunked.detach()) - reference_value)
    model.zero_grad(set_to_none=True)
    return {'completed': True, 'gradient_tensors': len(expected), 'loss_absolute_difference': difference,
            'max_gradient_absolute_difference': maximum, 'gradient_atol': 2e-6, 'gradient_rtol': 2e-4}
