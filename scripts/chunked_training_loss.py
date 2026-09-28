"""Research comparison helpers for the shared chunked vocabulary loss."""
from dmn.chunked_loss import loss_for


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
