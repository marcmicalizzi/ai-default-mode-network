"""Prepare the explicit static training placement without a large GPU cast spike."""
import os

ALLOCATOR = 'backend:native,max_split_size_mb:128'


def configure_allocator():
    # Apply only inside the disposable GPU worker, before importing Torch.
    # Keep large temporary weight buffers from becoming fragmented reservations.
    os.environ['PYTORCH_ALLOC_CONF'] = ALLOCATOR
    os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)


def prepare(model, *, large_tensor_bytes=64 * 1024**2, gradient_checkpointing=True):
    import torch
    from accelerate.hooks import remove_hook_from_module
    from peft import prepare_model_for_kbit_training
    # The experiments assign the entire decoder to one GPU and unused frozen
    # vision components to CPU. Remove inference-time dispatch/offload hooks,
    # restoring tensors before inspecting or hashing frozen weights. Accelerate
    # detach restores their original device, which can be CUDA even for a CPU
    # mapping, so explicitly reapply CPU placement for unused components.
    # This is not a general auto-offloaded training path.
    remove_hook_from_module(model, recurse=True)
    for name, device in getattr(model, 'hf_device_map', {}).items():
        if device == 'cpu':
            if name not in {'model.vision_tower', 'model.embed_vision'}:
                raise ValueError('research preparation supports CPU placement only for unused vision components')
            model.get_submodule(name).to('cpu')
    moved = []
    for name, parameter in model.named_parameters():
        if (parameter.device.type == 'cuda' and parameter.dtype in (torch.bfloat16, torch.float16)
                and parameter.__class__.__name__ != 'Params4bit'
                and parameter.numel() * parameter.element_size() >= large_tensor_bytes):
            device = parameter.device
            # Preserve the Parameter object and any tied references. Stage its
            # existing half-precision data in RAM, release the GPU allocation,
            # then do the same F32 conversion PEFT would perform. Avoid holding
            # both the old 2.6 GiB embedding and new 5.25 GiB embedding on GPU.
            parameter.data = parameter.detach().cpu()
            torch.cuda.empty_cache()
            # Convert on CPU first. Combining dtype/device in one .to() can
            # allocate another half-precision staging tensor on CUDA.
            parameter.data = parameter.data.float().to(device=device)
            moved.append(name)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={'use_reentrant': False})
    return model, moved
