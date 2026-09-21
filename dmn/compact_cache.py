"""Narrow experimental retirement policy for Gemma4 STANDARD sliding windows."""
from __future__ import annotations


def validate_retirements(length, ranges, window=1):
    """Validate the whole progressively shifted plan before any native mutation.

    Keeping a contiguous recent window ensures that no previously masked (and
    possibly evicted) local entry is brought back into the attention window.
    This condition is sufficient, not necessarily the least restrictive one.
    """
    if type(window) is not int or window < 1:
        raise ValueError("invalid local window")
    for start, count in ranges:
        if (type(start) is not int or type(count) is not int or
                not 0 <= start < start + count < length or length - start - count < window):
            raise ValueError("context retirement must preserve the complete recent local window")
        length -= count


def gemma4_retirement_window(config, metadata, native_window, binding_version):
    """Fail closed outside the reviewed native implementation and KV geometry.

    llama-cpp-python 0.3.35 pins llama.cpp 4df29be4. A custom native build must
    retain that implementation; binding version alone cannot certify its source.
    """
    if not config.experimental_compact_swa:
        return 1
    if binding_version != "0.3.35" or metadata("general.architecture") != "gemma4":
        raise ValueError("experimental compact retirement requires pinned Gemma4 native support")
    if native_window <= 0:
        raise ValueError("experimental compact retirement requires a local attention window")
    # Shared-KV variants and other cache formats require separate validation.
    shared = metadata("gemma4.attention.shared_kv_layers")
    if shared not in ("", "0"):
        raise ValueError("shared-KV Gemma4 variants are not validated for compact retirement")
    if config.type_k == "q8_0":
        for name in ("gemma4.attention.key_length", "gemma4.attention.key_length_swa"):
            try:
                head = int(metadata(name))
            except ValueError as exc:
                raise ValueError("cannot establish quantized K shift geometry") from exc
            if head <= 0 or head % 64:
                raise ValueError("quantized K retirement requires head dimensions divisible by 64")
    return native_window
