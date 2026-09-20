from __future__ import annotations

import tempfile
from pathlib import Path

from .backend import LlamaBackend
from .storage import write_durable


def verify_native(config, steps=24, report_path=None):
    """Fresh native context restoration versus uninterrupted continuation.

    Replaying a prompt is forbidden during load. Then compare all next sampled
    tokens and logits produced by actual decode, not only the cached logits.
    """
    import numpy as np
    backend = LlamaBackend(config)
    progress = {"unshifted_continuation_verified": False, "context_shift_restore_verified": False}
    try:
        with tempfile.TemporaryDirectory(prefix="dmn-native-verification-") as temp:
            saved = Path(temp)
            backend.eval(backend.tokenize("Once upon a time, a small bird watched the seasons change. " * 5, initial=True))
            for _ in range(8):
                backend.eval([backend.sample()])
            token_count = len(backend.tokens)
            backend.save(saved)
            fingerprint = backend.fingerprint
            expected = []
            for _ in range(steps):
                token = backend.sample()
                backend.eval([token])
                expected.append((token, backend.logits.copy()))
            backend.close()
            backend = LlamaBackend(config)
            original_eval = backend.eval

            def forbidden_eval(*args, **kwargs):
                raise AssertionError("checkpoint restore attempted prompt reevaluation")

            backend.eval = forbidden_eval
            evidence = backend.load(saved)
            backend.eval = original_eval
            max_error = 0.0
            for token, logits in expected:
                actual_token = backend.sample()
                if actual_token != token:
                    raise AssertionError(f"sampled continuation differs: {actual_token} != {token}")
                backend.eval([actual_token])
                error = float(np.max(np.abs(logits - backend.logits)))
                max_error = max(max_error, error)
                np.testing.assert_allclose(backend.logits, logits, rtol=1e-5, atol=1e-5)
            progress.update(unshifted_continuation_verified=True, maximum_logit_absolute_error=max_error)
            # Exercise native context retirement and a checkpoint *after* the
            # shift is applied. Compare its subsequent causal continuation too.
            shift_verified = False
            if backend.can_shift:
                backend.shift(4, min(16, len(backend.tokens) - 5))
                backend.eval(backend.tokenize("\nTime passed. "))
                backend.save(saved)
                probe = backend.sample()
                backend.eval([probe])
                expected_shift_logits = backend.logits.copy()
                backend.close()
                backend = LlamaBackend(config)
                backend.load(saved)
                if backend.sample() != probe:
                    raise AssertionError("shifted checkpoint sampler differs")
                backend.eval([probe])
                progress["shifted_maximum_logit_absolute_error"] = float(np.max(np.abs(backend.logits - expected_shift_logits)))
                np.testing.assert_allclose(backend.logits, expected_shift_logits, rtol=1e-5, atol=1e-5)
                shift_verified = True
            report = {"verified": True, "method": "fresh_native_context_no_prompt_eval_then_decode_comparison",
                      **evidence, **progress, "checkpoint_tokens": token_count, "continuation_tokens_compared": steps,
                      "maximum_logit_absolute_error": max_error, "context_shift_restore_verified": shift_verified,
                      "fingerprint": fingerprint,
                      "limits": "Evidence for this exact model/build/config; not cross-build equivalence or a claim about subjective continuity."}
            if report_path:
                report_path.parent.mkdir(parents=True, exist_ok=True)
                write_durable(report_path, report)
            return report
    except Exception as exc:
        if report_path:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            write_durable(report_path, {"verified": False, **progress,
                "error": str(exc), "fingerprint": backend.fingerprint})
        raise
    finally:
        backend.close()
