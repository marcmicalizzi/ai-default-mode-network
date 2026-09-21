"""Optional libmtmd image input on the existing native sequence, owned by inference."""
from __future__ import annotations

import ctypes as C
import io
import os
from contextlib import contextmanager
from pathlib import Path
from .attachments import ImageInputError

# Negative sentinel counts occupied visual positions. Never pass it to decode,
# sampling penalties, training, or retained-token reconstruction.
IMAGE_POSITION = -1


class NativeVision:
    def __init__(self, backend):
        import llama_cpp
        from llama_cpp import mtmd_cpp as api
        from .backend import sha256_file
        from .native_logging import _callback
        if llama_cpp.__version__ != "0.3.35":
            raise ValueError("vision requires the pinned llama-cpp-python 0.3.35 MTMD ABI")
        self.backend, self.api, self.ctx = backend, api, None
        # Do not allow an unrecorded override to mix two native library builds.
        native_root = Path(backend.api._lib._name).resolve().parent
        if Path(api._libmtmd._name).resolve().parent != native_root:
            raise ValueError("MTMD must come from the same native directory as llama")
        path = Path(backend.config.vision_projector_path).resolve()
        if not path.is_file():
            raise ValueError("vision projector file does not exist")
        params = api.mtmd_context_params_default()
        params.use_gpu = False  # Separate, bounded CPU projector; no extra GPU claim.
        params.n_threads = backend.config.n_threads
        params.warmup = False
        params.print_timings = False
        if _callback:
            api.mtmd_helper_log_set(_callback, None)
        self.ctx = api.mtmd_init_from_file(os.fsencode(path), backend.model, params)
        if not self.ctx:
            raise ValueError("MTMD could not load a compatible vision projector")
        try:
            if not api.mtmd_support_vision(self.ctx) or api.mtmd_decode_use_mrope(self.ctx):
                raise ValueError("vision currently requires a non-M-RoPE image model")
            self.fingerprint = {"projector_sha256": sha256_file(path),
                                "binding_sha256": sha256_file(Path(api.__file__)),
                                "position_format": "negative_visual_slots_v1", "projector_device": "cpu"}
        except BaseException:
            self.close()
            raise

    @contextmanager
    def prepare(self, images):
        """Preprocess without decoding into the model; all native buffers are scoped."""
        from PIL import Image
        api, bitmaps = self.api, []
        chunks = api.mtmd_input_chunks_init()
        if not chunks:
            raise ImageInputError("could not allocate image chunks")
        try:
            for image in images:
                try:
                    with Image.open(io.BytesIO(image.data)) as picture:
                        rgb = picture.convert("RGB")
                        try:
                            pixels = rgb.tobytes()
                            raw = (C.c_uint8 * len(pixels)).from_buffer_copy(pixels)
                            bitmap = api.mtmd_bitmap_init(rgb.width, rgb.height, raw)
                        finally:
                            rgb.close()
                except (OSError, ValueError) as exc:
                    raise ImageInputError("image pixel decoding failed") from exc
                if not bitmap:
                    raise ImageInputError("could not decode image")
                bitmaps.append(bitmap)
            # No untrusted text reaches MTMD's special-token/marker parser.
            marker = api.mtmd_default_marker()
            prompt = b"\n".join(marker for _ in bitmaps) + b"\n"
            text = api.mtmd_input_text(prompt, len(prompt), False, True)
            pointers = (api.mtmd_bitmap_p_ctypes * len(bitmaps))(*bitmaps)
            if api.mtmd_tokenize(self.ctx, chunks, C.byref(text), pointers, len(bitmaps)):
                raise ImageInputError("vision preprocessing failed")
            slots = []
            for index in range(api.mtmd_input_chunks_size(chunks)):
                chunk = api.mtmd_input_chunks_get(chunks, index)
                count = api.mtmd_input_chunk_get_n_tokens(chunk)
                if count != api.mtmd_input_chunk_get_n_pos(chunk):
                    raise ImageInputError("image position layout is unsupported")
                kind = api.mtmd_input_chunk_get_type(chunk)
                if kind == api.MTMD_INPUT_CHUNK_TYPE_TEXT:
                    length = C.c_size_t()
                    tokens = api.mtmd_input_chunk_get_tokens_text(chunk, C.byref(length))
                    if length.value != count:
                        raise ImageInputError("invalid MTMD text chunk")
                    slots.extend(tokens[:count])
                elif kind == api.MTMD_INPUT_CHUNK_TYPE_IMAGE:
                    slots.extend([IMAGE_POSITION] * count)
                else:
                    raise ImageInputError("only image attachments are supported")
            if IMAGE_POSITION not in slots:
                raise ImageInputError("projector produced no visual positions")
            yield PreparedImages(self, chunks, slots)
        finally:
            api.mtmd_input_chunks_free(chunks)
            for bitmap in bitmaps:
                api.mtmd_bitmap_free(bitmap)

    def close(self):
        if self.ctx:
            self.api.mtmd_free(self.ctx)
            self.ctx = None


class PreparedImages:
    def __init__(self, vision, chunks, slots):
        self.vision, self.chunks, self.slots = vision, chunks, slots
        self.positions = len(slots)

    def evaluate(self):
        vision, backend = self.vision, self.vision.backend
        before = len(backend.tokens)
        if before + self.positions > backend.n_ctx:
            raise ValueError("images would overflow context")
        end = C.c_int32()
        code = vision.api.mtmd_helper_eval_chunks(vision.ctx, backend.ctx, self.chunks,
            before, 0, backend.config.n_batch, True, C.byref(end))
        if code or end.value != before + self.positions:
            # A partial decode must stop inference; never acknowledge the event.
            raise RuntimeError("native image decode failed; restore the last committed checkpoint")
        backend.tokens.extend(self.slots)
        backend.decoded_tokens += self.positions
        backend.decode_calls += 1
        pointer = backend.api.llama_get_logits_ith(backend.ctx, -1)
        if not pointer:
            raise RuntimeError("image decode did not return logits")
        backend.logits = backend.np.ctypeslib.as_array(pointer, shape=(backend.n_vocab,)).copy()
