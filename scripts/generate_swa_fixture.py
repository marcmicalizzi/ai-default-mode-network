"""Generate tiny random Gemma4-shaped weights for cache mechanics, no trained model."""
from pathlib import Path
import struct
import numpy as np

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
OUT = parser.parse_args().output
if OUT.exists():
    raise SystemExit("fixture already exists")
def u(fmt, *v): return struct.pack("<" + fmt, *v)
def string(s):
    raw = s.encode()
    return u("Q", len(raw)) + raw
def val(kind, v):
    if kind == 8: return string(v)
    if kind == 9:
        elem, items = v
        return u("IQ", elem, len(items)) + b"".join(val(elem, x) for x in items)
    return u({4:"I",5:"i",6:"f",7:"?"}[kind], v)
metadata = []
def put(k, t, v): metadata.append(string(k) + u("I", t) + val(t,v))
put("general.architecture",8,"gemma4")
put("general.name",8,"DMN random Gemma4 cache-mechanics fixture; not a trained model")
for k,v in {"block_count":6,"context_length":4096,"embedding_length":64,
            "feed_forward_length":128,"attention.head_count":2,"attention.head_count_kv":1,
            "attention.key_length":128,"attention.value_length":128,
            "attention.key_length_swa":64,"attention.value_length_swa":64,
            "rope.dimension_count":128,"rope.dimension_count_swa":64,
            "attention.sliding_window":64,"attention.shared_kv_layers":0,
            "embedding_length_per_layer_input":0}.items(): put("gemma4."+k,4,v)
for k,v in {"attention.layer_norm_rms_epsilon":1e-6,"rope.freq_base":10000.0,
            "rope.freq_base_swa":10000.0,"final_logit_softcapping":30.0}.items(): put("gemma4."+k,6,v)
put("gemma4.attention.sliding_window_pattern",9,(7,[True]*5+[False]))
tokens = ["<unk>","<s>","</s>"] + [f"<0x{x:02X}>" for x in range(256)] + ["▁"]
put("tokenizer.ggml.model",8,"llama")
put("tokenizer.ggml.tokens",9,(8,tokens))
put("tokenizer.ggml.scores",9,(6,[0.0]*len(tokens)))
put("tokenizer.ggml.token_type",9,(5,[2,3,3]+[6]*256+[1]))
for k,v in {"bos_token_id":1,"eos_token_id":2,"unknown_token_id":0}.items(): put("tokenizer.ggml."+k,4,v)
put("tokenizer.ggml.add_bos_token",7,True)
rng=np.random.default_rng(73291)
tensors=[]
def tensor(name, shape, ones=False):
    data=(np.ones(shape) if ones else rng.normal(0,0.05,shape)).astype("<f4")
    tensors.append((name,data))
tensor("token_embd.weight",(len(tokens),64))
tensor("output_norm.weight",(64,),True)
for layer in range(6):
    head=64 if layer<5 else 128
    for name in ("attn_norm","post_attention_norm","ffn_norm","post_ffw_norm"):
        tensor(f"blk.{layer}.{name}.weight",(64,),True)
    for name in ("attn_q_norm","attn_k_norm"):
        tensor(f"blk.{layer}.{name}.weight",(head,),True)
    tensor(f"blk.{layer}.attn_q.weight",(head*2,64))
    tensor(f"blk.{layer}.attn_k.weight",(head,64))
    tensor(f"blk.{layer}.attn_v.weight",(head,64))
    tensor(f"blk.{layer}.attn_output.weight",(64,head*2))
    for name in ("ffn_gate","ffn_up"): tensor(f"blk.{layer}.{name}.weight",(128,64))
    tensor(f"blk.{layer}.ffn_down.weight",(64,128))
    if layer==5: tensor("rope_freqs.weight",(head//2,),True)
header=b"GGUF"+u("IQQ",3,len(tensors),len(metadata))+b"".join(metadata)
offset=0
for name,data in tensors:
    header+=string(name)+u("I",data.ndim)+u("Q"*data.ndim,*reversed(data.shape))+u("IQ",0,offset)
    offset+=(data.nbytes+31)//32*32
with OUT.open("xb") as f:
    f.write(header)
    f.write(b"\0"*((-len(header))%32))
    for _,data in tensors:
        f.write(data.tobytes())
        f.write(b"\0"*((-data.nbytes)%32))
print({"file":str(OUT),"bytes":OUT.stat().st_size,"trained_weights":False})
