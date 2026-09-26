"""Generate one reply from a chat model steered along the Healing Axis or one of its sub-axes.

The vector ../<axis>/<model>/vector.safetensors, scaled by alpha * scale (scale is stored in the file),
is added to the output of decoder block (layer - 1) at generated-token positions only; alpha = 0 runs
the unsteered model. Model ids, layers, default alphas and dtypes come from steering_config.json.
Requires torch, transformers (5.13 or newer) and safetensors. Llama-3.1-8B is gated on Hugging Face.

Example:
    python steer.py --model Qwen3-4B --client "I can't stop thinking about what my boss said."
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
AXES = ("HealingAxis", "Reflect_SubAxis", "Explore_SubAxis", "Direct_SubAxis", "Challenge_SubAxis")
# Decoder-only models expose model.layers; the Gemma-3 multimodal wrapper exposes model.language_model.layers.
BLOCK_PATHS = ("model.layers", "model.language_model.layers", "language_model.model.layers",
               "model.model.layers", "transformer.h")


def decoder_blocks(model):
    for path in BLOCK_PATHS:
        obj = model
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise RuntimeError("could not locate the decoder blocks of this model")


class Steerer:
    """Forward hook adding delta at generated-token positions only; prompt positions are never changed."""

    def __init__(self, block, delta, prompt_len):
        self.delta, self.prompt_len = delta, prompt_len
        block.register_forward_hook(self.hook)

    def hook(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        seq_len = hidden.shape[1]
        start = 0 if seq_len == 1 else self.prompt_len  # a decoding step is steered, the prompt pass is not
        if start < seq_len:
            delta = self.delta.to(device=hidden.device, dtype=hidden.dtype)
            hidden[:, start:seq_len, :] = hidden[:, start:seq_len, :] + delta
        return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden


def main():
    models = json.loads((HERE / "steering_config.json").read_text())["models"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=list(models))
    parser.add_argument("--axis", default="HealingAxis", choices=AXES)
    parser.add_argument("--alpha", type=float, help="steering strength in units of scale (default: the model's alpha)")
    parser.add_argument("--client", required=True, help="the client statement the model replies to")
    parser.add_argument("--system", help="optional system prompt (default: none)")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--device", help="default: cuda, else mps, else cpu")
    args = parser.parse_args()

    spec = models[args.model]
    alpha = spec["alpha"] if args.alpha is None else args.alpha
    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    with safe_open(HERE.parent / args.axis / args.model / "vector.safetensors", framework="pt") as handle:
        vector, meta = handle.get_tensor("vector"), handle.metadata()
    if meta["model"] != spec["hf_id"] or int(meta["layer"]) != spec["layer"]:
        raise ValueError("vector file does not match steering_config.json")

    tokenizer = AutoTokenizer.from_pretrained(spec["hf_id"])
    model = AutoModelForCausalLM.from_pretrained(spec["hf_id"], dtype=getattr(torch, spec["dtype"]))
    model = model.to(device).eval()
    messages = [{"role": "system", "content": args.system}] if args.system else []
    messages.append({"role": "user", "content": args.client})
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    if alpha != 0:  # float32 delta, cast to the model dtype inside the hook
        delta = (alpha * float(meta["scale"]) * vector).to(device)
        Steerer(decoder_blocks(model)[spec["layer"] - 1], delta, input_ids.shape[1])
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                max_new_tokens=args.max_new_tokens, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
    reply = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
    print(reply.rsplit("</think>", 1)[1].strip() if "</think>" in reply else reply)


if __name__ == "__main__":
    main()
