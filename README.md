# HealingAxis

Steering vectors and evaluation materials accompanying the HealingAxis paper.

## Contents

- `section_2_HealingAxis/`: Healing Axis, Reflect, Explore, Direct, and Challenge vectors for six models, plus the steering runner and configuration.
- `appendix_F_SkilledTherapyBenchmark/`: the psychotherapy-skill judge prompt.
- `appendix_F_SkilledTherapyBenchmark_on_GoodBadTherapy/`: judge-validation results for 4,604 client–response interactions.

## Run steering

From the repository root, install the runner's dependencies:

```sh
python -m pip install torch 'transformers>=5.13' safetensors
```

Generate a reply with Qwen3-4B:

```sh
python section_2_HealingAxis/steeringConfig/steer.py \
  --model Qwen3-4B \
  --client "I can't stop thinking about what my boss said."
```

The runner downloads the model from Hugging Face and prints the generated reply. Use `--alpha 0` for an unsteered reply. Select a sub-axis with `--axis Reflect_SubAxis`, `Explore_SubAxis`, `Direct_SubAxis`, or `Challenge_SubAxis`. `--system` supplies an optional system prompt; the default has none.

Supported model names: `Qwen3-4B`, `Qwen3-8B`, `Llama-3.1-8B`, `Gemma-3-12B`, `Gemma-3-27B`, and `OLMo-3-7B`. Llama-3.1-8B requires access to its gated Hugging Face model.

## Configuration

`section_2_HealingAxis/steeringConfig/steering_config.json` defines each model's checkpoint, layer, default steering strength, and dtype. Each vector file stores its model, layer, axis, and scale.

The runner adds `alpha × scale × vector` to the output of decoder block `layer - 1` during token generation. Prompt positions are unchanged. Generation is greedy, with a default maximum of 2,048 new tokens. The runner selects CUDA, then Apple MPS, then CPU; `--device` overrides this choice.

## Dataset

[GoodBadTherapy on Hugging Face](https://huggingface.co/datasets/ICLR-2027-13513/HealingAxis_GoodBadTherapy).
