# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0
"""Idempotently add optional 4-bit QLoRA + gradient checkpointing to OpenVLA-OFT.

Patches ONLY the user's local external/openvla-oft/vla-scripts/finetune.py.
Upstream normal bf16 training remains unchanged when --use_qlora=False.
"""
import argparse
from pathlib import Path

MARKER = "use_qlora: bool = False"
IMPORT_PEFT = "from peft import LoraConfig, PeftModel, get_peft_model"
IMPORT_HF = "from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor"

ORIGINAL_LOAD = '''    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device_id)'''

QUANT_LOAD = '''    # 4-bit NF4 quantization is opt-in and loads weights directly on this GPU.
    # .to() is invalid for 4-bit bitsandbytes models.
    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ) if cfg.use_qlora else None
    )
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map={"": device_id} if cfg.use_qlora else None,
    )
    if cfg.use_qlora:
        vla = prepare_model_for_kbit_training(
            vla, use_gradient_checkpointing=cfg.gradient_checkpointing
        )
        vla.config.use_cache = False
    else:
        vla = vla.to(device_id)
        if cfg.gradient_checkpointing:
            vla.gradient_checkpointing_enable()
            vla.config.use_cache = False'''

def replace_one(content, original, replacement, label):
    if content.count(original) != 1:
        raise RuntimeError(
            f"Cannot patch {label}: expected one matching upstream block, "
            f"found {content.count(original)}. Upstream may have changed."
        )
    return content.replace(original, replacement, 1)


def patch_source(content):
    if MARKER in content:
        # Already patched. Do not modify another upstream revision by accident.
        if "prepare_model_for_kbit_training(" not in content:
            raise RuntimeError("Partial QLoRA patch detected")
        return content, False
    content = replace_one(
        content, IMPORT_PEFT,
        IMPORT_PEFT + ", prepare_model_for_kbit_training", "PEFT import"
    )
    content = replace_one(
        content, IMPORT_HF, IMPORT_HF + ", BitsAndBytesConfig", "HF import"
    )
    config_line = "    lora_dropout: float = 0.0"
    pos = content.find(config_line)
    if pos < 0:
        raise RuntimeError("OpenVLA-OFT FinetuneConfig LoRA options not found")
    line_end = content.index("\n", pos)
    content = (content[:line_end+1] +
               "    use_qlora: bool = False\n"
               "    gradient_checkpointing: bool = False\n" +
               content[line_end+1:])
    content = replace_one(content, ORIGINAL_LOAD, QUANT_LOAD, "model loading")
    content = replace_one(
        content, '            target_modules="all-linear",',
        '            target_modules=["q_proj", "v_proj"] if cfg.use_qlora else "all-linear",',
        "LoRA target modules"
    )
    # Upstream loops one additional optimizer step for max_steps=1.
    content = replace_one(
        content, "            if log_step == cfg.max_steps:",
        "            if (batch_idx + 1) // cfg.grad_accumulation_steps >= cfg.max_steps:",
        "optimizer step stop"
    )
    # Save at the first successful optimizer step, rather than never writing
    # a checkpoint from a one-step smoke test.
    content = replace_one(
        content, "            if gradient_step_idx > 0 and log_step % cfg.save_freq == 0:",
        "            if ((batch_idx + 1) % cfg.grad_accumulation_steps == 0\n"
        "                    and (gradient_step_idx + 1) % cfg.save_freq == 0):",
        "checkpoint save"
    )
    return content, True


def patch(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"OpenVLA-OFT trainer not found: {path}")
    before = path.read_text()
    after, modified = patch_source(before)
    compile(after, str(path), "exec")
    if modified:
        path.write_text(after)
    return modified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--openvla-oft", type=Path, default=Path("external/openvla-oft")
    )
    args = parser.parse_args()
    trainer = args.openvla_oft / "vla-scripts/finetune.py"
    changed = patch(trainer)
    print("QLoRA patch applied." if changed else "QLoRA patch already applied.")
    print(f"Trainer: {trainer}")


if __name__ == "__main__":
    main()
