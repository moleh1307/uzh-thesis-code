"""Offline execution identity captured for review, then required before scoring."""
import importlib.metadata
import platform
from pathlib import Path

from specificity_validation import digest, file_hash, strict_json


def checkpoint_identity(directory):
    root = Path(directory).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("model must be an existing local checkpoint directory")
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in
                   {".json", ".safetensors", ".model", ".txt", ".jinja", ".tiktoken"})
    names = {str(p.relative_to(root)) for p in files}
    if not {"config.json", "generation_config.json", "tokenizer_config.json"} <= names:
        raise ValueError("checkpoint/config/tokenizer files are incomplete")
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("local safetensors weights required")
    for index in root.glob("*.safetensors.index.json"):
        weight_map = strict_json(index.read_text()).get("weight_map", {})
        if not weight_map:
            raise ValueError("empty safetensors index")
        for shard in weight_map.values():
            if not isinstance(shard, str) or shard not in names or ".." in Path(shard).parts:
                raise ValueError("safetensors index references an unhashed or external shard")
    # Hash every local resource, including files reached through snapshot symlinks.
    return {str(p.relative_to(root)): file_hash(p) for p in files}


def runtime_identity(torch, transformers):
    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__,
            "accelerate": importlib.metadata.version("accelerate"),
            "safetensors": importlib.metadata.version("safetensors"),
            "tokenizers": importlib.metadata.version("tokenizers"),
            "cuda": torch.version.cuda,
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}


def prepare_backend(args):
    import torch
    import transformers

    settings = strict_json(args.settings.read_text())
    profile = None if args.capture_execution_profile else strict_json(args.execution_profile.read_text())
    if profile is not None and profile.get("status") != "approved":
        raise ValueError("execution profile is not reviewed and approved")
    if args.revision != settings["model"]["revision"]:
        raise ValueError("revision does not match approved settings")
    if args.max_new_tokens != settings["decoding"]["max_new_tokens"]:
        raise ValueError("max_new_tokens does not match approved settings")
    if any(settings["decoding"].get(k) != v for k, v in
           {"do_sample": False, "temperature": None, "top_p": None, "top_k": None,
            "seed": 0, "input_truncation": False}.items()):
        raise ValueError("settings are incompatible with unchanged greedy decoding policy")
    if any(settings["model"].get(k) != v for k, v in
           {"precision": "bfloat16", "quantization": "none", "device_map": "auto"}.items()):
        raise ValueError("settings are incompatible with unchanged local model loading policy")
    identity = {"model_id": settings["model"]["id"], "revision": args.revision,
                "settings_sha256": file_hash(args.settings),
                "checkpoint_files": checkpoint_identity(args.model),
                "runtime": runtime_identity(torch, transformers)}
    for key, expected in settings["runtime"]["reference_versions"].items():
        if identity["runtime"].get(key) != expected:
            raise ValueError(f"runtime {key} differs from approved settings")
    for key, expected in settings["migration_notes"].get("gpu_runtime_packages_additionally_recorded", {}).items():
        if identity["runtime"].get(key) != expected:
            raise ValueError(f"runtime {key} differs from approved settings")
    if profile is not None:
        for key, value in identity.items():
            if profile["identity"].get(key) != value:
                raise ValueError(f"execution identity drift: {key}")
    kwargs = {"local_files_only": True, "trust_remote_code": False}
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, **kwargs)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, use_safetensors=True, **kwargs)
    model.eval()
    config = model.generation_config
    if config.repetition_penalty != settings["migration_notes"]["inherited_checkpoint_generation"]["repetition_penalty"]:
        raise ValueError("inherited repetition_penalty differs; not silently overridden")
    if config.num_beams != 1 or config.num_return_sequences != 1:
        raise ValueError("only single-beam single-result greedy generation is approved")
    config.do_sample = False
    config.max_new_tokens = args.max_new_tokens
    config.use_cache = True
    config.pad_token_id = tokenizer.eos_token_id
    for key in ("temperature", "top_p", "top_k"):
        setattr(config, key, None)
    identity.update({
        "tokenizer": {"chat_template": tokenizer.get_chat_template(),
                      "vocab_sha256": digest(tokenizer.get_vocab()),
                      "special_ids": sorted(tokenizer.all_special_ids),
                      "eos_token_id": tokenizer.eos_token_id,
                      "model_max_length": tokenizer.model_max_length},
        "generation_config": config.to_dict(),
        "device_map": {k: str(v) for k, v in model.hf_device_map.items()},
        "dtype": str(model.dtype), "quantized": bool(getattr(model, "is_quantized", False)),
        "model_context_limit": model.config.max_position_embeddings,
    })
    if identity["dtype"] != "torch.bfloat16" or identity["quantized"]:
        raise ValueError("only unquantized bfloat16 execution is approved")
    if profile is not None and profile["identity"] != identity:
        raise ValueError("execution identity drift: tokenizer/generation/device placement")
    if checkpoint_identity(args.model) != identity["checkpoint_files"]:
        raise ValueError("checkpoint changed during backend loading")
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    return tokenizer, model, identity
