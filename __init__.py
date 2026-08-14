import logging
import re

import comfy.model_detection
import comfy.sd
import comfy.utils
import folder_paths

from .merge_nodes import AnimaExpandedModelMerge, AnimaExpandedModelMergeBlocks

# Store the original UNet config detection function
orig_detect_unet_config = comfy.model_detection.detect_unet_config

def patched_detect_unet_config(state_dict, key_prefix, metadata=None):
    try:
        dit_config = orig_detect_unet_config(state_dict, key_prefix, metadata)
    except Exception as e:
        raise e

    try:
        if dit_config is not None and dit_config.get("image_model") == "anima":
            max_block = -1
            prefix = f"{key_prefix}blocks."

            # Scan keys to detect actual block count
            for k in state_dict.keys():
                if k.startswith(prefix):
                    parts = k[len(prefix):].split(".")
                    if parts and parts[0].isdigit():
                        max_block = max(max_block, int(parts[0]))

            if max_block != -1:
                actual_blocks = max_block + 1
                if actual_blocks != dit_config.get("num_blocks"):
                    print(f"[Anima 2.9B Patch] Dynamically patching Anima blocks from {dit_config.get('num_blocks')} to {actual_blocks}")
                    dit_config["num_blocks"] = actual_blocks
    except Exception as patch_err:
        print(f"[Anima 2.9B Patch] Warning: Failed to apply dynamic block patch: {patch_err}")

    return dit_config

# Apply patch
try:
    comfy.model_detection.detect_unet_config = patched_detect_unet_config
    print("[Anima 2.9B Patch] Successfully loaded ComfyUI model patch.")
except Exception as init_err:
    print(f"[Anima 2.9B Patch] Error: Failed to load ComfyUI model patch: {init_err}")

_DOTTED_BLOCK_RE = re.compile(
    r"^(?P<prefix>(?:(?:base_model\.model\.)?transformer\.)?"
    r"(?:diffusion_model\.)?(?:blocks|transformer_blocks)\.)"
    r"(?P<index>\d+)(?=\.)"
)
_KOHYA_BLOCK_RE = re.compile(
    r"^(?P<prefix>(?:lora_unet_|lycoris_)?(?:blocks|transformer_blocks)_)(?P<index>\d+)(?=_)"
)

# Anima 2.9B was made by inserting 12 new blocks into the 28-block Anima B1
# network. These are the exact destinations of the original blocks, verified
# by byte-identical weights in the base and 2.9B checkpoints.
_ANIMA_B1_TO_29B_BLOCKS = (
    0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19,
    20, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35, 37, 38, 39,
)


def _block_reference(key):
    """Return the block index match used by common Anima LoRA formats."""
    match = _DOTTED_BLOCK_RE.search(key)
    if match is None:
        match = _KOHYA_BLOCK_RE.search(key)
    return match


def _replace_block(key, match, new_index):
    return f"{key[:match.start('index')]}{new_index}{key[match.end('index'):]}"


def _model_block_count(model):
    indices = []
    for key in model.model.state_dict().keys():
        match = re.search(r"(?:^|\.)diffusion_model\.blocks\.(\d+)\.", key)
        if match:
            indices.append(int(match.group(1)))
    return max(indices) + 1 if indices else 0


def _depth_resample_lora(lora, target_blocks):
    """Stretch a shallower Anima LoRA over a deeper, shape-compatible model.

    Anima B1 LoRAs normally contain 28 blocks while Anima 2.9B contains 40.
    Every target block receives the adapter from the nearest relative depth in
    the source network. Non-block tensors (input/output layers and metadata
    tensors) are kept unchanged.
    """
    grouped = {}
    passthrough = {}
    source_indices = set()
    for key, value in lora.items():
        match = _block_reference(key)
        if match is None:
            passthrough[key] = value
            continue
        source_index = int(match.group("index"))
        source_indices.add(source_index)
        grouped.setdefault(source_index, []).append((key, match, value))

    if not source_indices:
        return lora, 0, 0

    source_blocks = max(source_indices) + 1
    if source_blocks >= target_blocks or target_blocks < 2 or source_blocks < 2:
        return lora, source_blocks, 0

    remapped = dict(passthrough)
    duplicated = 0
    denominator = target_blocks - 1
    for target_index in range(target_blocks):
        # Integer round-to-nearest avoids Python's banker's rounding.
        source_index = (target_index * (source_blocks - 1) + denominator // 2) // denominator
        for key, match, value in grouped.get(source_index, ()):
            remapped[_replace_block(key, match, target_index)] = value
            duplicated += 1

    return remapped, source_blocks, duplicated


def _preserved_block_lora(lora, target_blocks):
    """Map B1 adapters only to their byte-identical blocks in Anima 2.9B."""
    source_indices = {
        int(match.group("index"))
        for key in lora
        if (match := _block_reference(key)) is not None
    }
    if not source_indices:
        return lora, 0, 0

    source_blocks = max(source_indices) + 1
    if source_blocks != len(_ANIMA_B1_TO_29B_BLOCKS) or target_blocks != 40:
        return lora, source_blocks, 0

    remapped = {}
    mapped = 0
    for key, value in lora.items():
        match = _block_reference(key)
        if match is None:
            remapped[key] = value
            continue
        source_index = int(match.group("index"))
        remapped[_replace_block(key, match, _ANIMA_B1_TO_29B_BLOCKS[source_index])] = value
        mapped += 1
    return remapped, source_blocks, mapped


class Anima29BLoraLoader:
    """Apply 28-block Anima LoRAs across the 40-block Anima 2.9B model."""

    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "block_mapping": (["preserved_blocks", "depth_resample", "native_first_blocks"],),
            },
            "optional": {
                "clip": (
                    "CLIP",
                    {
                        "tooltip": (
                            "Optional. Leave disconnected for model-only LoRAs. "
                            "When omitted, strength_clip is ignored."
                        )
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "load_lora"
    CATEGORY = "loaders/Anima 2.9B"
    DESCRIPTION = (
        "Loads an Anima LoRA. preserved_blocks maps the 28 B1 adapters to "
        "their exact matching blocks in Anima 2.9B and skips the 12 inserted "
        "blocks. depth_resample is retained as a compatibility alias for the "
        "safe preserved-block mapping. The CLIP input is optional."
    )

    def load_lora(
        self,
        model,
        lora_name,
        strength_model,
        strength_clip,
        block_mapping,
        clip=None,
    ):
        effective_strength_clip = strength_clip if clip is not None else 0.0
        if strength_model == 0 and effective_strength_clip == 0:
            return model, clip

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        if self.loaded_lora is None or self.loaded_lora[0] != lora_path:
            lora, metadata = comfy.utils.load_torch_file(
                lora_path, safe_load=True, return_metadata=True
            )
            self.loaded_lora = (lora_path, lora, metadata)
        else:
            _, lora, metadata = self.loaded_lora

        target_blocks = _model_block_count(model)
        adapted_lora = lora
        if block_mapping in ("preserved_blocks", "depth_resample") and target_blocks:
            # Earlier releases exposed depth_resample as the default. Treat it
            # as a compatibility alias so saved workflows automatically use
            # the corrected, non-duplicating map.
            adapted_lora, source_blocks, mapped_tensors = _preserved_block_lora(lora, target_blocks)
            if source_blocks and source_blocks < target_blocks:
                logging.info(
                    "[Anima 2.9B LoRA] mapped %s from %d to its preserved blocks in %d-block Anima 2.9B (%d tensors)",
                    lora_name, source_blocks, target_blocks, mapped_tensors,
                )
            elif source_blocks:
                logging.info(
                    "[Anima 2.9B LoRA] %s already has %d blocks; using native mapping",
                    lora_name, source_blocks,
                )
            else:
                logging.warning(
                    "[Anima 2.9B LoRA] No block keys recognized in %s; using native mapping",
                    lora_name,
                )

        return comfy.sd.load_lora_for_models(
            model, clip, adapted_lora, strength_model, effective_strength_clip,
            lora_metadata=metadata,
        )


# ComfyUI Custom Node registrations
NODE_CLASS_MAPPINGS = {
    "Anima29BLoraLoader": Anima29BLoraLoader,
    "AnimaExpandedModelMerge": AnimaExpandedModelMerge,
    "AnimaExpandedModelMergeBlocks": AnimaExpandedModelMergeBlocks,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Anima29BLoraLoader": "Anima 2.9B LoRA Loader",
    "AnimaExpandedModelMerge": "Anima 2.9B + Anima Base Merge",
    "AnimaExpandedModelMergeBlocks": "Anima 2.9B + Anima Base Merge Blocks",
}
