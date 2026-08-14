import json
import logging
import re
from pathlib import Path

import comfy.model_management
import comfy.utils


_MANIFEST_PATH = Path(__file__).with_name("expand_manifest.json")
with _MANIFEST_PATH.open("r", encoding="utf-8") as manifest_file:
    EXPAND_MANIFEST = json.load(manifest_file)

OLD_BLOCK_COUNT = int(EXPAND_MANIFEST["old_block_count"])
NEW_BLOCK_COUNT = int(EXPAND_MANIFEST["new_block_count"])
INSERTION_POSITIONS = frozenset(int(x) for x in EXPAND_MANIFEST["insertion_positions"])
def _build_preserved_block_map():
    mapping = []
    next_base = 0
    for expanded_index in range(NEW_BLOCK_COUNT):
        if expanded_index in INSERTION_POSITIONS:
            mapping.append(None)
        else:
            mapping.append(next_base)
            next_base += 1
    if next_base != OLD_BLOCK_COUNT:
        raise RuntimeError(
            f"Invalid expansion manifest: mapped {next_base} preserved blocks, "
            f"expected {OLD_BLOCK_COUNT}"
        )
    return tuple(mapping)


PRESERVED_BLOCK_TO_BASE = _build_preserved_block_map()
_MAIN_BLOCK_RE = re.compile(r"^diffusion_model\.blocks\.(\d+)\.")


def _block_count(keys):
    indices = []
    for key in keys:
        match = _MAIN_BLOCK_RE.match(key)
        if match:
            indices.append(int(match.group(1)))
    return max(indices) + 1 if indices else 0


def _source_key(expanded_key):
    match = _MAIN_BLOCK_RE.match(expanded_key)
    if match is None:
        return expanded_key, None, None
    expanded_index = int(match.group(1))
    if expanded_index >= len(PRESERVED_BLOCK_TO_BASE):
        return None, expanded_index, None
    base_index = PRESERVED_BLOCK_TO_BASE[expanded_index]
    if base_index is None:
        return None, expanded_index, None
    start, end = match.span(1)
    return (
        f"{expanded_key[:start]}{base_index}{expanded_key[end:]}",
        expanded_index,
        base_index,
    )


def _base_tensor(key_patch):
    """Return the unpatched tensor at the head of get_key_patches()."""
    return key_patch[0][0]


class AnimaExpandedModelMerge:
    """Architecture-aware merge between Anima Base and Anima 2.9B."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anima_2_9b": (
                    "MODEL",
                    {"tooltip": "Anima 2.9B. The 12 expanded blocks are kept from this model."},
                ),
                "anima_base": (
                    "MODEL",
                    {"tooltip": "Anima Base or an Anima Base fine-tune with 28 blocks."},
                ),
                "ratio": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "merge"
    CATEGORY = "model/merging/Anima"
    DESCRIPTION = (
        "Merges Anima Base with Anima 2.9B using the correct 28-to-40 block map."
    )

    def merge(
        self,
        anima_2_9b,
        anima_base,
        ratio,
        **legacy_inputs,
    ):
        expanded_patches = anima_2_9b.get_key_patches("diffusion_model.")
        base_patches = anima_base.get_key_patches("diffusion_model.")

        expanded_blocks = _block_count(expanded_patches.keys())
        base_blocks = _block_count(base_patches.keys())
        if expanded_blocks != NEW_BLOCK_COUNT:
            raise ValueError(
                f"anima_2_9b must have {NEW_BLOCK_COUNT} blocks; got {expanded_blocks}."
            )
        if base_blocks != OLD_BLOCK_COUNT:
            raise ValueError(
                f"anima_base must have {OLD_BLOCK_COUNT} blocks; got {base_blocks}."
            )

        if ratio == 1.0:
            return (anima_2_9b.clone(),)

        merged = anima_2_9b.clone()
        progress = comfy.utils.ProgressBar(len(expanded_patches))

        for expanded_key, expanded_patch in expanded_patches.items():
            comfy.model_management.throw_exception_if_processing_interrupted()
            source_key, expanded_index, _ = _source_key(expanded_key)
            # The 12 blocks inserted by Anima 2.9B have no counterpart in
            # Anima Base and remain untouched, like unmatched keys in
            # ComfyUI's ModelMergeSimple.
            if source_key is None or source_key not in base_patches:
                progress.update(1)
                continue

            expanded_tensor = _base_tensor(expanded_patch)
            source_tensor = _base_tensor(base_patches[source_key])
            if expanded_tensor.shape != source_tensor.shape:
                logging.warning(
                    "[Anima Expanded Merge] Shape mismatch: %s %s != %s %s",
                    expanded_key,
                    tuple(expanded_tensor.shape),
                    source_key,
                    tuple(source_tensor.shape),
                )
                progress.update(1)
                continue

            merged.add_patches(
                {expanded_key: base_patches[source_key]},
                1.0 - ratio,
                ratio,
            )
            progress.update(1)

        return (merged,)


class AnimaExpandedModelMergeBlocks:
    """Per-block merge with the Anima Base blocks mapped into Anima 2.9B."""

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "anima_2_9b": (
                "MODEL",
                {"tooltip": "Anima 2.9B with 40 blocks."},
            ),
            "anima_base": (
                "MODEL",
                {"tooltip": "Anima Base or an Anima Base fine-tune with 28 blocks."},
            ),
        }
        for expanded_index, base_index in enumerate(PRESERVED_BLOCK_TO_BASE):
            name = f"blocks.{expanded_index}."
            if base_index is None:
                required[name] = (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 1.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Exclusive Anima 2.9B block; fixed at 1.0.",
                    },
                )
            else:
                required[name] = (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            f"Anima 2.9B block {expanded_index} / "
                            f"Anima Base block {base_index}."
                        ),
                    },
                )
        return {"required": required}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "merge"
    CATEGORY = "model/merging/Anima"
    DESCRIPTION = (
        "Merges each preserved Anima Base block into its matching Anima 2.9B "
        "block. The 12 exclusive Anima 2.9B blocks are visible and fixed."
    )

    def merge(self, anima_2_9b, anima_base, **ratios):
        expanded_patches = anima_2_9b.get_key_patches("diffusion_model.")
        base_patches = anima_base.get_key_patches("diffusion_model.")

        expanded_blocks = _block_count(expanded_patches.keys())
        base_blocks = _block_count(base_patches.keys())
        if expanded_blocks != NEW_BLOCK_COUNT:
            raise ValueError(
                f"anima_2_9b must have {NEW_BLOCK_COUNT} blocks; got {expanded_blocks}."
            )
        if base_blocks != OLD_BLOCK_COUNT:
            raise ValueError(
                f"anima_base must have {OLD_BLOCK_COUNT} blocks; got {base_blocks}."
            )

        merged = anima_2_9b.clone()
        progress = comfy.utils.ProgressBar(len(expanded_patches))

        for expanded_key, expanded_patch in expanded_patches.items():
            comfy.model_management.throw_exception_if_processing_interrupted()
            source_key, expanded_index, _ = _source_key(expanded_key)
            if expanded_index is None or source_key is None or source_key not in base_patches:
                progress.update(1)
                continue

            ratio = float(ratios.get(f"blocks.{expanded_index}.", 1.0))
            if ratio >= 1.0:
                progress.update(1)
                continue

            expanded_tensor = _base_tensor(expanded_patch)
            source_tensor = _base_tensor(base_patches[source_key])
            if expanded_tensor.shape != source_tensor.shape:
                logging.warning(
                    "[Anima Block Merge] Shape mismatch: %s %s != %s %s",
                    expanded_key,
                    tuple(expanded_tensor.shape),
                    source_key,
                    tuple(source_tensor.shape),
                )
                progress.update(1)
                continue

            merged.add_patches(
                {expanded_key: base_patches[source_key]},
                1.0 - ratio,
                ratio,
            )
            progress.update(1)

        return (merged,)


class AnimaExpandedModelMergeSections:
    """Three-section merge for the 40-block Anima 2.9B architecture."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anima_2_9b": (
                    "MODEL",
                    {"tooltip": "Anima 2.9B with 40 blocks."},
                ),
                "anima_base": (
                    "MODEL",
                    {"tooltip": "Anima Base or an Anima Base fine-tune with 28 blocks."},
                ),
                "input": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Anima 2.9B blocks 0-12 / Anima Base blocks 0-8.",
                    },
                ),
                "middle": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Anima 2.9B blocks 13-26 / Anima Base blocks 9-18.",
                    },
                ),
                "out": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Anima 2.9B blocks 27-39 / Anima Base blocks 19-27.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "merge"
    CATEGORY = "model/merging/Anima"
    DESCRIPTION = (
        "Merges the input, middle, and output thirds of Anima Base into the "
        "matching preserved blocks of Anima 2.9B."
    )

    @staticmethod
    def _section_ratio(expanded_index, input_ratio, middle_ratio, out_ratio):
        if expanded_index <= 12:
            return input_ratio
        if expanded_index <= 26:
            return middle_ratio
        return out_ratio

    def merge(self, anima_2_9b, anima_base, input, middle, out):
        expanded_patches = anima_2_9b.get_key_patches("diffusion_model.")
        base_patches = anima_base.get_key_patches("diffusion_model.")

        expanded_blocks = _block_count(expanded_patches.keys())
        base_blocks = _block_count(base_patches.keys())
        if expanded_blocks != NEW_BLOCK_COUNT:
            raise ValueError(
                f"anima_2_9b must have {NEW_BLOCK_COUNT} blocks; got {expanded_blocks}."
            )
        if base_blocks != OLD_BLOCK_COUNT:
            raise ValueError(
                f"anima_base must have {OLD_BLOCK_COUNT} blocks; got {base_blocks}."
            )

        merged = anima_2_9b.clone()
        progress = comfy.utils.ProgressBar(len(expanded_patches))

        for expanded_key, expanded_patch in expanded_patches.items():
            comfy.model_management.throw_exception_if_processing_interrupted()
            source_key, expanded_index, _ = _source_key(expanded_key)
            if expanded_index is None or source_key is None or source_key not in base_patches:
                progress.update(1)
                continue

            ratio = self._section_ratio(expanded_index, input, middle, out)
            if ratio >= 1.0:
                progress.update(1)
                continue

            expanded_tensor = _base_tensor(expanded_patch)
            source_tensor = _base_tensor(base_patches[source_key])
            if expanded_tensor.shape != source_tensor.shape:
                logging.warning(
                    "[Anima Section Merge] Shape mismatch: %s %s != %s %s",
                    expanded_key,
                    tuple(expanded_tensor.shape),
                    source_key,
                    tuple(source_tensor.shape),
                )
                progress.update(1)
                continue

            merged.add_patches(
                {expanded_key: base_patches[source_key]},
                1.0 - ratio,
                ratio,
            )
            progress.update(1)

        return (merged,)
