This patches the ComfyUI model loader on start-up so Anima checkpoints with
additional transformer blocks can be detected.

It also provides **Anima 2.9B LoRA Loader**. Standard Anima B1 LoRAs contain
28 blocks, while Anima 2.9B contains 40. The default `preserved_blocks` mode
maps the adapters only to the byte-identical B1 blocks retained in 2.9B and
leaves its 12 inserted blocks untouched. The former `depth_resample` value is
accepted as a compatibility alias, so existing workflows receive the fix.
`native_first_blocks` keeps ComfyUI's original behavior for comparison.

The `clip` input is optional. Leave it disconnected for the usual model-only
Anima LoRAs; in that case `strength_clip` is ignored. Connect it only when the
LoRA actually contains text-encoder weights and use the `clip` output.

## Anima 2.9B + Anima Base Merge

This node merges Anima Base with Anima 2.9B using the correct block mapping.

Connect Anima 2.9B to `anima_2_9b` and Anima Base to `anima_base`. Like ComfyUI's
`ModelMergeSimple`, `ratio` 1.0 keeps Anima 2.9B and 0.0 applies Anima Base to
the compatible weights. The 12 blocks exclusive to Anima 2.9B remain unchanged.

Connect the model output to **Save Diffusion Model** to save the result.

**Anima 2.9B + Anima Base Merge Blocks** provides one ratio for each of the
40 Anima 2.9B blocks. Only the 28 blocks that exist in Anima Base can be
changed; the 12 exclusive Anima 2.9B blocks are shown fixed at 1.0.
