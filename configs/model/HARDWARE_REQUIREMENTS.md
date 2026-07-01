# Hardware Requirements per Model

| Model | Method | Min VRAM | Recommended |
|---|---|---|---|
| TinyLlama 1.1B | LoRA (fp16) | 4 GB | 8 GB GPU or CPU |
| Qwen2.5 1.5B | LoRA (fp16) | 6 GB | 8 GB GPU or CPU |
| Llama 3.1 8B | QLoRA (4-bit) | 8 GB | 16-24 GB GPU |

## Notes

- **TinyLlama / Qwen2.5**: Can also run on CPU (expect ~10-30x slower training)
- **Llama 3.1 8B + QLoRA**: Reduces VRAM from ~16 GB (full fp16) to ~6-8 GB
- `gradient_checkpointing: true` further reduces peak VRAM by ~30-40%
- Inference (not training) requires ~half the VRAM listed above
