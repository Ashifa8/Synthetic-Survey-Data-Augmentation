# Checkpoints Directory

This directory stores pre-trained DGNN synthesiser weights after running `train.py`.

## Expected Files

| File | Size (approx.) | Description |
|------|---------------|-------------|
| `ctgan_30k.pkl` | ~50–80 MB | CTGAN trained for 30,000 epochs |
| `tvae_30k.pkl` | ~20–40 MB | TVAE trained for 30,000 epochs |
| `copulagan_30k.pkl` | ~50–80 MB | CopulaGAN trained for 30,000 epochs |

## Generating Checkpoints

Run the full training pipeline to produce these files:

```bash
python train.py --config config.yaml
```

Training time (Google Colab GPU — NVIDIA T4/A100):
- CTGAN : ~20 min 39 sec
- TVAE  : ~41 sec
- CopulaGAN : ~20 min 30 sec

## Loading Checkpoints

To skip retraining and use saved weights directly:

```bash
python train.py --skip-synthesis
```

Or for inference only:

```bash
python inference.py --model copulagan --n-rows 200
python inference.py --model all --n-rows 100 --out-dir results/generated
```

## Notes

- Checkpoints are saved using SDV's built-in `.save()` / `.load()` API (not raw `pickle`).
- Ensure the SDV version matches `requirements.txt` (`sdv>=1.9.0`) when loading weights
  across environments — mismatched versions may cause deserialization errors.
- Checkpoint files are excluded from version control via `.gitignore`.
  Re-run `train.py` on the target machine to regenerate them.
