# test45 — test33 with NequIP irreps up to l=3

`test45.py` is `test33.py` (DM2-style NequIP denoising autoencoder, β-cristobalite SiO2 2×2×2 supercell,
1536 atoms) with the NequIP irreps raised to l≤3:

| | test33 | test45 |
|---|---|---|
| `irreps_hidden` | `64x0e + 32x1e` | `64x0e + 32x1e + 16x2e + 8x3e` |
| `irreps_edge` | `4x0e + 4x1e + 2x2e` | `4x0e + 4x1e + 2x2e + 2x3e` |
| parameters | 572,768 | 990,432 |

Everything else (training on one duplicated reference structure with fresh rattle noise σ≤0.75 Å,
annealed denoising 2900 + 100 polish steps from σ=1.0 Å) is unchanged. Paths were made relative to
the file: input `data/silica_beta_cristobalite_init.data`, outputs in `figures/`, `checkpoints/`, `results/`.

Note: generation starts from the ideal crystal (as in test33), not from random positions.

## Setup and run

```bash
git clone https://github.com/haru2225/test45.git && cd test45
python -m venv ~/venvs/test45 && source ~/venvs/test45/bin/activate

# torch: match your CUDA (example: CUDA 12.4)
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# DM2 provides the `graphite` package (NequIP, RattleParticles, ...)
git clone https://github.com/digital-synthesis-lab/DM2.git ~/DM2 && pip install -e ~/DM2 --no-deps

python test45.py          # ~30,000 updates; use a GPU
# on a PBS cluster: qsub -P PROJECT_ID run_test45.pbs
```

`DM2` is MIT-licensed (Tim Hsu; Digital Synthesis Lab @ UCLA).
If `checkpoints/test45_sio2_crystal_2x2x2_nequip.pt` exists, training is skipped and generation runs
from it; delete it to retrain. Outputs: loss/bond/angle figures in `figures/`, the Si-only generated
trajectory in `results/crystal_generated_test45_2x2x2/`.
