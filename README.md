# test45 — test33 with NequIP irreps up to l=3

`test45.py` is the supercomputer version of test33 (DM2 NequIP denoising autoencoder, β-cristobalite SiO2
2×2×2 supercell, 1536 atoms; resumable training and generation) with the NequIP irreps raised to l≤3:

| | test33 | test45 |
|---|---|---|
| `irreps_hidden` | `64x0e + 32x1e` | `64x0e + 32x1e + 16x2e + 8x3e` |
| `irreps_edge` | `4x0e + 4x1e + 2x2e` | `4x0e + 4x1e + 2x2e + 2x3e` |
| parameters | 572,768 | 990,432 |

Only the irreps and default paths differ (checkpoint `checkpoints/test45_*.pt`, output `test45-output/`,
crystal data bundled in `data/`). Base script: `DM2/demo/demo_generating/test33.py`.

## スパコンでの実行（test33 と同じ: Singularity + DM2 をバインド）

DM2 (graphite) はイメージに入れず、ソースを `PYTHONPATH` で渡します。

```bash
git clone https://github.com/haru2225/test45.git
git clone https://github.com/digital-synthesis-lab/DM2.git      # test45 と同じ階層に置く
cd test45

module load singularity     # サイトのモジュール名に合わせる
singularity build test45-pytorch-2.5.0-cu124.sif Singularity.def
# Apptainer: apptainer build test45-pytorch-2.5.0-cu124.sif Singularity.def

qsub -P PROJECT_ID run_test45.pbs
# DM2 が別の場所なら: qsub -P PROJECT_ID -v DM2_ROOT=/abs/path/DM2 run_test45.pbs
```

- 学習・生成とも途中保存されます(walltime 切れ = 時間予算 11.5 h で保存して終了)。同じ `qsub` を再投入すると続きから再開します。
- 最終 checkpoint `checkpoints/test45_sio2_crystal_2x2x2_nequip.pt` があれば学習をスキップします。
- 主な環境変数: `NUM_UPDATES`, `DUPLICATE`, `GEN_NOISY_STEPS`, `GEN_POLISH_STEPS`, `TIME_BUDGET_HOURS`,
  `SIF_IMAGE`, `OUTPUT_DIR`, `CHECKPOINT_PATH`。
- 出力: `test45-output/` (`metrics.json`, 結合長/角度/損失の図, `final_structure.extxyz`, Si のみの CG 軌道)。

`DM2` は MIT (Tim Hsu; Digital Synthesis Lab @ UCLA)。
