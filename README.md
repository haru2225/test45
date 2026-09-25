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

## スパコンでの実行（Singularity / Apptainer, PBS, GPU 1台）

```bash
git clone https://github.com/haru2225/test45.git && cd test45

# 1. コンテナのビルド（初回のみ。ビルド可能な Linux 環境とネット接続が必要）
module load singularity            # サイトのモジュール名に合わせる（apptainer の場合も同様）
singularity build test45.sif Singularity.def
# Apptainer: apptainer build test45.sif Singularity.def
# ビルドできないサイトでは、手元の Linux で作った test45.sif を scp する

# 2. GPU での動作確認（任意）: 更新数を減らして短時間で
sed -i 's/^NUM_UPDATES = 30_000/NUM_UPDATES = 300/' test45.py
qsub -P PROJECT_ID -l walltime=00:30:00 run_test45.pbs
rm -f checkpoints/*.pt && sed -i 's/^NUM_UPDATES = 300/NUM_UPDATES = 30_000/' test45.py

# 3. 本番（walltime 20 時間）
qsub -P PROJECT_ID run_test45.pbs
qstat -u $USER
tail -f test45.o*
```

コードは実行時にバインドマウントされるので、`test45.py` を編集してもイメージの再ビルドは不要です。
DM2（graphite）はコミット `1c9b6d3` に固定してイメージに入れています。

チェックポイント `checkpoints/test45_sio2_crystal_2x2x2_nequip.pt` があれば学習をスキップして生成に進みます
（途中再開は未実装: walltime 切れで途中の pt が残った場合は、それが「学習済み」扱いになるので注意）。
再学習する場合は削除してください。出力: `figures/`（損失・結合長・角度）、`results/`（Si のみの生成軌道）。

## ローカル / venv での実行

```bash
python -m venv ~/venvs/test45 && source ~/venvs/test45/bin/activate
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu124   # CUDA に合わせる
pip install -r requirements.txt
git clone https://github.com/digital-synthesis-lab/DM2.git ~/DM2 && pip install -e ~/DM2 --no-deps
python test45.py
```

`DM2` は MIT（Tim Hsu; Digital Synthesis Lab @ UCLA）。
