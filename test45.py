# =============================================================================
# test45.py -- test33.py with NequIP irreps raised to l<=3 (hidden 64x0e+32x1e+16x2e+8x3e,
# edge 4x0e+4x1e+2x2e+2x3e; test33: l<=1 hidden / l<=2 edge). Nothing else changed except paths
# (made relative to this file, data bundled in data/, outputs in results/). Original test33 header follows.
# test33.py -- test32.py's DM2-based pipeline (NequIP denoising autoencoder,
# graphite package reused directly, CGMD bridge), retargeted from GLASS to
# 3D SiO2 CRYSTAL (beta-cristobalite) generation -- on request:
# "DM2と同じ設計で結晶構造の生成...入力を3Dsilca結晶にした". Also see
# test33_quickcheck.py, a reduced-scale copy sized to finish quickly on a
# REAL CUDA GPU (on request: "スパコンですぐ終わるようなCUDAでのうまくいって
# るか見るtestケースの実装もしたい") -- distinct from this session's usual
# local CPU/MPS smoke tests, which only check for code bugs; the quickcheck
# variant is meant to give an actual signal on whether generation quality is
# trending toward the reference on real hardware, before committing to the
# full ~30,000-update run.
#
# WHY CRYSTAL SHOULD BE EASIER, NOT HARDER, FOR THIS DENOISING SCHEME: DM2's
# training trick (duplicate a few reference structures, learn to reverse
# FRESH random noise applied to them every step) rewards regularity -- a
# crystal has ONE highly symmetric structure with (nearly) IDENTICAL local
# environments at every site, whereas glass has genuine local disorder (ring-
# size variation) the network also has to capture. This is the opposite of
# this session's own from-scratch SiO2 crystal work (test14.py-test20.py),
# where the LJ-crystal/SiO2-crystal architectures suffered badly from
# same-species collapse (Si-Si/O-O atoms landing on top of each other,
# diagnosed in test17.py -- see that file's module docstring) precisely
# BECAUSE every same-species atom's local environment looked identical to
# the network, giving it no signal to keep them apart when noise placed two
# of them close together. DM2's NequIP, unlike this session's simpler
# encoders, has genuine higher-order (l>0, via e3nn irreps like '32x1e')
# geometric features from real angular/tensor-product convolutions over an
# actual periodic radius graph -- whether that's enough to avoid the SAME
# same-species-collapse failure mode when starting from pure noise (as
# opposed to DM2's own generation protocol, which anneals from a
# MODERATE-sigma rattle of an ALREADY-correct structure, not a uniform-
# random start) is an open, genuinely interesting question this file lets
# the user actually test, not something assumed to be fixed by architecture.
#
# REFERENCE DATA CHANGE (originally the only substantive change from
# test32.py): instead of DM2's own 3000-atom glass `.dat` files, this file
# uses `md/silica_beta_cristobalite_init.data` -- ALREADY a LAMMPS-data-
# format 192-atom (64 Si + 128 O) beta-cristobalite supercell, verified
# readable by ASE exactly as DM2's loader expects (`ase.io.read(...,
# format='lammps-data')`, no conversion needed). This is the SAME initial
# structure used to seed this repo's own existing LAMMPS MD trajectories
# (md/traj_0.lammpstrj etc.), which the EXISTING crystal CG-SiO2 pipeline
# (src/scoremd's SilicaCGDataset, data/silica_cg/crystal_md) already
# consumes -- keeping this file's crystal consistent with that lineage
# rather than reusing test14.py's independently-built idealized structure.
#
# 2x2x2 SUPERCELL FIX (revised twice: the raw 192-atom cell's first full run
# generated badly wrong structures -- narrow bond/angle histograms collapsed
# onto ~4 discrete, non-tetrahedral values instead of the reference's
# continuous spread around 108.82 deg. Root cause #1, CONFIRMED: the cell is
# only 13.573 A across, L/2=6.79 A, SMALLER than DM2's own LARGE_CUTOFF=10.0
# A (tuned for their 3000-atom, ~36 A glass box) -- verified with ASE's
# primitive_neighbor_list that 69.5% of atom pairs get connected through 2-4
# DISTINCT periodic images simultaneously at cutoff=10.0 (a genuine minimum-
# image-convention violation; 0% at cutoff<=6.5). A first fix attempt just
# shrank LARGE_CUTOFF to 6.5 (keeping the 192-atom cell) -- confirmed to
# eliminate the duplicate-image corruption, but reported to leave generation
# equally broken, meaning the duplicate-image bug was real but NOT the whole
# story. Replicating the unit cell 2x2x2 instead (CRYSTAL_REPLICATE below)
# gives a ~27.1 A cell (L/2=13.6 A) and restores DM2's own UNMODIFIED
# defaults (CUTOFF=5.0/LARGE_CUTOFF=10.0, same as test32.py's glass) with
# their full original rattle margin (5.0 A, vs only 1.5 A when just shrinking
# LARGE_CUTOFF on the small cell) -- eliminates the same duplicate-image
# issue AND gives the network a receptive field (num_convs=3 * CUTOFF=5.0 =
# ~15 A) that no longer spans essentially the entire periodic cell, so
# generation is no longer implicitly forced to reconstruct the whole box's
# long-range phase/tiling from a receptive field roughly as big as the box
# itself. 1536 atoms (192*8) is still well under test32.py's 3000-atom
# glass, so NUM_UPDATES/DUPLICATE/BATCH_SIZE are kept unchanged -- the epoch
# count only depends on DUPLICATE and BATCH_SIZE (via TRAIN_EPOCHS below),
# not atom count.
#
# EVERYTHING ELSE (NequIP architecture/hyperparameters, RattleParticles/
# DownselectEdges transforms, the training loop, the two-phase annealed-
# denoising generation procedure, the Si-only CGMD bridge writing into
# data/silica_cg/ in SilicaCGDataset's own format) is UNCHANGED from
# test32.py -- see that file's module docstring for the full DM2 lineage
# and method description.
# =============================================================================

import time
import warnings
from functools import partial
from pathlib import Path

import ase.io
import matplotlib.pyplot as plt
import numpy as np
import torch

# e3nn==0.4.4 (DM2's pinned version) loads its own Wigner-3j constants via a
# bare `torch.load(...)` with no `weights_only` argument -- PyTorch >=2.6
# changed that default to True, which then rejects the `slice` objects
# stored in e3nn's constants.pt. This is e3nn's own internal loading, not
# anything from this repo, so allowlisting the one flagged global (not a
# blanket weights_only=False) is the narrowest fix. Must run before the
# `from graphite...` imports below, since those trigger e3nn's import-time
# `torch.load` call.
torch.serialization.add_safe_globals([slice])

from ase.neighborlist import primitive_neighbor_list
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from tqdm import trange

from graphite.nn.basis import bessel
from graphite.nn.models.e3nn_nequip import NequIP
from graphite.transforms import DownselectEdges, RattleParticles

warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

torch.manual_seed(1337)
run_started_at = time.perf_counter()

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Using device:", device)
if device.type != "cuda":
    print(
        "WARNING: no CUDA GPU detected -- this file's default hyperparameters (NUM_UPDATES=30_000) "
        "are sized for a real GPU/supercomputer. The 1536-atom (2x2x2) crystal is cheaper than "
        "test32.py's 3000-atom glass, but still slow to train fully on CPU/MPS. Fine for a quick "
        "smoke test, not for the real run."
    )

TOY_MODEL_DIR = Path(__file__).resolve().parent
DM2_DIR = None  # unused; graphite is imported from the installed package (see README)  # fixed location, not
# relative to __file__ -- this file's own smoke-test copies (this session's
# established pattern of writing a reduced-scale copy elsewhere to test
# before the real run) would otherwise resolve DM2_DIR to wherever the copy
# lives, not where DM2 actually is.
FIGURES_DIR = TOY_MODEL_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = TOY_MODEL_DIR / "checkpoints" / "test45_sio2_crystal_2x2x2_nequip.pt"  # new
# name -- deliberately distinct from both the original broken checkpoint
# (LARGE_CUTOFF=10.0 on the 192-atom cell) and the cutoff-shrink attempt
# (LARGE_CUTOFF=6.5, same 192-atom cell), so this run trains fresh on the
# replicated cell instead of the CHECKPOINT_PATH.exists() branch silently
# resuming old, differently-shaped weights.
CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)


def save_figure(name: str) -> Path:
    path = FIGURES_DIR / name
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print("Saved figure:", path)
    return path


# =============================================================================
# Graph construction + embedding -- DM2's own demo code, reused verbatim
# (ase_graph/InitialEmbedding/PeriodicStructureDataset are defined directly
# in DM2/demo/demo_training/denoiser_train_unconditional.py, not inside the
# graphite package itself, so they're reproduced here rather than imported).
# =============================================================================


def ase_graph(data: Data, cutoff: float) -> Data:
    """Real periodic radius graph via ASE's neighbor list (sparse, cutoff-
    based -- not this session's other files' dense all-pairs approach)."""
    i, j, D = primitive_neighbor_list(
        "ijD", cutoff=cutoff, pbc=data.pbc, cell=data.cell, positions=data.pos.numpy(), numbers=data.numbers,
    )
    data.edge_index = torch.tensor(np.stack((i, j)), dtype=torch.long)
    data.edge_attr = torch.tensor(D, dtype=torch.float)
    return data


class PeriodicStructureDataset(Dataset):
    """DM2's own trick: a handful of KNOWN-GOOD reference structures,
    duplicated many times -- data diversity for training comes from fresh
    RattleParticles noise every step, not from having many distinct base
    structures."""

    def __init__(self, atoms_list, large_cutoff: float, duplicate: int = 128):
        super().__init__(None, transform=None, pre_transform=None)
        self.dataset = []
        for atoms in atoms_list:
            x = LabelEncoder().fit_transform(atoms.numbers)
            data = Data(
                x=torch.tensor(x).long(),
                pos=torch.tensor(atoms.positions).float(),
                cell=atoms.cell,
                pbc=atoms.pbc,
                numbers=atoms.numbers,
            )
            data = ase_graph(data, large_cutoff)
            self.dataset.append(data)
        self.dataset = [d.clone() for d in self.dataset for _ in range(duplicate)]

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset[idx]


class InitialEmbedding(nn.Module):
    """Species embedding (x2, matching NequIP's node_x/node_z convention)
    + Bessel radial-basis edge embedding. NO sigma/time conditioning --
    the denoiser is trained sigma-agnostic (see module docstring)."""

    def __init__(self, num_species: int, cutoff: float):
        super().__init__()
        self.embed_node_x = nn.Embedding(num_species, 8)
        self.embed_node_z = nn.Embedding(num_species, 8)
        self.embed_edge = partial(bessel, start=0.0, end=cutoff, num_basis=16)

    def forward(self, data):
        data.h_node_x = self.embed_node_x(data.x)
        data.h_node_z = self.embed_node_z(data.x)
        data.h_edge = self.embed_edge(data.edge_attr.norm(dim=-1))
        return data


def loss_fn(model, data):
    pred_dx = model(data)
    return torch.nn.functional.mse_loss(pred_dx, data.dx)


def train_one_epoch(loader, model, optimizer, device, rattle_particles, downselect_edges, pin_memory):
    model.train()
    total_loss = 0.0
    for data in loader:
        optimizer.zero_grad(set_to_none=True)
        data = data.to(device, non_blocking=pin_memory)
        data = rattle_particles(data)
        data = downselect_edges(data)
        loss = loss_fn(model, data)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def valid_one_epoch(loader, model, device, rattle_particles, downselect_edges, pin_memory):
    model.eval()
    total_loss = 0.0
    for data in loader:
        data = data.to(device, non_blocking=pin_memory)
        data = rattle_particles(data)
        data = downselect_edges(data)
        loss = loss_fn(model, data)
        total_loss += loss.item()
    return total_loss / len(loader)


# =============================================================================
# Configuration -- DM2's own recommended values (denoiser_train_unconditional.py),
# unchanged (this file targets a real GPU/supercomputer, per the user's own
# instruction -- not scaled down for local hardware).
# =============================================================================

NUM_SPECIES = 2
PIN_MEMORY = (device.type == "cuda")
NUM_WORKERS = 0
BATCH_SIZE = 16
LARGE_CUTOFF = 10.0  # DM2's own default, restored -- see module docstring's "2x2x2 SUPERCELL FIX"
CUTOFF = 5.0
LEARN_RATE = 2e-4
NUM_UPDATES = 30_000
TRAIN_RATIO = 0.9
SIGMA_MAX = 0.75
DUPLICATE = 128

# test33.py: real beta-cristobalite crystal (192-atom unit cell, replicated
# 2x2x2 -- see module docstring's "2x2x2 SUPERCELL FIX"), not DM2's own
# glass data -- this is this repo's OWN LAMMPS initial structure (already
# used to seed md/traj_0.lammpstrj etc.), verified readable in DM2's own
# expected format (see module docstring).
CRYSTAL_DATA_PATH = TOY_MODEL_DIR / "data" / "silica_beta_cristobalite_init.data"  # fixed location, see DM2_DIR's comment above for why
CRYSTAL_REPLICATE = (2, 2, 2)  # 13.573 A unit cell -> ~27.146 A supercell, L/2=13.6 A > LARGE_CUTOFF=10.0

crystal_unit = ase.io.read(str(CRYSTAL_DATA_PATH), format="lammps-data")
crystal_atoms = crystal_unit.repeat(CRYSTAL_REPLICATE)
ideal_atoms_list = [crystal_atoms]
print(f"Loaded 1 reference SiO2 crystal structure: unit cell {len(crystal_unit)} atoms "
      f"({np.diag(np.asarray(crystal_unit.cell))} A), replicated {CRYSTAL_REPLICATE} -> "
      f"{len(crystal_atoms)} atoms ({np.diag(np.asarray(crystal_atoms.cell))} A) from {CRYSTAL_DATA_PATH}.")

init_embed = InitialEmbedding(num_species=NUM_SPECIES, cutoff=CUTOFF)
model = NequIP(
    init_embed=init_embed,
    irreps_node_x="8x0e",
    irreps_node_z="8x0e",
    irreps_hidden="64x0e + 32x1e + 16x2e + 8x3e",  # test45: l up to 3 (test33: up to l=1)
    irreps_edge="4x0e + 4x1e + 2x2e + 2x3e",  # test45: l up to 3 (test33: up to l=2)
    irreps_out="1x1e",
    num_convs=3,
    radial_neurons=[16, 64],
    num_neighbors=12,
).to(device)

rattle_particles = RattleParticles(sigma_max=SIGMA_MAX)
downselect_edges = DownselectEdges(cutoff=CUTOFF)

dataset = PeriodicStructureDataset(ideal_atoms_list, large_cutoff=LARGE_CUTOFF, duplicate=DUPLICATE)
num_train = int(TRAIN_RATIO * len(dataset))
num_valid = len(dataset) - num_train
ds_train, ds_valid = torch.utils.data.random_split(dataset, [num_train, num_valid])
train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
valid_loader = DataLoader(ds_valid, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

num_samples = len(dataset)
TRAIN_EPOCHS = max(1, int(NUM_UPDATES / (num_samples / BATCH_SIZE)))
print(f"{TRAIN_EPOCHS} epochs -> ~{NUM_UPDATES} optimizer updates "
      f"({num_samples} samples, batch {BATCH_SIZE}).")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARN_RATE)

if CHECKPOINT_PATH.exists():
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    print(f"Resumed from existing checkpoint: {CHECKPOINT_PATH} -- skipping training.")
else:
    print("No existing checkpoint found -- training from scratch.")
    L_train, L_valid = [], []
    checkpoint_every = max(1, TRAIN_EPOCHS // 20)
    for epoch in trange(TRAIN_EPOCHS, desc="Training NequIP denoiser"):
        train_loss = train_one_epoch(train_loader, model, optimizer, device, rattle_particles, downselect_edges, PIN_MEMORY)
        valid_loss = valid_one_epoch(valid_loader, model, device, rattle_particles, downselect_edges, PIN_MEMORY)
        L_train.append(train_loss)
        L_valid.append(valid_loss)
        if (epoch + 1) % checkpoint_every == 0:
            print(f"  epoch {epoch + 1}/{TRAIN_EPOCHS}  train={train_loss:.4f}  valid={valid_loss:.4f}  "
                  f"elapsed={(time.perf_counter() - run_started_at) / 60:.1f}min")
            torch.save(model.state_dict(), CHECKPOINT_PATH)  # periodic checkpoint -- long run, supercomputer job can be preempted

    plt.figure(figsize=(10, 3.5))
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(L_train, label="train")
    ax1.plot(L_valid, label="valid")
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax2 = plt.subplot(1, 2, 2)
    ax2.semilogy(L_train, label="train")
    ax2.semilogy(L_valid, label="valid")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    save_figure("test45_sio2_crystal_2x2x2_loss.png")

    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"Saved model checkpoint: {CHECKPOINT_PATH}")


# =============================================================================
# Generation -- DM2's own two-phase annealed denoising, reused directly
# (demo_generating/denoise_generate_unconditional.py's
# denoise_snapshot_with_noise_gpu / denoise_snapshot_gpu, ported here without
# the CUDA-only hardcoding so it also runs -- slowly -- on CPU/MPS).
# =============================================================================


@torch.no_grad()
def denoise_with_noise(atoms, model, steps: int, max_sigma: float, cutoff: float, large_cutoff: float):
    """Annealed-Langevin-style generation: repeatedly rattle at a DECREASING
    sigma, then subtract the model's predicted correction (which itself
    includes undoing that step's fresh noise, `model(data) + noisy_data.dx`)."""
    model.eval()
    sigmas = torch.linspace(max_sigma, 0.001, steps, device=device)
    x = torch.tensor(LabelEncoder().fit_transform(atoms.numbers), device=device).long()
    data = Data(x=x, pos=torch.tensor(atoms.positions, dtype=torch.float32, device=device),
                cell=torch.tensor(np.asarray(atoms.cell), dtype=torch.float32, device=device), pbc=atoms.pbc, numbers=atoms.numbers)

    pos_traj = [atoms.positions.copy()]
    for i, sigma in enumerate(sigmas, 1):
        data = ase_graph_gpu(data, cutoff=large_cutoff)
        # DM2's own per-step rattle: fresh noise AT THIS STEP'S sigma (not a
        # random range like training's RattleParticles -- generation anneals
        # a KNOWN, decreasing sigma schedule). `model(data)` sees the CLEAN
        # (not-yet-rattled) current position -- matching DM2's own
        # `model(data) + noisy_data.dx` exactly (noisy_data there is only
        # ever used for its freshly-drawn `dx`, never fed to the model).
        dx = sigma * torch.randn_like(data.pos)
        disp = model(data) + dx
        data.pos = data.pos - disp
        pos_traj.append(data.pos.detach().cpu().numpy())
        if i % max(1, steps // 10) == 0:
            print(f"  denoise-with-noise step {i}/{steps} (sigma={sigma.item():.4f})")
    return pos_traj


@torch.no_grad()
def denoise_polish(atoms, model, steps: int, cutoff: float, large_cutoff: float):
    """Pure iterative refinement, no added noise -- DM2's 'polish' phase."""
    model.eval()
    x = torch.tensor(LabelEncoder().fit_transform(atoms.numbers), device=device).long()
    data = Data(x=x, pos=torch.tensor(atoms.positions, dtype=torch.float32, device=device),
                cell=torch.tensor(np.asarray(atoms.cell), dtype=torch.float32, device=device), pbc=atoms.pbc, numbers=atoms.numbers)
    pos_traj = []
    for i in range(1, steps + 1):
        data = ase_graph_gpu(data, cutoff=large_cutoff)
        disp = model(data)
        data.pos = data.pos - disp
        pos_traj.append(data.pos.detach().cpu().numpy())
        if i % max(1, steps // 10) == 0:
            print(f"  polish step {i}/{steps}")
    return pos_traj


def ase_graph_gpu(data: Data, cutoff: float) -> Data:
    pos_cpu = data.pos.detach().cpu().numpy()
    cell_cpu = data.cell.detach().cpu().numpy() if torch.is_tensor(data.cell) else np.asarray(data.cell)
    i, j, D = primitive_neighbor_list("ijD", cutoff=cutoff, pbc=data.pbc, cell=cell_cpu, positions=pos_cpu, numbers=data.numbers)
    data.edge_index = torch.from_numpy(np.stack((i, j))).long().to(device)
    data.edge_attr = torch.from_numpy(D).float().to(device)  # cast to float32 BEFORE .to(device) -- MPS rejects float64 even transiently
    return data


GEN_STEPS_NOISY = 2900  # DM2's own demo default
GEN_STEPS_POLISH = 100  # DM2's own demo default
GEN_MAX_SIGMA = 1.0

start_atoms = ideal_atoms_list[0]
print("Generating SiO2 crystal via DM2's own annealed denoising...")
pos_traj = denoise_with_noise(start_atoms, model, steps=GEN_STEPS_NOISY, max_sigma=GEN_MAX_SIGMA, cutoff=CUTOFF, large_cutoff=LARGE_CUTOFF)
polished = denoise_polish(
    type("A", (), {"positions": pos_traj[-1], "numbers": start_atoms.numbers, "cell": start_atoms.cell, "pbc": start_atoms.pbc})(),
    model, steps=GEN_STEPS_POLISH, cutoff=CUTOFF, large_cutoff=LARGE_CUTOFF,
)
pos_traj.extend(polished)
print(f"Generation done: {len(pos_traj)} frames.")

box = np.diag(np.asarray(start_atoms.cell))  # (3,), Angstrom, orthorhombic cell
final_pos = pos_traj[-1] - box * np.floor(pos_traj[-1] / box)  # wrap into cell


def bond_and_angle_stats(pos, numbers, box, cutoff=2.2):
    """Nearest-neighbor Si-O bond length + O-Si-O angle, via a DYNAMIC
    (not fixed-topology) nearest-neighbor search -- test26.py's module
    docstring explains why glass needs this (no fixed topology to rely on);
    reused here for the crystal too, since it works generically and avoids
    needing a separate fixed-topology construction (test14.py's approach)
    just for this diagnostic."""
    si_idx = np.where(numbers == 14)[0]
    o_idx = np.where(numbers == 8)[0]
    dr = pos[si_idx, None, :] - pos[None, o_idx, :]
    dr -= box * np.round(dr / box)
    dist = np.linalg.norm(dr, axis=-1)  # (n_si, n_o)
    bonds = []
    angles = []
    for si_local, si_global in enumerate(si_idx):
        nn_o_local = np.argsort(dist[si_local])[:4]  # 4 nearest O -> tetrahedral SiO4
        bonds.extend(dist[si_local, nn_o_local].tolist())
        vecs = dr[si_local, nn_o_local]
        vecs = vecs / np.linalg.norm(vecs, axis=-1, keepdims=True)
        for a in range(4):
            for b in range(a + 1, 4):
                cos_t = np.clip((vecs[a] * vecs[b]).sum(), -1, 1)
                angles.append(np.degrees(np.arccos(cos_t)))
    return np.array(bonds), np.array(angles)


gen_bonds, gen_angles = bond_and_angle_stats(final_pos, np.asarray(start_atoms.numbers), box)
ref_bonds, ref_angles = bond_and_angle_stats(
    start_atoms.positions - box * np.floor(start_atoms.positions / box), np.asarray(start_atoms.numbers), box,
)
print(f"Reference: Si-O bond = {ref_bonds.mean():.3f} +/- {ref_bonds.std():.3f} A, "
      f"O-Si-O angle = {ref_angles.mean():.2f} +/- {ref_angles.std():.2f} deg")
print(f"Generated: Si-O bond = {gen_bonds.mean():.3f} +/- {gen_bonds.std():.3f} A, "
      f"O-Si-O angle = {gen_angles.mean():.2f} +/- {gen_angles.std():.2f} deg")

plt.figure()
plt.hist(ref_bonds, bins=60, density=True, histtype="step", label="reference crystal", linewidth=2)
plt.hist(gen_bonds, bins=60, density=True, histtype="step", label="DM2-generated crystal", linewidth=2)
plt.axvline(1.61, color="gray", linestyle=":")
plt.xlabel("Si-O distance (A)")
plt.legend()
plt.title("SiO2 crystal (DM2 NequIP denoiser) -- Si-O bond length")
save_figure("test45_sio2_crystal_2x2x2_bond_comparison.png")

plt.figure()
plt.hist(ref_angles, bins=60, density=True, histtype="step", label="reference crystal", linewidth=2)
plt.hist(gen_angles, bins=60, density=True, histtype="step", label="DM2-generated crystal", linewidth=2)
plt.axvline(109.47, color="gray", linestyle=":")
plt.xlabel("O-Si-O angle (deg)")
plt.legend()
plt.title("SiO2 crystal (DM2 NequIP denoiser) -- O-Si-O angle")
save_figure("test45_sio2_crystal_2x2x2_angle_comparison.png")


# =============================================================================
# CGMD bridge: extract a Si-only coarse-grained trajectory from the
# generated denoising trajectory, in the SAME format this repo's EXISTING
# CG-SiO2 pipeline (src/scoremd, data/silica_cg/*) already consumes -- see
# module docstring. Mirrors scripts/lammps_traj_to_cg.py's own convention
# exactly: Si-only, wrapped into the cell, Angstrom -> nm.
# =============================================================================

numbers_np = np.asarray(start_atoms.numbers)
si_mask = numbers_np == 14
box_ang = box  # (3,), Angstrom

cg_frames = []
for pos in pos_traj:
    wrapped = pos - box_ang * np.floor(pos / box_ang)
    cg_frames.append(wrapped[si_mask])
cg_positions_ang = np.stack(cg_frames, axis=0)  # (n_frames, n_si, 3)
cg_positions_nm = (cg_positions_ang / 10.0).astype(np.float32)  # Angstrom -> nm, matches lammps_traj_to_cg.py
cg_cell_nm = (box_ang / 10.0).astype(np.float32)

CG_OUT_DIR = TOY_MODEL_DIR / "results" / "crystal_generated_test45_2x2x2"
CG_OUT_DIR.mkdir(parents=True, exist_ok=True)
np.save(CG_OUT_DIR / "positions.npy", cg_positions_nm)
np.savez(
    CG_OUT_DIR / "manifest.npz",
    positions=cg_positions_nm,
    cell_nm=cg_cell_nm,
    labels=np.full(cg_positions_nm.shape[0], 11, dtype=np.int32),  # 11 = "DM2-generated crystal, 2x2x2 supercell fix" -- distinct from label=5 (broken, 192-atom, LARGE_CUTOFF=10.0) and label=10 (192-atom, LARGE_CUTOFF=6.5, also insufficient)
    source="DM2 NequIP denoiser (test33.py) -- annealed denoising trajectory of a beta-cristobalite SiO2 crystal, Si-only CG extraction",
)
print(f"Wrote CGMD bridge dataset: {CG_OUT_DIR}/positions.npy  shape {cg_positions_nm.shape}  "
      f"cell {cg_cell_nm} nm  (ready for src/scoremd's SilicaCGDataset)")

print(f"Total wall-clock: {(time.perf_counter() - run_started_at) / 60:.1f} min")
