"""exp_prot_control smoke：测 cuda:2 上三种 route 的吞吐与复现性（临时脚本）。"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch

_CODE = Path("/data/nfs/IC/baokun_yu/ConfuMPNN/code")
sys.path.insert(0, str(_CODE))
sys.path.insert(0, str(_CODE.parent / "LigandMPNN"))

from data_utils import featurize, parse_PDB
from src.condition_embedding import make_condition_vector
from src.conditioned_sampler import conditioned_sample
from src.differentiable_charge import net_charge
from src.guided_sampler import GuidedSampler
from src.charge_lookahead import make_dynamic_callback
from run_guided import load_model, load_condition_encoder, seq_to_string, load_calibration

DEV = torch.device("cuda:2")
W = "/data/nfs/IC/baokun_yu/ConfuMPNN/MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt"
ENC = "/data/nfs/IC/baokun_yu/ConfuMPNN/output/finetune_v12_2/finetune_epoch030.pt"
CAL = "/data/nfs/IC/baokun_yu/ConfuMPNN/output/charge_calibration_v12_2.json"
PDB = "/data/nfs/IC/baokun_yu/ConfuMPNN/data/validation_pdbs/1AZM.pdb"
pH = 7.4

def featurize_pdb(path):
    protein_dict, *_ = parse_PDB(path, device="cpu")
    L = protein_dict["X"].shape[0]
    native = seq_to_string(protein_dict["S"].reshape(-1).cpu().numpy())
    protein_dict["chain_mask"] = torch.ones(L, dtype=torch.int32)
    fd = featurize(protein_dict, cutoff_for_score=8.0, model_type="protein_mpnn",
                   use_atom_context=False, number_of_ligand_atoms=0)
    fd["batch_size"] = 1; fd["temperature"] = 0.3
    fd["bias"] = torch.zeros(1, L, 21)
    return fd, L, native, float(net_charge(native, pH))

fd, L, native, qnat = featurize_pdb(PDB)
tgt_native = int(round(qnat))
print(f"1AZM L={L} native_q={qnat:+.2f} round={tgt_native}")

model = load_model(W, DEV, model_type="auto")
enc = load_condition_encoder(ENC, DEV)
slope, off, mode, _ = load_calibration(CAL, "1AZM")
print("calib 1AZM:", mode, slope, off)

# ---- route A (bare) timing, 20 seq ----
torch.manual_seed(0)
t0 = time.time()
for k in range(20):
    torch.manual_seed(424242 + k)
    fd["randn"] = torch.randn(1, L)
    cond_vec = make_condition_vector(pH, net_charge=tgt_native)  # unused when enc=None
    out = conditioned_sample(model, None, fd, cond_vec, device=DEV)
    s = seq_to_string(out["S"][0].cpu().numpy())
    q = float(net_charge(s, pH))
    if k < 3: print(f"  A[{k}] q={q:+.2f}")
print(f"route A: 20 seq in {time.time()-t0:.2f}s -> {20/(time.time()-t0)*1000:.0f} ms/seq")

# ---- route B (conditioned) timing, 20 seq ----
t0 = time.time()
tgt_eff = (tgt_native - off) / slope
for k in range(20):
    torch.manual_seed(424242 + k)
    fd["randn"] = torch.randn(1, L)
    cond_vec = make_condition_vector(pH, net_charge=tgt_eff)
    out = conditioned_sample(model, enc, fd, cond_vec, device=DEV)
    s = seq_to_string(out["S"][0].cpu().numpy())
    q = float(net_charge(s, pH))
    if k < 3: print(f"  B[{k}] q={q:+.2f}")
print(f"route B: 20 seq in {time.time()-t0:.2f}s -> {20/(time.time()-t0)*1000:.0f} ms/seq")

# ---- route C (bias) timing, 5 seq ----
sampler = GuidedSampler(model, device=DEV)
cb = make_dynamic_callback(pH=pH, target_charge=tgt_native, structure_filter=None, strength=0.5)
t0 = time.time()
qs = []
for k in range(5):
    torch.manual_seed(424242 + k)
    fd["randn"] = torch.randn(1, L)
    out = sampler.sample(fd, bias_callback=cb)
    s = seq_to_string(out["S"][0].cpu().numpy())
    q = float(net_charge(s, pH)); qs.append(q)
    if k < 5: print(f"  C[{k}] q={q:+.2f}")
dt = time.time()-t0
print(f"route C: 5 seq in {dt:.2f}s -> {5/dt*1000:.0f} ms/seq, mean_q={np.mean(qs):+.2f}")

# ---- 复现性检查 route B: same seed twice ----
fd["randn"] = torch.randn(1, L); fd["randn"] = torch.randn(1, L)
torch.manual_seed(424242 + 999)
fd["randn"] = torch.randn(1, L)
cond_vec = make_condition_vector(pH, net_charge=tgt_eff)
o1 = conditioned_sample(model, enc, fd, cond_vec, device=DEV)
s1 = seq_to_string(o1["S"][0].cpu().numpy())
torch.manual_seed(424242 + 999)
fd["randn"] = torch.randn(1, L)
o2 = conditioned_sample(model, enc, fd, cond_vec, device=DEV)
s2 = seq_to_string(o2["S"][0].cpu().numpy())
print("repro routeB same seed equal:", s1 == s2)
print("DONE")
