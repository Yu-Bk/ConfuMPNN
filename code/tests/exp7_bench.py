"""exp7 speed benchmark: protein(MoMPNN v12.2) & ligand(LigandMPNN v14) on cuda:6."""
import sys, time, torch
from pathlib import Path
from data_utils import featurize, parse_PDB
from src.condition_embedding import make_condition_vector
from src.conditioned_sampler import conditioned_sample
from src.differentiable_charge import net_charge
from run_guided import load_model, load_condition_encoder, seq_to_string

DEV = torch.device("cuda:6")
def bench(tag, pdb, weights, enc, cal, use_ligand, n=12):
    pH=7.4
    protein_dict, *_ = parse_PDB(str(pdb), device="cpu")
    L = protein_dict["X"].shape[0]
    native = seq_to_string(protein_dict["S"].reshape(-1).cpu().numpy())
    qnat = float(net_charge(native, pH))
    protein_dict["chain_mask"] = torch.ones(L, dtype=torch.int32)
    fd = featurize(protein_dict, cutoff_for_score=8.0,
                   model_type="ligand_mpnn" if use_ligand else "protein_mpnn",
                   use_atom_context=use_ligand,
                   number_of_ligand_atoms=(25 if use_ligand else 0))
    fd["batch_size"]=1; fd["temperature"]=0.3; fd["bias"]=torch.zeros(1,L,21)
    model=load_model(weights, DEV, model_type="auto")
    en=load_condition_encoder(enc, DEV)
    tgt=float(round(qnat))
    # calibrate
    cal_slope, cal_off, mode, lab = None,None,None,None
    try:
        from run_guided import load_calibration
        cal_slope, cal_off, mode, lab = load_calibration(cal, pdb.stem)
        if cal_slope is None:
            cal_slope, cal_off, mode, lab = load_calibration(cal, pdb.stem, force_global=True)
    except Exception as e:
        print(' cal err', e)
    tgt_eff = float((tgt-cal_off)/cal_slope) if cal_slope else float(tgt)
    t0=time.time()
    for k in range(n):
        torch.manual_seed(1000+k)
        fd["randn"]=torch.randn(1,L)
        cv=make_condition_vector(pH, net_charge=tgt_eff)
        out=conditioned_sample(model, en, fd, cv, device=DEV)
        s=seq_to_string(out["S"][0].cpu().numpy())
        _=float(net_charge(s,pH))
    dt=time.time()-t0
    print(f"[{tag}] L={L} qnat={qnat:+.2f} cal={mode} {n} seq in {dt:.1f}s -> {dt/n*1000:.0f} ms/seq", flush=True)
    return dt/n*1000

if __name__=="__main__":
    from pathlib import Path
    ROOT=Path("/data/nfs/IC/baokun_yu/ConfuMPNN")
    W_PROT=str(ROOT/"MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt")
    ENC_PROT=str(ROOT/"output/finetune_v12_2/finetune_epoch030.pt")
    CAL_PROT=str(ROOT/"output/charge_calibration_v12_2.json")
    W_LIG=str(ROOT/"LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt")
    ENC_LIG=str(ROOT/"output/finetune_ligand_v14_rna/finetune_epoch050.pt")
    CAL_LIG=str(ROOT/"output/charge_calibration_v14_ligand_clean.json")
    print("== prot 1BJ4 (L470 slow case) ==", flush=True)
    bench("prot-1BJ4", ROOT/"data/validation_pdbs/1BJ4.pdb", W_PROT, ENC_PROT, CAL_PROT, False, n=8)
    print("== lig 2FEO (L221) ==", flush=True)
    bench("lig-2FEO", ROOT/"data/validation_pdbs/2FEO.pdb", W_LIG, ENC_LIG, CAL_LIG, True, n=8)
