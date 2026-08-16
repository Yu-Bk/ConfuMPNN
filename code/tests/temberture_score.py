"""TemBERTureTm 热稳打分：3 replica 平均熔解温度。"""
import csv
import sys

sys.path.insert(0, "/data/nfs/IC/baokun_yu/ConfuMPNN/TemBERTure/temBERTure")
from temBERTure import TemBERTure  # noqa: E402

TM = "/data/nfs/IC/baokun_yu/ConfuMPNN/TemBERTure/temBERTure/temBERTure_TM"


def read_fa(path):
    seqs = []
    name = None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            name = line[1:]
        elif line:
            seqs.append((name, line))
    return seqs


replicas = []
for r in ["replica1", "replica2", "replica3"]:
    print(f"加载 {r}...", flush=True)
    m = TemBERTure(adapter_path=f"{TM}/{r}/", device="cpu",
                   batch_size=16, task="regression")
    replicas.append(m)

seqs = read_fa("/tmp/score_input.fa")
rows = []
for name, seq in seqs:
    tms = [float(m.predict(seq)[0]) for m in replicas]
    mean_tm = sum(tms) / len(tms)
    rows.append([name, len(seq), round(mean_tm, 2), tms])
    print(f"{name:<12} Tm={mean_tm:.2f}  replicas={[round(t,1) for t in tms]}", flush=True)

with open("/tmp/tm_scores.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["name", "length", "mean_tm", "tm_replicas"])
    w.writerows(rows)
print("已写 /tmp/tm_scores.csv")
