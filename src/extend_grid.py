"""Правило расширения сетки (одинаковое для всех): если лучший lr на краю сетки, добавляется ещё один шаг ×3 (или ×1/3), 3 seed."""
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment as E

def main():
    a = pd.read_csv(E.RES / "A_lr_tuning.csv")
    jobs = []
    for o, g in a.groupby("opt"):
        m = g.groupby("lr").val_f1.mean()
        lrs = sorted(m.index)
        if m.idxmax() == lrs[-1]:
            jobs += [{"phase": "A_ext", "opt": o, "lr": lrs[-1] * 3, "seed": s, "noise_probe": False} for s in E.TUNE_SEEDS]
        elif m.idxmax() == lrs[0]:
            jobs += [{"phase": "A_ext", "opt": o, "lr": lrs[0] / 3, "seed": s, "noise_probe": False} for s in E.TUNE_SEEDS]
    print("extension jobs:", sorted({(j["opt"], j["lr"]) for j in jobs}))
    ext = E.save(E.pmap(jobs), "A_lr_tuning_ext")
    allr = pd.concat([a, ext])
    allr.to_csv(E.RES / "A_lr_tuning_all.csv", index=False)
    best = allr.groupby(["opt", "lr"]).val_f1.mean().reset_index().sort_values("val_f1").groupby("opt").tail(1)
    json.dump(dict(zip(best.opt, best.lr)), open(E.RES / "best_lr.json", "w"), indent=2)
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
